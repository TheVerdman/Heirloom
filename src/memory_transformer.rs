//! Memory-augmented transformer integration built on the same tensor, autograd,
//! module, optimizer, and CUDA paths as the dense reference model.
//!
//! This is a bounded systems prototype: exact lookup is suitable for small
//! fixtures, product-key lookup is an inspectable candidate path, and sparse
//! updates expose selected rows explicitly. It is not a production-scale ANN
//! index or a claim of pretrained model quality.

use crate::amp::{self, AmpBf16OpDecision};
use crate::nn::{
    sample_token, CausalSelfAttention, Embedding, FeedForward, GeneratedTokenStep,
    GenerationFinishReason, GenerationOptions, GenerationOutput, LayerNorm, Linear, LmEvalMetrics,
    Module, SparseAdamWRowsUpdate,
};
use crate::rng::HeirloomRng;
use crate::{DType, Device, Result, Tensor, TensorError};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashSet;

/// Which parameter subsets an optimizer step may update.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum MemoryUpdatePolicy {
    #[default]
    Full,
    MemoryOnly,
    SparseRows,
    Frozen,
}

/// Sparse-memory fine-tuning policy applied to dense and memory parameters.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SmftMode {
    #[default]
    Disabled,
    FreezeDenseUpdateMemory,
    MaskedMemoryRows,
}

/// Memory candidate lookup implementation.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLookupKind {
    #[default]
    Exact,
    ProductKey,
}

/// Configuration and optimizer policy for one memory feed-forward layer.
///
/// The current implementation requires one memory head, positive dimensions,
/// `memory_top_k <= memory_slots`, and square slot counts for product-key
/// lookup. [`MemoryTransformerConfig::validate`] enforces the full contract.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryLayerConfig {
    pub memory_slots: usize,
    pub memory_key_dim: usize,
    pub memory_value_dim: usize,
    pub memory_top_k: usize,
    pub memory_heads: usize,
    #[serde(default)]
    pub memory_lookup: MemoryLookupKind,
    pub shared_memory: bool,
    pub memory_plus: bool,
    pub memory_update_policy: MemoryUpdatePolicy,
    pub smft_mode: SmftMode,
}

/// Complete decoder and sparse-memory layout for [`MemoryTransformerLm`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryTransformerConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_hidden: usize,
    pub memory_layer_indices: Vec<usize>,
    pub memory_slots: usize,
    pub memory_key_dim: usize,
    pub memory_value_dim: usize,
    pub memory_top_k: usize,
    pub memory_heads: usize,
    #[serde(default)]
    pub memory_lookup: MemoryLookupKind,
    pub shared_memory: bool,
    pub memory_plus: bool,
    pub memory_update_policy: MemoryUpdatePolicy,
    pub smft_mode: SmftMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryForwardPrecisionPolicy {
    F32,
    Bf16Activations,
    AmpBf16,
}

/// Per-layer summary of rows selected by the most recent forward pass.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryLayerSelectionReport {
    pub block_index: usize,
    pub selected_row_events: usize,
    pub unique_selected_rows: usize,
    pub selected_rows_device: String,
}

/// Aggregate, serializable memory-selection evidence.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemorySelectionReport {
    pub shared_memory: bool,
    pub memory_lookup: MemoryLookupKind,
    pub configured_memory_layers: usize,
    pub captured_memory_layers: usize,
    pub memory_top_k: usize,
    pub selected_row_events: usize,
    pub unique_selected_rows: usize,
    pub selected_rows_device: Option<String>,
    pub layers: Vec<MemoryLayerSelectionReport>,
}

/// Dense row-access counts used to construct or audit sparse update masks.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryAccessCounts {
    pub memory_slots: usize,
    pub total_events: u64,
    pub unique_rows: usize,
    pub row_counts: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SmftRowScore {
    pub row: usize,
    pub foreground_count: u64,
    pub background_count: u64,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SmftRowMask {
    pub memory_slots: usize,
    pub trainable_rows: Vec<usize>,
    pub frozen_rows: usize,
    pub trainable_fraction: f64,
    pub scores: Vec<SmftRowScore>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SmftProductKeyMaskPolicy {
    ConservativeAllSlots,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SmftProductKeyMaskProjection {
    pub policy: SmftProductKeyMaskPolicy,
    pub side: usize,
    pub value_trainable_rows: usize,
    pub left_trainable_rows: Vec<usize>,
    pub right_trainable_rows: Vec<usize>,
    pub half_key_rows_are_conservative: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SmftAccessReport {
    pub counts: MemoryAccessCounts,
    pub top_rows: Vec<SmftRowScore>,
}

impl MemoryAccessCounts {
    pub fn empty(memory_slots: usize) -> Self {
        Self {
            memory_slots,
            total_events: 0,
            unique_rows: 0,
            row_counts: vec![0; memory_slots],
        }
    }

    pub fn from_rows(memory_slots: usize, rows: impl IntoIterator<Item = i64>) -> Result<Self> {
        let mut counts = Self::empty(memory_slots);
        for row in rows {
            if row < 0 || row as usize >= memory_slots {
                return Err(TensorError::InvalidOperation(format!(
                    "memory access row {row} is out of range for memory_slots {memory_slots}"
                )));
            }
            counts.row_counts[row as usize] += 1;
            counts.total_events += 1;
        }
        counts.unique_rows = counts.row_counts.iter().filter(|count| **count > 0).count();
        Ok(counts)
    }

    pub fn from_row_counts(memory_slots: usize, row_counts: Vec<u64>) -> Result<Self> {
        let counts = Self {
            memory_slots,
            total_events: row_counts.iter().sum(),
            unique_rows: row_counts.iter().filter(|count| **count > 0).count(),
            row_counts,
        };
        counts.validate()?;
        Ok(counts)
    }

    pub fn validate(&self) -> Result<()> {
        if self.row_counts.len() != self.memory_slots {
            return Err(TensorError::Shape(format!(
                "MemoryAccessCounts row_counts length {} does not match memory_slots {}",
                self.row_counts.len(),
                self.memory_slots
            )));
        }
        let total_events = self.row_counts.iter().sum::<u64>();
        if self.total_events != total_events {
            return Err(TensorError::InvalidOperation(format!(
                "MemoryAccessCounts total_events {} does not match row_counts sum {}",
                self.total_events, total_events
            )));
        }
        let unique_rows = self.row_counts.iter().filter(|count| **count > 0).count();
        if self.unique_rows != unique_rows {
            return Err(TensorError::InvalidOperation(format!(
                "MemoryAccessCounts unique_rows {} does not match non-zero row count {}",
                self.unique_rows, unique_rows
            )));
        }
        Ok(())
    }

    pub fn merge_in(&mut self, other: &MemoryAccessCounts) -> Result<()> {
        self.validate()?;
        other.validate()?;
        if self.memory_slots != other.memory_slots {
            return Err(TensorError::Shape(format!(
                "MemoryAccessCounts merge memory_slots mismatch: {} vs {}",
                self.memory_slots, other.memory_slots
            )));
        }
        for (left, right) in self.row_counts.iter_mut().zip(other.row_counts.iter()) {
            *left = left.checked_add(*right).ok_or_else(|| {
                TensorError::InvalidOperation("MemoryAccessCounts merge overflow".to_string())
            })?;
        }
        self.total_events = self.row_counts.iter().sum();
        self.unique_rows = self.row_counts.iter().filter(|count| **count > 0).count();
        Ok(())
    }

    pub fn top_rows(&self, limit: usize) -> Vec<SmftRowScore> {
        let mut rows = self
            .row_counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(row, count)| SmftRowScore {
                row,
                foreground_count: *count,
                background_count: 0,
                score: *count as f64,
            })
            .collect::<Vec<_>>();
        sort_smft_scores(&mut rows);
        rows.truncate(limit);
        rows
    }

    pub fn smft_mask_against(
        &self,
        background: &MemoryAccessCounts,
        trainable_fraction: f64,
        min_rows: usize,
    ) -> Result<SmftRowMask> {
        if self.memory_slots != background.memory_slots {
            return Err(TensorError::Shape(format!(
                "SMFT foreground/background memory_slots mismatch: {} vs {}",
                self.memory_slots, background.memory_slots
            )));
        }
        if !(0.0..=1.0).contains(&trainable_fraction) || !trainable_fraction.is_finite() {
            return Err(TensorError::InvalidOperation(format!(
                "SMFT trainable_fraction must be finite in [0, 1], got {trainable_fraction}"
            )));
        }
        let foreground_total = self.total_events.max(1) as f64;
        let background_total = background.total_events.max(1) as f64;
        let slots = self.memory_slots.max(1) as f64;
        let mut scores = self
            .row_counts
            .iter()
            .zip(background.row_counts.iter())
            .enumerate()
            .filter(|(_, (foreground, _))| **foreground > 0)
            .map(|(row, (foreground, background_count))| {
                let tf = *foreground as f64 / foreground_total;
                let idf = ((background_total + slots) / (*background_count as f64 + 1.0)).ln();
                SmftRowScore {
                    row,
                    foreground_count: *foreground,
                    background_count: *background_count,
                    score: tf * idf,
                }
            })
            .collect::<Vec<_>>();
        sort_smft_scores(&mut scores);
        let requested = ((self.unique_rows as f64) * trainable_fraction).ceil() as usize;
        let trainable_count = requested
            .max(min_rows.min(self.unique_rows))
            .min(scores.len());
        let trainable_rows = scores
            .iter()
            .take(trainable_count)
            .map(|score| score.row)
            .collect::<Vec<_>>();
        Ok(SmftRowMask {
            memory_slots: self.memory_slots,
            trainable_rows,
            frozen_rows: self.memory_slots.saturating_sub(trainable_count),
            trainable_fraction,
            scores,
        })
    }
}

impl SmftRowMask {
    pub fn validate(&self) -> Result<()> {
        if !self.trainable_fraction.is_finite()
            || self.trainable_fraction < 0.0
            || self.trainable_fraction > 1.0
        {
            return Err(TensorError::InvalidOperation(format!(
                "SMFT trainable_fraction must be finite and between 0 and 1, got {}",
                self.trainable_fraction
            )));
        }
        let mut seen = HashSet::new();
        for &row in &self.trainable_rows {
            if row >= self.memory_slots {
                return Err(TensorError::InvalidOperation(format!(
                    "SMFT trainable row {row} is out of range for memory_slots {}",
                    self.memory_slots
                )));
            }
            if !seen.insert(row) {
                return Err(TensorError::InvalidOperation(format!(
                    "SMFT trainable row {row} appears more than once"
                )));
            }
        }
        let expected_frozen_rows = self.memory_slots.saturating_sub(self.trainable_rows.len());
        if self.frozen_rows != expected_frozen_rows {
            return Err(TensorError::InvalidOperation(format!(
                "SMFT frozen_rows {} does not match memory_slots {} minus trainable rows {}",
                self.frozen_rows,
                self.memory_slots,
                self.trainable_rows.len()
            )));
        }
        Ok(())
    }

    pub fn product_key_projection(&self) -> Result<SmftProductKeyMaskProjection> {
        self.validate()?;
        let side = product_key_side(self.memory_slots)?;
        if side * side != self.memory_slots {
            return Err(TensorError::InvalidOperation(format!(
                "product-key SMFT mask projection requires square memory_slots, got {}",
                self.memory_slots
            )));
        }
        let mut trainable_slots = vec![false; self.memory_slots];
        for &row in &self.trainable_rows {
            trainable_slots[row] = true;
        }

        let mut left_trainable_rows = Vec::new();
        for left in 0..side {
            if (0..side).all(|right| trainable_slots[left * side + right]) {
                left_trainable_rows.push(left);
            }
        }

        let mut right_trainable_rows = Vec::new();
        for right in 0..side {
            if (0..side).all(|left| trainable_slots[left * side + right]) {
                right_trainable_rows.push(right);
            }
        }

        Ok(SmftProductKeyMaskProjection {
            policy: SmftProductKeyMaskPolicy::ConservativeAllSlots,
            side,
            value_trainable_rows: self.trainable_rows.len(),
            left_trainable_rows,
            right_trainable_rows,
            half_key_rows_are_conservative: true,
        })
    }

    pub fn to_tensor(&self, device: Device) -> Result<Tensor> {
        self.validate()?;
        let mut mask = vec![false; self.memory_slots];
        for &row in &self.trainable_rows {
            let Some(slot) = mask.get_mut(row) else {
                return Err(TensorError::InvalidOperation(format!(
                    "SMFT trainable row {row} is out of range for memory_slots {}",
                    self.memory_slots
                )));
            };
            *slot = true;
        }
        Tensor::from_bool(mask, &[self.memory_slots], false)?.to_device(device)
    }
}

impl SmftProductKeyMaskProjection {
    pub fn left_tensor(&self, device: Device) -> Result<Tensor> {
        self.mask_tensor(&self.left_trainable_rows, device)
    }

    pub fn right_tensor(&self, device: Device) -> Result<Tensor> {
        self.mask_tensor(&self.right_trainable_rows, device)
    }

    fn mask_tensor(&self, rows: &[usize], device: Device) -> Result<Tensor> {
        let mut mask = vec![false; self.side];
        for &row in rows {
            if row >= self.side {
                return Err(TensorError::InvalidOperation(format!(
                    "product-key SMFT projected row {row} is out of range for side {}",
                    self.side
                )));
            }
            mask[row] = true;
        }
        Tensor::from_bool(mask, &[self.side], false)?.to_device(device)
    }
}

impl MemoryTransformerConfig {
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            block_size: 64,
            n_layers: 32,
            d_model: 64,
            n_heads: 4,
            ff_hidden: 256,
            memory_layer_indices: vec![8, 16, 24],
            memory_slots: 1024,
            memory_key_dim: 32,
            memory_value_dim: 64,
            memory_top_k: 4,
            memory_heads: 1,
            memory_lookup: MemoryLookupKind::Exact,
            shared_memory: true,
            memory_plus: true,
            memory_update_policy: MemoryUpdatePolicy::Full,
            smft_mode: SmftMode::Disabled,
        }
    }

    pub fn memory_layer_config(&self) -> MemoryLayerConfig {
        MemoryLayerConfig {
            memory_slots: self.memory_slots,
            memory_key_dim: self.memory_key_dim,
            memory_value_dim: self.memory_value_dim,
            memory_top_k: self.memory_top_k,
            memory_heads: self.memory_heads,
            memory_lookup: self.memory_lookup.clone(),
            shared_memory: self.shared_memory,
            memory_plus: self.memory_plus,
            memory_update_policy: self.memory_update_policy.clone(),
            smft_mode: self.smft_mode.clone(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0
            || self.block_size == 0
            || self.n_layers == 0
            || self.d_model == 0
            || self.ff_hidden == 0
        {
            return Err(TensorError::InvalidOperation(
                "vocab_size, block_size, n_layers, d_model, and ff_hidden must be non-zero"
                    .to_string(),
            ));
        }
        if self.n_heads == 0 || !self.d_model.is_multiple_of(self.n_heads) {
            return Err(TensorError::Shape(format!(
                "d_model {} must be divisible by n_heads {}",
                self.d_model, self.n_heads
            )));
        }
        if self.memory_slots == 0
            || self.memory_key_dim == 0
            || self.memory_value_dim == 0
            || self.memory_top_k == 0
        {
            return Err(TensorError::InvalidOperation(
                "memory_slots, memory_key_dim, memory_value_dim, and memory_top_k must be non-zero"
                    .to_string(),
            ));
        }
        if self.memory_top_k > self.memory_slots {
            return Err(TensorError::InvalidOperation(format!(
                "memory_top_k {} cannot exceed memory_slots {}",
                self.memory_top_k, self.memory_slots
            )));
        }
        if self.memory_heads != 1 {
            return Err(TensorError::InvalidOperation(format!(
                "memory_heads={} requested, but the first Heirloom memory layer supports one memory head",
                self.memory_heads
            )));
        }
        if self.memory_lookup == MemoryLookupKind::ProductKey {
            if !self.memory_key_dim.is_multiple_of(2) {
                return Err(TensorError::InvalidOperation(format!(
                    "product-key memory lookup requires even memory_key_dim, got {}",
                    self.memory_key_dim
                )));
            }
            let side = product_key_side(self.memory_slots)?;
            if side * side != self.memory_slots {
                return Err(TensorError::InvalidOperation(format!(
                    "product-key memory lookup requires square memory_slots, got {}",
                    self.memory_slots
                )));
            }
        }
        let mut seen = HashSet::new();
        for &index in &self.memory_layer_indices {
            if index >= self.n_layers {
                return Err(TensorError::InvalidOperation(format!(
                    "memory layer index {index} is outside n_layers {}",
                    self.n_layers
                )));
            }
            if !seen.insert(index) {
                return Err(TensorError::InvalidOperation(format!(
                    "duplicate memory layer index {index}"
                )));
            }
        }
        Ok(())
    }

    fn memory_layer_set(&self) -> HashSet<usize> {
        self.memory_layer_indices.iter().copied().collect()
    }
}

#[derive(Clone)]
struct MemoryTables {
    key: Tensor,
    product_key_left: Option<Tensor>,
    product_key_right: Option<Tensor>,
    value: Tensor,
}

impl MemoryTables {
    fn new(config: &MemoryLayerConfig, rng: &mut HeirloomRng) -> Result<Self> {
        let key_stddev = (1.0 / config.memory_key_dim.max(1) as f32).sqrt();
        let product_key_side = if config.memory_lookup == MemoryLookupKind::ProductKey {
            Some(product_key_side(config.memory_slots)?)
        } else {
            None
        };
        let product_key_half_dim = config.memory_key_dim / 2;
        let product_key_stddev = (1.0 / product_key_half_dim.max(1) as f32).sqrt();
        Ok(Self {
            key: rng.normal_tensor(
                &[config.memory_slots, config.memory_key_dim],
                0.0,
                key_stddev,
                config.memory_lookup == MemoryLookupKind::Exact,
            )?,
            product_key_left: product_key_side
                .map(|side| {
                    rng.normal_tensor(&[side, product_key_half_dim], 0.0, product_key_stddev, true)
                })
                .transpose()?,
            product_key_right: product_key_side
                .map(|side| {
                    rng.normal_tensor(&[side, product_key_half_dim], 0.0, product_key_stddev, true)
                })
                .transpose()?,
            value: rng.normal_tensor(
                &[config.memory_slots, config.memory_value_dim],
                0.0,
                0.02,
                true,
            )?,
        })
    }

    fn to_device(&self, device: Device) -> Result<Self> {
        Ok(Self {
            key: self.key.to_device(device)?,
            product_key_left: self
                .product_key_left
                .as_ref()
                .map(|key| key.to_device(device))
                .transpose()?,
            product_key_right: self
                .product_key_right
                .as_ref()
                .map(|key| key.to_device(device))
                .transpose()?,
            value: self.value.to_device(device)?,
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        if let (Some(left), Some(right)) = (&self.product_key_left, &self.product_key_right) {
            parameters.push(left.clone());
            parameters.push(right.clone());
        } else {
            parameters.push(self.key.clone());
        }
        parameters.push(self.value.clone());
        parameters
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        if let (Some(left), Some(right)) = (&self.product_key_left, &self.product_key_right) {
            out.push((format!("{prefix}product_key_left"), left.clone()));
            out.push((format!("{prefix}product_key_right"), right.clone()));
        } else {
            out.push((format!("{prefix}key"), self.key.clone()));
        }
        out.push((format!("{prefix}value"), self.value.clone()));
        out
    }

    fn product_key_tables(&self) -> Result<(&Tensor, &Tensor)> {
        match (&self.product_key_left, &self.product_key_right) {
            (Some(left), Some(right)) => Ok((left, right)),
            _ => Err(TensorError::InvalidOperation(
                "product-key memory lookup requires product_key_left/product_key_right tables"
                    .to_string(),
            )),
        }
    }
}

/// Sparse top-k memory replacement for a transformer's dense feed-forward path.
///
/// The module records selected rows after forward so the optimizer and evidence
/// reports can use the same routing decisions that produced the output.
pub struct MemoryFeedForward {
    pub config: MemoryLayerConfig,
    query_proj: Linear,
    gate_proj: Option<Linear>,
    out_proj: Linear,
    memory: MemoryTables,
    include_memory_parameters: bool,
    last_selected_indices: RefCell<Option<Vec<i64>>>,
    last_selected_rows: RefCell<Option<Tensor>>,
}

impl MemoryFeedForward {
    /// Constructs validated projections and memory tables on CPU.
    pub fn new(d_model: usize, config: MemoryLayerConfig, rng: &mut HeirloomRng) -> Result<Self> {
        Self::new_with_tables(d_model, config, rng, None, true)
    }

    fn new_with_tables(
        d_model: usize,
        config: MemoryLayerConfig,
        rng: &mut HeirloomRng,
        memory: Option<MemoryTables>,
        include_memory_parameters: bool,
    ) -> Result<Self> {
        if config.memory_heads != 1 {
            return Err(TensorError::InvalidOperation(format!(
                "memory_heads={} requested, but MemoryFeedForward supports one memory head",
                config.memory_heads
            )));
        }
        let memory = match memory {
            Some(memory) => memory,
            None => MemoryTables::new(&config, rng)?,
        };
        let gate_proj = if config.memory_plus {
            Some(Linear::new_with_rng(d_model, config.memory_value_dim, rng)?)
        } else {
            None
        };
        Ok(Self {
            query_proj: Linear::new_with_rng(d_model, config.memory_key_dim, rng)?,
            gate_proj,
            out_proj: Linear::new_with_rng(config.memory_value_dim, d_model, rng)?,
            config,
            memory,
            include_memory_parameters,
            last_selected_indices: RefCell::new(None),
            last_selected_rows: RefCell::new(None),
        })
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        self.to_device_with_shared(device, None)
    }

    fn to_device_with_shared(&self, device: Device, shared: Option<&MemoryTables>) -> Result<Self> {
        Ok(Self {
            config: self.config.clone(),
            query_proj: self.query_proj.to_device(device)?,
            gate_proj: self
                .gate_proj
                .as_ref()
                .map(|gate| gate.to_device(device))
                .transpose()?,
            out_proj: self.out_proj.to_device(device)?,
            memory: match shared {
                Some(memory) => memory.clone(),
                None => self.memory.to_device(device)?,
            },
            include_memory_parameters: self.include_memory_parameters,
            last_selected_indices: RefCell::new(None),
            last_selected_rows: RefCell::new(None),
        })
    }

    pub fn last_selected_indices(&self) -> Option<Vec<i64>> {
        self.last_selected_indices.borrow().clone()
    }

    pub fn last_selected_rows(&self) -> Option<Tensor> {
        self.last_selected_rows.borrow().clone()
    }

    pub fn forward_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let input_shape = input.shape();
        let Some(&last_dim) = input_shape.last() else {
            return Err(TensorError::Shape(
                "MemoryFeedForward expected non-scalar input".to_string(),
            ));
        };
        let outer = input.numel() / last_dim;
        let flat_input = input.reshape(&[outer, last_dim])?;
        let query = self.query_proj.forward(&flat_input)?;
        let indices = match input.device() {
            Device::Cpu => self.select_topk_indices_cpu(&query)?,
            Device::Cuda(_) => self.select_topk_indices_cuda(&query)?,
        };
        *self.last_selected_rows.borrow_mut() = Some(indices.reshape(&[indices.numel()])?);
        let selected_scores = match self.config.memory_lookup {
            MemoryLookupKind::Exact => indices.memory_selected_scores(&query, &self.memory.key)?,
            MemoryLookupKind::ProductKey => {
                let (left_keys, right_keys) = self.memory.product_key_tables()?;
                indices.memory_product_key_selected_scores(&query, left_keys, right_keys)?
            }
        };
        let weights = selected_scores.softmax_dim(1)?;
        let weighted = indices.memory_weighted_value(&weights, &self.memory.value)?;
        let memory_value = if let Some(gate_proj) = &self.gate_proj {
            let gate = gate_proj.forward(&flat_input)?.gelu()?;
            weighted.mul(&gate)?
        } else {
            weighted
        };
        let projected = self.out_proj.forward(&memory_value)?;
        let mut output_shape = input_shape;
        *output_shape.last_mut().ok_or_else(|| {
            TensorError::Shape("MemoryFeedForward expected non-scalar input".to_string())
        })? = last_dim;
        let output = projected.reshape(&output_shape)?;
        let kernel_path = self.lookup_kernel_path(input.device());
        record_memory_amp_decision(prefix, input, kernel_path, None);
        Ok(output)
    }

    pub fn forward_bf16_activations_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        bf16_activation_roundtrip(self.forward_named(input, prefix)?)
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        self.forward_named(input, prefix)
    }

    fn select_topk_indices_cpu(&self, query: &Tensor) -> Result<Tensor> {
        match self.config.memory_lookup {
            MemoryLookupKind::Exact => self.select_exact_topk_indices_cpu(query),
            MemoryLookupKind::ProductKey => self.select_product_key_indices_cpu(query),
        }
    }

    fn select_exact_topk_indices_cpu(&self, query: &Tensor) -> Result<Tensor> {
        if query.device() != Device::Cpu || self.memory.key.device() != Device::Cpu {
            return Err(TensorError::Device(
                "MemoryFeedForward CUDA lookup requires dedicated CUDA kernels; CPU fallback is disabled"
                    .to_string(),
            ));
        }
        let shape = query.shape();
        if shape.len() != 2 || shape[1] != self.config.memory_key_dim {
            return Err(TensorError::Shape(format!(
                "memory query expected [tokens, {}], got {:?}",
                self.config.memory_key_dim, shape
            )));
        }
        let tokens = shape[0];
        let query_data = query.data_f32()?;
        let key_data = self.memory.key.data_f32()?;
        let mut indices = Vec::with_capacity(tokens * self.config.memory_top_k);
        for token in 0..tokens {
            let query_row = &query_data
                [token * self.config.memory_key_dim..(token + 1) * self.config.memory_key_dim];
            let mut scores = Vec::with_capacity(self.config.memory_slots);
            for slot in 0..self.config.memory_slots {
                let key_row = &key_data
                    [slot * self.config.memory_key_dim..(slot + 1) * self.config.memory_key_dim];
                let score = query_row
                    .iter()
                    .zip(key_row.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                scores.push((slot, score));
            }
            scores.sort_by(|(left_index, left_score), (right_index, right_score)| {
                right_score
                    .partial_cmp(left_score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left_index.cmp(right_index))
            });
            indices.extend(
                scores
                    .iter()
                    .take(self.config.memory_top_k)
                    .map(|(slot, _)| *slot as i64),
            );
        }
        *self.last_selected_indices.borrow_mut() = Some(indices.clone());
        Tensor::from_i64(indices, &[tokens, self.config.memory_top_k], false)
    }

    fn select_product_key_indices_cpu(&self, query: &Tensor) -> Result<Tensor> {
        let (left_keys, right_keys) = self.memory.product_key_tables()?;
        if query.device() != Device::Cpu
            || left_keys.device() != Device::Cpu
            || right_keys.device() != Device::Cpu
        {
            return Err(TensorError::Device(
                "MemoryFeedForward product-key CPU lookup requires CPU query and half-key tensors"
                    .to_string(),
            ));
        }
        let shape = query.shape();
        if shape.len() != 2 || shape[1] != self.config.memory_key_dim {
            return Err(TensorError::Shape(format!(
                "memory query expected [tokens, {}], got {:?}",
                self.config.memory_key_dim, shape
            )));
        }
        let side = product_key_side(self.config.memory_slots)?;
        if side * side != self.config.memory_slots {
            return Err(TensorError::InvalidOperation(format!(
                "product-key memory lookup requires square memory_slots, got {}",
                self.config.memory_slots
            )));
        }
        if !self.config.memory_key_dim.is_multiple_of(2) {
            return Err(TensorError::InvalidOperation(format!(
                "product-key memory lookup requires even memory_key_dim, got {}",
                self.config.memory_key_dim
            )));
        }
        let half_dim = self.config.memory_key_dim / 2;
        let tokens = shape[0];
        let query_data = query.data_f32()?;
        let left_key_data = left_keys.data_f32()?;
        let right_key_data = right_keys.data_f32()?;
        let beam = product_key_beam(side, self.config.memory_top_k);
        let mut indices = Vec::with_capacity(tokens * self.config.memory_top_k);
        for token in 0..tokens {
            let query_row = &query_data
                [token * self.config.memory_key_dim..(token + 1) * self.config.memory_key_dim];
            let mut left_scores = Vec::with_capacity(side);
            let mut right_scores = Vec::with_capacity(side);
            for left in 0..side {
                let key_row = &left_key_data[left * half_dim..(left + 1) * half_dim];
                left_scores.push((left, dot(&query_row[..half_dim], key_row)));
            }
            for right in 0..side {
                let key_row = &right_key_data[right * half_dim..(right + 1) * half_dim];
                right_scores.push((right, dot(&query_row[half_dim..], key_row)));
            }
            sort_scores_desc_index_asc(&mut left_scores);
            sort_scores_desc_index_asc(&mut right_scores);

            let mut candidates = Vec::with_capacity(beam * beam);
            for &(left, left_score) in left_scores.iter().take(beam) {
                for &(right, right_score) in right_scores.iter().take(beam) {
                    let slot = left * side + right;
                    candidates.push((slot, left_score + right_score));
                }
            }
            candidates.sort_by(|(left_index, left_score), (right_index, right_score)| {
                right_score
                    .partial_cmp(left_score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left_index.cmp(right_index))
            });
            candidates.dedup_by_key(|(slot, _)| *slot);
            indices.extend(
                candidates
                    .iter()
                    .take(self.config.memory_top_k)
                    .map(|(slot, _)| *slot as i64),
            );
        }
        *self.last_selected_indices.borrow_mut() = Some(indices.clone());
        Tensor::from_i64(indices, &[tokens, self.config.memory_top_k], false)
    }

    fn select_topk_indices_cuda(&self, query: &Tensor) -> Result<Tensor> {
        if !matches!(query.device(), Device::Cuda(_))
            || !matches!(self.memory.key.device(), Device::Cuda(_))
        {
            return Err(TensorError::Device(
                "select_topk_indices_cuda expected query and memory keys on CUDA".to_string(),
            ));
        }
        let shape = query.shape();
        if shape.len() != 2 || shape[1] != self.config.memory_key_dim {
            return Err(TensorError::Shape(format!(
                "memory query expected [tokens, {}], got {:?}",
                self.config.memory_key_dim, shape
            )));
        }
        let indices = match self.config.memory_lookup {
            MemoryLookupKind::Exact => {
                let key_shape = self.memory.key.shape();
                if key_shape != vec![self.config.memory_slots, self.config.memory_key_dim] {
                    return Err(TensorError::Shape(format!(
                        "memory key table expected [{}, {}], got {:?}",
                        self.config.memory_slots, self.config.memory_key_dim, key_shape
                    )));
                }
                let scores = query.matmul(&self.memory.key.transpose()?)?;
                scores.topk_indices_dim1(self.config.memory_top_k)?
            }
            MemoryLookupKind::ProductKey => {
                let (left_keys, right_keys) = self.memory.product_key_tables()?;
                let side = product_key_side(self.config.memory_slots)?;
                let half_dim = self.config.memory_key_dim / 2;
                if left_keys.shape() != vec![side, half_dim]
                    || right_keys.shape() != vec![side, half_dim]
                {
                    return Err(TensorError::Shape(format!(
                        "product-key half tables expected [{side}, {half_dim}], got left={:?} right={:?}",
                        left_keys.shape(),
                        right_keys.shape()
                    )));
                }
                let query_left = query.narrow(1, 0, half_dim)?;
                let query_right = query.narrow(1, half_dim, half_dim)?;
                let left_scores = query_left.matmul(&left_keys.transpose()?)?;
                let right_scores = query_right.matmul(&right_keys.transpose()?)?;
                left_scores.cuda_memory_product_key_topk_indices_from_side_scores(
                    &right_scores,
                    self.config.memory_slots,
                    self.config.memory_key_dim,
                    self.config.memory_top_k,
                    product_key_beam(side, self.config.memory_top_k),
                )?
            }
        };
        *self.last_selected_indices.borrow_mut() = None;
        Ok(indices)
    }

    fn lookup_kernel_path(&self, device: Device) -> &'static str {
        match (device, &self.config.memory_lookup) {
            (Device::Cpu, MemoryLookupKind::Exact) => "cpu_exact_topk_memory_lookup",
            (Device::Cpu, MemoryLookupKind::ProductKey) => {
                "cpu_product_key_candidate_topk_memory_lookup"
            }
            (Device::Cuda(_), MemoryLookupKind::Exact) => "cuda_exact_topk_memory_lookup",
            (Device::Cuda(_), MemoryLookupKind::ProductKey) => {
                "cuda_product_key_candidate_topk_memory_lookup"
            }
        }
    }
}

impl Module for MemoryFeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_named(input, "memory")
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        if self.include_memory_parameters {
            parameters.extend(self.memory.parameters());
        }
        parameters.extend(self.query_proj.parameters());
        if let Some(gate) = &self.gate_proj {
            parameters.extend(gate.parameters());
        }
        parameters.extend(self.out_proj.parameters());
        parameters
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        if self.include_memory_parameters {
            out.extend(self.memory.named_parameters(&format!("{prefix}memory.")));
        }
        out.extend(
            self.query_proj
                .named_parameters(&format!("{prefix}query_proj.")),
        );
        if let Some(gate) = &self.gate_proj {
            out.extend(gate.named_parameters(&format!("{prefix}gate_proj.")));
        }
        out.extend(
            self.out_proj
                .named_parameters(&format!("{prefix}out_proj.")),
        );
        out
    }
}

enum MemoryBlockFeedForward {
    Dense(FeedForward),
    Memory(MemoryFeedForward),
}

impl MemoryBlockFeedForward {
    fn to_device(&self, device: Device, shared: Option<&MemoryTables>) -> Result<Self> {
        match self {
            Self::Dense(feed_forward) => Ok(Self::Dense(feed_forward.to_device(device)?)),
            Self::Memory(memory) => Ok(Self::Memory(memory.to_device_with_shared(device, shared)?)),
        }
    }

    fn forward_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        match self {
            Self::Dense(feed_forward) => feed_forward.forward(input),
            Self::Memory(memory) => memory.forward_named(input, prefix),
        }
    }

    fn forward_bf16_activations_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        match self {
            Self::Dense(feed_forward) => feed_forward.forward_bf16_activations(input),
            Self::Memory(memory) => memory.forward_bf16_activations_named(input, prefix),
        }
    }

    fn forward_amp_bf16_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        match self {
            Self::Dense(feed_forward) => feed_forward.forward_amp_bf16_named(input, prefix),
            Self::Memory(memory) => memory.forward_amp_bf16_named(input, prefix),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        match self {
            Self::Dense(feed_forward) => feed_forward.parameters(),
            Self::Memory(memory) => memory.parameters(),
        }
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            Self::Dense(feed_forward) => feed_forward.named_parameters(&format!("{prefix}dense.")),
            Self::Memory(memory) => memory.named_parameters(&format!("{prefix}memory.")),
        }
    }

    fn memory_selected_rows(&self) -> Vec<Tensor> {
        match self {
            Self::Dense(_) => Vec::new(),
            Self::Memory(memory) => memory.last_selected_rows().into_iter().collect(),
        }
    }
}

/// Transformer block containing either dense or memory feed-forward routing.
pub struct MemoryTransformerBlock {
    pub index: usize,
    ln1: LayerNorm,
    attention: CausalSelfAttention,
    ln2: LayerNorm,
    feed_forward: MemoryBlockFeedForward,
}

impl MemoryTransformerBlock {
    fn new(
        index: usize,
        config: &MemoryTransformerConfig,
        rng: &mut HeirloomRng,
        shared_memory: Option<MemoryTables>,
    ) -> Result<Self> {
        let feed_forward = if config.memory_layer_indices.contains(&index) {
            MemoryBlockFeedForward::Memory(MemoryFeedForward::new_with_tables(
                config.d_model,
                config.memory_layer_config(),
                rng,
                shared_memory,
                !config.shared_memory,
            )?)
        } else {
            MemoryBlockFeedForward::Dense(FeedForward::new(config.d_model, config.ff_hidden, rng)?)
        };
        Ok(Self {
            index,
            ln1: LayerNorm::new(config.d_model)?,
            attention: CausalSelfAttention::new(config.d_model, config.n_heads, rng)?,
            ln2: LayerNorm::new(config.d_model)?,
            feed_forward,
        })
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        self.to_device_with_shared(device, None)
    }

    fn to_device_with_shared(&self, device: Device, shared: Option<&MemoryTables>) -> Result<Self> {
        Ok(Self {
            index: self.index,
            ln1: self.ln1.to_device(device)?,
            attention: self.attention.to_device(device)?,
            ln2: self.ln2.to_device(device)?,
            feed_forward: self.feed_forward.to_device(device, shared)?,
        })
    }

    pub fn forward_bf16_activations_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let ln1 = bf16_activation_roundtrip(self.ln1.forward(input)?)?;
        let attention = bf16_activation_roundtrip(self.attention.forward_bf16_activations(&ln1)?)?;
        let residual = bf16_activation_roundtrip(input.add(&attention)?)?;
        let ln2 = bf16_activation_roundtrip(self.ln2.forward(&residual)?)?;
        let feed_forward = bf16_activation_roundtrip(
            self.feed_forward
                .forward_bf16_activations_named(&ln2, &format!("{prefix}.feed_forward"))?,
        )?;
        residual.add(&feed_forward)
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let ln1 = bf16_activation_roundtrip(
            self.ln1
                .forward_amp_bf16_named(input, &format!("{prefix}.ln1"))?,
        )?;
        let attention = bf16_activation_roundtrip(
            self.attention
                .forward_amp_bf16_named(&ln1, &format!("{prefix}.attention"))?,
        )?;
        let residual = bf16_activation_roundtrip(input.add(&attention)?)?;
        let ln2 = bf16_activation_roundtrip(
            self.ln2
                .forward_amp_bf16_named(&residual, &format!("{prefix}.ln2"))?,
        )?;
        let feed_forward = bf16_activation_roundtrip(
            self.feed_forward
                .forward_amp_bf16_named(&ln2, &format!("{prefix}.feed_forward"))?,
        )?;
        residual.add(&feed_forward)
    }
}

impl Module for MemoryTransformerBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let attention = self.attention.forward(&self.ln1.forward(input)?)?;
        let residual = input.add(&attention)?;
        let feed_forward = self
            .feed_forward
            .forward_named(&self.ln2.forward(&residual)?, "feed_forward")?;
        residual.add(&feed_forward)
    }

    fn parameters(&self) -> Vec<Tensor> {
        [
            self.ln1.parameters(),
            self.attention.parameters(),
            self.ln2.parameters(),
            self.feed_forward.parameters(),
        ]
        .concat()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.ln1.named_parameters(&format!("{prefix}ln1.")));
        out.extend(
            self.attention
                .named_parameters(&format!("{prefix}attention.")),
        );
        out.extend(self.ln2.named_parameters(&format!("{prefix}ln2.")));
        out.extend(
            self.feed_forward
                .named_parameters(&format!("{prefix}feed_forward.")),
        );
        out
    }
}

/// Decoder-only language model with memory layers at explicitly selected block
/// indices.
///
/// Use [`MemoryTransformerLm::memory_selection_report`] and
/// [`MemoryTransformerLm::memory_sparse_adamw_updates`] to inspect and apply
/// sparse routing rather than inferring it from successful execution.
pub struct MemoryTransformerLm {
    pub config: MemoryTransformerConfig,
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<MemoryTransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    shared_memory: Option<MemoryTables>,
}

impl MemoryTransformerLm {
    /// Validates `config` and constructs a CPU model with deterministic RNG
    /// consumption.
    pub fn new(config: MemoryTransformerConfig, rng: &mut HeirloomRng) -> Result<Self> {
        config.validate()?;
        let memory_layer_set = config.memory_layer_set();
        let shared_memory = if config.shared_memory && !memory_layer_set.is_empty() {
            Some(MemoryTables::new(&config.memory_layer_config(), rng)?)
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(config.n_layers);
        for index in 0..config.n_layers {
            let block_shared = if memory_layer_set.contains(&index) && config.shared_memory {
                shared_memory.clone()
            } else {
                None
            };
            blocks.push(MemoryTransformerBlock::new(
                index,
                &config,
                rng,
                block_shared,
            )?);
        }
        Ok(Self {
            token_embedding: Embedding::new(config.vocab_size, config.d_model, rng)?,
            position_embedding: Embedding::new(config.block_size, config.d_model, rng)?,
            blocks,
            ln_f: LayerNorm::new(config.d_model)?,
            lm_head: Linear::new_with_rng(config.d_model, config.vocab_size, rng)?,
            shared_memory,
            config,
        })
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        let shared_memory = self
            .shared_memory
            .as_ref()
            .map(|memory| memory.to_device(device))
            .transpose()?;
        let blocks = self
            .blocks
            .iter()
            .map(|block| block.to_device_with_shared(device, shared_memory.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.to_device(device)?,
            position_embedding: self.position_embedding.to_device(device)?,
            blocks,
            ln_f: self.ln_f.to_device(device)?,
            lm_head: self.lm_head.to_device(device)?,
            shared_memory,
        })
    }

    pub fn device(&self) -> Device {
        self.token_embedding.weight().device()
    }

    pub fn blocks_len(&self) -> usize {
        self.blocks.len()
    }

    pub fn memory_sparse_adamw_updates(&self) -> Result<Vec<SparseAdamWRowsUpdate>> {
        if self.config.memory_update_policy == MemoryUpdatePolicy::Frozen
            || self.config.memory_layer_indices.is_empty()
        {
            return Ok(Vec::new());
        }
        let named = self.named_parameters("");
        if self.config.shared_memory {
            let selected_rows = self
                .blocks
                .iter()
                .flat_map(|block| block.feed_forward.memory_selected_rows())
                .collect::<Vec<_>>();
            if selected_rows.is_empty() {
                return Ok(Vec::new());
            }
            let selected_rows = Tensor::concat_i64_flat(&selected_rows)?;
            if self.config.memory_lookup == MemoryLookupKind::ProductKey {
                let side = product_key_side(self.config.memory_slots)?;
                let half_dim = self.config.memory_key_dim / 2;
                let (left_rows, right_rows) = selected_rows.memory_product_key_half_rows(side)?;
                return Ok(vec![
                    SparseAdamWRowsUpdate {
                        parameter_index: parameter_index_by_name(
                            &named,
                            "shared_memory.product_key_left",
                        )?,
                        selected_rows: left_rows,
                        row_mask: None,
                        rows: side,
                        row_dim: half_dim,
                    },
                    SparseAdamWRowsUpdate {
                        parameter_index: parameter_index_by_name(
                            &named,
                            "shared_memory.product_key_right",
                        )?,
                        selected_rows: right_rows,
                        row_mask: None,
                        rows: side,
                        row_dim: half_dim,
                    },
                    SparseAdamWRowsUpdate {
                        parameter_index: parameter_index_by_name(&named, "shared_memory.value")?,
                        selected_rows,
                        row_mask: None,
                        rows: self.config.memory_slots,
                        row_dim: self.config.memory_value_dim,
                    },
                ]);
            }
            return Ok(vec![
                SparseAdamWRowsUpdate {
                    parameter_index: parameter_index_by_name(&named, "shared_memory.key")?,
                    selected_rows: selected_rows.clone(),
                    row_mask: None,
                    rows: self.config.memory_slots,
                    row_dim: self.config.memory_key_dim,
                },
                SparseAdamWRowsUpdate {
                    parameter_index: parameter_index_by_name(&named, "shared_memory.value")?,
                    selected_rows,
                    row_mask: None,
                    rows: self.config.memory_slots,
                    row_dim: self.config.memory_value_dim,
                },
            ]);
        }

        let mut updates = Vec::new();
        for block in &self.blocks {
            let MemoryBlockFeedForward::Memory(memory) = &block.feed_forward else {
                continue;
            };
            let Some(selected_rows) = memory.last_selected_rows() else {
                continue;
            };
            if self.config.memory_lookup == MemoryLookupKind::ProductKey {
                let side = product_key_side(self.config.memory_slots)?;
                let half_dim = self.config.memory_key_dim / 2;
                let (left_rows, right_rows) = selected_rows.memory_product_key_half_rows(side)?;
                updates.push(SparseAdamWRowsUpdate {
                    parameter_index: parameter_index_by_name(
                        &named,
                        &format!(
                            "block.{}.feed_forward.memory.memory.product_key_left",
                            block.index
                        ),
                    )?,
                    selected_rows: left_rows,
                    row_mask: None,
                    rows: side,
                    row_dim: half_dim,
                });
                updates.push(SparseAdamWRowsUpdate {
                    parameter_index: parameter_index_by_name(
                        &named,
                        &format!(
                            "block.{}.feed_forward.memory.memory.product_key_right",
                            block.index
                        ),
                    )?,
                    selected_rows: right_rows,
                    row_mask: None,
                    rows: side,
                    row_dim: half_dim,
                });
            } else {
                updates.push(SparseAdamWRowsUpdate {
                    parameter_index: parameter_index_by_name(
                        &named,
                        &format!("block.{}.feed_forward.memory.memory.key", block.index),
                    )?,
                    selected_rows: selected_rows.clone(),
                    row_mask: None,
                    rows: self.config.memory_slots,
                    row_dim: self.config.memory_key_dim,
                });
            }
            updates.push(SparseAdamWRowsUpdate {
                parameter_index: parameter_index_by_name(
                    &named,
                    &format!("block.{}.feed_forward.memory.memory.value", block.index),
                )?,
                selected_rows,
                row_mask: None,
                rows: self.config.memory_slots,
                row_dim: self.config.memory_value_dim,
            });
        }
        Ok(updates)
    }

    pub fn memory_sparse_adamw_updates_with_mask(
        &self,
        mask: &SmftRowMask,
    ) -> Result<Vec<SparseAdamWRowsUpdate>> {
        if mask.memory_slots != self.config.memory_slots {
            return Err(TensorError::InvalidOperation(format!(
                "SMFT mask memory_slots {} does not match model memory_slots {}",
                mask.memory_slots, self.config.memory_slots
            )));
        }
        let row_mask = mask.to_tensor(self.device())?;
        let mut updates = self.memory_sparse_adamw_updates()?;
        if self.config.memory_lookup == MemoryLookupKind::ProductKey {
            let projection = mask.product_key_projection()?;
            let left_mask = projection.left_tensor(self.device())?;
            let right_mask = projection.right_tensor(self.device())?;
            let named = self.named_parameters("");
            for update in &mut updates {
                let Some((name, _)) = named.get(update.parameter_index) else {
                    return Err(TensorError::InvalidOperation(format!(
                        "product-key SMFT mask projection found out-of-range parameter index {}",
                        update.parameter_index
                    )));
                };
                if name.ends_with("product_key_left") {
                    update.row_mask = Some(left_mask.clone());
                } else if name.ends_with("product_key_right") {
                    update.row_mask = Some(right_mask.clone());
                } else if name.ends_with("value") {
                    update.row_mask = Some(row_mask.clone());
                } else {
                    return Err(TensorError::InvalidOperation(format!(
                        "product-key SMFT mask projection cannot classify sparse update parameter {name}"
                    )));
                }
            }
            return Ok(updates);
        }
        for update in &mut updates {
            update.row_mask = Some(row_mask.clone());
        }
        Ok(updates)
    }

    pub fn smft_product_key_mask_projection(
        &self,
        mask: &SmftRowMask,
    ) -> Result<Option<SmftProductKeyMaskProjection>> {
        if self.config.memory_lookup == MemoryLookupKind::ProductKey {
            Ok(Some(mask.product_key_projection()?))
        } else {
            Ok(None)
        }
    }

    pub fn memory_table_parameter_indices(&self) -> Result<Vec<usize>> {
        if self.config.memory_layer_indices.is_empty() {
            return Ok(Vec::new());
        }
        let named = self.named_parameters("");
        if self.config.shared_memory {
            if self.config.memory_lookup == MemoryLookupKind::ProductKey {
                return Ok(vec![
                    parameter_index_by_name(&named, "shared_memory.product_key_left")?,
                    parameter_index_by_name(&named, "shared_memory.product_key_right")?,
                    parameter_index_by_name(&named, "shared_memory.value")?,
                ]);
            }
            return Ok(vec![
                parameter_index_by_name(&named, "shared_memory.key")?,
                parameter_index_by_name(&named, "shared_memory.value")?,
            ]);
        }
        let mut indices = Vec::new();
        for block_index in &self.config.memory_layer_indices {
            if self.config.memory_lookup == MemoryLookupKind::ProductKey {
                indices.push(parameter_index_by_name(
                    &named,
                    &format!("block.{block_index}.feed_forward.memory.memory.product_key_left"),
                )?);
                indices.push(parameter_index_by_name(
                    &named,
                    &format!("block.{block_index}.feed_forward.memory.memory.product_key_right"),
                )?);
            } else {
                indices.push(parameter_index_by_name(
                    &named,
                    &format!("block.{block_index}.feed_forward.memory.memory.key"),
                )?);
            }
            indices.push(parameter_index_by_name(
                &named,
                &format!("block.{block_index}.feed_forward.memory.memory.value"),
            )?);
        }
        Ok(indices)
    }

    /// Returns selected-row counts from the most recent memory forward pass.
    pub fn memory_selection_report(&self) -> Result<MemorySelectionReport> {
        let mut layers = Vec::new();
        let mut all_rows = Vec::new();
        let mut selected_rows_device = None;
        for block in &self.blocks {
            let MemoryBlockFeedForward::Memory(memory) = &block.feed_forward else {
                continue;
            };
            let Some(selected_rows) = memory.last_selected_rows() else {
                continue;
            };
            let device = memory_device_label(selected_rows.device());
            selected_rows_device.get_or_insert_with(|| device.clone());
            let rows = selected_rows.data_i64()?;
            let unique_selected_rows = unique_i64_count(&rows);
            all_rows.extend(rows.iter().copied());
            layers.push(MemoryLayerSelectionReport {
                block_index: block.index,
                selected_row_events: selected_rows.numel(),
                unique_selected_rows,
                selected_rows_device: device,
            });
        }
        Ok(MemorySelectionReport {
            shared_memory: self.config.shared_memory,
            memory_lookup: self.config.memory_lookup.clone(),
            configured_memory_layers: self.config.memory_layer_indices.len(),
            captured_memory_layers: layers.len(),
            memory_top_k: self.config.memory_top_k,
            selected_row_events: all_rows.len(),
            unique_selected_rows: unique_i64_count(&all_rows),
            selected_rows_device,
            layers,
        })
    }

    pub fn memory_access_counts(&self) -> Result<MemoryAccessCounts> {
        let mut counts = MemoryAccessCounts::empty(self.config.memory_slots);
        for block in &self.blocks {
            let MemoryBlockFeedForward::Memory(memory) = &block.feed_forward else {
                continue;
            };
            let Some(selected_rows) = memory.last_selected_rows() else {
                continue;
            };
            let layer_counts = match selected_rows.device() {
                Device::Cpu => MemoryAccessCounts::from_rows(
                    self.config.memory_slots,
                    selected_rows.data_i64()?,
                )?,
                Device::Cuda(_) => MemoryAccessCounts::from_row_counts(
                    self.config.memory_slots,
                    selected_rows.cuda_i64_memory_access_counts(self.config.memory_slots)?,
                )?,
            };
            counts.merge_in(&layer_counts)?;
        }
        Ok(counts)
    }

    pub fn smft_access_report(&self, top_rows: usize) -> Result<SmftAccessReport> {
        let counts = self.memory_access_counts()?;
        Ok(SmftAccessReport {
            top_rows: counts.top_rows(top_rows),
            counts,
        })
    }

    pub fn smft_row_mask_against(
        &self,
        background: &MemoryAccessCounts,
        trainable_fraction: f64,
        min_rows: usize,
    ) -> Result<SmftRowMask> {
        self.memory_access_counts()?
            .smft_mask_against(background, trainable_fraction, min_rows)
    }

    /// Computes autoregressive cross-entropy through the standard autograd path.
    pub fn loss(&self, input: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward(input)?;
        let shape = logits.shape();
        let flat_logits = logits.reshape(&[shape[0] * shape[1], shape[2]])?;
        let flat_targets = targets.reshape(&[shape[0] * shape[1]])?;
        flat_logits.cross_entropy_for_logits_tensor(&flat_targets)
    }

    pub fn forward_bf16_activations(&self, input: &Tensor) -> Result<Tensor> {
        let shape = self.validate_token_input(input)?;
        let (batch, time) = (shape[0], shape[1]);
        let token = bf16_activation_roundtrip(self.token_embedding.forward(input)?)?;
        let position_ids = position_ids_tensor(batch, time, input.device())?;
        let position = bf16_activation_roundtrip(self.position_embedding.forward(&position_ids)?)?;
        let mut hidden = bf16_activation_roundtrip(token.add(&position)?)?;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = bf16_activation_roundtrip(
                block.forward_bf16_activations_named(&hidden, &format!("block.{index}"))?,
            )?;
        }
        let hidden = bf16_activation_roundtrip(self.ln_f.forward(&hidden)?)?;
        self.lm_head.forward(&hidden)
    }

    pub fn loss_bf16_activations(&self, input: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward_bf16_activations(input)?;
        let shape = logits.shape();
        let flat_logits = logits.reshape(&[shape[0] * shape[1], shape[2]])?;
        let flat_targets = targets.reshape(&[shape[0] * shape[1]])?;
        flat_logits.cross_entropy_for_logits_tensor(&flat_targets)
    }

    pub fn forward_amp_bf16(&self, input: &Tensor) -> Result<Tensor> {
        let shape = self.validate_token_input(input)?;
        let (batch, time) = (shape[0], shape[1]);
        let token = bf16_activation_roundtrip(
            self.token_embedding
                .forward_amp_bf16_named(input, "memory_transformer.token_embedding")?,
        )?;
        let position_ids = position_ids_tensor(batch, time, input.device())?;
        let position = bf16_activation_roundtrip(
            self.position_embedding
                .forward_amp_bf16_named(&position_ids, "memory_transformer.position_embedding")?,
        )?;
        let mut hidden = bf16_activation_roundtrip(token.add(&position)?)?;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden =
                bf16_activation_roundtrip(block.forward_amp_bf16_named(
                    &hidden,
                    &format!("memory_transformer.block.{index}"),
                )?)?;
        }
        let hidden = bf16_activation_roundtrip(
            self.ln_f
                .forward_amp_bf16_named(&hidden, "memory_transformer.ln_f")?,
        )?;
        self.lm_head
            .forward_amp_bf16_named(&hidden, "memory_transformer.lm_head")
    }

    pub fn loss_amp_bf16(&self, input: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward_amp_bf16(input)?;
        let shape = logits.shape();
        let flat_logits = logits.reshape(&[shape[0] * shape[1], shape[2]])?;
        let flat_targets = targets.reshape(&[shape[0] * shape[1]])?;
        flat_logits.cross_entropy_for_logits_tensor(&flat_targets)
    }

    pub fn generate_greedy(
        &self,
        prefix: &[usize],
        max_new_tokens: usize,
        eos_id: usize,
    ) -> Result<Vec<usize>> {
        let mut rng = HeirloomRng::new(0);
        Ok(self
            .generate(
                prefix,
                &GenerationOptions::greedy(max_new_tokens, eos_id),
                &mut rng,
            )?
            .tokens)
    }

    pub fn generate(
        &self,
        prefix: &[usize],
        options: &GenerationOptions,
        rng: &mut HeirloomRng,
    ) -> Result<GenerationOutput> {
        self.generate_with_activation_policy(
            prefix,
            options,
            rng,
            MemoryForwardPrecisionPolicy::F32,
        )
    }

    pub fn generate_bf16_activations(
        &self,
        prefix: &[usize],
        options: &GenerationOptions,
        rng: &mut HeirloomRng,
    ) -> Result<GenerationOutput> {
        self.generate_with_activation_policy(
            prefix,
            options,
            rng,
            MemoryForwardPrecisionPolicy::Bf16Activations,
        )
    }

    pub fn generate_amp_bf16(
        &self,
        prefix: &[usize],
        options: &GenerationOptions,
        rng: &mut HeirloomRng,
    ) -> Result<GenerationOutput> {
        self.generate_with_activation_policy(
            prefix,
            options,
            rng,
            MemoryForwardPrecisionPolicy::AmpBf16,
        )
    }

    fn generate_with_activation_policy(
        &self,
        prefix: &[usize],
        options: &GenerationOptions,
        rng: &mut HeirloomRng,
        precision: MemoryForwardPrecisionPolicy,
    ) -> Result<GenerationOutput> {
        options.validate()?;
        if prefix.is_empty() {
            return Err(TensorError::InvalidOperation(
                "generation prefix must contain at least one token".to_string(),
            ));
        }

        let mut tokens = prefix.to_vec();
        let mut steps = Vec::with_capacity(options.max_new_tokens);
        let mut finish_reason = GenerationFinishReason::MaxNewTokens;
        for _ in 0..options.max_new_tokens {
            let logits = self.next_token_logits_with_activation_policy(&tokens, precision)?;
            let (token_id, probability, log_probability) =
                sample_token(&logits, &tokens, options, rng)?;
            tokens.push(token_id);
            steps.push(GeneratedTokenStep {
                token_id,
                probability,
                log_probability,
            });
            if token_id == options.eos_id {
                finish_reason = GenerationFinishReason::Eos;
                break;
            }
        }
        Ok(GenerationOutput {
            new_token_count: steps.len(),
            tokens,
            finish_reason,
            rng_state: rng.state(),
            steps,
        })
    }

    pub fn evaluate_token_loss(
        &self,
        tokens: &[usize],
        batch_size: usize,
        max_batches: Option<usize>,
    ) -> Result<LmEvalMetrics> {
        self.evaluate_token_loss_with_activation_policy(
            tokens,
            batch_size,
            max_batches,
            MemoryForwardPrecisionPolicy::F32,
        )
    }

    pub fn evaluate_token_loss_bf16_activations(
        &self,
        tokens: &[usize],
        batch_size: usize,
        max_batches: Option<usize>,
    ) -> Result<LmEvalMetrics> {
        self.evaluate_token_loss_with_activation_policy(
            tokens,
            batch_size,
            max_batches,
            MemoryForwardPrecisionPolicy::Bf16Activations,
        )
    }

    pub fn evaluate_token_loss_amp_bf16(
        &self,
        tokens: &[usize],
        batch_size: usize,
        max_batches: Option<usize>,
    ) -> Result<LmEvalMetrics> {
        self.evaluate_token_loss_with_activation_policy(
            tokens,
            batch_size,
            max_batches,
            MemoryForwardPrecisionPolicy::AmpBf16,
        )
    }

    fn evaluate_token_loss_with_activation_policy(
        &self,
        tokens: &[usize],
        batch_size: usize,
        max_batches: Option<usize>,
        precision: MemoryForwardPrecisionPolicy,
    ) -> Result<LmEvalMetrics> {
        if batch_size == 0 {
            return Err(TensorError::InvalidOperation(
                "eval batch_size must be greater than zero".to_string(),
            ));
        }
        if max_batches.is_some_and(|limit| limit == 0) {
            return Err(TensorError::InvalidOperation(
                "max_batches must be greater than zero when set".to_string(),
            ));
        }
        if tokens.len() <= self.config.block_size {
            return Err(TensorError::InvalidOperation(format!(
                "eval needs more than block_size tokens, got {} tokens and block_size {}",
                tokens.len(),
                self.config.block_size
            )));
        }

        let mut offset = 0;
        let mut batches = 0;
        let mut examples = 0;
        let mut predicted_tokens = 0;
        let mut loss_sum = 0.0;
        while offset + self.config.block_size < tokens.len() {
            if max_batches.is_some_and(|limit| batches >= limit) {
                break;
            }
            let mut inputs = Vec::with_capacity(batch_size * self.config.block_size);
            let mut targets = Vec::with_capacity(batch_size * self.config.block_size);
            let mut actual_batch = 0;
            while actual_batch < batch_size && offset + self.config.block_size < tokens.len() {
                for item in 0..self.config.block_size {
                    inputs.push(tokens[offset + item] as i64);
                    targets.push(tokens[offset + item + 1] as i64);
                }
                offset += self.config.block_size;
                actual_batch += 1;
            }
            if actual_batch == 0 {
                break;
            }

            let device = self.device();
            let input = Tensor::from_i64(inputs, &[actual_batch, self.config.block_size], false)?
                .to_device(device)?;
            let target = Tensor::from_i64(targets, &[actual_batch, self.config.block_size], false)?
                .to_device(device)?;
            let loss = crate::no_grad(|| match precision {
                MemoryForwardPrecisionPolicy::F32 => self.loss(&input, &target),
                MemoryForwardPrecisionPolicy::Bf16Activations => {
                    self.loss_bf16_activations(&input, &target)
                }
                MemoryForwardPrecisionPolicy::AmpBf16 => self.loss_amp_bf16(&input, &target),
            })?;
            let weight = actual_batch * self.config.block_size;
            let loss_value = if precision == MemoryForwardPrecisionPolicy::AmpBf16 {
                amp::with_cuda_host_staging_allowed(
                    "amp-bf16 memory eval loss scalar logging",
                    || loss.data()[0] as f64,
                )
            } else {
                loss.data()[0] as f64
            };
            amp::record_amp_bf16_finite_check(
                "memory_eval_loss",
                "loss",
                loss.dtype(),
                loss.device(),
                loss_value,
            )?;
            if !loss_value.is_finite() {
                return Err(TensorError::Autograd(format!(
                    "non-finite memory eval loss: {loss_value}"
                )));
            }
            loss_sum += loss_value * weight as f64;
            predicted_tokens += weight;
            examples += actual_batch;
            batches += 1;
        }
        if predicted_tokens == 0 {
            return Err(TensorError::InvalidOperation(
                "eval produced zero predicted tokens".to_string(),
            ));
        }
        let loss = loss_sum / predicted_tokens as f64;
        Ok(LmEvalMetrics {
            batches,
            examples,
            tokens: predicted_tokens,
            loss,
            perplexity: loss.exp(),
        })
    }

    pub fn next_token_logits(&self, tokens: &[usize]) -> Result<Vec<f64>> {
        self.next_token_logits_with_activation_policy(tokens, MemoryForwardPrecisionPolicy::F32)
    }

    pub fn next_token_logits_bf16_activations(&self, tokens: &[usize]) -> Result<Vec<f64>> {
        self.next_token_logits_with_activation_policy(
            tokens,
            MemoryForwardPrecisionPolicy::Bf16Activations,
        )
    }

    pub fn next_token_logits_amp_bf16(&self, tokens: &[usize]) -> Result<Vec<f64>> {
        self.next_token_logits_with_activation_policy(tokens, MemoryForwardPrecisionPolicy::AmpBf16)
    }

    fn next_token_logits_with_activation_policy(
        &self,
        tokens: &[usize],
        precision: MemoryForwardPrecisionPolicy,
    ) -> Result<Vec<f64>> {
        if tokens.is_empty() {
            return Err(TensorError::InvalidOperation(
                "next_token_logits requires at least one token".to_string(),
            ));
        }
        let start = tokens.len().saturating_sub(self.config.block_size);
        let window = tokens[start..]
            .iter()
            .map(|token| *token as i64)
            .collect::<Vec<_>>();
        let input = Tensor::from_i64(window, &[1, tokens.len() - start], false)?
            .to_device(self.device())?;
        let logits = crate::no_grad(|| match precision {
            MemoryForwardPrecisionPolicy::F32 => self.forward(&input),
            MemoryForwardPrecisionPolicy::Bf16Activations => self.forward_bf16_activations(&input),
            MemoryForwardPrecisionPolicy::AmpBf16 => self.forward_amp_bf16(&input),
        })?;
        let shape = logits.shape();
        let classes = shape[2];
        let data = if precision == MemoryForwardPrecisionPolicy::AmpBf16 {
            amp::with_cuda_host_staging_allowed(
                "amp-bf16 memory generation logits inspection",
                || logits.data_f64(),
            )
        } else {
            logits.data_f64()
        };
        if precision == MemoryForwardPrecisionPolicy::AmpBf16 {
            let logits_are_finite = data.iter().all(|value| value.is_finite());
            let max_abs = if logits_are_finite {
                data.iter().map(|value| value.abs()).fold(0.0, f64::max)
            } else {
                f64::NAN
            };
            amp::record_amp_bf16_finite_check(
                "memory_generation_logits_max_abs",
                "next_token_logits",
                logits.dtype(),
                logits.device(),
                max_abs,
            )?;
            if !logits_are_finite {
                return Err(TensorError::Autograd(
                    "non-finite memory generation logits under amp-bf16".to_string(),
                ));
            }
        }
        let row_start = (shape[1] - 1) * classes;
        Ok(data[row_start..row_start + classes].to_vec())
    }

    fn validate_token_input(&self, input: &Tensor) -> Result<Vec<usize>> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "MemoryTransformerLm expects token input [batch, time], got {:?}",
                shape
            )));
        }
        if shape[1] > self.config.block_size {
            return Err(TensorError::Shape(format!(
                "input time {} exceeds block_size {}",
                shape[1], self.config.block_size
            )));
        }
        Ok(shape)
    }
}

fn product_key_side(memory_slots: usize) -> Result<usize> {
    let side = (memory_slots as f64).sqrt() as usize;
    if side == 0 {
        return Err(TensorError::InvalidOperation(
            "product-key memory lookup requires non-zero memory_slots".to_string(),
        ));
    }
    Ok(side)
}

fn product_key_beam(side: usize, top_k: usize) -> usize {
    side.min(top_k.max(ceil_sqrt(top_k)))
}

fn ceil_sqrt(value: usize) -> usize {
    if value <= 1 {
        return value;
    }
    let mut root = (value as f64).sqrt() as usize;
    while root * root < value {
        root += 1;
    }
    root
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn sort_scores_desc_index_asc(scores: &mut [(usize, f32)]) {
    scores.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_index.cmp(right_index))
    });
}

fn sort_smft_scores(scores: &mut [SmftRowScore]) {
    scores.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.foreground_count.cmp(&left.foreground_count))
            .then_with(|| left.background_count.cmp(&right.background_count))
            .then_with(|| left.row.cmp(&right.row))
    });
}

fn parameter_index_by_name(named: &[(String, Tensor)], name: &str) -> Result<usize> {
    named
        .iter()
        .position(|(candidate, _)| candidate == name)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "memory sparse AdamW could not find parameter named {name}"
            ))
        })
}

impl Module for MemoryTransformerLm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = self.validate_token_input(input)?;
        let (batch, time) = (shape[0], shape[1]);
        let token = self.token_embedding.forward(input)?;
        let position_ids = position_ids_tensor(batch, time, input.device())?;
        let position = self.position_embedding.forward(&position_ids)?;
        let mut hidden = token.add(&position)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let hidden = self.ln_f.forward(&hidden)?;
        self.lm_head.forward(&hidden)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.extend(self.token_embedding.parameters());
        parameters.extend(self.position_embedding.parameters());
        if let Some(memory) = &self.shared_memory {
            parameters.extend(memory.parameters());
        }
        for block in &self.blocks {
            parameters.extend(block.parameters());
        }
        parameters.extend(self.ln_f.parameters());
        parameters.extend(self.lm_head.parameters());
        parameters
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(
            self.token_embedding
                .named_parameters(&format!("{prefix}token_embedding.")),
        );
        out.extend(
            self.position_embedding
                .named_parameters(&format!("{prefix}position_embedding.")),
        );
        if let Some(memory) = &self.shared_memory {
            out.extend(memory.named_parameters(&format!("{prefix}shared_memory.")));
        }
        for (index, block) in self.blocks.iter().enumerate() {
            out.extend(block.named_parameters(&format!("{prefix}block.{index}.")));
        }
        out.extend(self.ln_f.named_parameters(&format!("{prefix}ln_f.")));
        out.extend(self.lm_head.named_parameters(&format!("{prefix}lm_head.")));
        out
    }
}

fn unique_i64_count(values: &[i64]) -> usize {
    values.iter().copied().collect::<HashSet<_>>().len()
}

fn memory_device_label(device: Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(id) => format!("cuda:{id}"),
    }
}

fn bf16_activation_roundtrip(tensor: Tensor) -> Result<Tensor> {
    tensor.to_dtype(DType::BFloat16)?.to_dtype(DType::F32)
}

fn position_ids_tensor(batch: usize, time: usize, device: Device) -> Result<Tensor> {
    let mut position_ids = Vec::with_capacity(batch * time);
    for _ in 0..batch {
        position_ids.extend((0..time).map(|index| index as i64));
    }
    Tensor::from_i64(position_ids, &[batch, time], false)?.to_device(device)
}

fn record_memory_amp_decision(
    prefix: &str,
    input: &Tensor,
    kernel_path: &str,
    fallback_reason: Option<&str>,
) {
    amp::record_amp_bf16_op_decision(AmpBf16OpDecision {
        op: format!("memory_lookup:{prefix}"),
        input_dtype: format!("{:?}", input.dtype()),
        input_device: match input.device() {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(device_id) => format!("cuda:{device_id}"),
        },
        compute_dtype: "f32".to_string(),
        accumulation_dtype: "f32".to_string(),
        output_dtype: "f32".to_string(),
        kernel_path: kernel_path.to_string(),
        tensor_core: false,
        fallback_reason: fallback_reason.map(str::to_string),
        finite_check: "memory_lookup_gradients_are_finite".to_string(),
    });
}
