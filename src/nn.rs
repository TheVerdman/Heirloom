//! Neural-network modules, generation helpers, and optimizers built on
//! [`Tensor`].

use crate::amp::{self, AmpBf16OpDecision, AmpBf16Policy};
use crate::rng::HeirloomRng;
use crate::{npy, DType, Device, Result, Tensor, TensorError};
use heirloom_kernels::cuda::{CudaBuffer, NcclCommunicator};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Minimal module contract used for composition, device movement, state dicts,
/// and optimization.
pub trait Module {
    /// Computes the module output without changing parameter identity.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    /// Returns trainable tensor handles in stable optimizer order.
    fn parameters(&self) -> Vec<Tensor>;
    /// Returns stable checkpoint names paired with the same parameter handles.
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.parameters()
            .into_iter()
            .enumerate()
            .map(|(index, parameter)| (format!("{prefix}param_{index}"), parameter))
            .collect()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AmpBf16TensorCoreCoverage {
    pub linear_totals: AmpBf16LinearCoverageTotals,
    pub linear_modules: Vec<AmpBf16LinearCoverage>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AmpBf16LinearCoverageTotals {
    pub calls: usize,
    pub tensor_core_calls: usize,
    pub fallback_calls: usize,
    pub unsupported_shape_calls: usize,
    pub unsupported_device_calls: usize,
    pub dtype_mismatch_calls: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AmpBf16LinearCoverage {
    pub module: String,
    pub calls: usize,
    pub tensor_core_calls: usize,
    pub fallback_calls: usize,
    pub unsupported_shape_calls: usize,
    pub unsupported_device_calls: usize,
    pub dtype_mismatch_calls: usize,
    pub last_m: usize,
    pub last_k: usize,
    pub last_n: usize,
    pub last_device: String,
    pub last_input_dtype: String,
    pub last_weight_dtype: String,
    pub last_bias_dtype: String,
    pub last_path: String,
}

impl AmpBf16LinearCoverage {
    fn new(module: &str) -> Self {
        Self {
            module: module.to_string(),
            calls: 0,
            tensor_core_calls: 0,
            fallback_calls: 0,
            unsupported_shape_calls: 0,
            unsupported_device_calls: 0,
            dtype_mismatch_calls: 0,
            last_m: 0,
            last_k: 0,
            last_n: 0,
            last_device: String::new(),
            last_input_dtype: String::new(),
            last_weight_dtype: String::new(),
            last_bias_dtype: String::new(),
            last_path: String::new(),
        }
    }
}

thread_local! {
    static AMP_BF16_LINEAR_COVERAGE: RefCell<BTreeMap<String, AmpBf16LinearCoverage>> =
        const { RefCell::new(BTreeMap::new()) };
}

pub fn reset_amp_bf16_tensor_core_coverage() {
    AMP_BF16_LINEAR_COVERAGE.with(|coverage| coverage.borrow_mut().clear());
    amp::reset_amp_bf16_runtime_reports();
}

pub fn amp_bf16_tensor_core_coverage() -> AmpBf16TensorCoreCoverage {
    AMP_BF16_LINEAR_COVERAGE.with(|coverage| {
        let modules: Vec<_> = coverage.borrow().values().cloned().collect();
        let mut totals = AmpBf16LinearCoverageTotals::default();
        for module in &modules {
            totals.calls += module.calls;
            totals.tensor_core_calls += module.tensor_core_calls;
            totals.fallback_calls += module.fallback_calls;
            totals.unsupported_shape_calls += module.unsupported_shape_calls;
            totals.unsupported_device_calls += module.unsupported_device_calls;
            totals.dtype_mismatch_calls += module.dtype_mismatch_calls;
        }
        AmpBf16TensorCoreCoverage {
            linear_totals: totals,
            linear_modules: modules,
        }
    })
}

pub fn amp_bf16_policy() -> AmpBf16Policy {
    AmpBf16Policy::cuda_training()
}

pub fn amp_bf16_op_decisions() -> Vec<AmpBf16OpDecision> {
    amp::amp_bf16_op_decisions()
}

pub fn amp_bf16_cuda_host_staging_events() -> Vec<amp::CudaHostStagingEvent> {
    amp::cuda_host_staging_events()
}

struct AmpBf16LinearRecord<'a> {
    module: &'a str,
    m: usize,
    k: usize,
    n: usize,
    device: Device,
    input_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    tensor_core_device: bool,
    tensor_core_shape: bool,
    used_tensor_core: bool,
}

fn record_amp_bf16_linear_coverage(record: AmpBf16LinearRecord<'_>) {
    let dtype_match = record.input_dtype == DType::F32
        && record.weight_dtype == DType::F32
        && record.bias_dtype == DType::F32;
    let path = if record.used_tensor_core {
        "tensor_core"
    } else if !record.tensor_core_device {
        "fallback_unsupported_device"
    } else if !record.tensor_core_shape {
        "fallback_unsupported_shape"
    } else if !dtype_match {
        "fallback_dtype_mismatch"
    } else {
        "fallback_unknown"
    };
    AMP_BF16_LINEAR_COVERAGE.with(|coverage| {
        let mut coverage = coverage.borrow_mut();
        let entry = coverage
            .entry(record.module.to_string())
            .or_insert_with(|| AmpBf16LinearCoverage::new(record.module));
        entry.calls += 1;
        if record.used_tensor_core {
            entry.tensor_core_calls += 1;
        } else {
            entry.fallback_calls += 1;
        }
        if !record.tensor_core_shape {
            entry.unsupported_shape_calls += 1;
        }
        if !record.tensor_core_device {
            entry.unsupported_device_calls += 1;
        }
        if !dtype_match {
            entry.dtype_mismatch_calls += 1;
        }
        entry.last_m = record.m;
        entry.last_k = record.k;
        entry.last_n = record.n;
        entry.last_device = device_label(record.device);
        entry.last_input_dtype = dtype_label(record.input_dtype);
        entry.last_weight_dtype = dtype_label(record.weight_dtype);
        entry.last_bias_dtype = dtype_label(record.bias_dtype);
        entry.last_path = path.to_string();
    });
}

fn child_module_name(prefix: &str, child: &str) -> String {
    if prefix.is_empty() {
        child.to_string()
    } else {
        format!("{prefix}.{child}")
    }
}

fn device_label(device: Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(device_id) => format!("cuda:{device_id}"),
    }
}

fn dtype_label(dtype: DType) -> String {
    format!("{dtype:?}")
}

fn amp_dtype_label(dtype: DType) -> String {
    amp::dtype_label(dtype)
}

fn amp_device_label(device: Device) -> String {
    amp::device_label(device)
}

fn record_amp_bf16_fp32_op_decision(
    op: impl Into<String>,
    input: &Tensor,
    kernel_path: &str,
    finite_check: &str,
) {
    amp::record_amp_bf16_op_decision(AmpBf16OpDecision {
        op: op.into(),
        input_dtype: amp_dtype_label(input.dtype()),
        input_device: amp_device_label(input.device()),
        compute_dtype: "f32".to_string(),
        accumulation_dtype: "f32".to_string(),
        output_dtype: "f32".to_string(),
        kernel_path: kernel_path.to_string(),
        tensor_core: false,
        fallback_reason: None,
        finite_check: finite_check.to_string(),
    });
}

fn record_amp_bf16_device_fp32_op_decision(
    op: impl Into<String>,
    input: &Tensor,
    cuda_kernel: &str,
    cpu_kernel: &str,
    finite_check: &str,
) {
    let kernel_path = match input.device() {
        Device::Cuda(_) => cuda_kernel,
        Device::Cpu => cpu_kernel,
    };
    record_amp_bf16_fp32_op_decision(op, input, kernel_path, finite_check);
}

fn bf16_activation_roundtrip(tensor: Tensor) -> Result<Tensor> {
    if tensor.dtype() != DType::F32 {
        return Err(TensorError::DType(format!(
            "BF16 activation policy currently expects f32 activations, got {:?}",
            tensor.dtype()
        )));
    }
    if tensor.device() != Device::Cpu {
        return tensor.cuda_f32_bf16_roundtrip();
    }
    tensor.to_dtype(DType::BFloat16)?.to_dtype(DType::F32)
}

fn ensure_amp_bf16_transformer_path_supported() -> Result<()> {
    if heirloom_kernels::cuda::tensor_core_requirement_enabled() {
        return Err(TensorError::Device(
            "HEIRLOOM_REQUIRE_TENSOR_CORES=1 requested a real Tensor Core AMP path, but this \
             projection shape/device is not handled by the BF16 Tensor Core GEMM path."
                .to_string(),
        ));
    }
    Ok(())
}

fn amp_bf16_attention_tensor_core_required() -> bool {
    matches!(
        std::env::var("HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

/// Affine projection with weight layout `[out_features, in_features]`.
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    pub const DEFAULT_SEED: u64 = 0x4845_4952_4C4F_4F4D;

    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        Self::new_with_seed(in_features, out_features, Self::DEFAULT_SEED)
    }

    pub fn new_with_seed(in_features: usize, out_features: usize, seed: u64) -> Result<Self> {
        let mut rng = HeirloomRng::new(seed);
        Self::new_with_rng(in_features, out_features, &mut rng)
    }

    pub fn new_with_rng(
        in_features: usize,
        out_features: usize,
        rng: &mut HeirloomRng,
    ) -> Result<Self> {
        let scale = (1.0 / in_features.max(1) as f32).sqrt();
        Ok(Self {
            in_features,
            out_features,
            weight: rng.uniform_tensor(&[out_features, in_features], -scale, scale, true)?,
            bias: Tensor::from_vec(vec![0.0; out_features], &[out_features], true)?,
        })
    }

    pub fn weight(&self) -> Tensor {
        self.weight.clone()
    }

    pub fn bias(&self) -> Tensor {
        self.bias.clone()
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        Ok(Self {
            in_features: self.in_features,
            out_features: self.out_features,
            weight: self.weight.to_device(device)?,
            bias: self.bias.to_device(device)?,
        })
    }

    pub fn forward_amp_bf16(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_amp_bf16_named(input, "linear")
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, module: &str) -> Result<Tensor> {
        let input_shape = input.shape();
        let Some(&last_dim) = input_shape.last() else {
            return Err(crate::TensorError::Shape(format!(
                "Linear expected input with last dim {}, got scalar",
                self.in_features
            )));
        };
        if last_dim != self.in_features {
            return Err(crate::TensorError::Shape(format!(
                "Linear expected input last dim {}, got {:?}",
                self.in_features, input_shape
            )));
        }
        let outer = input.numel() / self.in_features;
        let tensor_core_shape = heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(
            outer,
            self.in_features,
            self.out_features,
        );
        let tensor_core_device = match input.device() {
            Device::Cuda(device_id) => {
                heirloom_kernels::cuda::device_supports_bf16_tensor_cores(device_id as i32)
                    .unwrap_or(false)
            }
            Device::Cpu => false,
        };
        let input_dtype = input.dtype();
        let weight_dtype = self.weight.dtype();
        let bias_dtype = self.bias.dtype();
        let can_try_tensor_core = tensor_core_device
            && input_dtype == DType::F32
            && weight_dtype == DType::F32
            && bias_dtype == DType::F32
            && tensor_core_shape;
        let can_try_fused_cp_async_bias = can_try_tensor_core
            && heirloom_kernels::cuda::tensor_core_cp_async_gemm_enabled()
            && heirloom_kernels::cuda::bf16_tensor_core_matmul_exact_tile_shape_supported(
                outer,
                self.in_features,
                self.out_features,
            );

        record_amp_bf16_linear_coverage(AmpBf16LinearRecord {
            module,
            m: outer,
            k: self.in_features,
            n: self.out_features,
            device: input.device(),
            input_dtype,
            weight_dtype,
            bias_dtype,
            tensor_core_device,
            tensor_core_shape,
            used_tensor_core: can_try_tensor_core,
        });
        amp::record_amp_bf16_op_decision(AmpBf16OpDecision {
            op: format!("linear:{module}"),
            input_dtype: amp_dtype_label(input_dtype),
            input_device: amp_device_label(input.device()),
            compute_dtype: if can_try_tensor_core {
                "bf16".to_string()
            } else {
                amp_dtype_label(input_dtype)
            },
            accumulation_dtype: "f32".to_string(),
            output_dtype: "f32".to_string(),
            kernel_path: if can_try_tensor_core {
                if can_try_fused_cp_async_bias {
                    "cuda_tensor_core_bf16_rhs_t_cp_async_bias_fused".to_string()
                } else {
                    "cuda_tensor_core_bf16_rhs_t".to_string()
                }
            } else {
                "fallback_linear_forward".to_string()
            },
            tensor_core: can_try_tensor_core,
            fallback_reason: if can_try_tensor_core {
                None
            } else if !tensor_core_device {
                Some("unsupported_device".to_string())
            } else if !tensor_core_shape {
                Some("unsupported_shape".to_string())
            } else if input_dtype != DType::F32
                || weight_dtype != DType::F32
                || bias_dtype != DType::F32
            {
                Some("dtype_mismatch".to_string())
            } else {
                Some("unknown".to_string())
            },
            finite_check: "optimizer_and_gradient_checks".to_string(),
        });

        if can_try_tensor_core {
            let flat = input.reshape(&[outer, self.in_features])?;
            let flat_bf16 = flat.to_dtype(DType::BFloat16)?;
            let weight_bf16 = self.weight.to_dtype(DType::BFloat16)?;
            let output = if can_try_fused_cp_async_bias {
                flat_bf16.matmul_bf16_tensor_core_rhs_t_bias(&weight_bf16, &self.bias)?
            } else {
                flat_bf16
                    .matmul_bf16_tensor_core_rhs_t(&weight_bf16)?
                    .add(&self.bias)?
            };
            let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
            output_shape.push(self.out_features);
            return output.reshape(&output_shape);
        }

        ensure_amp_bf16_transformer_path_supported()?;
        self.forward(input)
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let Some(&last_dim) = input_shape.last() else {
            return Err(crate::TensorError::Shape(format!(
                "Linear expected input with last dim {}, got scalar",
                self.in_features
            )));
        };
        if last_dim != self.in_features {
            return Err(crate::TensorError::Shape(format!(
                "Linear expected input last dim {}, got {:?}",
                self.in_features, input_shape
            )));
        }
        let outer = input.numel() / self.in_features;
        let flat = input.reshape(&[outer, self.in_features])?;
        let weight_t = self.weight.transpose()?;
        let output = flat.matmul(&weight_t)?.add(&self.bias)?;
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(self.out_features);
        output.reshape(&output_shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{prefix}weight"), self.weight.clone()),
            (format!("{prefix}bias"), self.bias.clone()),
        ]
    }
}

#[derive(Default)]
/// Elementwise rectified-linear activation.
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }

    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

#[derive(Default)]
/// Elementwise Gaussian error linear unit.
pub struct Gelu;

impl Gelu {
    pub fn new() -> Self {
        Self
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, module: &str) -> Result<Tensor> {
        record_amp_bf16_device_fp32_op_decision(
            format!("gelu:{module}"),
            input,
            "cuda_gelu_f32",
            "cpu_gelu_f32",
            "gradient_is_finite",
        );
        self.forward(input)
    }
}

impl Module for Gelu {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.gelu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }

    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

/// Trainable embedding table indexed by an integer tensor.
pub struct Embedding {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    weight: Tensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, rng: &mut HeirloomRng) -> Result<Self> {
        Ok(Self {
            num_embeddings,
            embedding_dim,
            weight: rng.normal_tensor(&[num_embeddings, embedding_dim], 0.0, 0.02, true)?,
        })
    }

    pub fn weight(&self) -> Tensor {
        self.weight.clone()
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        Ok(Self {
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
            weight: self.weight.to_device(device)?,
        })
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, module: &str) -> Result<Tensor> {
        let kernel_path = match input.device() {
            Device::Cuda(_) => "cuda_embedding_f32",
            Device::Cpu => "cpu_embedding_f32",
        };
        amp::record_amp_bf16_op_decision(AmpBf16OpDecision {
            op: format!("embedding:{module}"),
            input_dtype: format!("indices:{} weight:f32", amp_dtype_label(input.dtype())),
            input_device: amp_device_label(input.device()),
            compute_dtype: "f32".to_string(),
            accumulation_dtype: "f32".to_string(),
            output_dtype: "f32".to_string(),
            kernel_path: kernel_path.to_string(),
            tensor_core: false,
            fallback_reason: None,
            finite_check: "embedding_gradient_scatter_add_is_finite".to_string(),
        });
        self.forward(input)
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.embedding(&self.weight)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![(format!("{prefix}weight"), self.weight.clone())]
    }
}

/// Layer normalization over the final dimension with trainable scale/bias.
pub struct LayerNorm {
    pub features: usize,
    pub eps: f64,
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    pub fn new(features: usize) -> Result<Self> {
        Ok(Self {
            features,
            eps: 1e-5,
            weight: Tensor::ones(&[features], true)?,
            bias: Tensor::zeros(&[features], true)?,
        })
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        Ok(Self {
            features: self.features,
            eps: self.eps,
            weight: self.weight.to_device(device)?,
            bias: self.bias.to_device(device)?,
        })
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, module: &str) -> Result<Tensor> {
        record_amp_bf16_device_fp32_op_decision(
            format!("layer_norm:{module}"),
            input,
            "cuda_layer_norm_f32",
            "cpu_layer_norm_f32",
            "layer_norm_gradient_is_finite",
        );
        self.forward(input)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.layer_norm_last_dim(&self.weight, &self.bias, self.eps)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{prefix}weight"), self.weight.clone()),
            (format!("{prefix}bias"), self.bias.clone()),
        ]
    }
}

/// Decoder-style causal multi-head self-attention.
pub struct CausalSelfAttention {
    pub d_model: usize,
    pub n_heads: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl CausalSelfAttention {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut HeirloomRng) -> Result<Self> {
        if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
            return Err(TensorError::Shape(format!(
                "d_model {d_model} must be divisible by n_heads {n_heads}"
            )));
        }
        Ok(Self {
            d_model,
            n_heads,
            q_proj: Linear::new_with_rng(d_model, d_model, rng)?,
            k_proj: Linear::new_with_rng(d_model, d_model, rng)?,
            v_proj: Linear::new_with_rng(d_model, d_model, rng)?,
            out_proj: Linear::new_with_rng(d_model, d_model, rng)?,
        })
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        Ok(Self {
            d_model: self.d_model,
            n_heads: self.n_heads,
            q_proj: self.q_proj.to_device(device)?,
            k_proj: self.k_proj.to_device(device)?,
            v_proj: self.v_proj.to_device(device)?,
            out_proj: self.out_proj.to_device(device)?,
        })
    }

    pub fn forward_bf16_activations(&self, input: &Tensor) -> Result<Tensor> {
        let query = bf16_activation_roundtrip(self.q_proj.forward(input)?)?;
        let key = bf16_activation_roundtrip(self.k_proj.forward(input)?)?;
        let value = bf16_activation_roundtrip(self.v_proj.forward(input)?)?;
        let attended =
            bf16_activation_roundtrip(query.causal_self_attention(&key, &value, self.n_heads)?)?;
        self.out_proj.forward(&attended)
    }

    pub fn forward_amp_bf16(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_amp_bf16_named(input, "attention")
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let query = bf16_activation_roundtrip(
            self.q_proj
                .forward_amp_bf16_named(input, &child_module_name(prefix, "q_proj"))?,
        )?;
        let key = bf16_activation_roundtrip(
            self.k_proj
                .forward_amp_bf16_named(input, &child_module_name(prefix, "k_proj"))?,
        )?;
        let value = bf16_activation_roundtrip(
            self.v_proj
                .forward_amp_bf16_named(input, &child_module_name(prefix, "v_proj"))?,
        )?;
        let attention_dims = attention_tensor_core_dims(&query, self.n_heads);
        let tensor_core_attention_device = match query.device() {
            Device::Cuda(device_id) => {
                heirloom_kernels::cuda::device_supports_bf16_tensor_cores(device_id as i32)
                    .unwrap_or(false)
            }
            Device::Cpu => false,
        };
        let tensor_core_attention_shape = attention_dims
            .is_some_and(heirloom_kernels::cuda::causal_attention_bf16_tensor_core_shape_supported);
        let attention_uses_tensor_core =
            tensor_core_attention_device && tensor_core_attention_shape;
        amp::record_amp_bf16_op_decision(AmpBf16OpDecision {
            op: format!("causal_attention:{prefix}"),
            input_dtype: amp_dtype_label(query.dtype()),
            input_device: amp_device_label(query.device()),
            compute_dtype: if attention_uses_tensor_core {
                "bf16".to_string()
            } else {
                amp_dtype_label(query.dtype())
            },
            accumulation_dtype: "f32".to_string(),
            output_dtype: "f32".to_string(),
            kernel_path: if attention_uses_tensor_core {
                "cuda_tensor_core_bf16_attention".to_string()
            } else {
                "fallback_causal_attention".to_string()
            },
            tensor_core: attention_uses_tensor_core,
            fallback_reason: if attention_uses_tensor_core {
                None
            } else if !tensor_core_attention_device {
                Some("unsupported_device".to_string())
            } else if !tensor_core_attention_shape {
                Some("unsupported_shape".to_string())
            } else {
                Some("unknown".to_string())
            },
            finite_check: "loss_gradient_optimizer_checks".to_string(),
        });
        let attended = if tensor_core_attention_device && tensor_core_attention_shape {
            query.causal_self_attention_amp_bf16_tensor_core(&key, &value, self.n_heads)?
        } else {
            if amp_bf16_attention_tensor_core_required() {
                return Err(TensorError::Device(format!(
                    "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 requested Tensor Core AMP attention, \
                     but support was device_supported={tensor_core_attention_device} \
                     shape_supported={tensor_core_attention_shape} for query shape {:?} n_heads={}",
                    query.shape(),
                    self.n_heads
                )));
            }
            query.causal_self_attention(&key, &value, self.n_heads)?
        };
        let attended = bf16_activation_roundtrip(attended)?;
        self.out_proj
            .forward_amp_bf16_named(&attended, &child_module_name(prefix, "out_proj"))
    }
}

fn attention_tensor_core_dims(
    query: &Tensor,
    n_heads: usize,
) -> Option<heirloom_kernels::cuda::CausalAttentionDims> {
    let shape = query.shape();
    if shape.len() != 3 || n_heads == 0 || !shape[2].is_multiple_of(n_heads) {
        return None;
    }
    Some(heirloom_kernels::cuda::CausalAttentionDims {
        batch: shape[0],
        time: shape[1],
        channels: shape[2],
        n_heads,
    })
}

impl Module for CausalSelfAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let query = self.q_proj.forward(input)?;
        let key = self.k_proj.forward(input)?;
        let value = self.v_proj.forward(input)?;
        let attended = query.causal_self_attention(&key, &value, self.n_heads)?;
        self.out_proj.forward(&attended)
    }

    fn parameters(&self) -> Vec<Tensor> {
        [
            self.q_proj.parameters(),
            self.k_proj.parameters(),
            self.v_proj.parameters(),
            self.out_proj.parameters(),
        ]
        .concat()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.q_proj.named_parameters(&format!("{prefix}q_proj.")));
        out.extend(self.k_proj.named_parameters(&format!("{prefix}k_proj.")));
        out.extend(self.v_proj.named_parameters(&format!("{prefix}v_proj.")));
        out.extend(
            self.out_proj
                .named_parameters(&format!("{prefix}out_proj.")),
        );
        out
    }
}

/// Transformer feed-forward sublayer (`Linear` → GELU → `Linear`).
pub struct FeedForward {
    fc1: Linear,
    gelu: Gelu,
    fc2: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, hidden: usize, rng: &mut HeirloomRng) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new_with_rng(d_model, hidden, rng)?,
            gelu: Gelu::new(),
            fc2: Linear::new_with_rng(hidden, d_model, rng)?,
        })
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        Ok(Self {
            fc1: self.fc1.to_device(device)?,
            gelu: Gelu::new(),
            fc2: self.fc2.to_device(device)?,
        })
    }

    pub fn forward_bf16_activations(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = bf16_activation_roundtrip(self.fc1.forward(input)?)?;
        let hidden = bf16_activation_roundtrip(self.gelu.forward(&hidden)?)?;
        self.fc2.forward(&hidden)
    }

    pub fn forward_amp_bf16(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_amp_bf16_named(input, "feed_forward")
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let hidden = bf16_activation_roundtrip(
            self.fc1
                .forward_amp_bf16_named(input, &child_module_name(prefix, "fc1"))?,
        )?;
        let hidden = bf16_activation_roundtrip(
            self.gelu
                .forward_amp_bf16_named(&hidden, &child_module_name(prefix, "gelu"))?,
        )?;
        self.fc2
            .forward_amp_bf16_named(&hidden, &child_module_name(prefix, "fc2"))
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.fc2
            .forward(&self.gelu.forward(&self.fc1.forward(input)?)?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.extend(self.fc1.named_parameters(&format!("{prefix}fc1.")));
        out.extend(self.fc2.named_parameters(&format!("{prefix}fc2.")));
        out
    }
}

/// Pre-normalized causal-attention and feed-forward residual block.
pub struct TransformerBlock {
    ln1: LayerNorm,
    attention: CausalSelfAttention,
    ln2: LayerNorm,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        ff_hidden: usize,
        rng: &mut HeirloomRng,
    ) -> Result<Self> {
        Ok(Self {
            ln1: LayerNorm::new(d_model)?,
            attention: CausalSelfAttention::new(d_model, n_heads, rng)?,
            ln2: LayerNorm::new(d_model)?,
            feed_forward: FeedForward::new(d_model, ff_hidden, rng)?,
        })
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        Ok(Self {
            ln1: self.ln1.to_device(device)?,
            attention: self.attention.to_device(device)?,
            ln2: self.ln2.to_device(device)?,
            feed_forward: self.feed_forward.to_device(device)?,
        })
    }

    pub fn forward_bf16_activations(&self, input: &Tensor) -> Result<Tensor> {
        let ln1 = bf16_activation_roundtrip(self.ln1.forward(input)?)?;
        let attention = bf16_activation_roundtrip(self.attention.forward_bf16_activations(&ln1)?)?;
        let residual = bf16_activation_roundtrip(input.add(&attention)?)?;
        let ln2 = bf16_activation_roundtrip(self.ln2.forward(&residual)?)?;
        let feed_forward =
            bf16_activation_roundtrip(self.feed_forward.forward_bf16_activations(&ln2)?)?;
        residual.add(&feed_forward)
    }

    pub fn forward_amp_bf16(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_amp_bf16_named(input, "block")
    }

    pub fn forward_amp_bf16_named(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let ln1 = bf16_activation_roundtrip(
            self.ln1
                .forward_amp_bf16_named(input, &child_module_name(prefix, "ln1"))?,
        )?;
        let attention = bf16_activation_roundtrip(
            self.attention
                .forward_amp_bf16_named(&ln1, &child_module_name(prefix, "attention"))?,
        )?;
        record_amp_bf16_device_fp32_op_decision(
            format!("residual_add:{}", child_module_name(prefix, "attention")),
            input,
            "cuda_add_f32",
            "cpu_add_f32",
            "residual_gradient_is_finite",
        );
        let residual = bf16_activation_roundtrip(input.add(&attention)?)?;
        let ln2 = bf16_activation_roundtrip(
            self.ln2
                .forward_amp_bf16_named(&residual, &child_module_name(prefix, "ln2"))?,
        )?;
        let feed_forward = bf16_activation_roundtrip(
            self.feed_forward
                .forward_amp_bf16_named(&ln2, &child_module_name(prefix, "feed_forward"))?,
        )?;
        record_amp_bf16_device_fp32_op_decision(
            format!("residual_add:{}", child_module_name(prefix, "feed_forward")),
            &residual,
            "cuda_add_f32",
            "cpu_add_f32",
            "residual_gradient_is_finite",
        );
        residual.add(&feed_forward)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let attention = self.attention.forward(&self.ln1.forward(input)?)?;
        let residual = input.add(&attention)?;
        let feed_forward = self.feed_forward.forward(&self.ln2.forward(&residual)?)?;
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

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Shape configuration for the bounded decoder-only reference model.
pub struct TinyTransformerConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_hidden: usize,
}

impl TinyTransformerConfig {
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            block_size: 64,
            d_model: 64,
            n_heads: 4,
            ff_hidden: 256,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub eos_id: usize,
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
}

impl GenerationOptions {
    pub fn greedy(max_new_tokens: usize, eos_id: usize) -> Self {
        Self {
            max_new_tokens,
            eos_id,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.temperature < 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "temperature must be non-negative, got {}",
                self.temperature
            )));
        }
        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                return Err(TensorError::InvalidOperation(
                    "top_k must be greater than zero when set".to_string(),
                ));
            }
        }
        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
                return Err(TensorError::InvalidOperation(format!(
                    "top_p must be in (0, 1], got {top_p}"
                )));
            }
        }
        if self.repetition_penalty < 1.0 {
            return Err(TensorError::InvalidOperation(format!(
                "repetition_penalty must be at least 1.0, got {}",
                self.repetition_penalty
            )));
        }
        if self.frequency_penalty < 0.0 || self.presence_penalty < 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "frequency_penalty and presence_penalty must be non-negative, got {}/{}",
                self.frequency_penalty, self.presence_penalty
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationFinishReason {
    Eos,
    MaxNewTokens,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratedTokenStep {
    pub token_id: usize,
    pub probability: f64,
    pub log_probability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub tokens: Vec<usize>,
    pub new_token_count: usize,
    pub finish_reason: GenerationFinishReason,
    pub rng_state: u64,
    pub steps: Vec<GeneratedTokenStep>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmEvalMetrics {
    pub batches: usize,
    pub examples: usize,
    pub tokens: usize,
    pub loss: f64,
    pub perplexity: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForwardPrecisionPolicy {
    F32,
    Bf16Activations,
    AmpBf16,
}

/// Small decoder-only language model used to exercise the complete runtime.
///
/// It is a systems-validation model, not a pretrained general-purpose model.
/// Parameters, autograd, AMP routing, evaluation, generation, and checkpointing
/// all use the same public runtime paths as the memory transformer.
pub struct TinyTransformerLm {
    pub config: TinyTransformerConfig,
    token_embedding: Embedding,
    position_embedding: Embedding,
    block: TransformerBlock,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl TinyTransformerLm {
    pub fn new(config: TinyTransformerConfig, rng: &mut HeirloomRng) -> Result<Self> {
        if config.block_size == 0 || config.vocab_size == 0 {
            return Err(TensorError::InvalidOperation(
                "vocab_size and block_size must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            token_embedding: Embedding::new(config.vocab_size, config.d_model, rng)?,
            position_embedding: Embedding::new(config.block_size, config.d_model, rng)?,
            block: TransformerBlock::new(config.d_model, config.n_heads, config.ff_hidden, rng)?,
            ln_f: LayerNorm::new(config.d_model)?,
            lm_head: Linear::new_with_rng(config.d_model, config.vocab_size, rng)?,
            config,
        })
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        Ok(Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.to_device(device)?,
            position_embedding: self.position_embedding.to_device(device)?,
            block: self.block.to_device(device)?,
            ln_f: self.ln_f.to_device(device)?,
            lm_head: self.lm_head.to_device(device)?,
        })
    }

    pub fn device(&self) -> Device {
        self.token_embedding.weight().device()
    }

    pub fn loss(&self, input: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward(input)?;
        let shape = logits.shape();
        let flat_logits = logits.reshape(&[shape[0] * shape[1], shape[2]])?;
        let flat_targets = targets.reshape(&[shape[0] * shape[1]])?;
        flat_logits.cross_entropy_for_logits_tensor(&flat_targets)
    }

    pub fn forward_bf16_activations(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "TinyTransformerLm expects token input [batch, time], got {:?}",
                shape
            )));
        }
        let (batch, time) = (shape[0], shape[1]);
        if time > self.config.block_size {
            return Err(TensorError::Shape(format!(
                "input time {time} exceeds block_size {}",
                self.config.block_size
            )));
        }
        let token = bf16_activation_roundtrip(self.token_embedding.forward(input)?)?;
        let position_ids = position_ids_tensor(batch, time, input.device())?;
        let position = bf16_activation_roundtrip(self.position_embedding.forward(&position_ids)?)?;
        let hidden = bf16_activation_roundtrip(token.add(&position)?)?;
        let hidden = bf16_activation_roundtrip(self.block.forward_bf16_activations(&hidden)?)?;
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
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "TinyTransformerLm expects token input [batch, time], got {:?}",
                shape
            )));
        }
        let (batch, time) = (shape[0], shape[1]);
        if time > self.config.block_size {
            return Err(TensorError::Shape(format!(
                "input time {time} exceeds block_size {}",
                self.config.block_size
            )));
        }
        let token = bf16_activation_roundtrip(
            self.token_embedding
                .forward_amp_bf16_named(input, "token_embedding")?,
        )?;
        let position_ids = position_ids_tensor(batch, time, input.device())?;
        let position = bf16_activation_roundtrip(
            self.position_embedding
                .forward_amp_bf16_named(&position_ids, "position_embedding")?,
        )?;
        record_amp_bf16_device_fp32_op_decision(
            "elementwise_add:token_plus_position",
            &token,
            "cuda_add_f32",
            "cpu_add_f32",
            "embedding_sum_gradient_is_finite",
        );
        let hidden = bf16_activation_roundtrip(token.add(&position)?)?;
        let hidden =
            bf16_activation_roundtrip(self.block.forward_amp_bf16_named(&hidden, "block")?)?;
        let hidden = bf16_activation_roundtrip(self.ln_f.forward_amp_bf16_named(&hidden, "ln_f")?)?;
        self.lm_head.forward_amp_bf16_named(&hidden, "lm_head")
    }

    pub fn loss_amp_bf16(&self, input: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward_amp_bf16(input)?;
        let shape = logits.shape();
        let flat_logits = logits.reshape(&[shape[0] * shape[1], shape[2]])?;
        let flat_targets = targets.reshape(&[shape[0] * shape[1]])?;
        amp::record_amp_bf16_op_decision(AmpBf16OpDecision {
            op: "cross_entropy_loss".to_string(),
            input_dtype: amp_dtype_label(flat_logits.dtype()),
            input_device: amp_device_label(flat_logits.device()),
            compute_dtype: "f32".to_string(),
            accumulation_dtype: "f32".to_string(),
            output_dtype: "f32".to_string(),
            kernel_path: match flat_logits.device() {
                Device::Cuda(_) => "cuda_cross_entropy_f32".to_string(),
                Device::Cpu => "cpu_cross_entropy_f32".to_string(),
            },
            tensor_core: false,
            fallback_reason: None,
            finite_check: "loss_is_finite".to_string(),
        });
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
        self.generate_with_activation_policy(prefix, options, rng, ForwardPrecisionPolicy::F32)
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
            ForwardPrecisionPolicy::Bf16Activations,
        )
    }

    pub fn generate_amp_bf16(
        &self,
        prefix: &[usize],
        options: &GenerationOptions,
        rng: &mut HeirloomRng,
    ) -> Result<GenerationOutput> {
        self.generate_with_activation_policy(prefix, options, rng, ForwardPrecisionPolicy::AmpBf16)
    }

    fn generate_with_activation_policy(
        &self,
        prefix: &[usize],
        options: &GenerationOptions,
        rng: &mut HeirloomRng,
        precision: ForwardPrecisionPolicy,
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
            ForwardPrecisionPolicy::F32,
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
            ForwardPrecisionPolicy::Bf16Activations,
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
            ForwardPrecisionPolicy::AmpBf16,
        )
    }

    fn evaluate_token_loss_with_activation_policy(
        &self,
        tokens: &[usize],
        batch_size: usize,
        max_batches: Option<usize>,
        precision: ForwardPrecisionPolicy,
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
                ForwardPrecisionPolicy::F32 => self.loss(&input, &target),
                ForwardPrecisionPolicy::Bf16Activations => {
                    self.loss_bf16_activations(&input, &target)
                }
                ForwardPrecisionPolicy::AmpBf16 => self.loss_amp_bf16(&input, &target),
            })?;
            let weight = actual_batch * self.config.block_size;
            let loss_value = if precision == ForwardPrecisionPolicy::AmpBf16 {
                amp::with_cuda_host_staging_allowed("amp-bf16 eval loss scalar logging", || {
                    loss.data()[0] as f64
                })
            } else {
                loss.data()[0] as f64
            };
            amp::record_amp_bf16_finite_check(
                "eval_loss",
                "loss",
                loss.dtype(),
                loss.device(),
                loss_value,
            )?;
            if !loss_value.is_finite() {
                return Err(TensorError::Autograd(format!(
                    "non-finite eval loss: {loss_value}"
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
        self.next_token_logits_with_activation_policy(tokens, ForwardPrecisionPolicy::F32)
    }

    pub fn next_token_logits_bf16_activations(&self, tokens: &[usize]) -> Result<Vec<f64>> {
        self.next_token_logits_with_activation_policy(
            tokens,
            ForwardPrecisionPolicy::Bf16Activations,
        )
    }

    pub fn next_token_logits_amp_bf16(&self, tokens: &[usize]) -> Result<Vec<f64>> {
        self.next_token_logits_with_activation_policy(tokens, ForwardPrecisionPolicy::AmpBf16)
    }

    fn next_token_logits_with_activation_policy(
        &self,
        tokens: &[usize],
        precision: ForwardPrecisionPolicy,
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
            ForwardPrecisionPolicy::F32 => self.forward(&input),
            ForwardPrecisionPolicy::Bf16Activations => self.forward_bf16_activations(&input),
            ForwardPrecisionPolicy::AmpBf16 => self.forward_amp_bf16(&input),
        })?;
        let shape = logits.shape();
        let classes = shape[2];
        let data = if precision == ForwardPrecisionPolicy::AmpBf16 {
            amp::with_cuda_host_staging_allowed("amp-bf16 generation logits inspection", || {
                logits.data_f64()
            })
        } else {
            logits.data_f64()
        };
        if precision == ForwardPrecisionPolicy::AmpBf16 {
            let logits_are_finite = data.iter().all(|value| value.is_finite());
            let max_abs = if logits_are_finite {
                data.iter().map(|value| value.abs()).fold(0.0, f64::max)
            } else {
                f64::NAN
            };
            amp::record_amp_bf16_finite_check(
                "generation_logits_max_abs",
                "next_token_logits",
                logits.dtype(),
                logits.device(),
                max_abs,
            )?;
            if !logits_are_finite {
                return Err(TensorError::Autograd(
                    "non-finite generation logits under amp-bf16".to_string(),
                ));
            }
        }
        let row_start = (shape[1] - 1) * classes;
        Ok(data[row_start..row_start + classes].to_vec())
    }
}

pub(crate) fn sample_token(
    logits: &[f64],
    context: &[usize],
    options: &GenerationOptions,
    rng: &mut HeirloomRng,
) -> Result<(usize, f64, f64)> {
    let mut adjusted = logits.to_vec();
    apply_repetition_controls(&mut adjusted, context, options);
    if options.temperature > 0.0 {
        for value in &mut adjusted {
            *value /= options.temperature;
        }
    }
    let keep = sampling_filter(&adjusted, options);
    for (index, value) in adjusted.iter_mut().enumerate() {
        if !keep[index] {
            *value = f64::NEG_INFINITY;
        }
    }
    let probabilities = softmax_logits(&adjusted);
    if options.temperature == 0.0 {
        let token = argmax_filtered(&adjusted);
        let probability = probabilities[token];
        return Ok((token, probability, probability.max(f64::MIN_POSITIVE).ln()));
    }

    let draw = rng.uniform_f32() as f64;
    let mut cumulative = 0.0;
    for (token, probability) in probabilities.iter().enumerate() {
        cumulative += *probability;
        if draw <= cumulative {
            return Ok((token, *probability, probability.max(f64::MIN_POSITIVE).ln()));
        }
    }
    let token = argmax_filtered(&adjusted);
    let probability = probabilities[token];
    Ok((token, probability, probability.max(f64::MIN_POSITIVE).ln()))
}

fn apply_repetition_controls(logits: &mut [f64], context: &[usize], options: &GenerationOptions) {
    let mut counts = HashMap::<usize, usize>::new();
    for &token in context {
        if token < logits.len() {
            *counts.entry(token).or_default() += 1;
        }
    }
    for (token, count) in counts {
        if options.repetition_penalty > 1.0 {
            if logits[token] > 0.0 {
                logits[token] /= options.repetition_penalty;
            } else {
                logits[token] *= options.repetition_penalty;
            }
        }
        logits[token] -= options.frequency_penalty * count as f64;
        logits[token] -= options.presence_penalty;
    }
}

fn sampling_filter(logits: &[f64], options: &GenerationOptions) -> Vec<bool> {
    let mut keep = vec![true; logits.len()];
    if let Some(top_k) = options.top_k {
        let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|(_, left), (_, right)| right.total_cmp(left));
        keep.fill(false);
        for (token, _) in ranked.into_iter().take(top_k.min(logits.len())) {
            keep[token] = true;
        }
    }

    if let Some(top_p) = options.top_p {
        if top_p < 1.0 {
            let filtered_logits = logits
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    if keep[index] {
                        *value
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>();
            let probabilities = softmax_logits(&filtered_logits);
            let mut ranked = probabilities
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            ranked.sort_by(|(_, left), (_, right)| right.total_cmp(left));
            keep.fill(false);
            let mut cumulative = 0.0;
            for (token, probability) in ranked {
                if probability == 0.0 {
                    continue;
                }
                keep[token] = true;
                cumulative += probability;
                if cumulative >= top_p {
                    break;
                }
            }
        }
    }

    if keep.iter().any(|value| *value) {
        keep
    } else {
        let mut fallback = vec![false; logits.len()];
        fallback[argmax_filtered(logits)] = true;
        fallback
    }
}

fn softmax_logits(logits: &[f64]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        let mut probabilities = vec![0.0; logits.len()];
        if !logits.is_empty() {
            probabilities[0] = 1.0;
        }
        return probabilities;
    }

    let mut sum = 0.0;
    let mut probabilities = Vec::with_capacity(logits.len());
    for &value in logits {
        let probability = if value.is_finite() {
            (value - max).exp()
        } else {
            0.0
        };
        sum += probability;
        probabilities.push(probability);
    }
    if sum == 0.0 {
        return probabilities;
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}

fn argmax_filtered(values: &[f64]) -> usize {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

impl Module for TinyTransformerLm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "TinyTransformerLm expects token input [batch, time], got {:?}",
                shape
            )));
        }
        let (batch, time) = (shape[0], shape[1]);
        if time > self.config.block_size {
            return Err(TensorError::Shape(format!(
                "input time {time} exceeds block_size {}",
                self.config.block_size
            )));
        }
        let token = self.token_embedding.forward(input)?;
        let positions = position_ids_tensor(batch, time, input.device())?;
        let position = self.position_embedding.forward(&positions)?;
        let hidden = token.add(&position)?;
        let hidden = self.block.forward(&hidden)?;
        let hidden = self.ln_f.forward(&hidden)?;
        self.lm_head.forward(&hidden)
    }

    fn parameters(&self) -> Vec<Tensor> {
        [
            self.token_embedding.parameters(),
            self.position_embedding.parameters(),
            self.block.parameters(),
            self.ln_f.parameters(),
            self.lm_head.parameters(),
        ]
        .concat()
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
        out.extend(self.block.named_parameters(&format!("{prefix}block.")));
        out.extend(self.ln_f.named_parameters(&format!("{prefix}ln_f.")));
        out.extend(self.lm_head.named_parameters(&format!("{prefix}lm_head.")));
        out
    }
}

fn position_ids_tensor(batch: usize, time: usize, device: Device) -> Result<Tensor> {
    let mut position_ids = Vec::with_capacity(batch * time);
    for _ in 0..batch {
        position_ids.extend((0..time).map(|index| index as i64));
    }
    Tensor::from_i64(position_ids, &[batch, time], false)?.to_device(device)
}

/// Ordered module composition whose parameters retain layer order.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    pub fn push<M>(&mut self, module: M)
    where
        M: Module + 'static,
    {
        self.layers.push(Box::new(module));
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut activation = input.clone();
        for layer in &self.layers {
            activation = layer.forward(&activation)?;
        }
        Ok(activation)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(index, layer)| layer.named_parameters(&format!("{prefix}{index}.")))
            .collect()
    }
}

pub fn mse_loss(prediction: &Tensor, target: &Tensor) -> Result<Tensor> {
    prediction.mse_loss(target)
}

pub fn cross_entropy_for_logits(logits: &Tensor, targets: &[usize]) -> Result<Tensor> {
    logits.cross_entropy_for_logits(targets)
}

pub fn sgd_step(parameters: &[Tensor], lr: f32) -> Result<()> {
    for parameter in parameters {
        parameter.apply_sgd(lr)?;
    }
    Ok(())
}

/// Common optimizer lifecycle for accumulated parameter gradients.
pub trait Optimizer {
    /// Clears gradients on all tracked parameters.
    fn zero_grad(&self);
    /// Applies one optimizer update or returns a validation/kernel error.
    fn step(&self) -> Result<()>;
}

/// Stateless stochastic-gradient descent over a fixed parameter list.
pub struct Sgd {
    parameters: Vec<Tensor>,
    lr: f32,
}

impl Sgd {
    pub fn new(parameters: Vec<Tensor>, lr: f32) -> Result<Self> {
        if lr <= 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "learning rate must be positive, got {lr}"
            )));
        }
        Ok(Self { parameters, lr })
    }

    pub fn parameters(&self) -> &[Tensor] {
        &self.parameters
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }
}

impl Optimizer for Sgd {
    fn zero_grad(&self) {
        for parameter in &self.parameters {
            parameter.zero_grad();
        }
    }

    fn step(&self) -> Result<()> {
        sgd_step(&self.parameters, self.lr)
    }
}

/// Serializable AdamW hyperparameters, step counter, and CPU moment buffers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamWState {
    pub step: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub clip_norm: Option<f64>,
    pub m: Vec<Vec<f64>>,
    pub v: Vec<Vec<f64>>,
}

/// AdamW optimizer with CPU and CUDA state plus explicit sparse-row update
/// entry points for memory tables.
pub struct AdamW {
    parameters: Vec<Tensor>,
    state: AdamWState,
    cuda_state: Vec<Option<AdamWCudaParamState>>,
}

/// Description of selected memory rows whose full gradients should be updated.
pub struct SparseAdamWRowsUpdate {
    pub parameter_index: usize,
    pub selected_rows: Tensor,
    pub row_mask: Option<Tensor>,
    pub rows: usize,
    pub row_dim: usize,
}

/// Description of selected memory rows with a compact row-gradient tensor.
pub struct SparseAdamWCompactRowsUpdate {
    pub parameter_index: usize,
    pub selected_rows: Tensor,
    pub compact_grad_rows: Tensor,
    pub rows: usize,
    pub row_dim: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DistributedGradientStats {
    pub all_reduce_calls: usize,
    pub all_reduce_bytes: usize,
}

struct AdamWCudaParamState {
    m: CudaBuffer,
    v: CudaBuffer,
}

impl AdamW {
    pub fn new(parameters: Vec<Tensor>, lr: f64) -> Result<Self> {
        if lr <= 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "AdamW learning rate must be positive, got {lr}"
            )));
        }
        let m = parameters
            .iter()
            .map(|parameter| vec![0.0; parameter.numel()])
            .collect::<Vec<_>>();
        let v = parameters
            .iter()
            .map(|parameter| vec![0.0; parameter.numel()])
            .collect::<Vec<_>>();
        Ok(Self {
            cuda_state: parameters.iter().map(|_| None).collect(),
            parameters,
            state: AdamWState {
                step: 0,
                lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
                clip_norm: None,
                m,
                v,
            },
        })
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self> {
        if weight_decay < 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "AdamW weight_decay must be non-negative, got {weight_decay}"
            )));
        }
        self.state.weight_decay = weight_decay;
        Ok(self)
    }

    pub fn with_clip_norm(mut self, clip_norm: Option<f64>) -> Result<Self> {
        if let Some(value) = clip_norm {
            if value <= 0.0 {
                return Err(TensorError::InvalidOperation(format!(
                    "clip_norm must be positive, got {value}"
                )));
            }
        }
        self.state.clip_norm = clip_norm;
        Ok(self)
    }

    pub fn state(&self) -> AdamWState {
        self.try_state()
            .expect("failed to materialize AdamW CUDA state")
    }

    pub fn try_state(&self) -> Result<AdamWState> {
        let mut state = self.state.clone();
        for (index, cuda_state) in self.cuda_state.iter().enumerate() {
            let Some(cuda_state) = cuda_state else {
                continue;
            };
            state.m[index] = cuda_state
                .m
                .to_f32()
                .map_err(cuda_error)?
                .into_iter()
                .map(f64::from)
                .collect();
            state.v[index] = cuda_state
                .v
                .to_f32()
                .map_err(cuda_error)?
                .into_iter()
                .map(f64::from)
                .collect();
        }
        Ok(state)
    }

    pub fn load_state(&mut self, state: AdamWState) -> Result<()> {
        if state.m.len() != self.parameters.len() || state.v.len() != self.parameters.len() {
            return Err(TensorError::InvalidOperation(
                "AdamW state parameter count mismatch".to_string(),
            ));
        }
        for (index, parameter) in self.parameters.iter().enumerate() {
            if state.m[index].len() != parameter.numel()
                || state.v[index].len() != parameter.numel()
            {
                return Err(TensorError::InvalidOperation(format!(
                    "AdamW state shape mismatch for parameter {index}"
                )));
            }
        }
        self.state = state;
        self.cuda_state = self.parameters.iter().map(|_| None).collect();
        Ok(())
    }

    pub fn step_index(&self) -> usize {
        self.state.step
    }

    pub fn all_reduce_cuda_gradients_nccl(
        &self,
        communicator: &mut NcclCommunicator,
        world_size: usize,
    ) -> Result<DistributedGradientStats> {
        if world_size == 0 {
            return Err(TensorError::InvalidOperation(
                "world_size must be positive for NCCL gradient all-reduce".to_string(),
            ));
        }
        if communicator.world_size() != world_size as i32 {
            return Err(TensorError::Device(format!(
                "NCCL communicator world_size={} does not match requested world_size={world_size}",
                communicator.world_size()
            )));
        }
        let mut stats = DistributedGradientStats::default();
        let scale = 1.0f32 / world_size as f32;
        for parameter in &self.parameters {
            if !matches!(parameter.device(), Device::Cuda(_)) {
                return Err(TensorError::Device(format!(
                    "NCCL gradient all-reduce requires CUDA parameters, got {:?}",
                    parameter.device()
                )));
            }
            let Some(grad) = parameter.cuda_grad_f32_buffer()? else {
                continue;
            };
            let reduced = communicator
                .all_reduce_sum_in_place_f32(&grad)
                .map_err(cuda_error)?;
            if reduced.calls == 0 {
                continue;
            }
            heirloom_kernels::cuda::scale_in_place_f32_buffer(&grad, scale).map_err(cuda_error)?;
            let Some(sum_squares) = parameter.cuda_grad_sum_squares_f32()? else {
                continue;
            };
            amp::record_amp_bf16_finite_check(
                "ddp_gradient_sum_squares_after_all_reduce",
                &format!("parameter:{}", parameter.id()),
                parameter.dtype(),
                parameter.device(),
                f64::from(sum_squares),
            )?;
            if !sum_squares.is_finite() {
                return Err(TensorError::Autograd(format!(
                    "non-finite CUDA gradient norm after NCCL all-reduce for parameter {}",
                    parameter.id()
                )));
            }
            stats.all_reduce_calls += reduced.calls;
            stats.all_reduce_bytes += reduced.bytes;
        }
        Ok(stats)
    }
}

impl Optimizer for AdamW {
    fn zero_grad(&self) {
        for parameter in &self.parameters {
            parameter.zero_grad();
        }
    }

    fn step(&self) -> Result<()> {
        Err(TensorError::InvalidOperation(
            "AdamW::step requires mutable optimizer; use step_mut".to_string(),
        ))
    }
}

impl AdamW {
    pub fn step_mut(&mut self) -> Result<()> {
        let clip_scale = self.clip_scale()?;
        if amp::is_amp_bf16_training() {
            let device = self
                .parameters
                .iter()
                .map(Tensor::device)
                .find(|device| matches!(device, Device::Cuda(_)))
                .unwrap_or(Device::Cpu);
            amp::record_amp_bf16_op_decision(AmpBf16OpDecision {
                op: "adamw_step".to_string(),
                input_dtype: "params:f32 gradients:f32 moments:f32".to_string(),
                input_device: amp_device_label(device),
                compute_dtype: "f32".to_string(),
                accumulation_dtype: "f32".to_string(),
                output_dtype: "f32".to_string(),
                kernel_path: match device {
                    Device::Cuda(_) => "cuda_adamw_f32".to_string(),
                    Device::Cpu => "cpu_adamw_f64_accumulation".to_string(),
                },
                tensor_core: false,
                fallback_reason: None,
                finite_check: "gradient_norm_optimizer_hyperparameters_and_updates".to_string(),
            });
            amp::record_amp_bf16_finite_check(
                "adamw_lr",
                "optimizer",
                DType::F32,
                device,
                self.state.lr,
            )?;
            amp::record_amp_bf16_finite_check(
                "adamw_clip_scale",
                "optimizer",
                DType::F32,
                device,
                clip_scale,
            )?;
        }
        self.state.step += 1;
        let bias1 = 1.0 - self.state.beta1.powi(self.state.step as i32);
        let bias2 = 1.0 - self.state.beta2.powi(self.state.step as i32);

        for index in 0..self.parameters.len() {
            let parameter = self.parameters[index].clone();
            if parameter.grad_tensor().is_none() {
                continue;
            }
            if !parameter.dtype().is_floating() {
                return Err(TensorError::DType(format!(
                    "AdamW only supports floating parameters, got {:?}",
                    parameter.dtype()
                )));
            }
            match parameter.device() {
                Device::Cpu => {
                    self.step_cpu_parameter(index, &parameter, clip_scale, bias1, bias2)?
                }
                Device::Cuda(_) => {
                    self.step_cuda_parameter(index, &parameter, clip_scale, bias1, bias2)?
                }
            }
        }
        Ok(())
    }

    pub fn step_parameter_indices_mut(&mut self, parameter_indices: &[usize]) -> Result<()> {
        if parameter_indices.is_empty() {
            return Ok(());
        }
        let parameter_indices =
            validate_optimizer_parameter_indices(parameter_indices, self.parameters.len())?;
        let clip_scale = self.clip_scale_for_indices(&parameter_indices)?;
        self.state.step += 1;
        let bias1 = 1.0 - self.state.beta1.powi(self.state.step as i32);
        let bias2 = 1.0 - self.state.beta2.powi(self.state.step as i32);

        for &index in &parameter_indices {
            let parameter = self.parameters[index].clone();
            if parameter.grad_tensor().is_none() {
                continue;
            }
            if !parameter.dtype().is_floating() {
                return Err(TensorError::DType(format!(
                    "AdamW only supports floating parameters, got {:?}",
                    parameter.dtype()
                )));
            }
            match parameter.device() {
                Device::Cpu => {
                    self.step_cpu_parameter(index, &parameter, clip_scale, bias1, bias2)?
                }
                Device::Cuda(_) => {
                    self.step_cuda_parameter(index, &parameter, clip_scale, bias1, bias2)?
                }
            }
        }
        Ok(())
    }

    pub fn step_sparse_rows_mut(&mut self, updates: &[SparseAdamWRowsUpdate]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        self.validate_sparse_row_updates(updates)?;

        let clip_scale = self.clip_scale_for_sparse_row_updates(updates)?;
        self.state.step += 1;
        let bias1 = 1.0 - self.state.beta1.powi(self.state.step as i32);
        let bias2 = 1.0 - self.state.beta2.powi(self.state.step as i32);
        let lr = self.state.lr as f32;
        let beta1 = self.state.beta1 as f32;
        let beta2 = self.state.beta2 as f32;
        let eps = self.state.eps as f32;
        let weight_decay = self.state.weight_decay as f32;
        let clip_scale = clip_scale as f32;
        let bias_correction1 = bias1 as f32;
        let bias_correction2 = bias2 as f32;

        for update in updates {
            let parameter = self.parameters[update.parameter_index].clone();
            if parameter.grad_tensor().is_none() {
                continue;
            }
            if parameter.dtype() != DType::F32 {
                return Err(TensorError::DType(format!(
                    "sparse-row AdamW currently supports f32 parameters only, got {:?}",
                    parameter.dtype()
                )));
            }
            match parameter.device() {
                Device::Cpu => self.step_cpu_sparse_rows_parameter(
                    update.parameter_index,
                    &parameter,
                    update,
                    f64::from(clip_scale),
                    bias1,
                    bias2,
                )?,
                Device::Cuda(_) => {
                    let cuda_state =
                        self.cuda_state_for_parameter(update.parameter_index, &parameter)?;
                    parameter.apply_adamw_cuda_sparse_rows_f32(
                        &update.selected_rows,
                        update.row_mask.as_ref(),
                        &cuda_state.m,
                        &cuda_state.v,
                        heirloom_kernels::cuda::SparseAdamWRowsDims {
                            selected_rows: update.selected_rows.numel(),
                            rows: update.rows,
                            row_dim: update.row_dim,
                        },
                        heirloom_kernels::cuda::AdamWParams {
                            lr,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            clip_scale,
                            bias_correction1,
                            bias_correction2,
                        },
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn step_cuda_sparse_rows_mut(&mut self, updates: &[SparseAdamWRowsUpdate]) -> Result<()> {
        self.step_sparse_rows_mut(updates)
    }

    pub fn step_cuda_sparse_compact_rows_mut(
        &mut self,
        updates: &[SparseAdamWCompactRowsUpdate],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let mut seen = HashSet::new();
        for update in updates {
            if update.parameter_index >= self.parameters.len() {
                return Err(TensorError::InvalidOperation(format!(
                    "compact sparse AdamW parameter index {} is out of range for {} parameters",
                    update.parameter_index,
                    self.parameters.len()
                )));
            }
            if !seen.insert(update.parameter_index) {
                return Err(TensorError::InvalidOperation(format!(
                    "duplicate compact sparse AdamW update for parameter index {}",
                    update.parameter_index
                )));
            }
            if update.selected_rows.ndim() != 1 {
                return Err(TensorError::Shape(format!(
                    "compact sparse AdamW selected_rows must be rank-1, got {:?}",
                    update.selected_rows.shape()
                )));
            }
            let expected_grad_shape = vec![update.selected_rows.numel(), update.row_dim];
            if update.compact_grad_rows.shape() != expected_grad_shape {
                return Err(TensorError::Shape(format!(
                    "compact sparse AdamW compact_grad_rows must have shape {:?}, got {:?}",
                    expected_grad_shape,
                    update.compact_grad_rows.shape()
                )));
            }
            if update.compact_grad_rows.device() != update.selected_rows.device() {
                return Err(TensorError::Device(format!(
                    "compact sparse AdamW selected_rows and compact_grad_rows must be on the same device, got {:?} and {:?}",
                    update.selected_rows.device(),
                    update.compact_grad_rows.device()
                )));
            }
            if update.compact_grad_rows.dtype() != DType::F32 {
                return Err(TensorError::DType(format!(
                    "compact sparse AdamW compact_grad_rows must be f32, got {:?}",
                    update.compact_grad_rows.dtype()
                )));
            }
        }

        let clip_scale = self.clip_scale_for_compact_row_updates(updates)?;
        self.state.step += 1;
        let bias1 = 1.0 - self.state.beta1.powi(self.state.step as i32);
        let bias2 = 1.0 - self.state.beta2.powi(self.state.step as i32);
        let lr = self.state.lr as f32;
        let beta1 = self.state.beta1 as f32;
        let beta2 = self.state.beta2 as f32;
        let eps = self.state.eps as f32;
        let weight_decay = self.state.weight_decay as f32;
        let clip_scale = clip_scale as f32;
        let bias_correction1 = bias1 as f32;
        let bias_correction2 = bias2 as f32;

        for update in updates {
            let parameter = self.parameters[update.parameter_index].clone();
            if parameter.dtype() != DType::F32 {
                return Err(TensorError::DType(format!(
                    "CUDA compact sparse-row AdamW currently supports f32 parameters only, got {:?}",
                    parameter.dtype()
                )));
            }
            if !matches!(parameter.device(), Device::Cuda(_)) {
                return Err(TensorError::Device(format!(
                    "CUDA compact sparse-row AdamW requires CUDA parameters, got {:?}",
                    parameter.device()
                )));
            }
            let cuda_state = self.cuda_state_for_parameter(update.parameter_index, &parameter)?;
            parameter.apply_adamw_cuda_sparse_rows_compact_grad_f32(
                &update.selected_rows,
                &update.compact_grad_rows,
                &cuda_state.m,
                &cuda_state.v,
                heirloom_kernels::cuda::SparseAdamWRowsDims {
                    selected_rows: update.selected_rows.numel(),
                    rows: update.rows,
                    row_dim: update.row_dim,
                },
                heirloom_kernels::cuda::AdamWParams {
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    clip_scale,
                    bias_correction1,
                    bias_correction2,
                },
            )?;
        }
        Ok(())
    }

    fn step_cpu_parameter(
        &mut self,
        index: usize,
        parameter: &Tensor,
        clip_scale: f64,
        bias1: f64,
        bias2: f64,
    ) -> Result<()> {
        let Some(grad) = parameter.grad_f64() else {
            return Ok(());
        };
        let mut values = parameter.data_f64();
        for item in 0..values.len() {
            let decayed_grad = grad[item] * clip_scale + self.state.weight_decay * values[item];
            self.state.m[index][item] = self.state.beta1 * self.state.m[index][item]
                + (1.0 - self.state.beta1) * decayed_grad;
            self.state.v[index][item] = self.state.beta2 * self.state.v[index][item]
                + (1.0 - self.state.beta2) * decayed_grad * decayed_grad;
            let m_hat = self.state.m[index][item] / bias1;
            let v_hat = self.state.v[index][item] / bias2;
            values[item] -= self.state.lr * m_hat / (v_hat.sqrt() + self.state.eps);
        }
        match parameter.dtype() {
            DType::F32 => parameter
                .copy_from_data(&values.iter().map(|value| *value as f32).collect::<Vec<_>>())?,
            DType::BFloat16 => parameter.copy_from_data_bf16(
                &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            )?,
            DType::F64 => parameter.copy_from_data_f64(&values)?,
            DType::I64 | DType::Bool => unreachable!("floating check above"),
        }
        Ok(())
    }

    fn step_cuda_parameter(
        &mut self,
        index: usize,
        parameter: &Tensor,
        clip_scale: f64,
        bias1: f64,
        bias2: f64,
    ) -> Result<()> {
        if parameter.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA AdamW currently supports f32 parameters only, got {:?}",
                parameter.dtype()
            )));
        }
        let lr = self.state.lr as f32;
        let beta1 = self.state.beta1 as f32;
        let beta2 = self.state.beta2 as f32;
        let eps = self.state.eps as f32;
        let weight_decay = self.state.weight_decay as f32;
        let clip_scale = clip_scale as f32;
        let bias_correction1 = bias1 as f32;
        let bias_correction2 = bias2 as f32;
        let cuda_state = self.cuda_state_for_parameter(index, parameter)?;
        parameter.apply_adamw_cuda_f32(
            &cuda_state.m,
            &cuda_state.v,
            heirloom_kernels::cuda::AdamWParams {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                clip_scale,
                bias_correction1,
                bias_correction2,
            },
        )?;
        Ok(())
    }

    fn step_cpu_sparse_rows_parameter(
        &mut self,
        index: usize,
        parameter: &Tensor,
        update: &SparseAdamWRowsUpdate,
        clip_scale: f64,
        bias1: f64,
        bias2: f64,
    ) -> Result<()> {
        let Some(grad) = parameter.grad_f64() else {
            return Ok(());
        };
        self.validate_sparse_row_parameter_shape(parameter, update)?;
        let active_rows = active_sparse_update_rows_cpu(update)?;
        if active_rows.is_empty() {
            return Ok(());
        }
        let mut values = parameter.data_f64();
        for row in active_rows {
            let row_start = row * update.row_dim;
            for item in row_start..row_start + update.row_dim {
                let decayed_grad = grad[item] * clip_scale + self.state.weight_decay * values[item];
                self.state.m[index][item] = self.state.beta1 * self.state.m[index][item]
                    + (1.0 - self.state.beta1) * decayed_grad;
                self.state.v[index][item] = self.state.beta2 * self.state.v[index][item]
                    + (1.0 - self.state.beta2) * decayed_grad * decayed_grad;
                let m_hat = self.state.m[index][item] / bias1;
                let v_hat = self.state.v[index][item] / bias2;
                values[item] -= self.state.lr * m_hat / (v_hat.sqrt() + self.state.eps);
            }
        }
        parameter.copy_from_data(&values.iter().map(|value| *value as f32).collect::<Vec<_>>())?;
        Ok(())
    }

    fn cuda_state_for_parameter(
        &mut self,
        index: usize,
        parameter: &Tensor,
    ) -> Result<&AdamWCudaParamState> {
        if self.cuda_state[index].is_none() {
            let Device::Cuda(device_id) = parameter.device() else {
                return Err(TensorError::Device(format!(
                    "AdamW CUDA state expected CUDA parameter, got {:?}",
                    parameter.device()
                )));
            };
            if self.state.m[index].len() != parameter.numel()
                || self.state.v[index].len() != parameter.numel()
            {
                return Err(TensorError::InvalidOperation(format!(
                    "AdamW state shape mismatch for CUDA parameter {index}"
                )));
            }
            let device_ordinal = i32::try_from(device_id).map_err(|_| {
                TensorError::Device(format!(
                    "CUDA device id {device_id} does not fit into a CUDA ordinal"
                ))
            })?;
            let m_data = self.state.m[index]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let v_data = self.state.v[index]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            self.cuda_state[index] = Some(AdamWCudaParamState {
                m: CudaBuffer::from_f32(device_ordinal, &m_data).map_err(cuda_error)?,
                v: CudaBuffer::from_f32(device_ordinal, &v_data).map_err(cuda_error)?,
            });
        }
        Ok(self.cuda_state[index]
            .as_ref()
            .expect("CUDA AdamW state initialized above"))
    }

    fn clip_scale(&self) -> Result<f64> {
        let indices = (0..self.parameters.len()).collect::<Vec<_>>();
        self.clip_scale_for_indices(&indices)
    }

    fn clip_scale_for_indices(&self, parameter_indices: &[usize]) -> Result<f64> {
        let Some(max_norm) = self.state.clip_norm else {
            return Ok(1.0);
        };
        let mut sum_sq = 0.0;
        for &index in parameter_indices {
            let parameter = self.parameters.get(index).ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "AdamW clip-scale parameter index {index} is out of range for {} parameters",
                    self.parameters.len()
                ))
            })?;
            match parameter.device() {
                Device::Cpu => {
                    if let Some(grad) = parameter.grad_f64() {
                        sum_sq += grad.into_iter().map(|value| value * value).sum::<f64>();
                    }
                }
                Device::Cuda(_) => {
                    if let Some(value) = parameter.cuda_grad_sum_squares_f32()? {
                        let value = f64::from(value);
                        amp::record_amp_bf16_finite_check(
                            "cuda_gradient_sum_squares",
                            &format!("parameter:{}", parameter.id()),
                            parameter.dtype(),
                            parameter.device(),
                            value,
                        )?;
                        if !value.is_finite() {
                            return Err(TensorError::Autograd(format!(
                                "non-finite CUDA gradient sum-of-squares for parameter {}: {value}",
                                parameter.id()
                            )));
                        }
                        sum_sq += value;
                    }
                }
            }
        }
        if !sum_sq.is_finite() || sum_sq < 0.0 {
            return Err(TensorError::Autograd(format!(
                "non-finite or negative global gradient sum-of-squares: {sum_sq}"
            )));
        }
        let norm = sum_sq.sqrt();
        let device = self
            .parameters
            .iter()
            .map(Tensor::device)
            .find(|device| matches!(device, Device::Cuda(_)))
            .unwrap_or(Device::Cpu);
        amp::record_amp_bf16_finite_check(
            "global_gradient_norm",
            "optimizer",
            DType::F32,
            device,
            norm,
        )?;
        if !norm.is_finite() {
            return Err(TensorError::Autograd(format!(
                "non-finite global gradient norm: {norm}"
            )));
        }
        let scale = if norm > max_norm {
            max_norm / (norm + 1e-12)
        } else {
            1.0
        };
        amp::record_amp_bf16_finite_check(
            "gradient_clip_scale",
            "optimizer",
            DType::F32,
            device,
            scale,
        )?;
        Ok(scale)
    }

    fn clip_scale_for_sparse_row_updates(&self, updates: &[SparseAdamWRowsUpdate]) -> Result<f64> {
        let Some(max_norm) = self.state.clip_norm else {
            return Ok(1.0);
        };
        let mut sum_sq = 0.0;
        for update in updates {
            let parameter = self.parameters.get(update.parameter_index).ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "sparse AdamW clip-scale parameter index {} is out of range for {} parameters",
                    update.parameter_index,
                    self.parameters.len()
                ))
            })?;
            if parameter.grad_tensor().is_none() {
                continue;
            }
            if parameter.dtype() != DType::F32 {
                return Err(TensorError::DType(format!(
                    "sparse-row AdamW clip scale currently supports f32 parameters only, got {:?}",
                    parameter.dtype()
                )));
            }
            match parameter.device() {
                Device::Cpu => {
                    self.validate_sparse_row_parameter_shape(parameter, update)?;
                    let Some(grad) = parameter.grad_f64() else {
                        continue;
                    };
                    for row in active_sparse_update_rows_cpu(update)? {
                        let row_start = row * update.row_dim;
                        sum_sq += grad[row_start..row_start + update.row_dim]
                            .iter()
                            .map(|value| value * value)
                            .sum::<f64>();
                    }
                }
                Device::Cuda(_) => {
                    let selected_mask =
                        update.selected_rows.cuda_selected_rows_mask(update.rows)?;
                    let active_mask = if let Some(row_mask) = update.row_mask.as_ref() {
                        selected_mask.cuda_bool_and(row_mask)?
                    } else {
                        selected_mask
                    };
                    let compact_rows = active_mask.cuda_bool_mask_to_i64_indices()?;
                    if compact_rows.numel() == 0 {
                        continue;
                    }
                    let Some(compact_grad_rows) =
                        parameter.cuda_memory_gather_selected_grad_rows_f32_i64(&compact_rows)?
                    else {
                        continue;
                    };
                    let value = f64::from(compact_grad_rows.cuda_f32_sum_squares()?);
                    amp::record_amp_bf16_finite_check(
                        "cuda_sparse_selected_gradient_sum_squares",
                        &format!("parameter:{}", update.parameter_index),
                        DType::F32,
                        compact_grad_rows.device(),
                        value,
                    )?;
                    if !value.is_finite() {
                        return Err(TensorError::Autograd(format!(
                            "non-finite CUDA sparse selected gradient sum-of-squares for parameter {}: {value}",
                            update.parameter_index
                        )));
                    }
                    sum_sq += value;
                }
            }
        }
        if !sum_sq.is_finite() || sum_sq < 0.0 {
            return Err(TensorError::Autograd(format!(
                "non-finite or negative sparse selected gradient sum-of-squares: {sum_sq}"
            )));
        }
        let norm = sum_sq.sqrt();
        let device = updates
            .iter()
            .map(|update| update.selected_rows.device())
            .find(|device| matches!(device, Device::Cuda(_)))
            .unwrap_or(Device::Cpu);
        amp::record_amp_bf16_finite_check(
            "sparse_selected_gradient_norm",
            "optimizer",
            DType::F32,
            device,
            norm,
        )?;
        if !norm.is_finite() {
            return Err(TensorError::Autograd(format!(
                "non-finite sparse selected gradient norm: {norm}"
            )));
        }
        let scale = if norm > max_norm {
            max_norm / (norm + 1e-12)
        } else {
            1.0
        };
        amp::record_amp_bf16_finite_check(
            "sparse_selected_gradient_clip_scale",
            "optimizer",
            DType::F32,
            device,
            scale,
        )?;
        Ok(scale)
    }

    fn validate_sparse_row_updates(&self, updates: &[SparseAdamWRowsUpdate]) -> Result<()> {
        let mut seen = HashSet::new();
        for update in updates {
            if update.parameter_index >= self.parameters.len() {
                return Err(TensorError::InvalidOperation(format!(
                    "sparse AdamW parameter index {} is out of range for {} parameters",
                    update.parameter_index,
                    self.parameters.len()
                )));
            }
            if !seen.insert(update.parameter_index) {
                return Err(TensorError::InvalidOperation(format!(
                    "duplicate sparse AdamW update for parameter index {}",
                    update.parameter_index
                )));
            }
            if update.selected_rows.ndim() != 1 {
                return Err(TensorError::Shape(format!(
                    "sparse AdamW selected_rows must be rank-1, got {:?}",
                    update.selected_rows.shape()
                )));
            }
            if let Some(row_mask) = &update.row_mask {
                if row_mask.ndim() != 1 || row_mask.numel() != update.rows {
                    return Err(TensorError::Shape(format!(
                        "sparse AdamW row_mask must be rank-1 with {} rows, got shape {:?}",
                        update.rows,
                        row_mask.shape()
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_sparse_row_parameter_shape(
        &self,
        parameter: &Tensor,
        update: &SparseAdamWRowsUpdate,
    ) -> Result<()> {
        let expected_shape = vec![update.rows, update.row_dim];
        if parameter.shape() != expected_shape {
            return Err(TensorError::Shape(format!(
                "sparse AdamW expected parameter shape {:?}, got {:?}",
                expected_shape,
                parameter.shape()
            )));
        }
        if self.state.m[update.parameter_index].len() != parameter.numel()
            || self.state.v[update.parameter_index].len() != parameter.numel()
        {
            return Err(TensorError::InvalidOperation(format!(
                "AdamW state shape mismatch for sparse parameter {}",
                update.parameter_index
            )));
        }
        Ok(())
    }

    fn clip_scale_for_compact_row_updates(
        &self,
        updates: &[SparseAdamWCompactRowsUpdate],
    ) -> Result<f64> {
        let Some(max_norm) = self.state.clip_norm else {
            return Ok(1.0);
        };
        let mut sum_sq = 0.0;
        for update in updates {
            let value = f64::from(update.compact_grad_rows.cuda_f32_sum_squares()?);
            amp::record_amp_bf16_finite_check(
                "cuda_compact_sparse_gradient_sum_squares",
                &format!("parameter:{}", update.parameter_index),
                DType::F32,
                update.compact_grad_rows.device(),
                value,
            )?;
            if !value.is_finite() {
                return Err(TensorError::Autograd(format!(
                    "non-finite CUDA compact sparse gradient sum-of-squares for parameter {}: {value}",
                    update.parameter_index
                )));
            }
            sum_sq += value;
        }
        if !sum_sq.is_finite() || sum_sq < 0.0 {
            return Err(TensorError::Autograd(format!(
                "non-finite or negative compact sparse gradient sum-of-squares: {sum_sq}"
            )));
        }
        let norm = sum_sq.sqrt();
        let device = updates
            .iter()
            .map(|update| update.compact_grad_rows.device())
            .find(|device| matches!(device, Device::Cuda(_)))
            .unwrap_or(Device::Cpu);
        amp::record_amp_bf16_finite_check(
            "compact_sparse_gradient_norm",
            "optimizer",
            DType::F32,
            device,
            norm,
        )?;
        if !norm.is_finite() {
            return Err(TensorError::Autograd(format!(
                "non-finite compact sparse gradient norm: {norm}"
            )));
        }
        let scale = if norm > max_norm {
            max_norm / (norm + 1e-12)
        } else {
            1.0
        };
        amp::record_amp_bf16_finite_check(
            "compact_sparse_gradient_clip_scale",
            "optimizer",
            DType::F32,
            device,
            scale,
        )?;
        Ok(scale)
    }
}

fn active_sparse_update_rows_cpu(update: &SparseAdamWRowsUpdate) -> Result<Vec<usize>> {
    if update.selected_rows.device() != Device::Cpu {
        return Err(TensorError::Device(format!(
            "CPU sparse-row AdamW expected CPU selected_rows, got {:?}",
            update.selected_rows.device()
        )));
    }
    let row_mask = if let Some(row_mask) = &update.row_mask {
        if row_mask.device() != Device::Cpu {
            return Err(TensorError::Device(format!(
                "CPU sparse-row AdamW expected CPU row_mask, got {:?}",
                row_mask.device()
            )));
        }
        Some(row_mask.data_bool()?)
    } else {
        None
    };
    let mut active = vec![false; update.rows];
    for row in update.selected_rows.data_i64()? {
        let row = usize::try_from(row).map_err(|_| {
            TensorError::InvalidOperation(format!(
                "sparse AdamW selected row {row} must be non-negative"
            ))
        })?;
        if row >= update.rows {
            return Err(TensorError::InvalidOperation(format!(
                "sparse AdamW selected row {row} is out of range for {} rows",
                update.rows
            )));
        }
        if row_mask.as_ref().is_none_or(|mask| mask[row]) {
            active[row] = true;
        }
    }
    Ok(active
        .iter()
        .enumerate()
        .filter_map(|(row, selected)| selected.then_some(row))
        .collect())
}

fn validate_optimizer_parameter_indices(
    parameter_indices: &[usize],
    parameter_count: usize,
) -> Result<Vec<usize>> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(parameter_indices.len());
    for &index in parameter_indices {
        if index >= parameter_count {
            return Err(TensorError::InvalidOperation(format!(
                "AdamW parameter index {index} is out of range for {parameter_count} parameters"
            )));
        }
        if !seen.insert(index) {
            return Err(TensorError::InvalidOperation(format!(
                "duplicate AdamW parameter index {index}"
            )));
        }
        out.push(index);
    }
    Ok(out)
}

fn cuda_error(error: heirloom_kernels::cuda::CudaError) -> TensorError {
    TensorError::Device(error.to_string())
}

/// Saves named parameters as NumPy files plus a manifest under `dir`.
pub fn save_state_dict(module: &dyn Module, dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)
        .map_err(|err| TensorError::Io(format!("failed to create {}: {err}", dir.display())))?;

    let named_parameters = module.named_parameters("");
    let mut manifest = String::new();
    for (index, (name, tensor)) in named_parameters.iter().enumerate() {
        let filename = format!("{index:04}_{}.npy", sanitize_state_name(name));
        let snapshot = if tensor.device() == Device::Cpu {
            tensor.clone()
        } else {
            tensor.cpu()?
        };
        npy::write_npy(dir.join(&filename), &snapshot)?;
        manifest.push_str(name);
        manifest.push('\t');
        manifest.push_str(&filename);
        manifest.push('\n');
    }

    fs::write(dir.join("state.tsv"), manifest).map_err(|err| {
        TensorError::Io(format!(
            "failed to write {}: {err}",
            dir.join("state.tsv").display()
        ))
    })?;
    Ok(())
}

/// Loads a state dict after validating parameter names, shapes, and dtypes.
pub fn load_state_dict(module: &dyn Module, dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    let manifest_path = dir.join("state.tsv");
    let manifest = fs::read_to_string(&manifest_path).map_err(|err| {
        TensorError::Io(format!("failed to read {}: {err}", manifest_path.display()))
    })?;

    let mut files_by_name = HashMap::new();
    for (line_number, line) in manifest.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (name, filename) = line.split_once('\t').ok_or_else(|| {
            TensorError::Io(format!(
                "invalid state manifest line {}: expected name<TAB>filename",
                line_number + 1
            ))
        })?;
        files_by_name.insert(name.to_string(), filename.to_string());
    }

    let mut consumed = HashSet::new();
    for (name, parameter) in module.named_parameters("") {
        let filename = files_by_name.get(&name).ok_or_else(|| {
            TensorError::InvalidOperation(format!("missing state_dict parameter {name:?}"))
        })?;
        let loaded = npy::read_npy(dir.join(filename), parameter.requires_grad())?;
        parameter.copy_from(&loaded)?;
        consumed.insert(name);
    }

    for name in files_by_name.keys() {
        if !consumed.contains(name) {
            return Err(TensorError::InvalidOperation(format!(
                "state_dict contains unexpected parameter {name:?}"
            )));
        }
    }
    Ok(())
}

fn sanitize_state_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
