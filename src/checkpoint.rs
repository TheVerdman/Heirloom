use crate::data::TokenDatasetState;
use crate::memory_transformer::{MemoryTransformerConfig, MemoryTransformerLm};
use crate::nn::{
    load_state_dict, save_state_dict, AdamW, AdamWState, Module, TinyTransformerConfig,
    TinyTransformerLm,
};
use crate::rng::HeirloomRng;
use crate::tokenizer::BpeTokenizer;
use crate::{Device, Result, TensorError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LmModelFamily {
    #[default]
    TinyTransformer,
    MemoryTransformer,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CheckpointFamilyProbe {
    #[serde(default)]
    model_family: LmModelFamily,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmCheckpointMetadata {
    #[serde(default)]
    pub model_family: LmModelFamily,
    pub config: TinyTransformerConfig,
    #[serde(default)]
    pub dataset_rng_state: u64,
    #[serde(default)]
    pub dataset_batches_seen: usize,
    #[serde(default)]
    pub dataset_manifest_path: Option<String>,
    pub step: usize,
}

impl LmCheckpointMetadata {
    pub fn dataset_state(&self) -> TokenDatasetState {
        TokenDatasetState {
            rng_state: self.dataset_rng_state,
            batches_seen: self.dataset_batches_seen,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryLmCheckpointMetadata {
    pub model_family: LmModelFamily,
    pub memory_config: MemoryTransformerConfig,
    #[serde(default)]
    pub dataset_rng_state: u64,
    #[serde(default)]
    pub dataset_batches_seen: usize,
    #[serde(default)]
    pub dataset_manifest_path: Option<String>,
    pub step: usize,
}

impl MemoryLmCheckpointMetadata {
    pub fn dataset_state(&self) -> TokenDatasetState {
        TokenDatasetState {
            rng_state: self.dataset_rng_state,
            batches_seen: self.dataset_batches_seen,
        }
    }
}

pub struct LoadedLmCheckpoint {
    pub model: TinyTransformerLm,
    pub optimizer: AdamW,
    pub tokenizer: BpeTokenizer,
    pub metadata: LmCheckpointMetadata,
}

pub struct LoadedMemoryLmCheckpoint {
    pub model: MemoryTransformerLm,
    pub optimizer: AdamW,
    pub tokenizer: BpeTokenizer,
    pub metadata: MemoryLmCheckpointMetadata,
}

pub fn save_lm_checkpoint(
    dir: impl AsRef<Path>,
    model: &TinyTransformerLm,
    optimizer: &AdamW,
    tokenizer: &BpeTokenizer,
    dataset_rng_state: u64,
) -> Result<()> {
    save_lm_checkpoint_with_dataset_state(
        dir,
        model,
        optimizer,
        tokenizer,
        TokenDatasetState {
            rng_state: dataset_rng_state,
            batches_seen: 0,
        },
        None,
    )
}

pub fn save_lm_checkpoint_with_dataset_state(
    dir: impl AsRef<Path>,
    model: &TinyTransformerLm,
    optimizer: &AdamW,
    tokenizer: &BpeTokenizer,
    dataset_state: TokenDatasetState,
    dataset_manifest_path: Option<String>,
) -> Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)
        .map_err(|err| TensorError::Io(format!("failed to create {}: {err}", dir.display())))?;

    save_state_dict(model as &dyn Module, dir.join("model"))?;
    tokenizer.save(dir.join("tokenizer.json"))?;
    write_json(
        dir.join("metadata.json"),
        &LmCheckpointMetadata {
            model_family: LmModelFamily::TinyTransformer,
            config: model.config.clone(),
            dataset_rng_state: dataset_state.rng_state,
            dataset_batches_seen: dataset_state.batches_seen,
            dataset_manifest_path,
            step: optimizer.step_index(),
        },
    )?;
    write_json(dir.join("optimizer.json"), &optimizer.try_state()?)
}

pub fn save_memory_lm_checkpoint_with_dataset_state(
    dir: impl AsRef<Path>,
    model: &MemoryTransformerLm,
    optimizer: &AdamW,
    tokenizer: &BpeTokenizer,
    dataset_state: TokenDatasetState,
    dataset_manifest_path: Option<String>,
) -> Result<()> {
    save_memory_lm_checkpoint_with_dataset_state_and_step(
        dir,
        model,
        optimizer,
        tokenizer,
        dataset_state,
        dataset_manifest_path,
        optimizer.step_index(),
    )
}

pub fn save_memory_lm_checkpoint_with_dataset_state_and_step(
    dir: impl AsRef<Path>,
    model: &MemoryTransformerLm,
    optimizer: &AdamW,
    tokenizer: &BpeTokenizer,
    dataset_state: TokenDatasetState,
    dataset_manifest_path: Option<String>,
    step: usize,
) -> Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)
        .map_err(|err| TensorError::Io(format!("failed to create {}: {err}", dir.display())))?;

    save_state_dict(model as &dyn Module, dir.join("model"))?;
    tokenizer.save(dir.join("tokenizer.json"))?;
    write_json(
        dir.join("metadata.json"),
        &MemoryLmCheckpointMetadata {
            model_family: LmModelFamily::MemoryTransformer,
            memory_config: model.config.clone(),
            dataset_rng_state: dataset_state.rng_state,
            dataset_batches_seen: dataset_state.batches_seen,
            dataset_manifest_path,
            step,
        },
    )?;
    write_json(dir.join("optimizer.json"), &optimizer.try_state()?)
}

pub fn load_lm_checkpoint(dir: impl AsRef<Path>) -> Result<LoadedLmCheckpoint> {
    load_lm_checkpoint_on_device(dir, Device::Cpu)
}

pub fn load_lm_checkpoint_on_device(
    dir: impl AsRef<Path>,
    device: Device,
) -> Result<LoadedLmCheckpoint> {
    let dir = dir.as_ref();
    ensure_checkpoint_family(dir, LmModelFamily::TinyTransformer)?;
    let metadata: LmCheckpointMetadata = read_json(dir.join("metadata.json"))?;
    let optimizer_state: AdamWState = read_json(dir.join("optimizer.json"))?;
    let tokenizer = BpeTokenizer::load(dir.join("tokenizer.json"))?;
    let mut rng = HeirloomRng::new(0);
    let cpu_model = TinyTransformerLm::new(metadata.config.clone(), &mut rng)?;
    load_state_dict(&cpu_model, dir.join("model"))?;
    let model = cpu_model.to_device(device)?;
    let mut optimizer = AdamW::new(model.parameters(), optimizer_state.lr)?;
    optimizer.load_state(optimizer_state)?;
    Ok(LoadedLmCheckpoint {
        model,
        optimizer,
        tokenizer,
        metadata,
    })
}

pub fn load_memory_lm_checkpoint_on_device(
    dir: impl AsRef<Path>,
    device: Device,
) -> Result<LoadedMemoryLmCheckpoint> {
    let dir = dir.as_ref();
    ensure_checkpoint_family(dir, LmModelFamily::MemoryTransformer)?;
    let metadata: MemoryLmCheckpointMetadata = read_json(dir.join("metadata.json"))?;
    let optimizer_state: AdamWState = read_json(dir.join("optimizer.json"))?;
    let tokenizer = BpeTokenizer::load(dir.join("tokenizer.json"))?;
    let mut rng = HeirloomRng::new(0);
    let cpu_model = MemoryTransformerLm::new(metadata.memory_config.clone(), &mut rng)?;
    load_state_dict(&cpu_model, dir.join("model"))?;
    let model = cpu_model.to_device(device)?;
    let mut optimizer = AdamW::new(model.parameters(), optimizer_state.lr)?;
    optimizer.load_state(optimizer_state)?;
    Ok(LoadedMemoryLmCheckpoint {
        model,
        optimizer,
        tokenizer,
        metadata,
    })
}

fn ensure_checkpoint_family(dir: &Path, expected: LmModelFamily) -> Result<()> {
    let probe: CheckpointFamilyProbe = read_json(dir.join("metadata.json"))?;
    if probe.model_family != expected {
        return Err(TensorError::InvalidOperation(format!(
            "checkpoint model_family is {:?}, but loader expected {:?}",
            probe.model_family, expected
        )));
    }
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(value)
        .map_err(|err| TensorError::Io(format!("failed to serialize {}: {err}", path.display())))?;
    fs::write(path, json)
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", path.display())))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let json = fs::read_to_string(path)
        .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
    serde_json::from_str(&json)
        .map_err(|err| TensorError::Io(format!("failed to parse {}: {err}", path.display())))
}
