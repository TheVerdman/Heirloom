use crate::rng::HeirloomRng;
use crate::tokenizer::{stable_hash_bytes, BpeTokenizer};
use crate::{Result, Tensor, TensorError};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const TINYSTORIES_VALID_URL: &str =
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt";
pub const DATASET_MANIFEST_FORMAT: &str = "heirloom.token_dataset";
pub const DATASET_MANIFEST_VERSION: u32 = 1;
pub const DATASET_MANIFEST_VERSION_V2: u32 = 2;
pub const TOKEN_SHARD_FORMAT: &str = "heirloom.token_shard";
pub const TOKEN_SHARD_VERSION: u32 = 1;
pub const DATASET_STORAGE_JSON: &str = "json";
pub const DATASET_STORAGE_BINARY_SHARDS: &str = "binary_shards";
pub const CORPUS_BLEND_MANIFEST_FORMAT: &str = "heirloom.corpus_blend";
pub const CORPUS_BLEND_MANIFEST_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenDatasetState {
    pub rng_state: u64,
    pub batches_seen: usize,
}

impl TokenDatasetState {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng_state: HeirloomRng::new(seed).state(),
            batches_seen: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedDataManifest {
    pub format: String,
    pub version: u32,
    #[serde(default = "default_storage")]
    pub storage: String,
    pub source_path: String,
    pub source_bytes: usize,
    pub source_hash: String,
    pub tokenizer_path: String,
    pub tokenizer_hash: String,
    #[serde(default)]
    pub train_tokens_path: String,
    #[serde(default)]
    pub valid_tokens_path: String,
    pub train_tokens: usize,
    pub valid_tokens: usize,
    #[serde(default)]
    pub train_hash: String,
    #[serde(default)]
    pub valid_hash: String,
    pub split: DataSplit,
    #[serde(default)]
    pub sources: Vec<PreparedDataSource>,
    #[serde(default)]
    pub train_shards: Vec<PreparedDataShard>,
    #[serde(default)]
    pub valid_shards: Vec<PreparedDataShard>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSplit {
    pub kind: String,
    pub valid_fraction: f64,
}

impl Default for DataSplit {
    fn default() -> Self {
        Self {
            kind: "contiguous_tail_valid".to_string(),
            valid_fraction: 0.1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedDataSource {
    pub source_id: String,
    pub path: String,
    pub bytes: usize,
    pub hash: String,
    pub tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedDataShard {
    pub split: String,
    pub metadata_path: String,
    pub tokens: usize,
    pub token_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CorpusBlendManifest {
    pub format: String,
    pub version: u32,
    pub blend_id: String,
    pub tokenizer_target_vocab_size: usize,
    pub tokenizer_family: String,
    pub sources: Vec<CorpusBlendSource>,
    pub local_source_count: usize,
    pub planned_source_count: usize,
    pub local_record_count: usize,
    pub local_bytes: usize,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CorpusBlendSource {
    pub source_id: String,
    pub display_name: String,
    pub kind: String,
    pub status: String,
    pub role: String,
    pub source_url: Option<String>,
    pub path: Option<String>,
    pub metadata_path: Option<String>,
    pub data_format: String,
    pub license: String,
    pub license_url: Option<String>,
    pub license_status: String,
    pub provenance: Vec<String>,
    pub sampling_weight: f64,
    pub include_in_tokenizer_training: bool,
    pub include_in_pretraining: bool,
    pub include_in_memory_trace_training: bool,
    pub synthetic: bool,
    pub local_bytes: Option<usize>,
    pub local_records: Option<usize>,
    pub content_hash: Option<String>,
    pub metadata_hash: Option<String>,
    #[serde(default)]
    pub split_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub domain_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub category_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub task_kind_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub extra: BTreeMap<String, JsonValue>,
}

fn default_storage() -> String {
    DATASET_STORAGE_JSON.to_string()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TokenShardDType {
    U16,
    U32,
}

impl TokenShardDType {
    pub fn bytes_per_token(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }

    fn max_token_id(self) -> usize {
        match self {
            Self::U16 => u16::MAX as usize,
            Self::U32 => u32::MAX as usize,
        }
    }

    fn for_tokens(tokens: &[usize]) -> Result<Self> {
        let max_token = tokens.iter().copied().max().unwrap_or(0);
        if max_token <= u16::MAX as usize {
            Ok(Self::U16)
        } else if max_token <= u32::MAX as usize {
            Ok(Self::U32)
        } else {
            Err(TensorError::InvalidOperation(format!(
                "token id {max_token} exceeds u32 token shard limit"
            )))
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenShardMetadata {
    pub format: String,
    pub version: u32,
    pub payload_path: String,
    pub dtype: TokenShardDType,
    pub token_count: usize,
    pub token_hash: String,
    pub payload_bytes: usize,
    pub payload_hash: String,
}

impl TokenShardMetadata {
    pub fn validate(&self) -> Result<()> {
        if self.format != TOKEN_SHARD_FORMAT {
            return Err(TensorError::InvalidOperation(format!(
                "unsupported token shard format {}, expected {}",
                self.format, TOKEN_SHARD_FORMAT
            )));
        }
        if self.version != TOKEN_SHARD_VERSION {
            return Err(TensorError::InvalidOperation(format!(
                "unsupported token shard version {}, expected {}",
                self.version, TOKEN_SHARD_VERSION
            )));
        }
        let expected_payload_bytes = self
            .token_count
            .checked_mul(self.dtype.bytes_per_token())
            .ok_or_else(|| {
                TensorError::InvalidOperation("token shard payload byte count overflow".to_string())
            })?;
        if self.payload_bytes != expected_payload_bytes {
            return Err(TensorError::InvalidOperation(format!(
                "token shard payload byte mismatch: metadata={} expected={}",
                self.payload_bytes, expected_payload_bytes
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PreparedTokenData {
    pub manifest_path: PathBuf,
    pub root: PathBuf,
    pub manifest: PreparedDataManifest,
}

#[derive(Clone, Debug)]
pub struct PreparedDataOptions {
    pub valid_fraction: f64,
    pub max_bytes: Option<usize>,
    pub shard_tokens: Option<usize>,
}

impl Default for PreparedDataOptions {
    fn default() -> Self {
        Self {
            valid_fraction: 0.1,
            max_bytes: None,
            shard_tokens: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TokenDataset {
    tokens: Vec<usize>,
    block_size: usize,
    rng: HeirloomRng,
    batches_seen: usize,
}

impl TokenDataset {
    pub fn new(tokens: Vec<usize>, block_size: usize, seed: u64) -> Result<Self> {
        if block_size == 0 {
            return Err(TensorError::InvalidOperation(
                "block_size must be greater than zero".to_string(),
            ));
        }
        if tokens.len() <= block_size {
            return Err(TensorError::InvalidOperation(format!(
                "dataset needs more than block_size tokens, got {} tokens and block_size {}",
                tokens.len(),
                block_size
            )));
        }
        Ok(Self {
            tokens,
            block_size,
            rng: HeirloomRng::new(seed),
            batches_seen: 0,
        })
    }

    pub fn with_rng_state(tokens: Vec<usize>, block_size: usize, rng_state: u64) -> Result<Self> {
        Self::with_state(
            tokens,
            block_size,
            TokenDatasetState {
                rng_state,
                batches_seen: 0,
            },
        )
    }

    pub fn with_state(
        tokens: Vec<usize>,
        block_size: usize,
        state: TokenDatasetState,
    ) -> Result<Self> {
        let mut dataset = Self::new(tokens, block_size, 0)?;
        dataset.rng = HeirloomRng::from_state(state.rng_state);
        dataset.batches_seen = state.batches_seen;
        Ok(dataset)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn rng_state(&self) -> u64 {
        self.rng.state()
    }

    pub fn state(&self) -> TokenDatasetState {
        TokenDatasetState {
            rng_state: self.rng.state(),
            batches_seen: self.batches_seen,
        }
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<(Tensor, Tensor)> {
        if batch_size == 0 {
            return Err(TensorError::InvalidOperation(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        let span = self.tokens.len() - self.block_size - 1;
        let mut inputs = Vec::with_capacity(batch_size * self.block_size);
        let mut targets = Vec::with_capacity(batch_size * self.block_size);

        for _ in 0..batch_size {
            let start = if span == 0 {
                0
            } else {
                (self.rng.next_u64() as usize) % span
            };
            for offset in 0..self.block_size {
                inputs.push(self.tokens[start + offset] as i64);
                targets.push(self.tokens[start + offset + 1] as i64);
            }
        }
        self.batches_seen += 1;

        Ok((
            Tensor::from_i64(inputs, &[batch_size, self.block_size], false)?,
            Tensor::from_i64(targets, &[batch_size, self.block_size], false)?,
        ))
    }

    pub fn deterministic_sharded_batch(
        tokens: &[usize],
        block_size: usize,
        batch_size: usize,
        seed: u64,
        global_step: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        if block_size == 0 {
            return Err(TensorError::InvalidOperation(
                "block_size must be greater than zero".to_string(),
            ));
        }
        if batch_size == 0 {
            return Err(TensorError::InvalidOperation(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        if world_size == 0 {
            return Err(TensorError::InvalidOperation(
                "world_size must be greater than zero".to_string(),
            ));
        }
        if rank >= world_size {
            return Err(TensorError::InvalidOperation(format!(
                "rank must be less than world_size, got rank={rank} world_size={world_size}"
            )));
        }
        if tokens.len() <= block_size {
            return Err(TensorError::InvalidOperation(format!(
                "dataset needs more than block_size tokens, got {} tokens and block_size {}",
                tokens.len(),
                block_size
            )));
        }

        let span = tokens.len() - block_size - 1;
        let mut inputs = Vec::with_capacity(batch_size * block_size);
        let mut targets = Vec::with_capacity(batch_size * block_size);
        for sample in 0..batch_size {
            let logical_sample = (global_step as u64)
                .wrapping_mul(world_size as u64)
                .wrapping_mul(batch_size as u64)
                .wrapping_add((rank as u64).wrapping_mul(batch_size as u64))
                .wrapping_add(sample as u64);
            let start = if span == 0 {
                0
            } else {
                (mix_u64(seed ^ logical_sample) as usize) % span
            };
            for offset in 0..block_size {
                inputs.push(tokens[start + offset] as i64);
                targets.push(tokens[start + offset + 1] as i64);
            }
        }

        Ok((
            Tensor::from_i64(inputs, &[batch_size, block_size], false)?,
            Tensor::from_i64(targets, &[batch_size, block_size], false)?,
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenDataSplit {
    Train,
    Valid,
}

impl TokenDataSplit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Valid => "valid",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamingTokenLoaderStats {
    pub batches: usize,
    pub samples: usize,
    pub tokens: usize,
    pub read_calls: usize,
    pub bytes_read: usize,
}

pub struct StreamingTokenDataset {
    shards: Vec<OpenTokenShard>,
    total_tokens: usize,
    block_size: usize,
    rng: HeirloomRng,
    batches_seen: usize,
    stats: StreamingTokenLoaderStats,
}

impl StreamingTokenDataset {
    pub fn open(
        prepared: &PreparedTokenData,
        split: TokenDataSplit,
        block_size: usize,
        state: TokenDatasetState,
    ) -> Result<Self> {
        if !prepared.manifest.is_binary_sharded() {
            return Err(TensorError::InvalidOperation(format!(
                "streaming token dataset requires manifest v2 with storage={DATASET_STORAGE_BINARY_SHARDS:?}, got version={} storage={:?}",
                prepared.manifest.version, prepared.manifest.storage
            )));
        }
        if block_size == 0 {
            return Err(TensorError::InvalidOperation(
                "block_size must be greater than zero".to_string(),
            ));
        }

        let mut shards = Vec::new();
        let mut total_tokens = 0usize;
        for shard in prepared.shards(split) {
            let metadata_path = prepared.resolve(&shard.metadata_path);
            let open = OpenTokenShard::open(&metadata_path, total_tokens)?;
            if open.metadata.token_count != shard.tokens
                || open.metadata.token_hash != shard.token_hash
            {
                return Err(TensorError::InvalidOperation(format!(
                    "{} shard metadata mismatch for {}: manifest len/hash={}/{}, shard len/hash={}/{}",
                    split.as_str(),
                    metadata_path.display(),
                    shard.tokens,
                    shard.token_hash,
                    open.metadata.token_count,
                    open.metadata.token_hash
                )));
            }
            total_tokens = total_tokens
                .checked_add(open.metadata.token_count)
                .ok_or_else(|| {
                    TensorError::InvalidOperation("streaming token count overflow".to_string())
                })?;
            shards.push(open);
        }

        if total_tokens <= block_size {
            return Err(TensorError::InvalidOperation(format!(
                "streaming dataset needs more than block_size tokens, got {total_tokens} tokens and block_size {block_size}"
            )));
        }

        Ok(Self {
            shards,
            total_tokens,
            block_size,
            rng: HeirloomRng::from_state(state.rng_state),
            batches_seen: state.batches_seen,
            stats: StreamingTokenLoaderStats::default(),
        })
    }

    pub fn len(&self) -> usize {
        self.total_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.total_tokens == 0
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn state(&self) -> TokenDatasetState {
        TokenDatasetState {
            rng_state: self.rng.state(),
            batches_seen: self.batches_seen,
        }
    }

    pub fn loader_stats(&self) -> StreamingTokenLoaderStats {
        self.stats
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<(Tensor, Tensor)> {
        if batch_size == 0 {
            return Err(TensorError::InvalidOperation(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        let span = self.total_tokens - self.block_size - 1;
        let mut inputs = Vec::with_capacity(batch_size * self.block_size);
        let mut targets = Vec::with_capacity(batch_size * self.block_size);
        for _ in 0..batch_size {
            let start = if span == 0 {
                0
            } else {
                (self.rng.next_u64() as usize) % span
            };
            self.push_window(start, &mut inputs, &mut targets)?;
        }
        self.batches_seen += 1;
        self.stats.batches += 1;
        self.stats.samples += batch_size;
        self.stats.tokens += batch_size * self.block_size;
        Ok((
            Tensor::from_i64(inputs, &[batch_size, self.block_size], false)?,
            Tensor::from_i64(targets, &[batch_size, self.block_size], false)?,
        ))
    }

    pub fn deterministic_sharded_batch(
        &mut self,
        batch_size: usize,
        seed: u64,
        global_step: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        if batch_size == 0 {
            return Err(TensorError::InvalidOperation(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        if world_size == 0 {
            return Err(TensorError::InvalidOperation(
                "world_size must be greater than zero".to_string(),
            ));
        }
        if rank >= world_size {
            return Err(TensorError::InvalidOperation(format!(
                "rank must be less than world_size, got rank={rank} world_size={world_size}"
            )));
        }
        let span = self.total_tokens - self.block_size - 1;
        let mut inputs = Vec::with_capacity(batch_size * self.block_size);
        let mut targets = Vec::with_capacity(batch_size * self.block_size);
        for sample in 0..batch_size {
            let logical_sample = (global_step as u64)
                .wrapping_mul(world_size as u64)
                .wrapping_mul(batch_size as u64)
                .wrapping_add((rank as u64).wrapping_mul(batch_size as u64))
                .wrapping_add(sample as u64);
            let start = if span == 0 {
                0
            } else {
                (mix_u64(seed ^ logical_sample) as usize) % span
            };
            self.push_window(start, &mut inputs, &mut targets)?;
        }
        self.stats.batches += 1;
        self.stats.samples += batch_size;
        self.stats.tokens += batch_size * self.block_size;
        Ok((
            Tensor::from_i64(inputs, &[batch_size, self.block_size], false)?,
            Tensor::from_i64(targets, &[batch_size, self.block_size], false)?,
        ))
    }

    pub fn sequential_batch_at(
        &mut self,
        token_offset: usize,
        batch_size: usize,
    ) -> Result<Option<(Tensor, Tensor, usize)>> {
        if batch_size == 0 {
            return Err(TensorError::InvalidOperation(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        if token_offset + self.block_size >= self.total_tokens {
            return Ok(None);
        }
        let mut inputs = Vec::with_capacity(batch_size * self.block_size);
        let mut targets = Vec::with_capacity(batch_size * self.block_size);
        let mut actual_batch = 0usize;
        let mut offset = token_offset;
        while actual_batch < batch_size && offset + self.block_size < self.total_tokens {
            self.push_window(offset, &mut inputs, &mut targets)?;
            offset += self.block_size;
            actual_batch += 1;
        }
        if actual_batch == 0 {
            return Ok(None);
        }
        self.stats.batches += 1;
        self.stats.samples += actual_batch;
        self.stats.tokens += actual_batch * self.block_size;
        Ok(Some((
            Tensor::from_i64(inputs, &[actual_batch, self.block_size], false)?,
            Tensor::from_i64(targets, &[actual_batch, self.block_size], false)?,
            actual_batch,
        )))
    }

    fn push_window(
        &mut self,
        start: usize,
        inputs: &mut Vec<i64>,
        targets: &mut Vec<i64>,
    ) -> Result<()> {
        let window = self.read_tokens(start, self.block_size + 1)?;
        for offset in 0..self.block_size {
            inputs.push(window[offset] as i64);
            targets.push(window[offset + 1] as i64);
        }
        Ok(())
    }

    fn read_tokens(&mut self, start: usize, len: usize) -> Result<Vec<usize>> {
        if start
            .checked_add(len)
            .is_none_or(|end| end > self.total_tokens)
        {
            return Err(TensorError::InvalidOperation(format!(
                "streaming token read out of range: start={start} len={len} total={}",
                self.total_tokens
            )));
        }
        let mut remaining = len;
        let mut cursor = start;
        let mut tokens = Vec::with_capacity(len);
        while remaining > 0 {
            let shard_index = self.find_shard(cursor)?;
            let shard = &mut self.shards[shard_index];
            let offset_in_shard = cursor - shard.start_token;
            let available = shard.metadata.token_count - offset_in_shard;
            let take = remaining.min(available);
            let mut chunk = shard.read_range(offset_in_shard, take)?;
            self.stats.read_calls += 1;
            self.stats.bytes_read += take * shard.metadata.dtype.bytes_per_token();
            tokens.append(&mut chunk);
            cursor += take;
            remaining -= take;
        }
        Ok(tokens)
    }

    fn find_shard(&self, token_index: usize) -> Result<usize> {
        self.shards
            .iter()
            .position(|shard| {
                token_index >= shard.start_token
                    && token_index < shard.start_token + shard.metadata.token_count
            })
            .ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "token index {token_index} is outside streaming shard ranges"
                ))
            })
    }
}

struct OpenTokenShard {
    metadata: TokenShardMetadata,
    start_token: usize,
    file: File,
}

impl OpenTokenShard {
    fn open(metadata_path: &Path, start_token: usize) -> Result<Self> {
        let metadata = load_token_shard_metadata(metadata_path)?;
        let root = metadata_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let payload_path = resolve_manifest_path(&root, &metadata.payload_path);
        validate_token_shard_payload_file(&payload_path, &metadata)?;
        let file = File::open(&payload_path).map_err(|err| {
            TensorError::Io(format!("failed to open {}: {err}", payload_path.display()))
        })?;
        Ok(Self {
            metadata,
            start_token,
            file,
        })
    }

    fn read_range(&mut self, start: usize, len: usize) -> Result<Vec<usize>> {
        if start
            .checked_add(len)
            .is_none_or(|end| end > self.metadata.token_count)
        {
            return Err(TensorError::InvalidOperation(format!(
                "token shard read out of range: start={start} len={len} token_count={}",
                self.metadata.token_count
            )));
        }
        let width = self.metadata.dtype.bytes_per_token();
        let byte_offset = start.checked_mul(width).ok_or_else(|| {
            TensorError::InvalidOperation("token shard seek overflow".to_string())
        })?;
        let byte_len = len.checked_mul(width).ok_or_else(|| {
            TensorError::InvalidOperation("token shard read length overflow".to_string())
        })?;
        let mut payload = vec![0u8; byte_len];
        self.file
            .seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|err| TensorError::Io(format!("failed to seek token shard: {err}")))?;
        self.file
            .read_exact(&mut payload)
            .map_err(|err| TensorError::Io(format!("failed to read token shard: {err}")))?;
        decode_token_shard_payload(&payload, self.metadata.dtype)
    }
}

fn mix_u64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl PreparedDataManifest {
    pub fn validate(&self) -> Result<()> {
        if self.format != DATASET_MANIFEST_FORMAT {
            return Err(TensorError::InvalidOperation(format!(
                "unsupported dataset manifest format {}, expected {}",
                self.format, DATASET_MANIFEST_FORMAT
            )));
        }
        if self.version != DATASET_MANIFEST_VERSION && self.version != DATASET_MANIFEST_VERSION_V2 {
            return Err(TensorError::InvalidOperation(format!(
                "unsupported dataset manifest version {}, expected {} or {}",
                self.version, DATASET_MANIFEST_VERSION, DATASET_MANIFEST_VERSION_V2
            )));
        }
        if !(0.0..1.0).contains(&self.split.valid_fraction) {
            return Err(TensorError::InvalidOperation(format!(
                "valid_fraction must be in [0, 1), got {}",
                self.split.valid_fraction
            )));
        }
        if self.train_tokens == 0 || self.valid_tokens == 0 {
            return Err(TensorError::InvalidOperation(format!(
                "prepared dataset must have non-empty train and valid splits, got train={} valid={}",
                self.train_tokens, self.valid_tokens
            )));
        }
        match self.version {
            DATASET_MANIFEST_VERSION => self.validate_v1(),
            DATASET_MANIFEST_VERSION_V2 => self.validate_v2(),
            _ => unreachable!("manifest version already checked"),
        }
    }

    pub fn is_binary_sharded(&self) -> bool {
        self.version == DATASET_MANIFEST_VERSION_V2 && self.storage == DATASET_STORAGE_BINARY_SHARDS
    }

    fn validate_v1(&self) -> Result<()> {
        if self.storage != DATASET_STORAGE_JSON {
            return Err(TensorError::InvalidOperation(format!(
                "dataset manifest v1 storage must be {DATASET_STORAGE_JSON:?}, got {:?}",
                self.storage
            )));
        }
        if self.train_tokens_path.is_empty() || self.valid_tokens_path.is_empty() {
            return Err(TensorError::InvalidOperation(
                "dataset manifest v1 requires train_tokens_path and valid_tokens_path".to_string(),
            ));
        }
        if self.train_hash.is_empty() || self.valid_hash.is_empty() {
            return Err(TensorError::InvalidOperation(
                "dataset manifest v1 requires train_hash and valid_hash".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_v2(&self) -> Result<()> {
        if self.storage != DATASET_STORAGE_BINARY_SHARDS {
            return Err(TensorError::InvalidOperation(format!(
                "dataset manifest v2 storage must be {DATASET_STORAGE_BINARY_SHARDS:?}, got {:?}",
                self.storage
            )));
        }
        if self.sources.is_empty() {
            return Err(TensorError::InvalidOperation(
                "dataset manifest v2 requires at least one source".to_string(),
            ));
        }
        if self.train_shards.is_empty() || self.valid_shards.is_empty() {
            return Err(TensorError::InvalidOperation(
                "dataset manifest v2 requires non-empty train_shards and valid_shards".to_string(),
            ));
        }
        let train_tokens = self
            .train_shards
            .iter()
            .map(|shard| shard.tokens)
            .sum::<usize>();
        let valid_tokens = self
            .valid_shards
            .iter()
            .map(|shard| shard.tokens)
            .sum::<usize>();
        if train_tokens != self.train_tokens || valid_tokens != self.valid_tokens {
            return Err(TensorError::InvalidOperation(format!(
                "dataset manifest v2 shard token counts do not match split totals: \
                 train_shards={train_tokens} train_total={} valid_shards={valid_tokens} valid_total={}",
                self.train_tokens, self.valid_tokens
            )));
        }
        Ok(())
    }
}

impl PreparedTokenData {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let manifest_path = path.as_ref().to_path_buf();
        let json = fs::read_to_string(&manifest_path).map_err(|err| {
            TensorError::Io(format!("failed to read {}: {err}", manifest_path.display()))
        })?;
        let manifest: PreparedDataManifest = serde_json::from_str(&json).map_err(|err| {
            TensorError::Io(format!(
                "failed to parse {}: {err}",
                manifest_path.display()
            ))
        })?;
        manifest.validate()?;
        let root = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self {
            manifest_path,
            root,
            manifest,
        })
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.resolve(&self.manifest.tokenizer_path)
    }

    pub fn train_tokens_path(&self) -> PathBuf {
        self.resolve(&self.manifest.train_tokens_path)
    }

    pub fn valid_tokens_path(&self) -> PathBuf {
        self.resolve(&self.manifest.valid_tokens_path)
    }

    pub fn load_tokenizer(&self) -> Result<BpeTokenizer> {
        let tokenizer = BpeTokenizer::load(self.tokenizer_path())?;
        let actual_hash = tokenizer.fingerprint()?;
        if actual_hash != self.manifest.tokenizer_hash {
            return Err(TensorError::InvalidOperation(format!(
                "tokenizer hash mismatch for prepared dataset: manifest={} actual={}",
                self.manifest.tokenizer_hash, actual_hash
            )));
        }
        Ok(tokenizer)
    }

    pub fn train_tokens(&self) -> Result<Vec<usize>> {
        if self.manifest.is_binary_sharded() {
            return self.materialize_shards(TokenDataSplit::Train);
        }
        let tokens = read_token_file(self.train_tokens_path())?;
        validate_loaded_tokens(
            "train",
            &tokens,
            self.manifest.train_tokens,
            &self.manifest.train_hash,
        )?;
        Ok(tokens)
    }

    pub fn valid_tokens(&self) -> Result<Vec<usize>> {
        if self.manifest.is_binary_sharded() {
            return self.materialize_shards(TokenDataSplit::Valid);
        }
        let tokens = read_token_file(self.valid_tokens_path())?;
        validate_loaded_tokens(
            "valid",
            &tokens,
            self.manifest.valid_tokens,
            &self.manifest.valid_hash,
        )?;
        Ok(tokens)
    }

    pub fn streaming_dataset(
        &self,
        split: TokenDataSplit,
        block_size: usize,
        state: TokenDatasetState,
    ) -> Result<StreamingTokenDataset> {
        StreamingTokenDataset::open(self, split, block_size, state)
    }

    pub fn storage_kind(&self) -> &str {
        &self.manifest.storage
    }

    fn materialize_shards(&self, split: TokenDataSplit) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        for shard in self.shards(split) {
            let shard_tokens = read_token_shard(self.resolve(&shard.metadata_path))?;
            validate_loaded_tokens(
                split.as_str(),
                &shard_tokens,
                shard.tokens,
                &shard.token_hash,
            )?;
            tokens.extend(shard_tokens);
        }
        let (expected_len, expected_hash) = match split {
            TokenDataSplit::Train => (self.manifest.train_tokens, &self.manifest.train_hash),
            TokenDataSplit::Valid => (self.manifest.valid_tokens, &self.manifest.valid_hash),
        };
        if !expected_hash.is_empty() {
            validate_loaded_tokens(split.as_str(), &tokens, expected_len, expected_hash)?;
        } else if tokens.len() != expected_len {
            return Err(TensorError::InvalidOperation(format!(
                "{} token count mismatch: expected {}, got {}",
                split.as_str(),
                expected_len,
                tokens.len()
            )));
        }
        Ok(tokens)
    }

    fn shards(&self, split: TokenDataSplit) -> &[PreparedDataShard] {
        match split {
            TokenDataSplit::Train => &self.manifest.train_shards,
            TokenDataSplit::Valid => &self.manifest.valid_shards,
        }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            path
        } else {
            self.root.join(path)
        }
    }
}

pub fn build_qb_native_corpus_blend_manifest(
    blend_id: impl Into<String>,
    tokenizer_target_vocab_size: usize,
    qb_root: Option<&Path>,
) -> Result<CorpusBlendManifest> {
    if tokenizer_target_vocab_size < crate::tokenizer::BYTE_OFFSET + crate::tokenizer::BYTE_VOCAB {
        return Err(TensorError::InvalidOperation(format!(
            "tokenizer target vocab size must be at least {}, got {}",
            crate::tokenizer::BYTE_OFFSET + crate::tokenizer::BYTE_VOCAB,
            tokenizer_target_vocab_size
        )));
    }

    let mut sources = planned_qb_native_sources();
    if let Some(qb_root) = qb_root {
        sources.extend(scan_qb_trace_sources(qb_root)?);
    }
    let local_source_count = sources
        .iter()
        .filter(|source| source.path.is_some())
        .count();
    let planned_source_count = sources.len().saturating_sub(local_source_count);
    let local_record_count = sources
        .iter()
        .filter_map(|source| source.local_records)
        .sum::<usize>();
    let local_bytes = sources
        .iter()
        .filter_map(|source| source.local_bytes)
        .sum::<usize>();
    let manifest = CorpusBlendManifest {
        format: CORPUS_BLEND_MANIFEST_FORMAT.to_string(),
        version: CORPUS_BLEND_MANIFEST_VERSION,
        blend_id: blend_id.into(),
        tokenizer_target_vocab_size,
        tokenizer_family: "heirloom.byte_bpe".to_string(),
        sources,
        local_source_count,
        planned_source_count,
        local_record_count,
        local_bytes,
        notes: vec![
            "Token dataset manifests describe prepared shards; this corpus blend manifest describes source-level provenance, licensing, and sampling policy.".to_string(),
            "TinyStories remains a smoke/regression source, not a default production pretraining blend member.".to_string(),
            "External source license fields are planning metadata and must be revalidated before production download or redistribution.".to_string(),
        ],
    };
    validate_corpus_blend_manifest(&manifest)?;
    Ok(manifest)
}

pub fn write_corpus_blend_manifest(
    manifest: &CorpusBlendManifest,
    out: impl AsRef<Path>,
) -> Result<()> {
    validate_corpus_blend_manifest(manifest)?;
    let out = out.as_ref();
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                TensorError::Io(format!("failed to create {}: {err}", parent.display()))
            })?;
        }
    }
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|err| TensorError::Io(format!("failed to serialize corpus blend: {err}")))?;
    fs::write(out, json + "\n")
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", out.display())))
}

fn validate_corpus_blend_manifest(manifest: &CorpusBlendManifest) -> Result<()> {
    if manifest.format != CORPUS_BLEND_MANIFEST_FORMAT {
        return Err(TensorError::InvalidOperation(format!(
            "unsupported corpus blend manifest format {}, expected {}",
            manifest.format, CORPUS_BLEND_MANIFEST_FORMAT
        )));
    }
    if manifest.version != CORPUS_BLEND_MANIFEST_VERSION {
        return Err(TensorError::InvalidOperation(format!(
            "unsupported corpus blend manifest version {}, expected {}",
            manifest.version, CORPUS_BLEND_MANIFEST_VERSION
        )));
    }
    if manifest.sources.is_empty() {
        return Err(TensorError::InvalidOperation(
            "corpus blend manifest requires at least one source".to_string(),
        ));
    }
    let mut ids = BTreeMap::<&str, usize>::new();
    for source in &manifest.sources {
        if source.source_id.trim().is_empty() {
            return Err(TensorError::InvalidOperation(
                "corpus blend source_id must not be empty".to_string(),
            ));
        }
        *ids.entry(&source.source_id).or_default() += 1;
        if source.sampling_weight < 0.0 || !source.sampling_weight.is_finite() {
            return Err(TensorError::InvalidOperation(format!(
                "source {} has invalid sampling_weight {}",
                source.source_id, source.sampling_weight
            )));
        }
        if source.path.is_some() && source.content_hash.is_none() {
            return Err(TensorError::InvalidOperation(format!(
                "local source {} requires content_hash",
                source.source_id
            )));
        }
    }
    if let Some((id, count)) = ids.into_iter().find(|(_, count)| *count > 1) {
        return Err(TensorError::InvalidOperation(format!(
            "duplicate corpus blend source_id {id} count={count}"
        )));
    }
    Ok(())
}

fn planned_qb_native_sources() -> Vec<CorpusBlendSource> {
    vec![
        CorpusBlendSource {
            source_id: "allenai.dolma.v1_7".to_string(),
            display_name: "Dolma v1.7".to_string(),
            kind: "external_pretraining_corpus".to_string(),
            status: "planned".to_string(),
            role: "general_language_backbone".to_string(),
            source_url: Some("https://huggingface.co/datasets/allenai/dolma".to_string()),
            path: None,
            metadata_path: None,
            data_format: "jsonl.gz".to_string(),
            license: "ODC-BY; original source licenses and terms also apply".to_string(),
            license_url: Some("https://opendatacommons.org/licenses/by/1-0/".to_string()),
            license_status: "odc_by_internal_attribution".to_string(),
            provenance: vec![
                "web".to_string(),
                "academic_publications".to_string(),
                "code".to_string(),
                "books".to_string(),
                "encyclopedic".to_string(),
            ],
            sampling_weight: 0.35,
            include_in_tokenizer_training: true,
            include_in_pretraining: true,
            include_in_memory_trace_training: false,
            synthetic: false,
            local_bytes: None,
            local_records: None,
            content_hash: None,
            metadata_hash: None,
            split_counts: BTreeMap::new(),
            domain_counts: BTreeMap::new(),
            category_counts: BTreeMap::new(),
            task_kind_counts: BTreeMap::new(),
            extra: BTreeMap::from([
                (
                    "reference".to_string(),
                    JsonValue::String("arxiv:2402.00159".to_string()),
                ),
                (
                    "default_version".to_string(),
                    JsonValue::String("v1_7".to_string()),
                ),
            ]),
        },
        CorpusBlendSource {
            source_id: "nvidia.nemotron_cc.high_actual".to_string(),
            display_name: "Nemotron-CC high actual".to_string(),
            kind: "external_pretraining_corpus".to_string(),
            status: "planned".to_string(),
            role: "high_quality_web_backbone".to_string(),
            source_url: Some(
                "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample"
                    .to_string(),
            ),
            path: None,
            metadata_path: None,
            data_format: "jsonl.zstd".to_string(),
            license: "NVIDIA Data Agreement for Model Training; internal training only"
                .to_string(),
            license_url: Some(
                "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/blob/main/LICENSE.md"
                    .to_string(),
            ),
            license_status: "nvidia_data_agreement_internal_training".to_string(),
            provenance: vec!["common_crawl".to_string(), "nvidia_filtering".to_string()],
            sampling_weight: 0.25,
            include_in_tokenizer_training: true,
            include_in_pretraining: true,
            include_in_memory_trace_training: false,
            synthetic: false,
            local_bytes: None,
            local_records: None,
            content_hash: None,
            metadata_hash: None,
            split_counts: BTreeMap::new(),
            domain_counts: BTreeMap::new(),
            category_counts: BTreeMap::new(),
            task_kind_counts: BTreeMap::new(),
            extra: BTreeMap::from([
                (
                    "reference".to_string(),
                    JsonValue::String("arxiv:2412.02595".to_string()),
                ),
                (
                    "partition_hint".to_string(),
                    JsonValue::String("quality=high/kind=actual/kind2=actual".to_string()),
                ),
                (
                    "scope_note".to_string(),
                    JsonValue::String(
                        "approval covers the NVIDIA sample artifact; full-size external objects need separate review"
                            .to_string(),
                    ),
                ),
            ]),
        },
        CorpusBlendSource {
            source_id: "allenai.dolma3_dolmino_mix-100B-1125".to_string(),
            display_name: "Dolma 3 Dolmino mix 100B (OLMo 3 stage 2)".to_string(),
            kind: "external_pretraining_corpus".to_string(),
            status: "planned".to_string(),
            role: "olmo3_dolmino_mix".to_string(),
            source_url: Some(
                "https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125"
                    .to_string(),
            ),
            path: None,
            metadata_path: None,
            data_format: "jsonl.zst".to_string(),
            license: "ODC-BY; attribution required; internal raw-slice training only".to_string(),
            license_url: Some("https://opendatacommons.org/licenses/by/1-0/".to_string()),
            license_status: "odc_by_internal_attribution".to_string(),
            provenance: vec![
                "dolma3_dolmino_mix".to_string(),
                "olmo3".to_string(),
                "dolmino".to_string(),
                "code".to_string(),
                "math".to_string(),
                "instruction".to_string(),
                "thinking".to_string(),
            ],
            sampling_weight: 0.20,
            include_in_tokenizer_training: true,
            include_in_pretraining: true,
            include_in_memory_trace_training: false,
            synthetic: false,
            local_bytes: None,
            local_records: None,
            content_hash: None,
            metadata_hash: None,
            split_counts: BTreeMap::new(),
            domain_counts: BTreeMap::new(),
            category_counts: BTreeMap::new(),
            task_kind_counts: BTreeMap::new(),
            extra: BTreeMap::from([
                (
                    "reference".to_string(),
                    JsonValue::String("arxiv:2512.13961".to_string()),
                ),
                (
                    "artifact_note".to_string(),
                    JsonValue::String(
                        "OLMo 3 curated slice uses the mixed-down 100B Dolmino artifact"
                            .to_string(),
                    ),
                ),
                (
                    "dataset_size_note".to_string(),
                    JsonValue::String("100B tokens; Hugging Face card lists 340 GB".to_string()),
                ),
            ]),
        },
        CorpusBlendSource {
            source_id: "nvidia.nemotron_cc_math".to_string(),
            display_name: "Nemotron-CC-Math".to_string(),
            kind: "external_pretraining_corpus".to_string(),
            status: "planned".to_string(),
            role: "math_science_code_reasoning".to_string(),
            source_url: Some(
                "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample"
                    .to_string(),
            ),
            path: None,
            metadata_path: None,
            data_format: "jsonl".to_string(),
            license: "NVIDIA Data Agreement for Model Training; internal training only"
                .to_string(),
            license_url: Some(
                "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/blob/main/LICENSE.md"
                    .to_string(),
            ),
            license_status: "nvidia_data_agreement_internal_training".to_string(),
            provenance: vec![
                "common_crawl".to_string(),
                "math_scientific_extraction".to_string(),
            ],
            sampling_weight: 0.10,
            include_in_tokenizer_training: true,
            include_in_pretraining: true,
            include_in_memory_trace_training: false,
            synthetic: false,
            local_bytes: None,
            local_records: None,
            content_hash: None,
            metadata_hash: None,
            split_counts: BTreeMap::new(),
            domain_counts: BTreeMap::new(),
            category_counts: BTreeMap::new(),
            task_kind_counts: BTreeMap::new(),
            extra: BTreeMap::from([
                (
                    "reference".to_string(),
                    JsonValue::String("arxiv:2508.15096".to_string()),
                ),
                (
                    "scope_note".to_string(),
                    JsonValue::String(
                        "approval covers the NVIDIA sample artifact; full-size external objects need separate review"
                            .to_string(),
                    ),
                ),
            ]),
        },
    ]
}

fn scan_qb_trace_sources(root: &Path) -> Result<Vec<CorpusBlendSource>> {
    let synthetic_root = root.join("synthetic");
    if !synthetic_root.exists() {
        return Err(TensorError::InvalidOperation(format!(
            "VECL-QB synthetic corpus root not found: {}",
            synthetic_root.display()
        )));
    }
    let mut dirs = fs::read_dir(&synthetic_root)
        .map_err(|err| {
            TensorError::Io(format!(
                "failed to read {}: {err}",
                synthetic_root.display()
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|err| {
            TensorError::Io(format!(
                "failed to scan {}: {err}",
                synthetic_root.display()
            ))
        })?;
    dirs.sort_by_key(|entry| entry.path());

    let mut sources = Vec::new();
    for entry in dirs {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(size_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let metadata_path = path.join("metadata.json");
        let corpus_path = path.join("corpus.jsonl");
        if !metadata_path.exists() || !corpus_path.exists() {
            continue;
        }
        sources.push(qb_trace_source_from_files(
            size_name,
            &metadata_path,
            &corpus_path,
        )?);
    }
    if sources.is_empty() {
        return Err(TensorError::InvalidOperation(format!(
            "no VECL-QB synthetic corpora found under {}",
            synthetic_root.display()
        )));
    }
    Ok(sources)
}

fn qb_trace_source_from_files(
    size_name: &str,
    metadata_path: &Path,
    corpus_path: &Path,
) -> Result<CorpusBlendSource> {
    let metadata_bytes = fs::read(metadata_path).map_err(|err| {
        TensorError::Io(format!("failed to read {}: {err}", metadata_path.display()))
    })?;
    let corpus_bytes = fs::read(corpus_path).map_err(|err| {
        TensorError::Io(format!("failed to read {}: {err}", corpus_path.display()))
    })?;
    let metadata: JsonValue = serde_json::from_slice(&metadata_bytes).map_err(|err| {
        TensorError::Io(format!(
            "failed to parse {}: {err}",
            metadata_path.display()
        ))
    })?;
    let record_count = count_jsonl_records(corpus_path)?;
    if let Some(actual_count) = metadata.get("actual_count").and_then(JsonValue::as_u64) {
        if actual_count as usize != record_count {
            return Err(TensorError::InvalidOperation(format!(
                "{} actual_count={} does not match corpus records={record_count}",
                metadata_path.display(),
                actual_count
            )));
        }
    }
    let validation = metadata
        .get("validation")
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();
    let split_counts = json_object_usize_map(validation.get("split_counts"));
    let domain_counts = json_object_usize_map(validation.get("domain_counts"));
    let category_counts = json_object_usize_map(validation.get("category_counts"));
    let task_kind_counts = json_object_usize_map(validation.get("task_kind_counts"));
    let executable_records = validation
        .get("executable_records")
        .and_then(JsonValue::as_u64)
        .or_else(|| {
            metadata
                .get("executable_records")
                .and_then(JsonValue::as_u64)
        });
    let supervised_records = validation
        .get("supervised_records")
        .and_then(JsonValue::as_u64)
        .or_else(|| {
            metadata
                .get("supervised_records")
                .and_then(JsonValue::as_u64)
        });
    let dataset_hash = metadata
        .get("dataset_hash")
        .and_then(JsonValue::as_str)
        .map(|value| JsonValue::String(value.to_string()));
    let training_export_diversity = metadata.get("training_export_diversity").cloned();
    let semantic_diversity = metadata.get("semantic_diversity").cloned();
    let mut extra = BTreeMap::new();
    if let Some(value) = dataset_hash {
        extra.insert("dataset_hash".to_string(), value);
    }
    if let Some(value) = executable_records {
        extra.insert("executable_records".to_string(), JsonValue::from(value));
    }
    if let Some(value) = supervised_records {
        extra.insert("supervised_records".to_string(), JsonValue::from(value));
    }
    if let Some(value) = training_export_diversity {
        extra.insert("training_export_diversity".to_string(), value);
    }
    if let Some(value) = semantic_diversity {
        extra.insert("semantic_diversity".to_string(), value);
    }

    Ok(CorpusBlendSource {
        source_id: format!("vecl_qb.synthetic.{size_name}"),
        display_name: format!("VECL-QB synthetic {size_name}"),
        kind: "local_qb_trace_corpus".to_string(),
        status: "available".to_string(),
        role: "tool_routing_memory_trace_supervision".to_string(),
        source_url: None,
        path: Some(corpus_path.display().to_string()),
        metadata_path: Some(metadata_path.display().to_string()),
        data_format: "jsonl".to_string(),
        license: "project-internal synthetic corpus; no production user data".to_string(),
        license_url: None,
        license_status: "internal_synthetic".to_string(),
        provenance: vec![
            "VECL-QB".to_string(),
            "synthetic_tool_use".to_string(),
            size_name.to_string(),
        ],
        sampling_weight: if size_name == "v1-hard" { 0.10 } else { 0.0 },
        include_in_tokenizer_training: true,
        include_in_pretraining: size_name == "v1-hard",
        include_in_memory_trace_training: true,
        synthetic: true,
        local_bytes: Some(corpus_bytes.len()),
        local_records: Some(record_count),
        content_hash: Some(stable_hash_bytes(&corpus_bytes)),
        metadata_hash: Some(stable_hash_bytes(&metadata_bytes)),
        split_counts,
        domain_counts,
        category_counts,
        task_kind_counts,
        extra,
    })
}

fn count_jsonl_records(path: &Path) -> Result<usize> {
    let file = File::open(path)
        .map_err(|err| TensorError::Io(format!("failed to open {}: {err}", path.display())))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line
            .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

fn json_object_usize_map(value: Option<&JsonValue>) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    if let Some(object) = value.and_then(JsonValue::as_object) {
        for (key, value) in object {
            if let Some(count) = value.as_u64() {
                map.insert(key.clone(), count as usize);
            }
        }
    }
    map
}

pub fn prepare_lm_data(
    input: impl AsRef<Path>,
    tokenizer: &BpeTokenizer,
    tokenizer_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: PreparedDataOptions,
) -> Result<PreparedTokenData> {
    tokenizer.validate()?;
    if !(0.0..1.0).contains(&options.valid_fraction) {
        return Err(TensorError::InvalidOperation(format!(
            "valid_fraction must be in [0, 1), got {}",
            options.valid_fraction
        )));
    }

    let input = input.as_ref();
    let tokenizer_path = tokenizer_path.as_ref();
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)
        .map_err(|err| TensorError::Io(format!("failed to create {}: {err}", out_dir.display())))?;

    let mut bytes = fs::read(input)
        .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", input.display())))?;
    if let Some(limit) = options.max_bytes {
        bytes.truncate(limit);
    }
    let text = String::from_utf8_lossy(&bytes);
    let tokens = tokenizer.encode(&text, true, true);
    if tokens.len() < 2 {
        return Err(TensorError::InvalidOperation(format!(
            "not enough tokens to prepare train/valid splits: {}",
            tokens.len()
        )));
    }
    let valid_len = ((tokens.len() as f64) * options.valid_fraction).round() as usize;
    let valid_len = valid_len.clamp(1, tokens.len().saturating_sub(1));
    let train_len = tokens.len().saturating_sub(valid_len);
    if train_len == 0 || valid_len == 0 {
        return Err(TensorError::InvalidOperation(format!(
            "not enough tokens to prepare train/valid splits: {}",
            tokens.len()
        )));
    }

    let train_tokens = tokens[..train_len].to_vec();
    let valid_tokens = tokens[train_len..].to_vec();
    let train_path = out_dir.join("train.tokens.json");
    let valid_path = out_dir.join("valid.tokens.json");
    let manifest_path = out_dir.join("manifest.json");
    write_token_file(&train_path, &train_tokens)?;
    write_token_file(&valid_path, &valid_tokens)?;

    let manifest = PreparedDataManifest {
        format: DATASET_MANIFEST_FORMAT.to_string(),
        version: DATASET_MANIFEST_VERSION,
        storage: DATASET_STORAGE_JSON.to_string(),
        source_path: input.display().to_string(),
        source_bytes: bytes.len(),
        source_hash: stable_hash_bytes(&bytes),
        tokenizer_path: path_for_manifest(tokenizer_path, out_dir),
        tokenizer_hash: tokenizer.fingerprint()?,
        train_tokens_path: "train.tokens.json".to_string(),
        valid_tokens_path: "valid.tokens.json".to_string(),
        train_tokens: train_tokens.len(),
        valid_tokens: valid_tokens.len(),
        train_hash: stable_hash_tokens(&train_tokens),
        valid_hash: stable_hash_tokens(&valid_tokens),
        split: DataSplit {
            kind: "contiguous_tail_valid".to_string(),
            valid_fraction: options.valid_fraction,
        },
        sources: vec![PreparedDataSource {
            source_id: source_id_from_path(input),
            path: input.display().to_string(),
            bytes: bytes.len(),
            hash: stable_hash_bytes(&bytes),
            tokens: tokens.len(),
        }],
        train_shards: Vec::new(),
        valid_shards: Vec::new(),
    };
    manifest.validate()?;
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|err| TensorError::Io(format!("failed to serialize manifest: {err}")))?;
    fs::write(&manifest_path, json + "\n").map_err(|err| {
        TensorError::Io(format!(
            "failed to write {}: {err}",
            manifest_path.display()
        ))
    })?;
    PreparedTokenData::load(manifest_path)
}

pub fn prepare_lm_data_binary_shards(
    inputs: &[PathBuf],
    tokenizer: &BpeTokenizer,
    tokenizer_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: PreparedDataOptions,
) -> Result<PreparedTokenData> {
    tokenizer.validate()?;
    if inputs.is_empty() {
        return Err(TensorError::InvalidOperation(
            "binary-shard data preparation requires at least one --input".to_string(),
        ));
    }
    if inputs.len() > 1 && options.max_bytes.is_some() {
        return Err(TensorError::InvalidOperation(
            "--max-bytes is supported only with a single --input".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&options.valid_fraction) {
        return Err(TensorError::InvalidOperation(format!(
            "valid_fraction must be in [0, 1), got {}",
            options.valid_fraction
        )));
    }

    let tokenizer_path = tokenizer_path.as_ref();
    let out_dir = out_dir.as_ref();
    let shards_dir = out_dir.join("shards");
    fs::create_dir_all(&shards_dir).map_err(|err| {
        TensorError::Io(format!("failed to create {}: {err}", shards_dir.display()))
    })?;

    let mut all_tokens = Vec::new();
    let mut source_bytes = 0usize;
    let mut source_hash_input = Vec::new();
    let mut sources = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut bytes = fs::read(input)
            .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", input.display())))?;
        if let Some(limit) = options.max_bytes {
            bytes.truncate(limit);
        }
        source_bytes += bytes.len();
        source_hash_input.extend_from_slice(input.display().to_string().as_bytes());
        source_hash_input.push(0);
        source_hash_input.extend_from_slice(&bytes);
        source_hash_input.push(0);
        let text = String::from_utf8_lossy(&bytes);
        let tokens = tokenizer.encode(&text, true, true);
        sources.push(PreparedDataSource {
            source_id: source_id_from_path(input),
            path: input.display().to_string(),
            bytes: bytes.len(),
            hash: stable_hash_bytes(&bytes),
            tokens: tokens.len(),
        });
        all_tokens.extend(tokens);
    }

    if all_tokens.len() < 2 {
        return Err(TensorError::InvalidOperation(format!(
            "not enough tokens to prepare train/valid splits: {}",
            all_tokens.len()
        )));
    }
    let valid_len = ((all_tokens.len() as f64) * options.valid_fraction).round() as usize;
    let valid_len = valid_len.clamp(1, all_tokens.len().saturating_sub(1));
    let train_len = all_tokens.len().saturating_sub(valid_len);
    if train_len == 0 || valid_len == 0 {
        return Err(TensorError::InvalidOperation(format!(
            "not enough tokens to prepare train/valid splits: {}",
            all_tokens.len()
        )));
    }

    let train_tokens = all_tokens[..train_len].to_vec();
    let valid_tokens = all_tokens[train_len..].to_vec();
    if matches!(options.shard_tokens, Some(0)) {
        return Err(TensorError::InvalidOperation(
            "--shard-tokens must be greater than zero".to_string(),
        ));
    }
    let train_shards = write_token_shard_set(
        &shards_dir,
        "train",
        &train_tokens,
        options.shard_tokens,
        out_dir,
    )?;
    let valid_shards = write_token_shard_set(
        &shards_dir,
        "valid",
        &valid_tokens,
        options.shard_tokens,
        out_dir,
    )?;

    let manifest_path = out_dir.join("manifest.json");
    let manifest = PreparedDataManifest {
        format: DATASET_MANIFEST_FORMAT.to_string(),
        version: DATASET_MANIFEST_VERSION_V2,
        storage: DATASET_STORAGE_BINARY_SHARDS.to_string(),
        source_path: inputs
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(","),
        source_bytes,
        source_hash: stable_hash_bytes(&source_hash_input),
        tokenizer_path: path_for_manifest(tokenizer_path, out_dir),
        tokenizer_hash: tokenizer.fingerprint()?,
        train_tokens_path: String::new(),
        valid_tokens_path: String::new(),
        train_tokens: train_tokens.len(),
        valid_tokens: valid_tokens.len(),
        train_hash: stable_hash_tokens(&train_tokens),
        valid_hash: stable_hash_tokens(&valid_tokens),
        split: DataSplit {
            kind: "contiguous_tail_valid".to_string(),
            valid_fraction: options.valid_fraction,
        },
        sources,
        train_shards,
        valid_shards,
    };
    manifest.validate()?;
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|err| TensorError::Io(format!("failed to serialize manifest: {err}")))?;
    fs::write(&manifest_path, json + "\n").map_err(|err| {
        TensorError::Io(format!(
            "failed to write {}: {err}",
            manifest_path.display()
        ))
    })?;
    PreparedTokenData::load(manifest_path)
}

fn write_token_shard_set(
    shards_dir: &Path,
    split: &str,
    tokens: &[usize],
    shard_tokens: Option<usize>,
    manifest_root: &Path,
) -> Result<Vec<PreparedDataShard>> {
    if tokens.is_empty() {
        return Err(TensorError::InvalidOperation(format!(
            "{split} split has no tokens to shard"
        )));
    }
    let chunk_size = shard_tokens.unwrap_or(tokens.len()).max(1);
    let mut shards = Vec::new();
    for (index, chunk) in tokens.chunks(chunk_size).enumerate() {
        let payload_path = shards_dir.join(format!("{split}-{index:06}.tokens.bin"));
        let metadata_path = shards_dir.join(format!("{split}-{index:06}.tokens.json"));
        let metadata = write_token_shard(&payload_path, &metadata_path, chunk)?;
        shards.push(PreparedDataShard {
            split: split.to_string(),
            metadata_path: path_for_manifest(&metadata_path, manifest_root),
            tokens: metadata.token_count,
            token_hash: metadata.token_hash,
        });
    }
    Ok(shards)
}

pub fn read_token_file(path: impl AsRef<Path>) -> Result<Vec<usize>> {
    let path = path.as_ref();
    let json = fs::read_to_string(path)
        .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
    serde_json::from_str(&json)
        .map_err(|err| TensorError::Io(format!("failed to parse {}: {err}", path.display())))
}

pub fn write_token_file(path: impl AsRef<Path>, tokens: &[usize]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            TensorError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }
    let json = serde_json::to_string(tokens)
        .map_err(|err| TensorError::Io(format!("failed to serialize {}: {err}", path.display())))?;
    fs::write(path, json)
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", path.display())))
}

pub fn load_token_shard_metadata(path: impl AsRef<Path>) -> Result<TokenShardMetadata> {
    let path = path.as_ref();
    let json = fs::read_to_string(path)
        .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
    let metadata: TokenShardMetadata = serde_json::from_str(&json)
        .map_err(|err| TensorError::Io(format!("failed to parse {}: {err}", path.display())))?;
    metadata.validate()?;
    Ok(metadata)
}

pub fn write_token_shard(
    payload_path: impl AsRef<Path>,
    metadata_path: impl AsRef<Path>,
    tokens: &[usize],
) -> Result<TokenShardMetadata> {
    let payload_path = payload_path.as_ref();
    let metadata_path = metadata_path.as_ref();
    if let Some(parent) = payload_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            TensorError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }
    if let Some(parent) = metadata_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            TensorError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }

    let dtype = TokenShardDType::for_tokens(tokens)?;
    let payload = encode_token_shard_payload(tokens, dtype)?;
    fs::write(payload_path, &payload).map_err(|err| {
        TensorError::Io(format!("failed to write {}: {err}", payload_path.display()))
    })?;

    let metadata_root = metadata_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let metadata = TokenShardMetadata {
        format: TOKEN_SHARD_FORMAT.to_string(),
        version: TOKEN_SHARD_VERSION,
        payload_path: path_for_manifest(payload_path, &metadata_root),
        dtype,
        token_count: tokens.len(),
        token_hash: stable_hash_tokens(tokens),
        payload_bytes: payload.len(),
        payload_hash: stable_hash_bytes(&payload),
    };
    metadata.validate()?;
    let json = serde_json::to_string_pretty(&metadata).map_err(|err| {
        TensorError::Io(format!("failed to serialize token shard metadata: {err}"))
    })?;
    fs::write(metadata_path, json + "\n").map_err(|err| {
        TensorError::Io(format!(
            "failed to write {}: {err}",
            metadata_path.display()
        ))
    })?;
    Ok(metadata)
}

pub fn read_token_shard(metadata_path: impl AsRef<Path>) -> Result<Vec<usize>> {
    let metadata_path = metadata_path.as_ref();
    let metadata = load_token_shard_metadata(metadata_path)?;
    let root = metadata_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let payload_path = resolve_manifest_path(&root, &metadata.payload_path);
    let payload = fs::read(&payload_path).map_err(|err| {
        TensorError::Io(format!("failed to read {}: {err}", payload_path.display()))
    })?;
    if payload.len() != metadata.payload_bytes {
        return Err(TensorError::InvalidOperation(format!(
            "token shard payload byte mismatch: metadata={} actual={}",
            metadata.payload_bytes,
            payload.len()
        )));
    }
    let payload_hash = stable_hash_bytes(&payload);
    if payload_hash != metadata.payload_hash {
        return Err(TensorError::InvalidOperation(format!(
            "token shard payload hash mismatch: metadata={} actual={}",
            metadata.payload_hash, payload_hash
        )));
    }

    let tokens = decode_token_shard_payload(&payload, metadata.dtype)?;
    if tokens.len() != metadata.token_count {
        return Err(TensorError::InvalidOperation(format!(
            "token shard length mismatch: metadata={} actual={}",
            metadata.token_count,
            tokens.len()
        )));
    }
    let token_hash = stable_hash_tokens(&tokens);
    if token_hash != metadata.token_hash {
        return Err(TensorError::InvalidOperation(format!(
            "token shard hash mismatch: metadata={} actual={}",
            metadata.token_hash, token_hash
        )));
    }
    Ok(tokens)
}

fn validate_token_shard_payload_file(path: &Path, metadata: &TokenShardMetadata) -> Result<()> {
    let file_metadata = fs::metadata(path)
        .map_err(|err| TensorError::Io(format!("failed to stat {}: {err}", path.display())))?;
    if file_metadata.len() != metadata.payload_bytes as u64 {
        return Err(TensorError::InvalidOperation(format!(
            "token shard payload byte mismatch: metadata={} actual={}",
            metadata.payload_bytes,
            file_metadata.len()
        )));
    }
    let payload_hash = stable_hash_file(path)?;
    if payload_hash != metadata.payload_hash {
        return Err(TensorError::InvalidOperation(format!(
            "token shard payload hash mismatch: metadata={} actual={}",
            metadata.payload_hash, payload_hash
        )));
    }
    Ok(())
}

fn validate_loaded_tokens(
    split: &str,
    tokens: &[usize],
    expected_len: usize,
    expected_hash: &str,
) -> Result<()> {
    let actual_hash = stable_hash_tokens(tokens);
    if tokens.len() != expected_len || actual_hash != expected_hash {
        return Err(TensorError::InvalidOperation(format!(
            "{split} token file mismatch: expected len/hash={expected_len}/{expected_hash}, got {}/{}",
            tokens.len(),
            actual_hash
        )));
    }
    Ok(())
}

fn encode_token_shard_payload(tokens: &[usize], dtype: TokenShardDType) -> Result<Vec<u8>> {
    let mut payload = Vec::with_capacity(tokens.len() * dtype.bytes_per_token());
    for &token in tokens {
        if token > dtype.max_token_id() {
            return Err(TensorError::InvalidOperation(format!(
                "token id {token} exceeds {:?} token shard limit",
                dtype
            )));
        }
        match dtype {
            TokenShardDType::U16 => payload.extend_from_slice(&(token as u16).to_le_bytes()),
            TokenShardDType::U32 => payload.extend_from_slice(&(token as u32).to_le_bytes()),
        }
    }
    Ok(payload)
}

fn decode_token_shard_payload(payload: &[u8], dtype: TokenShardDType) -> Result<Vec<usize>> {
    let width = dtype.bytes_per_token();
    if !payload.len().is_multiple_of(width) {
        return Err(TensorError::InvalidOperation(format!(
            "token shard payload byte length {} is not divisible by token width {}",
            payload.len(),
            width
        )));
    }
    let mut tokens = Vec::with_capacity(payload.len() / width);
    match dtype {
        TokenShardDType::U16 => {
            for chunk in payload.chunks_exact(2) {
                tokens.push(u16::from_le_bytes([chunk[0], chunk[1]]) as usize);
            }
        }
        TokenShardDType::U32 => {
            for chunk in payload.chunks_exact(4) {
                tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize);
            }
        }
    }
    Ok(tokens)
}

fn stable_hash_tokens(tokens: &[usize]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len() * std::mem::size_of::<u64>());
    for &token in tokens {
        bytes.extend_from_slice(&(token as u64).to_le_bytes());
    }
    stable_hash_bytes(&bytes)
}

fn stable_hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .map_err(|err| TensorError::Io(format!("failed to open {}: {err}", path.display())))?;
    let mut hash = 0xcbf29ce484222325u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
        if read == 0 {
            break;
        }
        for &byte in &buffer[..read] {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

fn path_for_manifest(path: &Path, manifest_root: &Path) -> String {
    let absolute_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let absolute_root =
        fs::canonicalize(manifest_root).unwrap_or_else(|_| manifest_root.to_path_buf());
    absolute_path
        .strip_prefix(absolute_root)
        .unwrap_or(&absolute_path)
        .display()
        .to_string()
}

fn source_id_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("source")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn resolve_manifest_path(root: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

pub fn read_text(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    fs::read_to_string(path)
        .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))
}

pub fn download_tinystories_valid(out: impl AsRef<Path>) -> Result<()> {
    let out = out.as_ref();
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            TensorError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }

    let response = ureq::get(TINYSTORIES_VALID_URL).call().map_err(|err| {
        TensorError::Io(format!("failed to download TinyStories-valid.txt: {err}"))
    })?;
    let mut reader = response.into_reader();
    let mut file = fs::File::create(out)
        .map_err(|err| TensorError::Io(format!("failed to create {}: {err}", out.display())))?;
    std::io::copy(&mut reader, &mut file)
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", out.display())))?;
    file.flush()
        .map_err(|err| TensorError::Io(format!("failed to flush {}: {err}", out.display())))
}
