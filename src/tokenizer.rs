use crate::{Result, TensorError};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

pub const PAD_ID: usize = 0;
pub const BOS_ID: usize = 1;
pub const EOS_ID: usize = 2;
pub const BYTE_OFFSET: usize = 3;
pub const BYTE_VOCAB: usize = 256;
pub const TOKENIZER_FORMAT: &str = "heirloom.byte_bpe";
pub const TOKENIZER_VERSION: u32 = 1;
pub const TOKENIZER_VERSION_V2: u32 = 2;
pub const PRODUCTION_TOKENIZER_VOCAB_SIZE: usize = 32_768;
pub const DEFAULT_TOKENIZER_SAMPLE_BYTES: u64 = 32 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeMerge {
    pub left: usize,
    pub right: usize,
    pub id: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReservedToken {
    pub id: usize,
    pub token: String,
    pub category: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeTokenizerTrainingConfig {
    pub target_vocab_size: usize,
    pub sample_bytes: u64,
    pub seed: u64,
    pub memory_limit_bytes: Option<u64>,
    pub source_blend_hash: String,
    pub sample_manifest_hash: String,
    pub trainer: String,
    #[serde(default)]
    pub digit_isolation: bool,
}

impl Default for BpeTokenizerTrainingConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 0,
            sample_bytes: 0,
            seed: 0,
            memory_limit_bytes: None,
            source_blend_hash: String::new(),
            sample_manifest_hash: String::new(),
            trainer: "heirloom.byte_bpe.v1".to_string(),
            digit_isolation: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BpeTokenizerValidationMetrics {
    pub training_bytes: u64,
    pub weighted_training_bytes: u64,
    pub training_sequences: u64,
    pub reserved_tokens: usize,
    pub byte_token_count: usize,
    pub learned_merge_count: usize,
    pub byte_coverage: f64,
    pub artifact_hash: String,
    #[serde(default)]
    pub initial_unique_sequences: usize,
    #[serde(default)]
    pub initial_pair_count: usize,
    #[serde(default)]
    pub final_pair_count: usize,
    #[serde(default)]
    pub heap_pops: u64,
    #[serde(default)]
    pub stale_heap_pops: u64,
    #[serde(default)]
    pub heap_rebuilds: u64,
    #[serde(default)]
    pub heap_rebuild_candidates_dropped: u64,
    #[serde(default)]
    pub affected_word_updates: u64,
    #[serde(default)]
    pub max_heap_len: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeTokenizerMetadata {
    pub format: String,
    pub version: u32,
    pub vocab_size: usize,
    pub byte_offset: usize,
    pub byte_vocab: usize,
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub training_bytes: usize,
    pub training_hash: String,
    #[serde(default)]
    pub tokenizer_id: String,
    #[serde(default)]
    pub source_blend_hash: String,
    #[serde(default)]
    pub sample_manifest_hash: String,
    #[serde(default)]
    pub reserved_registry_hash: String,
    #[serde(default)]
    pub artifact_hash: String,
    #[serde(default)]
    pub training_config: BpeTokenizerTrainingConfig,
    #[serde(default)]
    pub validation: BpeTokenizerValidationMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeTokenizer {
    #[serde(default = "legacy_metadata")]
    metadata: BpeTokenizerMetadata,
    id_to_bytes: Vec<Vec<u8>>,
    merges: Vec<BpeMerge>,
    #[serde(default)]
    reserved_tokens: Vec<ReservedToken>,
    #[serde(skip)]
    merge_lookup: OnceLock<HashMap<(usize, usize), usize>>,
    #[serde(skip)]
    reserved_lookup: OnceLock<ReservedLookup>,
}

type ReservedMatch = (usize, Vec<u8>);
type ReservedLookup = HashMap<u8, Vec<ReservedMatch>>;

#[derive(Clone, Debug)]
pub struct BpeTrainingSample {
    pub source_id: String,
    pub bytes: Vec<u8>,
    pub weight: u64,
}

#[derive(Clone, Debug)]
pub struct BpeTokenizerV2Options {
    pub tokenizer_id: String,
    pub vocab_size: usize,
    pub sample_bytes: u64,
    pub seed: u64,
    pub memory_limit_bytes: Option<u64>,
    pub source_blend_hash: String,
    pub sample_manifest_hash: String,
    pub digit_isolation: bool,
    pub require_exact_vocab: bool,
}

impl Default for BpeTokenizerV2Options {
    fn default() -> Self {
        Self {
            tokenizer_id: "heirloom-byte-bpe-32768-v2".to_string(),
            vocab_size: PRODUCTION_TOKENIZER_VOCAB_SIZE,
            sample_bytes: DEFAULT_TOKENIZER_SAMPLE_BYTES,
            seed: 0,
            memory_limit_bytes: None,
            source_blend_hash: String::new(),
            sample_manifest_hash: String::new(),
            digit_isolation: true,
            require_exact_vocab: false,
        }
    }
}

impl BpeTokenizer {
    pub fn train(text: &str, vocab_size: usize) -> Result<Self> {
        Self::train_v1_bytes(text.as_bytes(), vocab_size)
    }

    pub fn train_v1_bytes(bytes: &[u8], vocab_size: usize) -> Result<Self> {
        validate_vocab_size(vocab_size, BYTE_OFFSET + BYTE_VOCAB)?;
        let reserved_tokens = Vec::new();
        let (id_to_bytes, merges, training_sequences) =
            train_bpe_merges(bytes, 1, vocab_size, &reserved_tokens, 2)?;
        let tokenizer = Self {
            metadata: BpeTokenizerMetadata {
                format: TOKENIZER_FORMAT.to_string(),
                version: TOKENIZER_VERSION,
                vocab_size: id_to_bytes.len(),
                byte_offset: BYTE_OFFSET,
                byte_vocab: BYTE_VOCAB,
                pad_id: PAD_ID,
                bos_id: BOS_ID,
                eos_id: EOS_ID,
                training_bytes: bytes.len(),
                training_hash: stable_hash_bytes(bytes),
                tokenizer_id: format!("heirloom-byte-bpe-v1-{}", id_to_bytes.len()),
                source_blend_hash: String::new(),
                sample_manifest_hash: String::new(),
                reserved_registry_hash: String::new(),
                artifact_hash: String::new(),
                training_config: BpeTokenizerTrainingConfig {
                    target_vocab_size: vocab_size,
                    sample_bytes: bytes.len() as u64,
                    seed: 0,
                    memory_limit_bytes: None,
                    source_blend_hash: String::new(),
                    sample_manifest_hash: String::new(),
                    trainer: "heirloom.byte_bpe.v1".to_string(),
                    digit_isolation: false,
                },
                validation: BpeTokenizerValidationMetrics {
                    training_bytes: bytes.len() as u64,
                    weighted_training_bytes: bytes.len() as u64,
                    training_sequences,
                    reserved_tokens: 0,
                    byte_token_count: BYTE_VOCAB,
                    learned_merge_count: merges.len(),
                    byte_coverage: 1.0,
                    artifact_hash: String::new(),
                    initial_unique_sequences: 0,
                    initial_pair_count: 0,
                    final_pair_count: 0,
                    heap_pops: 0,
                    stale_heap_pops: 0,
                    heap_rebuilds: 0,
                    heap_rebuild_candidates_dropped: 0,
                    affected_word_updates: 0,
                    max_heap_len: 0,
                },
            },
            id_to_bytes,
            merges,
            reserved_tokens,
            merge_lookup: OnceLock::new(),
            reserved_lookup: OnceLock::new(),
        };
        tokenizer.validate()?;
        Ok(tokenizer)
    }

    pub fn train_v2(
        samples: &[BpeTrainingSample],
        reserved_tokens: Vec<ReservedToken>,
        options: BpeTokenizerV2Options,
    ) -> Result<Self> {
        let base_vocab = BYTE_OFFSET + BYTE_VOCAB + reserved_tokens.len();
        validate_vocab_size(options.vocab_size, base_vocab)?;
        validate_reserved_tokens(&reserved_tokens)?;

        let mut sequences = HashMap::<Vec<usize>, u64>::new();
        let mut training_bytes = 0u64;
        let mut weighted_training_bytes = 0u64;
        let mut training_sequences = 0u64;
        let mut hash_input = Vec::new();

        for sample in samples {
            if sample.weight == 0 || sample.bytes.is_empty() {
                continue;
            }
            training_bytes = training_bytes
                .checked_add(sample.bytes.len() as u64)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(
                        "tokenizer training byte count overflowed".to_string(),
                    )
                })?;
            weighted_training_bytes = weighted_training_bytes
                .checked_add((sample.bytes.len() as u64).saturating_mul(sample.weight))
                .ok_or_else(|| {
                    TensorError::InvalidOperation(
                        "tokenizer weighted byte count overflowed".to_string(),
                    )
                })?;
            hash_input.extend_from_slice(sample.source_id.as_bytes());
            hash_input.push(0);
            hash_input.extend_from_slice(&sample.weight.to_le_bytes());
            hash_input.extend_from_slice(&sample.bytes);
            hash_input.push(0);
            hash_input.push(u8::from(options.digit_isolation));
            for segment in
                byte_segments_for_bpe(&sample.bytes, &reserved_tokens, options.digit_isolation)
            {
                for chunk in byte_chunks_bytes(segment) {
                    if chunk.is_empty() {
                        continue;
                    }
                    let sequence = chunk
                        .iter()
                        .map(|byte| BYTE_OFFSET + *byte as usize)
                        .collect::<Vec<_>>();
                    *sequences.entry(sequence).or_default() += sample.weight;
                    training_sequences += 1;
                }
            }
        }
        if sequences.is_empty() {
            return Err(TensorError::InvalidOperation(
                "tokenizer v2 training requires at least one non-empty byte sequence".to_string(),
            ));
        }

        let (id_to_bytes, merges, trainer_stats) =
            train_bpe_merges_incremental(sequences, options.vocab_size, &reserved_tokens, 1)?;
        if options.require_exact_vocab && id_to_bytes.len() != options.vocab_size {
            return Err(TensorError::InvalidOperation(format!(
                "tokenizer v2 trainer produced vocab={} but exact vocab={} was required",
                id_to_bytes.len(),
                options.vocab_size
            )));
        }

        let reserved_registry_hash = stable_hash_reserved_tokens(&reserved_tokens)?;
        let tokenizer = Self {
            metadata: BpeTokenizerMetadata {
                format: TOKENIZER_FORMAT.to_string(),
                version: TOKENIZER_VERSION_V2,
                vocab_size: id_to_bytes.len(),
                byte_offset: BYTE_OFFSET,
                byte_vocab: BYTE_VOCAB,
                pad_id: PAD_ID,
                bos_id: BOS_ID,
                eos_id: EOS_ID,
                training_bytes: training_bytes as usize,
                training_hash: stable_hash_bytes(&hash_input),
                tokenizer_id: options.tokenizer_id,
                source_blend_hash: options.source_blend_hash.clone(),
                sample_manifest_hash: options.sample_manifest_hash.clone(),
                reserved_registry_hash,
                artifact_hash: String::new(),
                training_config: BpeTokenizerTrainingConfig {
                    target_vocab_size: options.vocab_size,
                    sample_bytes: options.sample_bytes,
                    seed: options.seed,
                    memory_limit_bytes: options.memory_limit_bytes,
                    source_blend_hash: options.source_blend_hash,
                    sample_manifest_hash: options.sample_manifest_hash,
                    trainer: if options.digit_isolation {
                        "heirloom.byte_bpe.native_incremental.digit_isolated.v2".to_string()
                    } else {
                        "heirloom.byte_bpe.native_incremental.v2".to_string()
                    },
                    digit_isolation: options.digit_isolation,
                },
                validation: BpeTokenizerValidationMetrics {
                    training_bytes,
                    weighted_training_bytes,
                    training_sequences,
                    reserved_tokens: reserved_tokens.len(),
                    byte_token_count: BYTE_VOCAB,
                    learned_merge_count: merges.len(),
                    byte_coverage: 1.0,
                    artifact_hash: String::new(),
                    initial_unique_sequences: trainer_stats.initial_unique_sequences,
                    initial_pair_count: trainer_stats.initial_pair_count,
                    final_pair_count: trainer_stats.final_pair_count,
                    heap_pops: trainer_stats.heap_pops,
                    stale_heap_pops: trainer_stats.stale_heap_pops,
                    heap_rebuilds: trainer_stats.heap_rebuilds,
                    heap_rebuild_candidates_dropped: trainer_stats.heap_rebuild_candidates_dropped,
                    affected_word_updates: trainer_stats.affected_word_updates,
                    max_heap_len: trainer_stats.max_heap_len,
                },
            },
            id_to_bytes,
            merges,
            reserved_tokens,
            merge_lookup: OnceLock::new(),
            reserved_lookup: OnceLock::new(),
        };
        let mut tokenizer = tokenizer;
        let artifact_hash = tokenizer.fingerprint_without_artifact_hash()?;
        tokenizer.metadata.artifact_hash = artifact_hash.clone();
        tokenizer.metadata.validation.artifact_hash = artifact_hash;
        tokenizer.validate()?;
        Ok(tokenizer)
    }

    pub fn metadata(&self) -> &BpeTokenizerMetadata {
        &self.metadata
    }

    pub fn tokenizer_id(&self) -> &str {
        &self.metadata.tokenizer_id
    }

    pub fn reserved_tokens(&self) -> &[ReservedToken] {
        &self.reserved_tokens
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    pub fn merges(&self) -> &[BpeMerge] {
        &self.merges
    }

    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<usize> {
        self.encode_bytes(text.as_bytes(), add_bos, add_eos)
    }

    pub fn encode_bytes(&self, bytes: &[u8], add_bos: bool, add_eos: bool) -> Vec<usize> {
        let mut output =
            Vec::with_capacity(bytes.len() + usize::from(add_bos) + usize::from(add_eos));
        if add_bos {
            output.push(BOS_ID);
        }
        let merge_lookup = self.merge_lookup();
        let digit_isolation = self.digit_isolation_enabled();
        let mut cache = HashMap::<Vec<u8>, Vec<usize>>::new();
        let mut index = 0usize;
        while index < bytes.len() {
            if let Some((id, len)) = self.match_reserved(bytes, index) {
                output.push(id);
                index += len;
                continue;
            }
            if digit_isolation && bytes[index].is_ascii_digit() {
                output.push(BYTE_OFFSET + bytes[index] as usize);
                index += 1;
                continue;
            }
            let start = index;
            index += 1;
            while index < bytes.len()
                && self.match_reserved(bytes, index).is_none()
                && !(digit_isolation && bytes[index].is_ascii_digit())
            {
                index += 1;
            }
            for chunk in byte_chunks_bytes(&bytes[start..index]) {
                let tokens = cache
                    .entry(chunk.to_vec())
                    .or_insert_with(|| encode_chunk_with_merge_lookup(chunk, merge_lookup));
                output.extend(tokens.iter().copied());
            }
        }
        if add_eos {
            output.push(EOS_ID);
        }
        output
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        self.decode_lossy_utf8(tokens)
    }

    pub fn decode_bytes(&self, tokens: &[usize]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &token in tokens {
            if matches!(token, PAD_ID | BOS_ID | EOS_ID) {
                continue;
            }
            if let Some(piece) = self.id_to_bytes.get(token) {
                bytes.extend_from_slice(piece);
            }
        }
        bytes
    }

    pub fn decode_lossy_utf8(&self, tokens: &[usize]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(tokens)).into_owned()
    }

    pub fn digit_isolation_enabled(&self) -> bool {
        self.metadata.training_config.digit_isolation
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.validate()?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|err| TensorError::Io(format!("failed to serialize tokenizer: {err}")))?;
        fs::write(path, json + "\n")
            .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", path.display())))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = fs::read_to_string(path)
            .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
        let mut tokenizer: Self = serde_json::from_str(&json)
            .map_err(|err| TensorError::Io(format!("failed to parse {}: {err}", path.display())))?;
        if tokenizer.metadata.format.is_empty() {
            tokenizer.metadata = inferred_legacy_metadata(tokenizer.id_to_bytes.len());
        }
        if tokenizer.metadata.tokenizer_id.is_empty() {
            tokenizer.metadata.tokenizer_id =
                format!("heirloom-byte-bpe-v{}", tokenizer.metadata.version);
        }
        tokenizer.validate()?;
        Ok(tokenizer)
    }

    pub fn fingerprint(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_vec(self)
            .map(|bytes| stable_hash_bytes(&bytes))
            .map_err(|err| TensorError::Io(format!("failed to serialize tokenizer: {err}")))
    }

    pub fn fingerprint_without_artifact_hash(&self) -> Result<String> {
        let mut clone = self.clone();
        clone.metadata.artifact_hash.clear();
        clone.metadata.validation.artifact_hash.clear();
        serde_json::to_vec(&clone)
            .map(|bytes| stable_hash_bytes(&bytes))
            .map_err(|err| TensorError::Io(format!("failed to serialize tokenizer: {err}")))
    }

    pub fn validate(&self) -> Result<()> {
        if self.metadata.format != TOKENIZER_FORMAT {
            return Err(TensorError::InvalidOperation(format!(
                "unsupported tokenizer format {}, expected {}",
                self.metadata.format, TOKENIZER_FORMAT
            )));
        }
        if self.metadata.version != TOKENIZER_VERSION
            && self.metadata.version != TOKENIZER_VERSION_V2
        {
            return Err(TensorError::InvalidOperation(format!(
                "unsupported tokenizer version {}, expected {} or {}",
                self.metadata.version, TOKENIZER_VERSION, TOKENIZER_VERSION_V2
            )));
        }
        if self.metadata.pad_id != PAD_ID
            || self.metadata.bos_id != BOS_ID
            || self.metadata.eos_id != EOS_ID
            || self.metadata.byte_offset != BYTE_OFFSET
            || self.metadata.byte_vocab != BYTE_VOCAB
        {
            return Err(TensorError::InvalidOperation(
                "tokenizer metadata special ids do not match runtime constants".to_string(),
            ));
        }
        if self.metadata.vocab_size != self.id_to_bytes.len() {
            return Err(TensorError::InvalidOperation(format!(
                "tokenizer metadata vocab_size={} does not match table len={}",
                self.metadata.vocab_size,
                self.id_to_bytes.len()
            )));
        }
        if self.id_to_bytes.len() < BYTE_OFFSET + BYTE_VOCAB {
            return Err(TensorError::InvalidOperation(format!(
                "tokenizer has {} entries, expected at least {}",
                self.id_to_bytes.len(),
                BYTE_OFFSET + BYTE_VOCAB
            )));
        }
        for special in [PAD_ID, BOS_ID, EOS_ID] {
            if !self.id_to_bytes[special].is_empty() {
                return Err(TensorError::InvalidOperation(format!(
                    "special token id {special} must not decode to bytes"
                )));
            }
        }
        for byte in 0..=u8::MAX {
            let id = BYTE_OFFSET + byte as usize;
            if self.id_to_bytes[id] != [byte] {
                return Err(TensorError::InvalidOperation(format!(
                    "byte token id {id} has invalid payload"
                )));
            }
        }
        let merge_start = if self.metadata.version == TOKENIZER_VERSION_V2 {
            validate_reserved_tokens(&self.reserved_tokens)?;
            for token in &self.reserved_tokens {
                if token.id >= self.id_to_bytes.len() {
                    return Err(TensorError::InvalidOperation(format!(
                        "reserved token id {} is outside vocab size {}",
                        token.id,
                        self.id_to_bytes.len()
                    )));
                }
                if self.id_to_bytes[token.id] != token.token.as_bytes() {
                    return Err(TensorError::InvalidOperation(format!(
                        "reserved token id {} payload does not match token string",
                        token.id
                    )));
                }
            }
            BYTE_OFFSET + BYTE_VOCAB + self.reserved_tokens.len()
        } else {
            if !self.reserved_tokens.is_empty() {
                return Err(TensorError::InvalidOperation(
                    "tokenizer v1 artifacts must not contain reserved_tokens".to_string(),
                ));
            }
            BYTE_OFFSET + BYTE_VOCAB
        };
        if self.id_to_bytes.len() != merge_start + self.merges.len() {
            return Err(TensorError::InvalidOperation(format!(
                "tokenizer vocab length {} does not equal merge_start {} + merges {}",
                self.id_to_bytes.len(),
                merge_start,
                self.merges.len()
            )));
        }
        for (expected_id, merge) in (merge_start..).zip(self.merges.iter()) {
            if merge.id != expected_id {
                return Err(TensorError::InvalidOperation(format!(
                    "merge id {} is not contiguous expected {}",
                    merge.id, expected_id
                )));
            }
            if merge.left >= merge.id
                || merge.right >= merge.id
                || merge.id >= self.id_to_bytes.len()
            {
                return Err(TensorError::InvalidOperation(format!(
                    "merge id {} references invalid parents ({}, {})",
                    merge.id, merge.left, merge.right
                )));
            }
            if self.metadata.version == TOKENIZER_VERSION_V2
                && (reserved_token_id(&self.reserved_tokens, merge.left).is_some()
                    || reserved_token_id(&self.reserved_tokens, merge.right).is_some())
            {
                return Err(TensorError::InvalidOperation(format!(
                    "merge id {} attempts to merge reserved token parent ({}, {})",
                    merge.id, merge.left, merge.right
                )));
            }
            let mut expected_bytes = self.id_to_bytes[merge.left].clone();
            expected_bytes.extend_from_slice(&self.id_to_bytes[merge.right]);
            if self.id_to_bytes[merge.id] != expected_bytes {
                return Err(TensorError::InvalidOperation(format!(
                    "merge id {} bytes do not match parent concatenation",
                    merge.id
                )));
            }
            if self.digit_isolation_enabled()
                && self.id_to_bytes[merge.id]
                    .iter()
                    .any(|byte| byte.is_ascii_digit())
            {
                return Err(TensorError::InvalidOperation(format!(
                    "merge id {} contains an ASCII digit despite digit isolation",
                    merge.id
                )));
            }
        }
        Ok(())
    }

    fn match_reserved(&self, bytes: &[u8], index: usize) -> Option<(usize, usize)> {
        let candidates = self.reserved_lookup().get(&bytes[index])?;
        for (id, needle) in candidates {
            if index + needle.len() <= bytes.len() && bytes[index..index + needle.len()] == **needle
            {
                return Some((*id, needle.len()));
            }
        }
        None
    }

    fn merge_lookup(&self) -> &HashMap<(usize, usize), usize> {
        self.merge_lookup.get_or_init(|| {
            self.merges
                .iter()
                .map(|merge| ((merge.left, merge.right), merge.id))
                .collect()
        })
    }

    fn reserved_lookup(&self) -> &ReservedLookup {
        self.reserved_lookup.get_or_init(|| {
            let mut lookup = ReservedLookup::new();
            for token in &self.reserved_tokens {
                let bytes = token.token.as_bytes();
                if let Some(&first) = bytes.first() {
                    lookup
                        .entry(first)
                        .or_default()
                        .push((token.id, bytes.to_vec()));
                }
            }
            for candidates in lookup.values_mut() {
                candidates.sort_by(|(left_id, left_bytes), (right_id, right_bytes)| {
                    right_bytes
                        .len()
                        .cmp(&left_bytes.len())
                        .then_with(|| left_id.cmp(right_id))
                });
            }
            lookup
        })
    }
}

pub fn default_reserved_tokens() -> Vec<ReservedToken> {
    let registry = [
        ("<|system|>", "chat", "system message boundary"),
        ("<|user|>", "chat", "user message boundary"),
        ("<|assistant|>", "chat", "assistant message boundary"),
        ("<|message_end|>", "chat", "message terminator"),
        ("<|tool_call|>", "tool", "tool call envelope"),
        ("<|tool_name|>", "tool", "tool name field"),
        ("<|tool_args|>", "tool", "tool argument field"),
        ("<|tool_result|>", "tool_result", "tool result envelope"),
        ("<|tool_error|>", "tool_result", "tool error envelope"),
        ("<|memory_read|>", "memory", "memory read marker"),
        ("<|memory_write|>", "memory", "memory write marker"),
        ("<|memory_row|>", "memory", "memory row reference"),
        ("<|smft|>", "smft", "SMFT marker"),
        ("<|smft_trainable|>", "smft", "SMFT trainable row marker"),
        ("<|smft_frozen|>", "smft", "SMFT frozen row marker"),
        ("<|trace|>", "trace", "trace envelope"),
        ("<|trace_step|>", "trace", "trace step marker"),
        ("<|state|>", "trace", "state field marker"),
        ("<|action|>", "trace", "action field marker"),
        ("<|evidence|>", "evidence_citation", "evidence marker"),
        ("<|citation|>", "evidence_citation", "citation marker"),
        ("<|claim|>", "evidence_citation", "claim marker"),
        ("<|document|>", "document_source", "document marker"),
        ("<|source|>", "document_source", "source marker"),
        ("<|source_id|>", "document_source", "source id marker"),
        ("<|time_now|>", "temporal", "current time marker"),
        ("<|event_time|>", "temporal", "event time marker"),
        ("<|elapsed_time|>", "temporal", "elapsed time marker"),
        ("<|sep|>", "separator", "separator marker"),
        ("<|record_end|>", "separator", "record terminator"),
        ("<|mask|>", "mask", "mask marker"),
        ("<|qb_artifact|>", "qb_reference", "QB artifact reference"),
        ("<|qb_claim|>", "qb_reference", "QB claim reference"),
        ("<|qb_authority|>", "qb_reference", "QB authority marker"),
        ("<|developer|>", "chat", "developer message boundary"),
        ("<|instruction|>", "chat", "instruction boundary"),
        ("<|response|>", "chat", "response boundary"),
        ("<|context|>", "chat", "context boundary"),
        ("<|turn|>", "chat", "conversation turn marker"),
        ("<|end_turn|>", "chat", "conversation turn terminator"),
        ("<|tool_id|>", "tool", "tool identifier field"),
        ("<|tool_input|>", "tool", "tool input payload"),
        ("<|tool_output|>", "tool", "tool output payload"),
        ("<|tool_status|>", "tool", "tool status field"),
        ("<|tool_call_id|>", "tool", "tool call identifier"),
        ("<|tool_schema|>", "tool", "tool schema marker"),
        ("<|tool_choice|>", "tool", "tool choice marker"),
        ("<|tool_observation|>", "tool", "tool observation marker"),
        ("<|tool_stdout|>", "tool_result", "tool stdout stream"),
        ("<|tool_stderr|>", "tool_result", "tool stderr stream"),
        ("<|tool_exit_code|>", "tool_result", "tool exit code"),
        ("<|tool_artifact|>", "tool_result", "tool-produced artifact"),
        ("<|memory_key|>", "memory", "memory key field"),
        ("<|memory_value|>", "memory", "memory value field"),
        ("<|memory_slot|>", "memory", "memory slot marker"),
        ("<|memory_query|>", "memory", "memory query marker"),
        ("<|memory_result|>", "memory", "memory result marker"),
        ("<|memory_update|>", "memory", "memory update marker"),
        ("<|memory_scope|>", "memory", "memory scope marker"),
        ("<|smft_mask|>", "smft", "SMFT mask marker"),
        ("<|smft_rank|>", "smft", "SMFT rank marker"),
        ("<|smft_row|>", "smft", "SMFT row marker"),
        ("<|smft_union|>", "smft", "SMFT row-union marker"),
        ("<|smft_delta|>", "smft", "SMFT delta marker"),
        ("<|trace_id|>", "trace", "trace id field"),
        ("<|trace_parent|>", "trace", "trace parent field"),
        ("<|trace_child|>", "trace", "trace child field"),
        ("<|reasoning|>", "trace", "reasoning field marker"),
        ("<|plan|>", "trace", "plan field marker"),
        ("<|step|>", "trace", "step field marker"),
        ("<|observation|>", "trace", "observation field marker"),
        ("<|decision|>", "trace", "decision field marker"),
        ("<|quote|>", "evidence_citation", "quote marker"),
        ("<|url|>", "evidence_citation", "URL marker"),
        (
            "<|source_title|>",
            "evidence_citation",
            "source title marker",
        ),
        ("<|source_date|>", "evidence_citation", "source date marker"),
        ("<|evidence_id|>", "evidence_citation", "evidence id marker"),
        ("<|claim_id|>", "evidence_citation", "claim id marker"),
        ("<|confidence|>", "evidence_citation", "confidence marker"),
        ("<|doc_id|>", "document_source", "document id marker"),
        ("<|doc_title|>", "document_source", "document title marker"),
        ("<|doc_url|>", "document_source", "document URL marker"),
        (
            "<|doc_author|>",
            "document_source",
            "document author marker",
        ),
        ("<|doc_date|>", "document_source", "document date marker"),
        ("<|section|>", "document_source", "document section marker"),
        ("<|chunk|>", "document_source", "document chunk marker"),
        ("<|current_date|>", "temporal", "current date marker"),
        ("<|current_time|>", "temporal", "current time marker"),
        ("<|timezone|>", "temporal", "timezone marker"),
        ("<|created_at|>", "temporal", "created-at timestamp"),
        ("<|updated_at|>", "temporal", "updated-at timestamp"),
        ("<|deadline|>", "temporal", "deadline marker"),
        ("<|duration|>", "temporal", "duration marker"),
        ("<|relative_time|>", "temporal", "relative time marker"),
        ("<|field|>", "separator", "field separator"),
        ("<|item|>", "separator", "item separator"),
        ("<|list_end|>", "separator", "list terminator"),
        ("<|json|>", "separator", "JSON block marker"),
        ("<|yaml|>", "separator", "YAML block marker"),
        ("<|code|>", "separator", "code block marker"),
        ("<|mask_span|>", "mask", "mask span marker"),
        ("<|mask_token|>", "mask", "mask token marker"),
        ("<|redacted|>", "mask", "redaction marker"),
        ("<|qb_task|>", "qb_reference", "QB task marker"),
        ("<|qb_route|>", "qb_reference", "QB route marker"),
        ("<|qb_specialist|>", "qb_reference", "QB specialist marker"),
        ("<|qb_tool|>", "qb_reference", "QB tool marker"),
        ("<|qb_memory|>", "qb_reference", "QB memory marker"),
        ("<|qb_evidence|>", "qb_reference", "QB evidence marker"),
        ("<|qb_trace|>", "qb_reference", "QB trace marker"),
        ("<|artifact|>", "artifact", "artifact marker"),
        ("<|artifact_id|>", "artifact", "artifact id marker"),
        ("<|artifact_path|>", "artifact", "artifact path marker"),
        ("<|artifact_hash|>", "artifact", "artifact hash marker"),
        ("<|policy|>", "governance", "policy marker"),
        ("<|license|>", "governance", "license marker"),
        ("<|attribution|>", "governance", "attribution marker"),
        ("<|privacy|>", "governance", "privacy marker"),
        ("<|provenance|>", "governance", "provenance marker"),
        ("<|review|>", "governance", "review marker"),
        ("<|audit|>", "governance", "audit marker"),
        ("<|schema|>", "schema", "schema marker"),
        ("<|schema_id|>", "schema", "schema id marker"),
        ("<|schema_version|>", "schema", "schema version marker"),
        ("<|field_name|>", "schema", "schema field-name marker"),
        ("<|field_value|>", "schema", "schema field-value marker"),
        ("<|type|>", "schema", "schema type marker"),
        ("<|required|>", "schema", "schema required-field marker"),
    ];
    registry
        .iter()
        .enumerate()
        .map(|(index, (token, category, description))| ReservedToken {
            id: BYTE_OFFSET + BYTE_VOCAB + index,
            token: (*token).to_string(),
            category: (*category).to_string(),
            description: (*description).to_string(),
        })
        .collect()
}

pub fn reserved_tokens_from_json_str(json: &str) -> Result<Vec<ReservedToken>> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|err| TensorError::Io(format!("failed to parse reserved token JSON: {err}")))?;
    let tokens_value = value.get("tokens").cloned().unwrap_or(value);
    let tokens: Vec<ReservedToken> = serde_json::from_value(tokens_value).map_err(|err| {
        TensorError::Io(format!("failed to parse reserved token registry: {err}"))
    })?;
    validate_reserved_tokens(&tokens)?;
    Ok(tokens)
}

pub fn validate_reserved_tokens(tokens: &[ReservedToken]) -> Result<()> {
    let mut ids = BTreeSet::new();
    let mut strings = BTreeSet::new();
    let mut categories = BTreeSet::new();
    for (index, token) in tokens.iter().enumerate() {
        let expected_id = BYTE_OFFSET + BYTE_VOCAB + index;
        if token.id != expected_id {
            return Err(TensorError::InvalidOperation(format!(
                "reserved token {} has id {}, expected contiguous id {}",
                token.token, token.id, expected_id
            )));
        }
        if token.token.is_empty() {
            return Err(TensorError::InvalidOperation(
                "reserved token string must not be empty".to_string(),
            ));
        }
        if token.category.trim().is_empty() {
            return Err(TensorError::InvalidOperation(format!(
                "reserved token {} has empty category",
                token.token
            )));
        }
        if !ids.insert(token.id) {
            return Err(TensorError::InvalidOperation(format!(
                "duplicate reserved token id {}",
                token.id
            )));
        }
        if !strings.insert(token.token.as_str()) {
            return Err(TensorError::InvalidOperation(format!(
                "duplicate reserved token string {}",
                token.token
            )));
        }
        categories.insert(token.category.as_str());
    }
    let required_categories = [
        "chat",
        "tool",
        "tool_result",
        "memory",
        "smft",
        "trace",
        "evidence_citation",
        "document_source",
        "temporal",
        "separator",
        "mask",
        "qb_reference",
    ];
    for category in required_categories {
        if !categories.contains(category) {
            return Err(TensorError::InvalidOperation(format!(
                "reserved token registry missing required category {category}"
            )));
        }
    }
    Ok(())
}

fn validate_vocab_size(vocab_size: usize, minimum: usize) -> Result<()> {
    if vocab_size < minimum {
        return Err(TensorError::InvalidOperation(format!(
            "BPE vocab size must be at least {minimum}, got {vocab_size}"
        )));
    }
    Ok(())
}

fn legacy_metadata() -> BpeTokenizerMetadata {
    BpeTokenizerMetadata {
        format: String::new(),
        version: TOKENIZER_VERSION,
        vocab_size: 0,
        byte_offset: BYTE_OFFSET,
        byte_vocab: BYTE_VOCAB,
        pad_id: PAD_ID,
        bos_id: BOS_ID,
        eos_id: EOS_ID,
        training_bytes: 0,
        training_hash: "legacy".to_string(),
        tokenizer_id: String::new(),
        source_blend_hash: String::new(),
        sample_manifest_hash: String::new(),
        reserved_registry_hash: String::new(),
        artifact_hash: String::new(),
        training_config: BpeTokenizerTrainingConfig::default(),
        validation: BpeTokenizerValidationMetrics::default(),
    }
}

fn inferred_legacy_metadata(vocab_size: usize) -> BpeTokenizerMetadata {
    BpeTokenizerMetadata {
        format: TOKENIZER_FORMAT.to_string(),
        version: TOKENIZER_VERSION,
        vocab_size,
        byte_offset: BYTE_OFFSET,
        byte_vocab: BYTE_VOCAB,
        pad_id: PAD_ID,
        bos_id: BOS_ID,
        eos_id: EOS_ID,
        training_bytes: 0,
        training_hash: "legacy".to_string(),
        tokenizer_id: "heirloom-byte-bpe-legacy".to_string(),
        source_blend_hash: String::new(),
        sample_manifest_hash: String::new(),
        reserved_registry_hash: String::new(),
        artifact_hash: String::new(),
        training_config: BpeTokenizerTrainingConfig::default(),
        validation: BpeTokenizerValidationMetrics::default(),
    }
}

pub(crate) fn stable_hash_bytes(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn stable_hash_reserved_tokens(tokens: &[ReservedToken]) -> Result<String> {
    serde_json::to_vec(tokens)
        .map(|bytes| stable_hash_bytes(&bytes))
        .map_err(|err| TensorError::Io(format!("failed to serialize reserved tokens: {err}")))
}

fn train_bpe_merges(
    bytes: &[u8],
    weight: u64,
    vocab_size: usize,
    reserved_tokens: &[ReservedToken],
    min_pair_count: u64,
) -> Result<(Vec<Vec<u8>>, Vec<BpeMerge>, u64)> {
    let mut id_to_bytes = initial_id_to_bytes(reserved_tokens);
    let mut sequences = HashMap::<Vec<usize>, u64>::new();
    let mut training_sequences = 0u64;
    for segment in byte_segments_without_reserved(bytes, reserved_tokens) {
        for chunk in byte_chunks_bytes(segment) {
            if chunk.is_empty() {
                continue;
            }
            let sequence = chunk
                .iter()
                .map(|byte| BYTE_OFFSET + *byte as usize)
                .collect::<Vec<_>>();
            *sequences.entry(sequence).or_default() += weight;
            training_sequences += 1;
        }
    }
    let mut merges = Vec::new();
    while id_to_bytes.len() < vocab_size && !sequences.is_empty() {
        let Some(((left, right), count)) = most_frequent_pair(&sequences) else {
            break;
        };
        if count < min_pair_count {
            break;
        }
        let id = id_to_bytes.len();
        let mut bytes = id_to_bytes[left].clone();
        bytes.extend_from_slice(&id_to_bytes[right]);
        id_to_bytes.push(bytes);
        merges.push(BpeMerge { left, right, id });
        sequences = replace_pair_in_sequences(sequences, left, right, id);
    }
    Ok((id_to_bytes, merges, training_sequences))
}

fn initial_id_to_bytes(reserved_tokens: &[ReservedToken]) -> Vec<Vec<u8>> {
    let mut id_to_bytes = vec![Vec::new(), Vec::new(), Vec::new()];
    for byte in 0..=u8::MAX {
        id_to_bytes.push(vec![byte]);
    }
    for token in reserved_tokens {
        id_to_bytes.push(token.token.as_bytes().to_vec());
    }
    id_to_bytes
}

fn most_frequent_pair(sequences: &HashMap<Vec<usize>, u64>) -> Option<((usize, usize), u64)> {
    let mut counts = HashMap::<(usize, usize), u64>::new();
    for (sequence, frequency) in sequences {
        for pair in sequence.windows(2) {
            *counts.entry((pair[0], pair[1])).or_default() += *frequency;
        }
    }
    counts.into_iter().max_by(
        |((left_a, right_a), count_a), ((left_b, right_b), count_b)| {
            count_a
                .cmp(count_b)
                .then_with(|| left_b.cmp(left_a))
                .then_with(|| right_b.cmp(right_a))
        },
    )
}

fn replace_pair_in_sequences(
    sequences: HashMap<Vec<usize>, u64>,
    left: usize,
    right: usize,
    replacement: usize,
) -> HashMap<Vec<usize>, u64> {
    let mut updated = HashMap::with_capacity(sequences.len());
    for (sequence, frequency) in sequences {
        let sequence = replace_pair(&sequence, left, right, replacement);
        *updated.entry(sequence).or_default() += frequency;
    }
    updated
}

fn replace_pair(sequence: &[usize], left: usize, right: usize, replacement: usize) -> Vec<usize> {
    let mut output = Vec::with_capacity(sequence.len());
    let mut index = 0;
    while index < sequence.len() {
        if index + 1 < sequence.len() && sequence[index] == left && sequence[index + 1] == right {
            output.push(replacement);
            index += 2;
        } else {
            output.push(sequence[index]);
            index += 1;
        }
    }
    output
}

fn encode_chunk_with_merge_lookup(
    chunk: &[u8],
    merge_lookup: &HashMap<(usize, usize), usize>,
) -> Vec<usize> {
    let mut tokens = chunk
        .iter()
        .map(|byte| BYTE_OFFSET + *byte as usize)
        .collect::<Vec<_>>();
    if tokens.len() < 2 || merge_lookup.is_empty() {
        return tokens;
    }

    let len = tokens.len();
    let none = usize::MAX;
    let mut previous = (0..len)
        .map(|index| if index == 0 { none } else { index - 1 })
        .collect::<Vec<_>>();
    let mut next = (0..len)
        .map(|index| if index + 1 == len { none } else { index + 1 })
        .collect::<Vec<_>>();
    let mut alive = vec![true; len];
    let mut heap = BinaryHeap::<std::cmp::Reverse<(usize, usize, usize, usize)>>::new();

    for index in 0..len - 1 {
        if let Some(&merge_id) = merge_lookup.get(&(tokens[index], tokens[index + 1])) {
            heap.push(std::cmp::Reverse((
                merge_id,
                index,
                tokens[index],
                tokens[index + 1],
            )));
        }
    }

    while let Some(std::cmp::Reverse((merge_id, left, left_token, right_token))) = heap.pop() {
        if left >= tokens.len() || !alive[left] || tokens[left] != left_token {
            continue;
        }
        let right = next[left];
        if right == none || right >= tokens.len() || !alive[right] || tokens[right] != right_token {
            continue;
        }
        if merge_lookup.get(&(tokens[left], tokens[right])).copied() != Some(merge_id) {
            continue;
        }

        tokens[left] = merge_id;
        alive[right] = false;
        let after = next[right];
        next[left] = after;
        if after != none {
            previous[after] = left;
        }

        let before = previous[left];
        if before != none {
            if let Some(&candidate) = merge_lookup.get(&(tokens[before], tokens[left])) {
                heap.push(std::cmp::Reverse((
                    candidate,
                    before,
                    tokens[before],
                    tokens[left],
                )));
            }
        }
        if after != none {
            if let Some(&candidate) = merge_lookup.get(&(tokens[left], tokens[after])) {
                heap.push(std::cmp::Reverse((
                    candidate,
                    left,
                    tokens[left],
                    tokens[after],
                )));
            }
        }
    }

    let mut output = Vec::new();
    let mut index = 0usize;
    while index != none {
        if alive[index] {
            output.push(tokens[index]);
        }
        index = next[index];
    }
    output
}

#[derive(Clone)]
struct BpeTrainerWord {
    tokens: Vec<usize>,
    weight: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PairCandidate {
    count: u64,
    left: usize,
    right: usize,
}

#[derive(Clone, Debug, Default)]
struct BpeTrainerStats {
    initial_unique_sequences: usize,
    initial_pair_count: usize,
    final_pair_count: usize,
    heap_pops: u64,
    stale_heap_pops: u64,
    heap_rebuilds: u64,
    heap_rebuild_candidates_dropped: u64,
    affected_word_updates: u64,
    max_heap_len: usize,
}

impl Ord for PairCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

impl PartialOrd for PairCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn train_bpe_merges_incremental(
    sequences: HashMap<Vec<usize>, u64>,
    vocab_size: usize,
    reserved_tokens: &[ReservedToken],
    min_pair_count: u64,
) -> Result<(Vec<Vec<u8>>, Vec<BpeMerge>, BpeTrainerStats)> {
    let mut id_to_bytes = initial_id_to_bytes(reserved_tokens);
    let initial_unique_sequences = sequences.len();
    let mut words = sequences
        .into_iter()
        .filter(|(tokens, weight)| tokens.len() > 1 && *weight > 0)
        .map(|(tokens, weight)| BpeTrainerWord { tokens, weight })
        .collect::<Vec<_>>();
    if words.is_empty() {
        return Ok((
            id_to_bytes,
            Vec::new(),
            BpeTrainerStats {
                initial_unique_sequences,
                ..BpeTrainerStats::default()
            },
        ));
    }

    let mut pair_counts = HashMap::<(usize, usize), u64>::new();
    let mut pair_words = HashMap::<(usize, usize), HashSet<usize>>::new();
    for (word_id, word) in words.iter().enumerate() {
        add_word_pair_counts_without_heap(
            word_id,
            &word.tokens,
            word.weight,
            &mut pair_counts,
            &mut pair_words,
        );
    }
    let initial_pair_count = pair_counts.len();
    let mut heap = BinaryHeap::<PairCandidate>::with_capacity(pair_counts.len());
    for (&(left, right), &count) in pair_counts.iter() {
        heap.push(PairCandidate { count, left, right });
    }
    let mut stats = BpeTrainerStats {
        initial_unique_sequences,
        initial_pair_count,
        max_heap_len: heap.len(),
        ..BpeTrainerStats::default()
    };

    let mut merges = Vec::new();
    while id_to_bytes.len() < vocab_size {
        let Some(candidate) = pop_current_best_pair(&mut heap, &pair_counts, &mut stats) else {
            break;
        };
        if candidate.count < min_pair_count {
            break;
        }
        if reserved_token_id(reserved_tokens, candidate.left).is_some()
            || reserved_token_id(reserved_tokens, candidate.right).is_some()
        {
            return Err(TensorError::InvalidOperation(
                "tokenizer v2 trainer attempted to merge a reserved token".to_string(),
            ));
        }

        let mut affected = pair_words
            .get(&(candidate.left, candidate.right))
            .map(|word_ids| word_ids.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        affected.sort_unstable();
        if affected.is_empty() {
            pair_counts.remove(&(candidate.left, candidate.right));
            continue;
        }

        let id = id_to_bytes.len();
        let mut bytes = id_to_bytes[candidate.left].clone();
        bytes.extend_from_slice(&id_to_bytes[candidate.right]);
        id_to_bytes.push(bytes);
        merges.push(BpeMerge {
            left: candidate.left,
            right: candidate.right,
            id,
        });

        for word_id in affected {
            if !words[word_id]
                .tokens
                .windows(2)
                .any(|pair| pair[0] == candidate.left && pair[1] == candidate.right)
            {
                continue;
            }
            stats.affected_word_updates = stats.affected_word_updates.saturating_add(1);
            let weight = words[word_id].weight;
            let old_tokens = std::mem::take(&mut words[word_id].tokens);
            subtract_word_pair_counts(
                word_id,
                &old_tokens,
                weight,
                &mut pair_counts,
                &mut pair_words,
                &mut heap,
            );
            let new_tokens = replace_pair(&old_tokens, candidate.left, candidate.right, id);
            add_word_pair_counts(
                word_id,
                &new_tokens,
                weight,
                &mut pair_counts,
                &mut pair_words,
                &mut heap,
            );
            stats.max_heap_len = stats.max_heap_len.max(heap.len());
            words[word_id].tokens = new_tokens;
        }
        maybe_rebuild_pair_heap(&mut heap, &pair_counts, &mut stats);
    }

    stats.final_pair_count = pair_counts.len();
    Ok((id_to_bytes, merges, stats))
}

fn pop_current_best_pair(
    heap: &mut BinaryHeap<PairCandidate>,
    pair_counts: &HashMap<(usize, usize), u64>,
    stats: &mut BpeTrainerStats,
) -> Option<PairCandidate> {
    while let Some(candidate) = heap.pop() {
        stats.heap_pops = stats.heap_pops.saturating_add(1);
        let current = pair_counts
            .get(&(candidate.left, candidate.right))
            .copied()
            .unwrap_or(0);
        if current == candidate.count && current > 0 {
            return Some(candidate);
        }
        stats.stale_heap_pops = stats.stale_heap_pops.saturating_add(1);
    }
    None
}

fn maybe_rebuild_pair_heap(
    heap: &mut BinaryHeap<PairCandidate>,
    pair_counts: &HashMap<(usize, usize), u64>,
    stats: &mut BpeTrainerStats,
) {
    let live_pairs = pair_counts.len().max(1);
    let rebuild_threshold = live_pairs.saturating_mul(16).max(100_000);
    if heap.len() <= rebuild_threshold {
        return;
    }
    let old_len = heap.len();
    heap.clear();
    heap.reserve(pair_counts.len());
    for (&(left, right), &count) in pair_counts.iter() {
        if count > 0 {
            heap.push(PairCandidate { count, left, right });
        }
    }
    stats.heap_rebuilds = stats.heap_rebuilds.saturating_add(1);
    stats.heap_rebuild_candidates_dropped = stats
        .heap_rebuild_candidates_dropped
        .saturating_add(old_len.saturating_sub(heap.len()) as u64);
}

fn add_word_pair_counts(
    word_id: usize,
    tokens: &[usize],
    weight: u64,
    pair_counts: &mut HashMap<(usize, usize), u64>,
    pair_words: &mut HashMap<(usize, usize), HashSet<usize>>,
    heap: &mut BinaryHeap<PairCandidate>,
) {
    if tokens.len() < 2 || weight == 0 {
        return;
    }
    for (key, delta) in pair_deltas(tokens, weight) {
        let count = pair_counts.entry(key).or_default();
        *count = count.saturating_add(delta);
        heap.push(PairCandidate {
            count: *count,
            left: key.0,
            right: key.1,
        });
        pair_words.entry(key).or_default().insert(word_id);
    }
}

fn add_word_pair_counts_without_heap(
    word_id: usize,
    tokens: &[usize],
    weight: u64,
    pair_counts: &mut HashMap<(usize, usize), u64>,
    pair_words: &mut HashMap<(usize, usize), HashSet<usize>>,
) {
    if tokens.len() < 2 || weight == 0 {
        return;
    }
    for (key, delta) in pair_deltas(tokens, weight) {
        let count = pair_counts.entry(key).or_default();
        *count = count.saturating_add(delta);
        pair_words.entry(key).or_default().insert(word_id);
    }
}

fn subtract_word_pair_counts(
    word_id: usize,
    tokens: &[usize],
    weight: u64,
    pair_counts: &mut HashMap<(usize, usize), u64>,
    pair_words: &mut HashMap<(usize, usize), HashSet<usize>>,
    heap: &mut BinaryHeap<PairCandidate>,
) {
    if tokens.len() < 2 || weight == 0 {
        return;
    }
    for (key, delta) in pair_deltas(tokens, weight) {
        if let Some(count) = pair_counts.get_mut(&key) {
            if *count <= delta {
                *count = 0;
            } else {
                *count -= delta;
                heap.push(PairCandidate {
                    count: *count,
                    left: key.0,
                    right: key.1,
                });
            }
        }
        if pair_counts.get(&key).copied().unwrap_or(0) == 0 {
            pair_counts.remove(&key);
        }
        let remove_entry = if let Some(word_ids) = pair_words.get_mut(&key) {
            word_ids.remove(&word_id);
            word_ids.is_empty()
        } else {
            false
        };
        if remove_entry {
            pair_words.remove(&key);
        }
    }
}

fn pair_deltas(tokens: &[usize], weight: u64) -> Vec<((usize, usize), u64)> {
    let mut pairs = tokens
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect::<Vec<_>>();
    pairs.sort_unstable();

    let mut deltas = Vec::new();
    let mut index = 0usize;
    while index < pairs.len() {
        let key = pairs[index];
        let mut occurrences = 1u64;
        index += 1;
        while index < pairs.len() && pairs[index] == key {
            occurrences = occurrences.saturating_add(1);
            index += 1;
        }
        deltas.push((key, occurrences.saturating_mul(weight)));
    }
    deltas
}

fn byte_chunks_bytes(bytes: &[u8]) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut current_is_whitespace = None;

    for (index, byte) in bytes.iter().enumerate() {
        let is_whitespace = byte.is_ascii_whitespace();
        if current_is_whitespace.is_some_and(|kind| kind != is_whitespace) {
            chunks.push(&bytes[start..index]);
            start = index;
        }
        current_is_whitespace = Some(is_whitespace);
    }

    if start < bytes.len() {
        chunks.push(&bytes[start..]);
    }
    chunks
}

fn byte_segments_without_reserved<'a>(
    bytes: &'a [u8],
    reserved_tokens: &'a [ReservedToken],
) -> Vec<&'a [u8]> {
    byte_segments_for_bpe(bytes, reserved_tokens, false)
}

fn byte_segments_for_bpe<'a>(
    bytes: &'a [u8],
    reserved_tokens: &'a [ReservedToken],
    digit_isolation: bool,
) -> Vec<&'a [u8]> {
    if reserved_tokens.is_empty() {
        if !digit_isolation {
            return vec![bytes];
        }
        return byte_segments_split_digits(bytes);
    }
    let mut segments = Vec::new();
    let mut index = 0usize;
    let mut start = 0usize;
    while index < bytes.len() {
        if let Some((_, len)) = match_reserved_registry(reserved_tokens, bytes, index) {
            if start < index {
                push_bpe_segment(&mut segments, &bytes[start..index], digit_isolation);
            }
            index += len;
            start = index;
        } else if digit_isolation && bytes[index].is_ascii_digit() {
            if start < index {
                push_bpe_segment(&mut segments, &bytes[start..index], digit_isolation);
            }
            segments.push(&bytes[index..index + 1]);
            index += 1;
            start = index;
        } else {
            index += 1;
        }
    }
    if start < bytes.len() {
        push_bpe_segment(&mut segments, &bytes[start..], digit_isolation);
    }
    segments
}

fn push_bpe_segment<'a>(segments: &mut Vec<&'a [u8]>, segment: &'a [u8], digit_isolation: bool) {
    if digit_isolation {
        segments.extend(byte_segments_split_digits(segment));
    } else if !segment.is_empty() {
        segments.push(segment);
    }
}

fn byte_segments_split_digits(bytes: &[u8]) -> Vec<&[u8]> {
    let mut segments = Vec::new();
    let mut index = 0usize;
    let mut start = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_digit() {
            if start < index {
                segments.push(&bytes[start..index]);
            }
            segments.push(&bytes[index..index + 1]);
            index += 1;
            start = index;
        } else {
            index += 1;
        }
    }
    if start < bytes.len() {
        segments.push(&bytes[start..]);
    }
    segments
}

fn match_reserved_registry(
    reserved_tokens: &[ReservedToken],
    bytes: &[u8],
    index: usize,
) -> Option<(usize, usize)> {
    reserved_tokens
        .iter()
        .filter_map(|token| {
            let needle = token.token.as_bytes();
            if needle.is_empty() || index + needle.len() > bytes.len() {
                return None;
            }
            (bytes[index..index + needle.len()] == *needle).then_some((token.id, needle.len()))
        })
        .max_by_key(|(id, len)| (*len, std::cmp::Reverse(*id)))
}

fn reserved_token_id(tokens: &[ReservedToken], id: usize) -> Option<usize> {
    let start = BYTE_OFFSET + BYTE_VOCAB;
    (id >= start && id < start + tokens.len()).then_some(id)
}

pub fn tokenizer_metadata_report(
    tokenizer: &BpeTokenizer,
    path: Option<&Path>,
) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "tokenizer_id": tokenizer.tokenizer_id(),
        "tokenizer_path": path.map(|path| path.display().to_string()),
        "format": &tokenizer.metadata().format,
        "version": tokenizer.metadata().version,
        "vocab_size": tokenizer.vocab_size(),
        "tokenizer_hash": tokenizer.fingerprint()?,
        "artifact_hash": &tokenizer.metadata().artifact_hash,
        "source_blend_hash": &tokenizer.metadata().source_blend_hash,
        "sample_manifest_hash": &tokenizer.metadata().sample_manifest_hash,
        "reserved_tokens": tokenizer.reserved_tokens().len(),
        "reserved_registry_hash": &tokenizer.metadata().reserved_registry_hash,
        "training_config": &tokenizer.metadata().training_config,
        "validation": &tokenizer.metadata().validation,
    }))
}

pub fn token_length_histogram(tokenizer: &BpeTokenizer) -> BTreeMap<String, usize> {
    let mut histogram = BTreeMap::new();
    for piece in &tokenizer.id_to_bytes {
        *histogram.entry(piece.len().to_string()).or_default() += 1;
    }
    histogram
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_encode_chunk(tokenizer: &BpeTokenizer, chunk: &[u8]) -> Vec<usize> {
        let mut tokens = chunk
            .iter()
            .map(|byte| BYTE_OFFSET + *byte as usize)
            .collect::<Vec<_>>();
        for merge in tokenizer.merges() {
            tokens = replace_pair(&tokens, merge.left, merge.right, merge.id);
        }
        tokens
    }

    #[test]
    fn fast_chunk_encoder_matches_sequential_merge_reference() {
        let tokenizer =
            BpeTokenizer::train_v1_bytes(b"aaaa aaab aaaab ababa banana bandana abracadabra", 512)
                .expect("train tokenizer");
        for chunk in [
            b"aaaa".as_slice(),
            b"aaaab".as_slice(),
            b"ababa".as_slice(),
            b"banana".as_slice(),
            b"abracadabra".as_slice(),
        ] {
            assert_eq!(
                encode_chunk_with_merge_lookup(chunk, tokenizer.merge_lookup()),
                reference_encode_chunk(&tokenizer, chunk)
            );
        }
    }
}
