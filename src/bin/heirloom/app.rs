use clap::{Parser, Subcommand, ValueEnum};
use heirloom::checkpoint::{
    load_lm_checkpoint_on_device, load_memory_lm_checkpoint_on_device,
    save_lm_checkpoint_with_dataset_state, save_memory_lm_checkpoint_with_dataset_state_and_step,
};
use heirloom::data::{
    build_qb_native_corpus_blend_manifest, download_tinystories_valid, prepare_lm_data,
    prepare_lm_data_binary_shards, read_text, write_corpus_blend_manifest, write_token_shard,
    CorpusBlendManifest, CorpusBlendSource, DataSplit, PreparedDataManifest, PreparedDataOptions,
    PreparedDataShard, PreparedDataSource, PreparedTokenData, StreamingTokenDataset,
    TokenDataSplit, TokenDataset, TokenDatasetState, TokenShardDType, CORPUS_BLEND_MANIFEST_FORMAT,
    CORPUS_BLEND_MANIFEST_VERSION, DATASET_MANIFEST_FORMAT, DATASET_MANIFEST_VERSION_V2,
    DATASET_STORAGE_BINARY_SHARDS,
};
use heirloom::memory_transformer::{
    MemoryAccessCounts, MemoryLookupKind, MemoryTransformerConfig, MemoryTransformerLm,
    MemoryUpdatePolicy, SmftMode, SmftRowMask,
};
use heirloom::nn::{
    amp_bf16_cuda_host_staging_events, amp_bf16_op_decisions, amp_bf16_policy,
    amp_bf16_tensor_core_coverage, reset_amp_bf16_tensor_core_coverage, AdamW,
    AmpBf16TensorCoreCoverage, DistributedGradientStats, GenerationOptions, GenerationOutput,
    LmEvalMetrics, Module, Optimizer, SparseAdamWCompactRowsUpdate, SparseAdamWRowsUpdate,
    TinyTransformerConfig, TinyTransformerLm,
};
use heirloom::rng::HeirloomRng;
use heirloom::tokenizer::{
    default_reserved_tokens, reserved_tokens_from_json_str, token_length_histogram,
    tokenizer_metadata_report, validate_reserved_tokens, BpeTokenizer, BpeTokenizerV2Options,
    BpeTrainingSample, ReservedToken, BOS_ID, BYTE_OFFSET, BYTE_VOCAB,
    DEFAULT_TOKENIZER_SAMPLE_BYTES, EOS_ID, PRODUCTION_TOKENIZER_VOCAB_SIZE, TOKENIZER_VERSION_V2,
};
use heirloom::{amp, no_grad, DType, Device, Result, Tensor, TensorError};
use heirloom_kernels::cuda;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DDP_PARAMETER_CHECKSUM_TOLERANCE: f64 = 1.0e-3;
const NCCL_PROBE_TOLERANCE: f64 = 1.0e-6;
const NCCL_UNIQUE_ID_HELPER_MARKER: &str = "HEIRLOOM_NCCL_UNIQUE_ID_HEX=";
const NCCL_UNIQUE_ID_HELPER_HOLD_SECS: u64 = 600;
const NCCL_UNIQUE_ID_HELPER_START_TIMEOUT_SECS: u64 = 30;

// The first committee-readiness split intentionally keeps one module namespace:
// these files share many private report/config types, while each file now has a
// bounded review responsibility and can be isolated further without changing
// the CLI surface.
include!("cli.rs");
include!("commands/data.rs");
include!("commands/tokenizer.rs");
include!("commands/gpu.rs");
include!("commands/readiness.rs");
include!("commands/padawan.rs");
include!("commands/training.rs");
include!("commands/internal.rs");
include!("commands/inference.rs");
include!("dispatch.rs");
include!("runtime.rs");
include!("data_pipeline.rs");
include!("experimental.rs");
include!("reports.rs");
include!("tests.rs");
