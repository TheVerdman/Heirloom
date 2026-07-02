#![recursion_limit = "256"]

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

#[derive(Parser)]
#[command(name = "heirloom")]
#[command(about = "Heirloom tensor/autograd training runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Data {
        #[command(subcommand)]
        command: DataCommands,
    },
    Tokenizer {
        #[command(subcommand)]
        command: TokenizerCommands,
    },
    Gpu {
        #[command(subcommand)]
        command: GpuCommands,
    },
    Readiness {
        #[command(subcommand)]
        command: ReadinessCommands,
    },
    Padawan {
        #[command(subcommand)]
        command: PadawanCommands,
    },
    TrainLm {
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        dataset_manifest: Option<PathBuf>,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 500)]
        steps: usize,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        #[arg(long, default_value_t = 1)]
        grad_accumulation_steps: usize,
        #[arg(long, default_value_t = 64)]
        block_size: usize,
        #[arg(long, default_value_t = 64)]
        d_model: usize,
        #[arg(long, default_value_t = 4)]
        n_heads: usize,
        #[arg(long, default_value_t = 256)]
        ff_hidden: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f64,
        #[arg(long, default_value_t = 1.0)]
        clip_norm: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long)]
        devices: Option<String>,
        #[arg(long, value_enum)]
        distributed: Option<DistributedMode>,
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        #[arg(long, default_value_t = false)]
        resume: bool,
        #[arg(long, default_value_t = 25)]
        log_every: usize,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 120)]
        ddp_init_timeout_secs: u64,
        #[arg(long, default_value_t = 1)]
        ddp_checksum_every: usize,
    },
    TrainMemoryLm {
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        dataset_manifest: Option<PathBuf>,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        #[arg(long, default_value_t = 1)]
        grad_accumulation_steps: usize,
        #[arg(long, default_value_t = 64)]
        block_size: usize,
        #[arg(long, default_value_t = 32)]
        n_layers: usize,
        #[arg(long, default_value_t = 64)]
        d_model: usize,
        #[arg(long, default_value_t = 4)]
        n_heads: usize,
        #[arg(long, default_value_t = 256)]
        ff_hidden: usize,
        #[arg(long, value_delimiter = ',', default_value = "8,16,24")]
        memory_layer_indices: Vec<usize>,
        #[arg(long, default_value_t = false)]
        disable_memory_layers: bool,
        #[arg(long, default_value_t = 1024)]
        memory_slots: usize,
        #[arg(long, default_value_t = 32)]
        memory_key_dim: usize,
        #[arg(long, default_value_t = 64)]
        memory_value_dim: usize,
        #[arg(long, default_value_t = 4)]
        memory_top_k: usize,
        #[arg(long, default_value_t = 1)]
        memory_heads: usize,
        #[arg(long, value_enum, default_value_t = MemoryLookupKind::Exact)]
        memory_lookup: MemoryLookupKind,
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        shared_memory: bool,
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        memory_plus: bool,
        #[arg(long, value_enum, default_value_t = MemoryUpdatePolicy::Full)]
        memory_update_policy: MemoryUpdatePolicy,
        #[arg(long, value_enum, default_value_t = SmftMode::Disabled)]
        smft_mode: SmftMode,
        #[arg(long)]
        smft_row_mask: Option<PathBuf>,
        #[arg(long)]
        smft_background_counts: Option<PathBuf>,
        #[arg(long)]
        smft_access_counts_out: Option<PathBuf>,
        #[arg(long)]
        smft_mask_out: Option<PathBuf>,
        #[arg(long, default_value_t = 0.1)]
        smft_trainable_fraction: f64,
        #[arg(long, default_value_t = 1)]
        smft_min_rows: usize,
        #[arg(long, default_value_t = 0)]
        smft_refresh_every: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f64,
        #[arg(long, default_value_t = 1.0)]
        clip_norm: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long)]
        devices: Option<String>,
        #[arg(long, value_enum)]
        distributed: Option<DistributedMode>,
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        #[arg(long, default_value_t = false)]
        resume: bool,
        #[arg(long, default_value_t = 25)]
        log_every: usize,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 120)]
        ddp_init_timeout_secs: u64,
        #[arg(long, default_value_t = 1)]
        ddp_checksum_every: usize,
    },
    #[command(name = "train-lm-rank", hide = true)]
    TrainLmRank {
        #[arg(long)]
        config: PathBuf,
    },
    #[command(name = "train-memory-lm-rank", hide = true)]
    TrainMemoryLmRank {
        #[arg(long)]
        config: PathBuf,
    },
    #[command(name = "nccl-probe-rank", hide = true)]
    NcclProbeRank {
        #[arg(long)]
        config: PathBuf,
    },
    #[command(name = "nccl-unique-id", hide = true)]
    NcclUniqueId {
        #[arg(long)]
        hold_secs: Option<u64>,
    },
    #[command(name = "launcher-test", hide = true)]
    LauncherTest {
        #[arg(long, default_value_t = 2)]
        ranks: usize,
        #[arg(long)]
        fail_rank: Option<usize>,
        #[arg(long)]
        hang_rank: Option<usize>,
        #[arg(long, default_value_t = 3)]
        timeout_secs: u64,
        #[arg(long, default_value_t = 5)]
        rank_start_timeout_secs: u64,
        #[arg(long, default_value_t = 1)]
        kill_grace_secs: u64,
        #[arg(long)]
        report: PathBuf,
    },
    #[command(name = "launcher-test-rank", hide = true)]
    LauncherTestRank {
        #[arg(long)]
        config: PathBuf,
    },
    Generate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        #[arg(long, default_value_t = 80)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        top_p: Option<f64>,
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f64,
        #[arg(long, default_value_t = 0.0)]
        frequency_penalty: f64,
        #[arg(long, default_value_t = 0.0)]
        presence_penalty: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    GenerateMemoryLm {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        #[arg(long, default_value_t = 80)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        top_p: Option<f64>,
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f64,
        #[arg(long, default_value_t = 0.0)]
        frequency_penalty: f64,
        #[arg(long, default_value_t = 0.0)]
        presence_penalty: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    EvalLm {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        dataset_manifest: Option<PathBuf>,
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        #[arg(long, value_enum, default_value_t = EvalSplit::Valid)]
        split: EvalSplit,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        #[arg(long)]
        max_batches: Option<usize>,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    EvalMemoryLm {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        dataset_manifest: Option<PathBuf>,
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        #[arg(long, value_enum, default_value_t = EvalSplit::Valid)]
        split: EvalSplit,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        #[arg(long)]
        max_batches: Option<usize>,
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EvalSplit {
    Train,
    Valid,
}

impl EvalSplit {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Valid => "valid",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
enum Precision {
    F32,
    Bf16,
    AmpBf16,
}

impl Precision {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::AmpBf16 => "amp-bf16",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum)]
enum DistributedMode {
    Nccl,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
enum NcclProbeKind {
    Spawn,
    CudaContext,
    NcclInit,
    AllReduce,
}

impl NcclProbeKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Spawn => "spawn",
            Self::CudaContext => "cuda-context",
            Self::NcclInit => "nccl-init",
            Self::AllReduce => "all-reduce",
        }
    }

    fn requires_cuda_context(self) -> bool {
        matches!(self, Self::CudaContext | Self::NcclInit | Self::AllReduce)
    }

    fn requires_nccl(self) -> bool {
        matches!(self, Self::NcclInit | Self::AllReduce)
    }

    fn requires_all_reduce(self) -> bool {
        matches!(self, Self::AllReduce)
    }
}

#[derive(Subcommand)]
enum DataCommands {
    TinystoriesValid {
        #[arg(long)]
        out: PathBuf,
    },
    Prepare {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, value_enum, default_value_t = DataPrepareFormat::Json)]
        format: DataPrepareFormat,
        #[arg(long, default_value_t = 0.1)]
        valid_fraction: f64,
        #[arg(long)]
        max_bytes: Option<usize>,
        #[arg(long)]
        shard_tokens: Option<usize>,
    },
    CorpusBlend {
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        qb_root: Option<PathBuf>,
        #[arg(long, default_value = "qb-native-pretraining-v1")]
        blend_id: String,
        #[arg(long, default_value_t = 32768)]
        tokenizer_vocab_size: usize,
    },
    MaterializeBlend {
        #[arg(long)]
        corpus_blend: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 20_000_000_000)]
        target_tokens: u64,
        #[arg(long, value_enum, default_value_t = MaterializeBlendMode::DryRun)]
        mode: MaterializeBlendMode,
        #[arg(long, default_value_t = 1107)]
        seed: u64,
        #[arg(long, default_value_t = 0.001)]
        valid_fraction: f64,
        #[arg(long, default_value_t = 1_000_000)]
        shard_tokens: usize,
        #[arg(long, default_value_t = 128 * 1024 * 1024)]
        text_shard_bytes: usize,
        #[arg(long)]
        max_source_bytes: Option<usize>,
        #[arg(long)]
        max_docs_per_source: Option<usize>,
        #[arg(long, default_value_t = 1)]
        min_doc_bytes: usize,
        #[arg(long)]
        max_doc_bytes: Option<usize>,
        #[arg(long, default_value_t = 2.0)]
        max_tokens_per_byte: f64,
        #[arg(long, value_enum, default_value_t = MaterializerCandidateTextMode::Retain)]
        candidate_text_mode: MaterializerCandidateTextMode,
        #[arg(long, default_value_t = 0.0)]
        candidate_retention_token_multiplier: f64,
        #[arg(long, default_value_t = 1024)]
        candidate_retention_min_docs: usize,
        #[arg(long, default_value_t = 10000)]
        candidate_prune_every: usize,
        #[arg(long, default_value_t = 0)]
        progress_every_records: usize,
        #[arg(long, default_value_t = 0)]
        progress_every_bytes: u64,
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        resume_checkpoint: bool,
        #[arg(long, default_value_t = 0)]
        checkpoint_every_records: usize,
        #[arg(long, default_value_t = 0)]
        checkpoint_every_bytes: u64,
        #[arg(long, hide = true)]
        checkpoint_stop_after_records: Option<usize>,
        #[arg(long)]
        allow_license_status: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum DataPrepareFormat {
    Json,
    BinaryShard,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum MaterializeBlendMode {
    DryRun,
    Sample,
    Full,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum MaterializerCandidateTextMode {
    Retain,
    Rescan,
}

fn candidate_text_mode_label(mode: MaterializerCandidateTextMode) -> &'static str {
    match mode {
        MaterializerCandidateTextMode::Retain => "retain",
        MaterializerCandidateTextMode::Rescan => "rescan",
    }
}

#[derive(Subcommand)]
enum TokenizerCommands {
    Train {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1024)]
        vocab_size: usize,
        #[arg(long)]
        max_bytes: Option<usize>,
    },
    TrainCorpus {
        #[arg(long)]
        corpus_blend: PathBuf,
        #[arg(long)]
        reserved_tokens: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        work_dir: PathBuf,
        #[arg(long, default_value_t = PRODUCTION_TOKENIZER_VOCAB_SIZE)]
        vocab_size: usize,
        #[arg(long, default_value_t = DEFAULT_TOKENIZER_SAMPLE_BYTES)]
        sample_bytes: u64,
        #[arg(long, default_value_t = 1107)]
        seed: u64,
        #[arg(long)]
        memory_limit_bytes: Option<u64>,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long)]
        allow_license_status: Vec<String>,
        #[arg(long, default_value_t = false)]
        require_exact_vocab: bool,
    },
    Validate {
        tokenizer: PathBuf,
    },
    Fertility {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        corpus_blend: PathBuf,
        #[arg(long)]
        report: PathBuf,
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        sample_bytes: u64,
        #[arg(long, default_value_t = 1107)]
        seed: u64,
        #[arg(long)]
        allow_license_status: Vec<String>,
    },
    BenchEncode {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 64 * 1024 * 1024)]
        max_bytes: usize,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long, default_value_t = false)]
        add_bos: bool,
        #[arg(long, default_value_t = false)]
        add_eos: bool,
    },
}

#[derive(Subcommand)]
enum GpuCommands {
    Info,
    Topology {
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Smoke {
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value_t = 1024)]
        len: usize,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    TensorCoreProbe {
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    TensorCoreMicrobench {
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, value_enum, default_value_t = TensorCoreMicrobenchSection::All)]
        section: TensorCoreMicrobenchSection,
        #[arg(long, default_value_t = 32)]
        iterations: usize,
        #[arg(long, default_value_t = 4)]
        warmup: usize,
        #[arg(long, default_value_t = 32)]
        m: usize,
        #[arg(long, default_value_t = 64)]
        k: usize,
        #[arg(long, default_value_t = 32)]
        n: usize,
        #[arg(long, default_value_t = 16)]
        attention_time: usize,
        #[arg(long, default_value_t = 16)]
        attention_head_dim: usize,
        #[arg(long, default_value_t = 1)]
        attention_heads: usize,
        #[arg(long, default_value_t = 1)]
        attention_batch: usize,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    NcclProbe {
        #[arg(long)]
        devices: String,
        #[arg(long, default_value_t = 1024)]
        len: usize,
        #[arg(long, value_enum, default_value_t = NcclProbeKind::AllReduce)]
        probe_kind: NcclProbeKind,
        #[arg(long, default_value_t = 120)]
        timeout_secs: u64,
        #[arg(long, default_value_t = 10)]
        rank_start_timeout_secs: u64,
        #[arg(long, default_value_t = 5)]
        kill_grace_secs: u64,
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum PadawanCommands {
    Validate {
        #[arg(long, required = true)]
        episodes: Vec<PathBuf>,
        #[arg(long)]
        artifact_root: Option<PathBuf>,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Verify {
        #[arg(long, required = true)]
        episodes: Vec<PathBuf>,
        #[arg(long)]
        artifact_root: Option<PathBuf>,
        #[arg(long, value_enum)]
        family: Vec<PadawanVerifierFamily>,
        #[arg(long)]
        allow_path_prefix: Vec<String>,
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ReadinessCommands {
    ValidateLearningSanity {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum PadawanVerifierFamily {
    All,
    CodePatch,
    JsonToolCall,
    EvidenceCitation,
    MemorySmft,
}

impl PadawanVerifierFamily {
    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::CodePatch => "code_patch",
            Self::JsonToolCall => "json_tool_call",
            Self::EvidenceCitation => "evidence_citation",
            Self::MemorySmft => "memory_smft",
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum TensorCoreMicrobenchSection {
    All,
    Gemm,
    Attention,
}

impl TensorCoreMicrobenchSection {
    fn label(self) -> &'static str {
        match self {
            TensorCoreMicrobenchSection::All => "all",
            TensorCoreMicrobenchSection::Gemm => "gemm",
            TensorCoreMicrobenchSection::Attention => "attention",
        }
    }

    fn includes_gemm(self) -> bool {
        matches!(
            self,
            TensorCoreMicrobenchSection::All | TensorCoreMicrobenchSection::Gemm
        )
    }

    fn includes_attention(self) -> bool {
        matches!(
            self,
            TensorCoreMicrobenchSection::All | TensorCoreMicrobenchSection::Attention
        )
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Data { command } => match command {
            DataCommands::TinystoriesValid { out } => {
                download_tinystories_valid(&out)?;
                println!(
                    "downloaded TinyStories validation split to {}",
                    out.display()
                );
            }
            DataCommands::Prepare {
                input,
                tokenizer,
                out_dir,
                format,
                valid_fraction,
                max_bytes,
                shard_tokens,
            } => {
                let tokenizer_model = BpeTokenizer::load(&tokenizer)?;
                let options = PreparedDataOptions {
                    valid_fraction,
                    max_bytes,
                    shard_tokens,
                };
                let prepared = match format {
                    DataPrepareFormat::Json => {
                        if shard_tokens.is_some() {
                            return Err(TensorError::InvalidOperation(
                                "--shard-tokens is supported only with --format binary-shard"
                                    .to_string(),
                            ));
                        }
                        if input.len() != 1 {
                            return Err(TensorError::InvalidOperation(format!(
                                "data prepare --format json accepts exactly one --input, got {}",
                                input.len()
                            )));
                        }
                        prepare_lm_data(&input[0], &tokenizer_model, &tokenizer, &out_dir, options)?
                    }
                    DataPrepareFormat::BinaryShard => prepare_lm_data_binary_shards(
                        &input,
                        &tokenizer_model,
                        &tokenizer,
                        &out_dir,
                        options,
                    )?,
                };
                println!(
                    "prepared data version={} storage={} train_tokens={} valid_tokens={} manifest={}",
                    prepared.manifest.version,
                    prepared.manifest.storage,
                    prepared.manifest.train_tokens,
                    prepared.manifest.valid_tokens,
                    prepared.manifest_path.display()
                );
            }
            DataCommands::CorpusBlend {
                out,
                qb_root,
                blend_id,
                tokenizer_vocab_size,
            } => {
                let manifest = build_qb_native_corpus_blend_manifest(
                    blend_id,
                    tokenizer_vocab_size,
                    qb_root.as_deref(),
                )?;
                write_corpus_blend_manifest(&manifest, &out)?;
                println!(
                    "wrote corpus blend version={} sources={} local_sources={} local_records={} out={}",
                    manifest.version,
                    manifest.sources.len(),
                    manifest.local_source_count,
                    manifest.local_record_count,
                    out.display()
                );
            }
            DataCommands::MaterializeBlend {
                corpus_blend,
                tokenizer,
                out_dir,
                target_tokens,
                mode,
                seed,
                valid_fraction,
                shard_tokens,
                text_shard_bytes,
                max_source_bytes,
                max_docs_per_source,
                min_doc_bytes,
                max_doc_bytes,
                max_tokens_per_byte,
                candidate_text_mode,
                candidate_retention_token_multiplier,
                candidate_retention_min_docs,
                candidate_prune_every,
                progress_every_records,
                progress_every_bytes,
                checkpoint_dir,
                resume_checkpoint,
                checkpoint_every_records,
                checkpoint_every_bytes,
                checkpoint_stop_after_records,
                allow_license_status,
            } => {
                let outcome = materialize_blend(MaterializeBlendConfig {
                    corpus_blend,
                    tokenizer,
                    out_dir,
                    target_tokens,
                    mode,
                    seed,
                    valid_fraction,
                    shard_tokens,
                    text_shard_bytes,
                    max_source_bytes,
                    max_docs_per_source,
                    min_doc_bytes,
                    max_doc_bytes,
                    max_tokens_per_byte,
                    candidate_text_mode,
                    candidate_retention_token_multiplier,
                    candidate_retention_min_docs,
                    candidate_prune_every,
                    progress_every_records,
                    progress_every_bytes,
                    checkpoint_dir,
                    resume_checkpoint,
                    checkpoint_every_records,
                    checkpoint_every_bytes,
                    checkpoint_stop_after_records,
                    allow_license_status,
                })?;
                println!(
                    "materialized blend mode={:?} selected_docs={} selected_tokens={} elapsed_ms={} tokens_per_second={:.2} prepared_manifest={}",
                    outcome.mode,
                    outcome.selected_docs,
                    outcome.selected_tokens,
                    outcome.total_elapsed_ms,
                    outcome.selected_tokens_per_second,
                    outcome
                        .prepared_manifest_path
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "none".to_string())
                );
            }
        },
        Commands::Tokenizer { command } => match command {
            TokenizerCommands::Train {
                input,
                out,
                vocab_size,
                max_bytes,
            } => {
                let text = maybe_truncate(read_text(&input)?, max_bytes);
                let tokenizer = BpeTokenizer::train(&text, vocab_size)?;
                tokenizer.save(&out)?;
                println!(
                    "trained tokenizer vocab={} merges={} -> {}",
                    tokenizer.vocab_size(),
                    tokenizer.merges().len(),
                    out.display()
                );
            }
            TokenizerCommands::TrainCorpus {
                corpus_blend,
                reserved_tokens,
                out,
                work_dir,
                vocab_size,
                sample_bytes,
                seed,
                memory_limit_bytes,
                report,
                allow_license_status,
                require_exact_vocab,
            } => {
                let outcome = train_tokenizer_from_corpus_blend(TokenizerCorpusTrainConfig {
                    corpus_blend,
                    reserved_tokens,
                    out,
                    work_dir,
                    vocab_size,
                    sample_bytes,
                    seed,
                    memory_limit_bytes,
                    report,
                    allow_license_status,
                    require_exact_vocab,
                })?;
                println!(
                    "trained corpus tokenizer version={} vocab={} reserved={} hash={} -> {}",
                    outcome.version,
                    outcome.vocab_size,
                    outcome.reserved_tokens,
                    outcome.tokenizer_hash,
                    outcome.tokenizer_path.display()
                );
            }
            TokenizerCommands::Validate { tokenizer } => {
                let tokenizer_model = BpeTokenizer::load(&tokenizer)?;
                tokenizer_model.validate()?;
                println!(
                    "validated tokenizer id={} version={} vocab={} hash={}",
                    tokenizer_model.tokenizer_id(),
                    tokenizer_model.metadata().version,
                    tokenizer_model.vocab_size(),
                    tokenizer_model.fingerprint()?
                );
            }
            TokenizerCommands::Fertility {
                tokenizer,
                corpus_blend,
                report,
                sample_bytes,
                seed,
                allow_license_status,
            } => {
                let tokenizer_model = BpeTokenizer::load(&tokenizer)?;
                let manifest: CorpusBlendManifest = read_json_typed(&corpus_blend)?;
                let samples = materialize_tokenizer_samples(
                    &manifest,
                    &allow_license_statuses(allow_license_status),
                    sample_bytes,
                    seed,
                    None,
                    corpus_blend.parent(),
                )?;
                let fertility_report = tokenizer_fertility_report(
                    &tokenizer_model,
                    Some(&tokenizer),
                    &manifest,
                    samples,
                )?;
                write_json_file(&report, fertility_report)?;
                println!(
                    "wrote tokenizer fertility report vocab={} report={}",
                    tokenizer_model.vocab_size(),
                    report.display()
                );
            }
            TokenizerCommands::BenchEncode {
                tokenizer,
                input,
                report,
                max_bytes,
                iterations,
                add_bos,
                add_eos,
            } => {
                let bench_report = tokenizer_encode_bench_report(TokenizerEncodeBenchConfig {
                    tokenizer_path: tokenizer.clone(),
                    input,
                    max_bytes,
                    iterations,
                    add_bos,
                    add_eos,
                })?;
                let tokenizer_model = BpeTokenizer::load(&tokenizer)?;
                if let Some(report_path) = report {
                    write_json_file(&report_path, bench_report.clone())?;
                    println!(
                        "tokenizer encode bench vocab={} docs={} bytes={} iterations={} tokens_per_second={:.2} bytes_per_second={:.2} report={}",
                        tokenizer_model.vocab_size(),
                        bench_report["sample"]["documents"],
                        bench_report["sample"]["bytes"],
                        iterations,
                        bench_report["throughput"]["tokens_per_second"].as_f64().unwrap_or(0.0),
                        bench_report["throughput"]["bytes_per_second"].as_f64().unwrap_or(0.0),
                        report_path.display()
                    );
                } else {
                    println!(
                        "tokenizer encode bench vocab={} docs={} bytes={} iterations={} tokens_per_second={:.2} bytes_per_second={:.2}",
                        tokenizer_model.vocab_size(),
                        bench_report["sample"]["documents"],
                        bench_report["sample"]["bytes"],
                        iterations,
                        bench_report["throughput"]["tokens_per_second"].as_f64().unwrap_or(0.0),
                        bench_report["throughput"]["bytes_per_second"].as_f64().unwrap_or(0.0)
                    );
                }
            }
        },
        Commands::Gpu { command } => match command {
            GpuCommands::Info => match cuda::system_info() {
                Ok(info) => {
                    println!("cuda_driver_loaded={}", info.driver_loaded);
                    println!("cuda_device_count={}", info.device_count);
                    for device in info.devices {
                        println!(
                            "device={} name={} pci_bus_id={} compute_capability={}.{}",
                            device.ordinal,
                            device.name,
                            device.pci_bus_id,
                            device.compute_capability_major,
                            device.compute_capability_minor
                        );
                    }
                }
                Err(err) => {
                    println!("cuda_available=false");
                    println!("reason={err}");
                }
            },
            GpuCommands::Topology { report } => {
                let info = cuda::system_info().map_err(|err| {
                    TensorError::Device(format!("CUDA topology info failed: {err}"))
                })?;
                let peers = cuda::peer_access_matrix().map_err(|err| {
                    TensorError::Device(format!("CUDA peer access matrix failed: {err}"))
                })?;
                println!("cuda_device_count={}", info.device_count);
                for device in &info.devices {
                    println!(
                        "device={} name={} pci_bus_id={} compute_capability={}.{}",
                        device.ordinal,
                        device.name,
                        device.pci_bus_id,
                        device.compute_capability_major,
                        device.compute_capability_minor
                    );
                }
                for peer in &peers {
                    println!(
                        "peer_access from={} to={} can_access={}",
                        peer.from_ordinal, peer.to_ordinal, peer.can_access
                    );
                }
                if let Some(report_path) = report {
                    write_cuda_topology_report(&report_path, &info, &peers)?;
                }
            }
            GpuCommands::Smoke {
                device,
                len,
                report,
            } => {
                let report_data = cuda::smoke_f32(device, len)
                    .map_err(|err| TensorError::Device(format!("CUDA smoke failed: {err}")))?;
                println!(
                    "cuda_smoke device={} name=\"{}\" len={} add_max_abs_error={} relu_max_abs_error={}",
                    report_data.device.ordinal,
                    report_data.device.name,
                    report_data.len,
                    report_data.add_max_abs_error,
                    report_data.relu_max_abs_error
                );
                if let Some(report_path) = report {
                    write_cuda_smoke_report(&report_path, &report_data)?;
                }
            }
            GpuCommands::TensorCoreProbe { device, report } => {
                let probe = cuda::bf16_mma_probe(device).map_err(|err| {
                    TensorError::Device(format!("CUDA Tensor Core probe failed: {err}"))
                })?;
                let counters = cuda::tensor_core_counters();
                println!(
                    "cuda_tensor_core_probe device={} expected_dot={} max_abs_error={} samples={:?}",
                    probe.device_ordinal,
                    probe.expected_dot,
                    probe.max_abs_error,
                    probe.samples.iter().take(8).collect::<Vec<_>>()
                );
                println!(
                    "tensor_core_counters bf16_mma_probe_calls={} bf16_tensor_core_matmul_calls={} bf16_tensor_core_matmul_forward_calls={} bf16_tensor_core_matmul_backward_calls={} bf16_scalar_matmul_fallback_calls={} bf16_tensor_core_attention_forward_calls={} bf16_tensor_core_attention_qk_matmul_calls={} bf16_tensor_core_attention_av_matmul_calls={} bf16_tensor_core_attention_backward_calls={} bf16_tensor_core_attention_score_grad_matmul_calls={} bf16_tensor_core_attention_dq_matmul_calls={} bf16_tensor_core_attention_dk_matmul_calls={} bf16_tensor_core_attention_dv_matmul_calls={}",
                    counters.bf16_mma_probe_calls,
                    counters.bf16_tensor_core_matmul_calls,
                    counters.bf16_tensor_core_matmul_forward_calls,
                    counters.bf16_tensor_core_matmul_backward_calls,
                    counters.bf16_scalar_matmul_fallback_calls,
                    counters.bf16_tensor_core_attention_forward_calls,
                    counters.bf16_tensor_core_attention_qk_matmul_calls,
                    counters.bf16_tensor_core_attention_av_matmul_calls,
                    counters.bf16_tensor_core_attention_backward_calls,
                    counters.bf16_tensor_core_attention_score_grad_matmul_calls,
                    counters.bf16_tensor_core_attention_dq_matmul_calls,
                    counters.bf16_tensor_core_attention_dk_matmul_calls,
                    counters.bf16_tensor_core_attention_dv_matmul_calls
                );
                if let Some(report_path) = report {
                    write_tensor_core_probe_report(&report_path, &probe, counters)?;
                }
            }
            GpuCommands::TensorCoreMicrobench {
                device,
                section,
                iterations,
                warmup,
                m,
                k,
                n,
                attention_time,
                attention_head_dim,
                attention_heads,
                attention_batch,
                report,
            } => {
                let bench = tensor_core_microbench_report(TensorCoreMicrobenchConfig {
                    device,
                    section,
                    iterations,
                    warmup,
                    m,
                    k,
                    n,
                    attention_time,
                    attention_head_dim,
                    attention_heads,
                    attention_batch,
                })?;
                let bench_passed = json_bool(&bench, "passed");
                let tensor_core_flash_status = bench["attention"]["tensor_core_flash_forward"]
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("missing");
                println!(
                    "tensor_core_microbench status={} section={} device={} gemm_tier={} gemm_ms={:.3} gemm_tflops={:.6} gemm_max_abs_error={:.6} attention_tier={} attention_current_ms={:.3} attention_scalar_flash_ms={:.3} attention_tc_flash_status={} attention_tc_flash_ms={:.3} attention_scalar_flash_speedup={:.3} attention_current_max_abs_error={:.6} attention_scalar_flash_max_abs_error={:.6} attention_tc_flash_max_abs_error={:.6}",
                    bench["status"].as_str().unwrap_or("unknown"),
                    bench["section"].as_str().unwrap_or("unknown"),
                    device,
                    bench["gemm"]["tier"].as_str().unwrap_or("unknown"),
                    bench["gemm"]["elapsed_ms"].as_f64().unwrap_or(0.0),
                    bench["gemm"]["tflops_per_second"].as_f64().unwrap_or(0.0),
                    bench["gemm"]["max_abs_error"].as_f64().unwrap_or(0.0),
                    bench["attention"]["tier"].as_str().unwrap_or("unknown"),
                    bench["attention"]["current_materialized"]["elapsed_ms"]
                        .as_f64()
                        .unwrap_or(0.0),
                    bench["attention"]["flash_forward"]["elapsed_ms"]
                        .as_f64()
                        .unwrap_or(0.0),
                    tensor_core_flash_status,
                    bench["attention"]["tensor_core_flash_forward"]["elapsed_ms"]
                        .as_f64()
                        .unwrap_or(0.0),
                    bench["attention"]["speedup_ratio_flash_vs_current"]
                        .as_f64()
                        .unwrap_or(0.0),
                    bench["attention"]["current_materialized"]["max_abs_error"]
                        .as_f64()
                        .unwrap_or(0.0),
                    bench["attention"]["flash_forward"]["max_abs_error"]
                        .as_f64()
                        .unwrap_or(0.0),
                    bench["attention"]["tensor_core_flash_forward"]["max_abs_error"]
                        .as_f64()
                        .unwrap_or(0.0),
                );
                if let Some(report_path) = report {
                    write_json_file(&report_path, bench.clone())?;
                }
                if !bench_passed {
                    return Err(TensorError::InvalidOperation(format!(
                        "gpu tensor-core-microbench failed: status={} attention_tensor_core_flash_status={}",
                        bench["status"].as_str().unwrap_or("unknown"),
                        tensor_core_flash_status
                    )));
                }
            }
            GpuCommands::NcclProbe {
                devices,
                len,
                probe_kind,
                timeout_secs,
                rank_start_timeout_secs,
                kill_grace_secs,
                report,
            } => {
                run_nccl_probe(NcclProbeConfig {
                    devices_value: devices,
                    len,
                    probe_kind,
                    timeout: Duration::from_secs(timeout_secs),
                    rank_start_timeout: Duration::from_secs(rank_start_timeout_secs),
                    kill_grace: Duration::from_secs(kill_grace_secs),
                    report,
                })?;
            }
        },
        Commands::Readiness { command } => match command {
            ReadinessCommands::ValidateLearningSanity { manifest, report } => {
                let summary = validate_learning_sanity_manifest(&manifest)?;
                if let Some(report_path) = report {
                    write_json_file(&report_path, summary.clone())?;
                }
                println!(
                    "learning_sanity status={} stages={} passed={} failed={} min_loss_reduction={:.6}",
                    summary["status"].as_str().unwrap_or("unknown"),
                    summary["stages"].as_array().map(Vec::len).unwrap_or(0),
                    summary["passed"].as_u64().unwrap_or(0),
                    summary["failed"].as_u64().unwrap_or(0),
                    summary["min_loss_reduction_observed"].as_f64().unwrap_or(0.0),
                );
                if summary["status"].as_str() != Some("passed") {
                    return Err(TensorError::InvalidOperation(
                        "learning sanity validation failed".to_string(),
                    ));
                }
            }
        },
        Commands::Padawan { command } => match command {
            PadawanCommands::Validate {
                episodes,
                artifact_root,
                report,
            } => {
                let summary = padawan_validate_report(&episodes, artifact_root.as_deref())?;
                if let Some(report_path) = report {
                    write_json_file(&report_path, summary.clone())?;
                }
                println!(
                    "validated padawan episodes={} sft_eligible={} smft_eligible={} max_guidance_level={}",
                    summary["episodes"].as_u64().unwrap_or(0),
                    summary["sft_eligible"].as_u64().unwrap_or(0),
                    summary["smft_eligible"].as_u64().unwrap_or(0),
                    summary["max_guidance_level"].as_u64().unwrap_or(0),
                );
            }
            PadawanCommands::Verify {
                episodes,
                artifact_root,
                family,
                allow_path_prefix,
                report,
            } => {
                let summary = padawan_verify_report(
                    &episodes,
                    artifact_root.as_deref(),
                    &family,
                    &allow_path_prefix,
                )?;
                if let Some(report_path) = report {
                    write_json_file(&report_path, summary.clone())?;
                }
                println!(
                    "padawan verifier harness status={} episodes={} families={} passed={} failed={} skipped={}",
                    summary["status"].as_str().unwrap_or("unknown"),
                    summary["episodes"].as_array().map(Vec::len).unwrap_or(0),
                    summary["families_requested"].as_array().map(Vec::len).unwrap_or(0),
                    summary["family_counts"]["passed"].as_u64().unwrap_or(0),
                    summary["family_counts"]["failed"].as_u64().unwrap_or(0),
                    summary["family_counts"]["skipped"].as_u64().unwrap_or(0),
                );
                if summary["status"].as_str() != Some("passed") {
                    return Err(TensorError::InvalidOperation(
                        "padawan verifier harness failed".to_string(),
                    ));
                }
            }
        },
        Commands::TrainLm {
            data,
            tokenizer,
            dataset_manifest,
            checkpoint,
            steps,
            batch_size,
            grad_accumulation_steps,
            block_size,
            d_model,
            n_heads,
            ff_hidden,
            lr,
            weight_decay,
            clip_norm,
            seed,
            device,
            devices,
            distributed,
            precision,
            resume,
            log_every,
            report,
            ddp_init_timeout_secs,
            ddp_checksum_every,
        } => {
            validate_grad_accumulation_steps(grad_accumulation_steps)?;
            if devices.is_some() || distributed.is_some() {
                run_distributed_train_lm(DistributedTrainLmConfig {
                    data,
                    tokenizer,
                    dataset_manifest,
                    checkpoint,
                    steps,
                    batch_size,
                    grad_accumulation_steps,
                    block_size,
                    d_model,
                    n_heads,
                    ff_hidden,
                    lr,
                    weight_decay,
                    clip_norm,
                    seed,
                    devices,
                    distributed,
                    precision,
                    resume,
                    log_every,
                    report,
                    ddp_init_timeout: Duration::from_secs(ddp_init_timeout_secs),
                    ddp_checksum_every,
                })?;
                return Ok(());
            }
            let train_device = parse_device(&device)?;
            ensure_precision_supported_for_device(precision, train_device)?;
            let prepared =
                load_prepared_from_cli_or_checkpoint(&dataset_manifest, &checkpoint, resume)?;
            let (model, mut optimizer, tokenizer, dataset_state, start_step) = if resume
                && checkpoint.exists()
            {
                let loaded = load_lm_checkpoint_on_device(&checkpoint, train_device)?;
                if let Some(prepared) = &prepared {
                    validate_checkpoint_tokenizer_matches_manifest(&loaded.tokenizer, prepared)?;
                }
                let dataset_state = loaded.metadata.dataset_state();
                let start_step = loaded.metadata.step;
                (
                    loaded.model,
                    loaded.optimizer,
                    loaded.tokenizer,
                    dataset_state,
                    start_step,
                )
            } else {
                let tokenizer = if let Some(prepared) = &prepared {
                    prepared.load_tokenizer()?
                } else {
                    let tokenizer_path = require_path(tokenizer.as_ref(), "--tokenizer")?;
                    BpeTokenizer::load(tokenizer_path)?
                };
                let config = TinyTransformerConfig {
                    vocab_size: tokenizer.vocab_size(),
                    block_size,
                    d_model,
                    n_heads,
                    ff_hidden,
                };
                let mut rng = HeirloomRng::new(seed);
                let model = TinyTransformerLm::new(config, &mut rng)?.to_device(train_device)?;
                let optimizer = AdamW::new(model.parameters(), lr)?
                    .with_weight_decay(weight_decay)?
                    .with_clip_norm(Some(clip_norm))?;
                (
                    model,
                    optimizer,
                    tokenizer,
                    TokenDatasetState::from_seed(seed),
                    0,
                )
            };

            let dense_flops_per_token = dense_training_flops_per_token_estimate(
                tiny_dense_parameter_estimate(&model.config),
            );
            let mut dataset = build_single_train_dataset(
                prepared.as_ref(),
                data.as_ref(),
                &tokenizer,
                model.config.block_size,
                dataset_state,
            )?;
            let mut initial_loss = None;
            let mut final_loss = 0.0;
            let mut timings = TrainingTimings::default();
            let train_start = Instant::now();
            cuda::reset_tensor_core_counters();
            cuda::reset_cuda_runtime_counters();
            reset_amp_bf16_tensor_core_coverage();

            let _amp_guard = (precision == Precision::AmpBf16).then(amp::enter_amp_bf16_training);
            for local_step in 0..steps {
                optimizer.zero_grad();
                let mut step_loss_sum = 0.0;
                for _micro_step in 0..grad_accumulation_steps {
                    let dataloader_start = Instant::now();
                    let (input, target) = dataset.next_batch(batch_size)?;
                    timings.add_dataloader(dataloader_start.elapsed());
                    let h2d_start = Instant::now();
                    let h2d_timer = start_cuda_compute_timer(train_device)?;
                    let input = input.to_device(train_device)?;
                    let target = target.to_device(train_device)?;
                    let h2d_cuda_ms = stop_cuda_timer(h2d_timer)?;
                    timings.add_host_to_device(h2d_start.elapsed());
                    timings.add_host_to_device_cuda_ms(h2d_cuda_ms);
                    let forward_backward_start = Instant::now();
                    let forward_backward_timer = start_cuda_compute_timer(train_device)?;
                    let loss = match precision {
                        Precision::F32 => model.loss(&input, &target)?,
                        Precision::Bf16 => model.loss_bf16_activations(&input, &target)?,
                        Precision::AmpBf16 => model.loss_amp_bf16(&input, &target)?,
                    };
                    let loss_value = training_loss_scalar(&loss, precision)?;
                    step_loss_sum += loss_value;
                    loss_for_grad_accumulation(&loss, grad_accumulation_steps)?.backward()?;
                    let forward_backward_cuda_ms = stop_cuda_timer(forward_backward_timer)?;
                    timings.add_forward_backward(forward_backward_start.elapsed());
                    timings.add_forward_backward_cuda_ms(forward_backward_cuda_ms);
                }
                let step_loss = step_loss_sum / grad_accumulation_steps as f64;
                initial_loss.get_or_insert(step_loss);
                final_loss = step_loss;
                let optimizer_start = Instant::now();
                let optimizer_timer = start_cuda_compute_timer(train_device)?;
                optimizer.step_mut()?;
                let optimizer_cuda_ms = stop_cuda_timer(optimizer_timer)?;
                timings.add_optimizer(optimizer_start.elapsed());
                timings.add_optimizer_cuda_ms(optimizer_cuda_ms);

                let global_step = start_step + local_step + 1;
                if log_every > 0 && (local_step == 0 || global_step % log_every == 0) {
                    println!("step={global_step} loss={step_loss:.6}");
                }
            }
            let train_elapsed = train_start.elapsed();
            let tensor_core_counters = cuda::tensor_core_counters();
            let cuda_runtime_counters = cuda::cuda_runtime_counters();
            let tensor_core_coverage = amp_bf16_tensor_core_coverage();
            let loader_report = dataset.loader_report();
            let tokens_seen = accumulated_training_tokens_seen(
                steps,
                batch_size,
                model.config.block_size,
                grad_accumulation_steps,
            )?;
            let performance_report = training_performance_report(TrainingPerformanceInput {
                tokens_seen,
                train_elapsed,
                timings,
                dense_flops_per_token,
                device_count: device_count_for_training(train_device),
                micro_batch_size: batch_size,
                grad_accumulation_steps,
                data_parallel_world_size: 1,
            });

            checkpoint_with_amp_staging_allowed(precision, || {
                save_lm_checkpoint_with_dataset_state(
                    &checkpoint,
                    &model,
                    &optimizer,
                    &tokenizer,
                    dataset.state(),
                    prepared
                        .as_ref()
                        .map(|prepared| prepared.manifest_path.display().to_string()),
                )
            })?;
            if let Some(report_path) = report {
                let checkpoint_tokenizer_path = checkpoint.join("tokenizer.json");
                write_train_report(TrainReportInput {
                    path: &report_path,
                    start_step,
                    final_step: optimizer.step_index(),
                    initial_loss: initial_loss.unwrap_or(final_loss),
                    final_loss,
                    checkpoint: checkpoint.display().to_string(),
                    device: train_device,
                    precision,
                    learning_rate: lr,
                    micro_batch_size: batch_size,
                    grad_accumulation_steps,
                    tensor_core_counters,
                    cuda_runtime_counters,
                    tensor_core_coverage,
                    loader_report,
                    performance_report,
                    tokenizer_report: tokenizer_metadata_report(
                        &tokenizer,
                        Some(&checkpoint_tokenizer_path),
                    )?,
                })?;
            }
            println!(
                "saved checkpoint={} precision={} start_loss={:.6} final_loss={:.6}",
                checkpoint.display(),
                precision.as_str(),
                initial_loss.unwrap_or(final_loss),
                final_loss
            );
        }
        Commands::TrainMemoryLm {
            data,
            tokenizer,
            dataset_manifest,
            checkpoint,
            steps,
            batch_size,
            grad_accumulation_steps,
            block_size,
            n_layers,
            d_model,
            n_heads,
            ff_hidden,
            memory_layer_indices,
            disable_memory_layers,
            memory_slots,
            memory_key_dim,
            memory_value_dim,
            memory_top_k,
            memory_heads,
            memory_lookup,
            shared_memory,
            memory_plus,
            memory_update_policy,
            smft_mode,
            smft_row_mask,
            smft_background_counts,
            smft_access_counts_out,
            smft_mask_out,
            smft_trainable_fraction,
            smft_min_rows,
            smft_refresh_every,
            lr,
            weight_decay,
            clip_norm,
            seed,
            device,
            devices,
            distributed,
            precision,
            resume,
            log_every,
            report,
            ddp_init_timeout_secs,
            ddp_checksum_every,
        } => {
            validate_grad_accumulation_steps(grad_accumulation_steps)?;
            let config = TrainMemoryLmConfig {
                data,
                tokenizer,
                dataset_manifest,
                checkpoint,
                steps,
                batch_size,
                grad_accumulation_steps,
                block_size,
                n_layers,
                d_model,
                n_heads,
                ff_hidden,
                memory_layer_indices: if disable_memory_layers {
                    Vec::new()
                } else {
                    memory_layer_indices
                },
                memory_slots,
                memory_key_dim,
                memory_value_dim,
                memory_top_k,
                memory_heads,
                memory_lookup,
                shared_memory,
                memory_plus,
                memory_update_policy,
                smft_mode,
                smft_row_mask,
                smft_background_counts,
                smft_access_counts_out,
                smft_mask_out,
                smft_trainable_fraction,
                smft_min_rows,
                smft_refresh_every,
                lr,
                weight_decay,
                clip_norm,
                seed,
                device,
                devices,
                distributed,
                precision,
                resume,
                log_every,
                report,
                ddp_init_timeout: Duration::from_secs(ddp_init_timeout_secs),
                ddp_checksum_every,
            };
            if config.devices.is_some() || config.distributed.is_some() {
                run_distributed_train_memory_lm(config)?;
                return Ok(());
            }
            run_train_memory_lm(config)?;
        }
        Commands::TrainLmRank { config } => {
            run_train_lm_rank(&config)?;
        }
        Commands::TrainMemoryLmRank { config } => {
            run_train_memory_lm_rank(&config)?;
        }
        Commands::NcclProbeRank { config } => {
            run_nccl_probe_rank(&config)?;
        }
        Commands::NcclUniqueId { hold_secs } => {
            let root = cuda::NcclUniqueIdRoot::new().map_err(cuda_error)?;
            println!(
                "{}{}",
                NCCL_UNIQUE_ID_HELPER_MARKER,
                root.unique_id().to_hex()
            );
            std::io::stdout().flush().map_err(|err| {
                TensorError::Io(format!(
                    "failed to flush NCCL unique-id helper stdout: {err}"
                ))
            })?;
            if let Some(seconds) = hold_secs {
                eprintln!("NCCL unique-id helper holding bootstrap endpoint for {seconds}s");
                std::thread::sleep(Duration::from_secs(seconds));
            }
        }
        Commands::LauncherTest {
            ranks,
            fail_rank,
            hang_rank,
            timeout_secs,
            rank_start_timeout_secs,
            kill_grace_secs,
            report,
        } => {
            run_launcher_test(LauncherTestConfig {
                ranks,
                fail_rank,
                hang_rank,
                timeout: Duration::from_secs(timeout_secs),
                rank_start_timeout: Duration::from_secs(rank_start_timeout_secs),
                kill_grace: Duration::from_secs(kill_grace_secs),
                report,
            })?;
        }
        Commands::LauncherTestRank { config } => {
            run_launcher_test_rank(&config)?;
        }
        Commands::Generate {
            checkpoint,
            prompt,
            device,
            precision,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            report,
        } => {
            let generation_device = parse_device(&device)?;
            ensure_precision_supported_for_device(precision, generation_device)?;
            let loaded = load_lm_checkpoint_on_device(&checkpoint, generation_device)?;
            let mut tokens = loaded.tokenizer.encode(&prompt, true, false);
            if tokens.is_empty() {
                tokens.push(BOS_ID);
            }
            let options = GenerationOptions {
                max_new_tokens,
                eos_id: EOS_ID,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
            };
            let mut rng = HeirloomRng::new(seed);
            if precision == Precision::AmpBf16 {
                reset_amp_bf16_tensor_core_coverage();
            }
            let _amp_guard = (precision == Precision::AmpBf16).then(amp::enter_amp_bf16_training);
            let output = match precision {
                Precision::F32 => loaded.model.generate(&tokens, &options, &mut rng)?,
                Precision::Bf16 => loaded
                    .model
                    .generate_bf16_activations(&tokens, &options, &mut rng)?,
                Precision::AmpBf16 => loaded
                    .model
                    .generate_amp_bf16(&tokens, &options, &mut rng)?,
            };
            let text = loaded.tokenizer.decode(&output.tokens);
            println!("{text}");
            if let Some(report_path) = report {
                let checkpoint_tokenizer_path = checkpoint.join("tokenizer.json");
                write_generation_report(GenerationReportInput {
                    path: &report_path,
                    command: "generate",
                    model_family: "tiny_transformer",
                    checkpoint: &checkpoint,
                    prompt: &prompt,
                    decoded_text: &text,
                    prompt_tokens: &tokens,
                    output: &output,
                    options: &options,
                    seed,
                    device: generation_device,
                    precision,
                    tokenizer_report: tokenizer_metadata_report(
                        &loaded.tokenizer,
                        Some(&checkpoint_tokenizer_path),
                    )?,
                })?;
            }
        }
        Commands::GenerateMemoryLm {
            checkpoint,
            prompt,
            device,
            precision,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            report,
        } => {
            let generation_device = parse_device(&device)?;
            ensure_precision_supported_for_device(precision, generation_device)?;
            let loaded = load_memory_lm_checkpoint_on_device(&checkpoint, generation_device)?;
            let mut tokens = loaded.tokenizer.encode(&prompt, true, false);
            if tokens.is_empty() {
                tokens.push(BOS_ID);
            }
            let options = GenerationOptions {
                max_new_tokens,
                eos_id: EOS_ID,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
            };
            let mut rng = HeirloomRng::new(seed);
            if precision == Precision::AmpBf16 {
                reset_amp_bf16_tensor_core_coverage();
            }
            let _amp_guard = (precision == Precision::AmpBf16).then(amp::enter_amp_bf16_training);
            let output = match precision {
                Precision::F32 => loaded.model.generate(&tokens, &options, &mut rng)?,
                Precision::Bf16 => loaded
                    .model
                    .generate_bf16_activations(&tokens, &options, &mut rng)?,
                Precision::AmpBf16 => loaded
                    .model
                    .generate_amp_bf16(&tokens, &options, &mut rng)?,
            };
            let text = loaded.tokenizer.decode(&output.tokens);
            println!("{text}");
            if let Some(report_path) = report {
                let checkpoint_tokenizer_path = checkpoint.join("tokenizer.json");
                write_generation_report(GenerationReportInput {
                    path: &report_path,
                    command: "generate-memory-lm",
                    model_family: "memory_transformer",
                    checkpoint: &checkpoint,
                    prompt: &prompt,
                    decoded_text: &text,
                    prompt_tokens: &tokens,
                    output: &output,
                    options: &options,
                    seed,
                    device: generation_device,
                    precision,
                    tokenizer_report: tokenizer_metadata_report(
                        &loaded.tokenizer,
                        Some(&checkpoint_tokenizer_path),
                    )?,
                })?;
            }
        }
        Commands::EvalLm {
            checkpoint,
            dataset_manifest,
            data,
            device,
            precision,
            split,
            batch_size,
            max_batches,
            report,
        } => {
            let eval_device = parse_device(&device)?;
            ensure_precision_supported_for_device(precision, eval_device)?;
            let loaded = load_lm_checkpoint_on_device(&checkpoint, eval_device)?;
            let prepared =
                load_prepared_from_cli_or_checkpoint(&dataset_manifest, &checkpoint, true)?;
            if let Some(prepared) = &prepared {
                validate_checkpoint_tokenizer_matches_manifest(&loaded.tokenizer, prepared)?;
            }
            if precision == Precision::AmpBf16 {
                reset_amp_bf16_tensor_core_coverage();
            }
            let _amp_guard = (precision == Precision::AmpBf16).then(amp::enter_amp_bf16_training);
            let (metrics, source, loader_report) = if let Some(prepared) = prepared
                .as_ref()
                .filter(|prepared| prepared.manifest.is_binary_sharded())
            {
                let (metrics, loader_report) = evaluate_streaming_lm(
                    &loaded.model,
                    prepared,
                    split,
                    batch_size,
                    max_batches,
                    precision,
                )?;
                (
                    metrics,
                    format!("{}:{}", prepared.manifest_path.display(), split.as_str()),
                    loader_report,
                )
            } else {
                let (tokens, source) =
                    load_eval_tokens(prepared.as_ref(), data.as_ref(), &loaded.tokenizer, split)?;
                let metrics = match precision {
                    Precision::F32 => {
                        loaded
                            .model
                            .evaluate_token_loss(&tokens, batch_size, max_batches)?
                    }
                    Precision::Bf16 => loaded.model.evaluate_token_loss_bf16_activations(
                        &tokens,
                        batch_size,
                        max_batches,
                    )?,
                    Precision::AmpBf16 => loaded.model.evaluate_token_loss_amp_bf16(
                        &tokens,
                        batch_size,
                        max_batches,
                    )?,
                };
                let loader_report = serde_json::json!({
                    "kind": if prepared.is_some() { "json_tokens_in_memory" } else { "raw_text_in_memory" },
                    "source": source.clone(),
                    "tokens_materialized": true,
                    "total_tokens": tokens.len(),
                });
                (metrics, source, loader_report)
            };
            println!(
                "split={} precision={} loss={:.6} perplexity={:.6} batches={} tokens={}",
                split.as_str(),
                precision.as_str(),
                metrics.loss,
                metrics.perplexity,
                metrics.batches,
                metrics.tokens
            );
            if let Some(report_path) = report {
                let checkpoint_tokenizer_path = checkpoint.join("tokenizer.json");
                write_eval_report(EvalReportInput {
                    path: &report_path,
                    command: "eval-lm",
                    model_family: "tiny_transformer",
                    checkpoint: &checkpoint,
                    dataset_manifest: prepared
                        .as_ref()
                        .map(|prepared| prepared.manifest_path.as_path()),
                    source,
                    split,
                    batch_size,
                    max_batches,
                    metrics: &metrics,
                    device: eval_device,
                    precision,
                    loader_report,
                    tokenizer_report: tokenizer_metadata_report(
                        &loaded.tokenizer,
                        Some(&checkpoint_tokenizer_path),
                    )?,
                })?;
            }
        }
        Commands::EvalMemoryLm {
            checkpoint,
            dataset_manifest,
            data,
            device,
            precision,
            split,
            batch_size,
            max_batches,
            report,
        } => {
            let eval_device = parse_device(&device)?;
            ensure_precision_supported_for_device(precision, eval_device)?;
            let loaded = load_memory_lm_checkpoint_on_device(&checkpoint, eval_device)?;
            let prepared =
                load_prepared_from_cli_or_checkpoint(&dataset_manifest, &checkpoint, true)?;
            if let Some(prepared) = &prepared {
                validate_checkpoint_tokenizer_matches_manifest(&loaded.tokenizer, prepared)?;
            }
            if precision == Precision::AmpBf16 {
                reset_amp_bf16_tensor_core_coverage();
            }
            let _amp_guard = (precision == Precision::AmpBf16).then(amp::enter_amp_bf16_training);
            let (metrics, source, loader_report) = if let Some(prepared) = prepared
                .as_ref()
                .filter(|prepared| prepared.manifest.is_binary_sharded())
            {
                let (metrics, loader_report) = evaluate_streaming_memory_lm(
                    &loaded.model,
                    prepared,
                    split,
                    batch_size,
                    max_batches,
                    precision,
                )?;
                (
                    metrics,
                    format!("{}:{}", prepared.manifest_path.display(), split.as_str()),
                    loader_report,
                )
            } else {
                let (tokens, source) =
                    load_eval_tokens(prepared.as_ref(), data.as_ref(), &loaded.tokenizer, split)?;
                let metrics = match precision {
                    Precision::F32 => {
                        loaded
                            .model
                            .evaluate_token_loss(&tokens, batch_size, max_batches)?
                    }
                    Precision::Bf16 => loaded.model.evaluate_token_loss_bf16_activations(
                        &tokens,
                        batch_size,
                        max_batches,
                    )?,
                    Precision::AmpBf16 => loaded.model.evaluate_token_loss_amp_bf16(
                        &tokens,
                        batch_size,
                        max_batches,
                    )?,
                };
                let loader_report = serde_json::json!({
                    "kind": if prepared.is_some() { "json_tokens_in_memory" } else { "raw_text_in_memory" },
                    "source": source.clone(),
                    "tokens_materialized": true,
                    "total_tokens": tokens.len(),
                });
                (metrics, source, loader_report)
            };
            println!(
                "split={} precision={} loss={:.6} perplexity={:.6} batches={} tokens={}",
                split.as_str(),
                precision.as_str(),
                metrics.loss,
                metrics.perplexity,
                metrics.batches,
                metrics.tokens
            );
            if let Some(report_path) = report {
                let checkpoint_tokenizer_path = checkpoint.join("tokenizer.json");
                write_eval_report(EvalReportInput {
                    path: &report_path,
                    command: "eval-memory-lm",
                    model_family: "memory_transformer",
                    checkpoint: &checkpoint,
                    dataset_manifest: prepared
                        .as_ref()
                        .map(|prepared| prepared.manifest_path.as_path()),
                    source,
                    split,
                    batch_size,
                    max_batches,
                    metrics: &metrics,
                    device: eval_device,
                    precision,
                    loader_report,
                    tokenizer_report: tokenizer_metadata_report(
                        &loaded.tokenizer,
                        Some(&checkpoint_tokenizer_path),
                    )?,
                })?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct DistributedTrainLmConfig {
    data: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    dataset_manifest: Option<PathBuf>,
    checkpoint: PathBuf,
    steps: usize,
    batch_size: usize,
    grad_accumulation_steps: usize,
    block_size: usize,
    d_model: usize,
    n_heads: usize,
    ff_hidden: usize,
    lr: f64,
    weight_decay: f64,
    clip_norm: f64,
    seed: u64,
    devices: Option<String>,
    distributed: Option<DistributedMode>,
    precision: Precision,
    resume: bool,
    log_every: usize,
    report: Option<PathBuf>,
    ddp_init_timeout: Duration,
    ddp_checksum_every: usize,
}

#[derive(Clone, Debug)]
struct TrainMemoryLmConfig {
    data: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    dataset_manifest: Option<PathBuf>,
    checkpoint: PathBuf,
    steps: usize,
    batch_size: usize,
    grad_accumulation_steps: usize,
    block_size: usize,
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    ff_hidden: usize,
    memory_layer_indices: Vec<usize>,
    memory_slots: usize,
    memory_key_dim: usize,
    memory_value_dim: usize,
    memory_top_k: usize,
    memory_heads: usize,
    memory_lookup: MemoryLookupKind,
    shared_memory: bool,
    memory_plus: bool,
    memory_update_policy: MemoryUpdatePolicy,
    smft_mode: SmftMode,
    smft_row_mask: Option<PathBuf>,
    smft_background_counts: Option<PathBuf>,
    smft_access_counts_out: Option<PathBuf>,
    smft_mask_out: Option<PathBuf>,
    smft_trainable_fraction: f64,
    smft_min_rows: usize,
    smft_refresh_every: usize,
    lr: f64,
    weight_decay: f64,
    clip_norm: f64,
    seed: u64,
    device: String,
    devices: Option<String>,
    distributed: Option<DistributedMode>,
    precision: Precision,
    resume: bool,
    log_every: usize,
    report: Option<PathBuf>,
    ddp_init_timeout: Duration,
    ddp_checksum_every: usize,
}

#[derive(Debug)]
struct TokenizerCorpusTrainConfig {
    corpus_blend: PathBuf,
    reserved_tokens: Option<PathBuf>,
    out: PathBuf,
    work_dir: PathBuf,
    vocab_size: usize,
    sample_bytes: u64,
    seed: u64,
    memory_limit_bytes: Option<u64>,
    report: Option<PathBuf>,
    allow_license_status: Vec<String>,
    require_exact_vocab: bool,
}

#[derive(Debug)]
struct TokenizerCorpusTrainOutcome {
    tokenizer_path: PathBuf,
    tokenizer_hash: String,
    version: u32,
    vocab_size: usize,
    reserved_tokens: usize,
}

#[derive(Debug)]
struct TokenizerEncodeBenchConfig {
    tokenizer_path: PathBuf,
    input: Vec<PathBuf>,
    max_bytes: usize,
    iterations: usize,
    add_bos: bool,
    add_eos: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TokenizerSampleSourceReport {
    source_id: String,
    display_name: String,
    path: Option<String>,
    source_url: Option<String>,
    license_status: String,
    sampling_weight: f64,
    quota_bytes: u64,
    sampled_bytes: u64,
    effective_weight: u64,
    exhausted: bool,
    records: usize,
    content_hash: Option<String>,
}

#[derive(Clone, Debug)]
struct MaterializedTokenizerSamples {
    samples: Vec<BpeTrainingSample>,
    source_reports: Vec<TokenizerSampleSourceReport>,
    sample_manifest_path: Option<PathBuf>,
    sample_manifest_hash: String,
    sampled_bytes: u64,
}

#[derive(Debug)]
struct MaterializeBlendConfig {
    corpus_blend: PathBuf,
    tokenizer: PathBuf,
    out_dir: PathBuf,
    target_tokens: u64,
    mode: MaterializeBlendMode,
    seed: u64,
    valid_fraction: f64,
    shard_tokens: usize,
    text_shard_bytes: usize,
    max_source_bytes: Option<usize>,
    max_docs_per_source: Option<usize>,
    min_doc_bytes: usize,
    max_doc_bytes: Option<usize>,
    max_tokens_per_byte: f64,
    candidate_text_mode: MaterializerCandidateTextMode,
    candidate_retention_token_multiplier: f64,
    candidate_retention_min_docs: usize,
    candidate_prune_every: usize,
    progress_every_records: usize,
    progress_every_bytes: u64,
    checkpoint_dir: Option<PathBuf>,
    resume_checkpoint: bool,
    checkpoint_every_records: usize,
    checkpoint_every_bytes: u64,
    checkpoint_stop_after_records: Option<usize>,
    allow_license_status: Vec<String>,
}

#[derive(Debug)]
struct MaterializeBlendOutcome {
    mode: MaterializeBlendMode,
    selected_docs: usize,
    selected_tokens: u64,
    total_elapsed_ms: u64,
    selected_tokens_per_second: f64,
    prepared_manifest_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MaterializedDoc {
    source_id: String,
    source_path: String,
    ordinal: usize,
    text: Option<String>,
    text_hash: String,
    text_bytes: usize,
    token_count: usize,
    tokens_per_byte: f64,
    score: f64,
    sample_key: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MaterializedDocRecord {
    source_id: String,
    source_path: String,
    ordinal: usize,
    text_hash: String,
    text_bytes: usize,
    token_count: usize,
    tokens_per_byte: f64,
    score: f64,
    sample_key: u64,
    split: String,
    text_shard_path: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct SourceCurationStats {
    source_id: String,
    display_name: String,
    path: Option<String>,
    license_status: String,
    sampling_weight: f64,
    token_quota: u64,
    source_file_count: usize,
    scanned_docs: usize,
    candidate_docs: usize,
    candidate_tokens: u64,
    candidate_bytes: u64,
    retained_candidate_docs: usize,
    retained_candidate_tokens: u64,
    retained_candidate_bytes: u64,
    pruned_candidate_docs: usize,
    pruned_candidate_tokens: u64,
    pruned_candidate_bytes: u64,
    selected_docs: usize,
    selected_tokens: u64,
    scanned_bytes: u64,
    selected_bytes: u64,
    written_docs: usize,
    written_tokens: u64,
    written_bytes: u64,
    scan_elapsed_ms: u64,
    selection_elapsed_ms: u64,
    write_elapsed_ms: u64,
    score_elapsed_ms: u64,
    tokenizer_encode_elapsed_ms: u64,
    hash_elapsed_ms: u64,
    scan_bytes_per_second: f64,
    scan_docs_per_second: f64,
    score_docs_per_second: f64,
    candidate_docs_per_second: f64,
    candidate_tokens_per_second: f64,
    tokenizer_encode_tokens_per_second: f64,
    tokenizer_encode_bytes_per_second: f64,
    write_bytes_per_second: f64,
    write_tokens_per_second: f64,
    candidate_retention_token_multiplier: f64,
    candidate_retention_min_docs: usize,
    candidate_prune_every: usize,
    candidate_retention_enabled: bool,
    candidate_retention_token_limit: Option<u64>,
    candidate_retention_exact_for_quota: bool,
    candidate_text_mode: String,
    candidate_text_retained: bool,
    retained_candidate_text_bytes: u64,
    limited_by_max_source_bytes: bool,
    limited_by_max_docs_per_source: bool,
    content_hash: String,
    exhausted: bool,
    rejections: BTreeMap<String, usize>,
    checkpoint_path: Option<String>,
    checkpoint_loaded: bool,
    checkpoint_completed: bool,
    checkpoint_saved_count: usize,
    checkpoint_cursor_file_index: usize,
    checkpoint_cursor_byte_offset: u64,
    checkpoint_completion_reason: Option<String>,
}

#[derive(Clone, Debug)]
struct SourceCandidates {
    stats: SourceCurationStats,
    candidates: Vec<MaterializedDoc>,
}

#[derive(Clone, Debug, Default)]
struct MaterializedOutputStats {
    written_docs: usize,
    written_tokens: u64,
    written_bytes: u64,
    train_tokens: usize,
    valid_tokens: usize,
    text_shards: usize,
    train_token_shards: usize,
    valid_token_shards: usize,
    per_source: BTreeMap<String, MaterializedSourceWriteStats>,
}

#[derive(Clone, Debug, Default)]
struct MaterializedSourceWriteStats {
    docs: usize,
    tokens: u64,
    bytes: u64,
    elapsed_ms: u64,
    bytes_per_second: f64,
    tokens_per_second: f64,
}

#[derive(Clone, Debug)]
struct MaterializedOutputReport {
    manifest_path: PathBuf,
    stats: MaterializedOutputStats,
}

#[derive(Clone, Copy, Debug, Default)]
struct MaterializerDocTiming {
    tokenize_elapsed: Duration,
    hash_elapsed: Duration,
}

#[derive(Clone, Debug)]
struct MaterializerDocReject {
    reason: String,
    timing: MaterializerDocTiming,
}

#[derive(Clone, Copy, Debug, Default)]
struct CandidatePruneStats {
    docs: usize,
    tokens: u64,
    bytes: u64,
}

#[derive(Clone, Debug)]
struct MaterializerCheckpointControl {
    dir: PathBuf,
    manifest_hash: String,
    tokenizer_hash: String,
    resume: bool,
    every_records: usize,
    every_bytes: u64,
    stop_after_records: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct MaterializerScanCursor {
    file_index: usize,
    byte_offset: u64,
    ordinal: usize,
    completed: bool,
    completion_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MaterializerSourceCheckpoint {
    format: String,
    version: u32,
    source_id: String,
    config_hash: String,
    saved_unix_ms: u128,
    cursor: MaterializerScanCursor,
    stats: SourceCurationStats,
    candidates: Vec<MaterializedDoc>,
    accepted_hashes: Vec<String>,
    content_hash_state: u64,
    source_files: Vec<String>,
}

struct StableFnv64 {
    value: u64,
}

impl StableFnv64 {
    fn new() -> Self {
        Self {
            value: 0xcbf29ce484222325,
        }
    }

    fn from_value(value: u64) -> Self {
        Self { value }
    }

    fn value(&self) -> u64 {
        self.value
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.value ^= byte as u64;
            self.value = self.value.wrapping_mul(0x100000001b3);
        }
    }

    fn update_token(&mut self, token: usize) {
        self.update(&(token as u64).to_le_bytes());
    }

    fn finish_hex(&self) -> String {
        format!("{:016x}", self.value)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DdpRankConfig {
    data: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    dataset_manifest: Option<PathBuf>,
    checkpoint: PathBuf,
    steps: usize,
    batch_size: usize,
    grad_accumulation_steps: usize,
    block_size: usize,
    d_model: usize,
    n_heads: usize,
    ff_hidden: usize,
    lr: f64,
    weight_decay: f64,
    clip_norm: f64,
    seed: u64,
    precision: Precision,
    resume: bool,
    log_every: usize,
    rank: usize,
    world_size: usize,
    device_id: usize,
    nccl_unique_id_hex: String,
    report_path: PathBuf,
    stage_path: PathBuf,
    save_checkpoint: bool,
    ddp_checksum_every: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DdpMemoryRankConfig {
    data: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    dataset_manifest: Option<PathBuf>,
    checkpoint: PathBuf,
    steps: usize,
    batch_size: usize,
    grad_accumulation_steps: usize,
    block_size: usize,
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    ff_hidden: usize,
    memory_layer_indices: Vec<usize>,
    memory_slots: usize,
    memory_key_dim: usize,
    memory_value_dim: usize,
    memory_top_k: usize,
    memory_heads: usize,
    memory_lookup: MemoryLookupKind,
    shared_memory: bool,
    memory_plus: bool,
    memory_update_policy: MemoryUpdatePolicy,
    smft_mode: SmftMode,
    smft_row_mask: Option<PathBuf>,
    lr: f64,
    weight_decay: f64,
    clip_norm: f64,
    seed: u64,
    precision: Precision,
    resume: bool,
    log_every: usize,
    rank: usize,
    world_size: usize,
    device_id: usize,
    nccl_unique_id_hex: String,
    report_path: PathBuf,
    stage_path: PathBuf,
    save_checkpoint: bool,
    ddp_checksum_every: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NcclProbeRankConfig {
    rank: usize,
    world_size: usize,
    device_id: usize,
    nccl_unique_id_hex: String,
    len: usize,
    probe_kind: NcclProbeKind,
    report_path: PathBuf,
    stage_path: PathBuf,
}

#[derive(Clone, Debug)]
struct NcclProbeConfig {
    devices_value: String,
    len: usize,
    probe_kind: NcclProbeKind,
    timeout: Duration,
    rank_start_timeout: Duration,
    kill_grace: Duration,
    report: Option<PathBuf>,
}

enum SingleTrainDataset {
    InMemory {
        dataset: TokenDataset,
        total_tokens: usize,
        kind: &'static str,
        source: String,
    },
    Streaming {
        dataset: StreamingTokenDataset,
        source: String,
    },
}

impl SingleTrainDataset {
    fn next_batch(&mut self, batch_size: usize) -> Result<(Tensor, Tensor)> {
        match self {
            Self::InMemory { dataset, .. } => dataset.next_batch(batch_size),
            Self::Streaming { dataset, .. } => dataset.next_batch(batch_size),
        }
    }

    fn state(&self) -> TokenDatasetState {
        match self {
            Self::InMemory { dataset, .. } => dataset.state(),
            Self::Streaming { dataset, .. } => dataset.state(),
        }
    }

    fn loader_report(&self) -> serde_json::Value {
        match self {
            Self::InMemory {
                dataset,
                total_tokens,
                kind,
                source,
            } => serde_json::json!({
                "kind": kind,
                "source": source,
                "tokens_materialized": true,
                "total_tokens": total_tokens,
                "block_size": dataset.block_size(),
                "batches_seen": dataset.state().batches_seen,
            }),
            Self::Streaming { dataset, source } => serde_json::json!({
                "kind": "binary_shard_streaming",
                "source": source,
                "tokens_materialized": false,
                "total_tokens": dataset.len(),
                "block_size": dataset.block_size(),
                "batches_seen": dataset.state().batches_seen,
                "stats": dataset.loader_stats(),
            }),
        }
    }
}

enum ShardedTrainDataset {
    InMemory {
        tokens: Vec<usize>,
        block_size: usize,
        kind: &'static str,
        source: String,
    },
    Streaming {
        dataset: StreamingTokenDataset,
        source: String,
    },
}

impl ShardedTrainDataset {
    fn deterministic_sharded_batch(
        &mut self,
        batch_size: usize,
        seed: u64,
        global_step: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::InMemory {
                tokens, block_size, ..
            } => TokenDataset::deterministic_sharded_batch(
                tokens,
                *block_size,
                batch_size,
                seed,
                global_step,
                rank,
                world_size,
            ),
            Self::Streaming { dataset, .. } => {
                dataset.deterministic_sharded_batch(batch_size, seed, global_step, rank, world_size)
            }
        }
    }

    fn loader_report(&self) -> serde_json::Value {
        match self {
            Self::InMemory {
                tokens,
                block_size,
                kind,
                source,
            } => serde_json::json!({
                "kind": kind,
                "source": source,
                "tokens_materialized": true,
                "total_tokens": tokens.len(),
                "block_size": block_size,
            }),
            Self::Streaming { dataset, source } => serde_json::json!({
                "kind": "binary_shard_streaming",
                "source": source,
                "tokens_materialized": false,
                "total_tokens": dataset.len(),
                "block_size": dataset.block_size(),
                "stats": dataset.loader_stats(),
            }),
        }
    }
}

fn build_single_train_dataset(
    prepared: Option<&PreparedTokenData>,
    data: Option<&PathBuf>,
    tokenizer: &BpeTokenizer,
    block_size: usize,
    state: TokenDatasetState,
) -> Result<SingleTrainDataset> {
    if let Some(prepared) = prepared {
        if prepared.manifest.is_binary_sharded() {
            let dataset = prepared.streaming_dataset(TokenDataSplit::Train, block_size, state)?;
            return Ok(SingleTrainDataset::Streaming {
                dataset,
                source: format!("{}:train", prepared.manifest_path.display()),
            });
        }
        let tokens = prepared.train_tokens()?;
        let total_tokens = tokens.len();
        let dataset = TokenDataset::with_state(tokens, block_size, state)?;
        return Ok(SingleTrainDataset::InMemory {
            dataset,
            total_tokens,
            kind: "json_tokens_in_memory",
            source: format!("{}:train", prepared.manifest_path.display()),
        });
    }

    let data_path = require_path(data, "--data")?;
    let text = read_text(data_path)?;
    let tokens = tokenizer.encode(&text, true, true);
    let total_tokens = tokens.len();
    let dataset = TokenDataset::with_state(tokens, block_size, state)?;
    Ok(SingleTrainDataset::InMemory {
        dataset,
        total_tokens,
        kind: "raw_text_in_memory",
        source: data_path.display().to_string(),
    })
}

fn build_sharded_train_dataset(
    prepared: Option<&PreparedTokenData>,
    data: Option<&PathBuf>,
    tokenizer: &BpeTokenizer,
    block_size: usize,
    seed: u64,
) -> Result<ShardedTrainDataset> {
    if let Some(prepared) = prepared {
        if prepared.manifest.is_binary_sharded() {
            let dataset = prepared.streaming_dataset(
                TokenDataSplit::Train,
                block_size,
                TokenDatasetState::from_seed(seed),
            )?;
            return Ok(ShardedTrainDataset::Streaming {
                dataset,
                source: format!("{}:train", prepared.manifest_path.display()),
            });
        }
        let tokens = prepared.train_tokens()?;
        return Ok(ShardedTrainDataset::InMemory {
            tokens,
            block_size,
            kind: "json_tokens_in_memory",
            source: format!("{}:train", prepared.manifest_path.display()),
        });
    }

    let data_path = require_path(data, "--data")?;
    let text = read_text(data_path)?;
    let tokens = tokenizer.encode(&text, true, true);
    Ok(ShardedTrainDataset::InMemory {
        tokens,
        block_size,
        kind: "raw_text_in_memory",
        source: data_path.display().to_string(),
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct TrainingTimings {
    dataloader: Duration,
    host_to_device: Duration,
    forward_backward: Duration,
    all_reduce: Duration,
    optimizer: Duration,
    host_to_device_cuda_ms: f64,
    forward_backward_cuda_ms: f64,
    all_reduce_cuda_ms: f64,
    optimizer_cuda_ms: f64,
}

impl TrainingTimings {
    fn add_dataloader(&mut self, elapsed: Duration) {
        self.dataloader += elapsed;
    }

    fn add_host_to_device(&mut self, elapsed: Duration) {
        self.host_to_device += elapsed;
    }

    fn add_host_to_device_cuda_ms(&mut self, elapsed_ms: Option<f64>) {
        self.host_to_device_cuda_ms += elapsed_ms.unwrap_or(0.0);
    }

    fn add_forward_backward(&mut self, elapsed: Duration) {
        self.forward_backward += elapsed;
    }

    fn add_forward_backward_cuda_ms(&mut self, elapsed_ms: Option<f64>) {
        self.forward_backward_cuda_ms += elapsed_ms.unwrap_or(0.0);
    }

    fn add_all_reduce(&mut self, elapsed: Duration) {
        self.all_reduce += elapsed;
    }

    fn add_all_reduce_cuda_ms(&mut self, elapsed_ms: Option<f64>) {
        self.all_reduce_cuda_ms += elapsed_ms.unwrap_or(0.0);
    }

    fn add_optimizer(&mut self, elapsed: Duration) {
        self.optimizer += elapsed;
    }

    fn add_optimizer_cuda_ms(&mut self, elapsed_ms: Option<f64>) {
        self.optimizer_cuda_ms += elapsed_ms.unwrap_or(0.0);
    }
}

const A100_SXM_BF16_PEAK_FLOPS: f64 = 312_000_000_000_000.0;

fn tiny_dense_parameter_estimate(config: &TinyTransformerConfig) -> u64 {
    let vocab = config.vocab_size as u64;
    let block = config.block_size as u64;
    let d = config.d_model as u64;
    let ff = config.ff_hidden as u64;
    let embeddings = vocab * d + block * d;
    let attention = 4 * d * d + 4 * d;
    let mlp = 2 * d * ff + ff + d;
    let norms = 4 * d;
    let lm_head = d * vocab + vocab;
    embeddings + attention + mlp + norms + lm_head
}

fn memory_dense_parameter_estimate(config: &MemoryTransformerConfig, vocab_size: usize) -> u64 {
    let vocab = vocab_size as u64;
    let block = config.block_size as u64;
    let layers = config.n_layers as u64;
    let d = config.d_model as u64;
    let ff = config.ff_hidden as u64;
    let embeddings = vocab * d + block * d;
    let per_layer_attention = 4 * d * d + 4 * d;
    let per_layer_mlp = 2 * d * ff + ff + d;
    let per_layer_norms = 4 * d;
    let final_norm = 2 * d;
    let lm_head = d * vocab + vocab;
    embeddings
        + layers * (per_layer_attention + per_layer_mlp + per_layer_norms)
        + final_norm
        + lm_head
}

fn dense_training_flops_per_token_estimate(dense_params: u64) -> f64 {
    6.0 * dense_params as f64
}

fn validate_grad_accumulation_steps(steps: usize) -> Result<()> {
    if steps == 0 {
        return Err(TensorError::InvalidOperation(
            "--grad-accumulation-steps must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn accumulated_training_tokens_seen(
    steps: usize,
    batch_size: usize,
    block_size: usize,
    grad_accumulation_steps: usize,
) -> Result<usize> {
    steps
        .checked_mul(grad_accumulation_steps)
        .and_then(|tokens| tokens.checked_mul(batch_size))
        .and_then(|tokens| tokens.checked_mul(block_size))
        .ok_or_else(|| {
            TensorError::InvalidOperation(
                "training token count overflowed usize while building performance report"
                    .to_string(),
            )
        })
}

fn ddp_sample_step(
    start_step: usize,
    local_step: usize,
    micro_step: usize,
    grad_accumulation_steps: usize,
) -> Result<usize> {
    start_step
        .checked_add(local_step)
        .and_then(|step| step.checked_mul(grad_accumulation_steps))
        .and_then(|step| step.checked_add(micro_step))
        .ok_or_else(|| {
            TensorError::InvalidOperation(
                "DDP sample step overflowed usize during gradient accumulation".to_string(),
            )
        })
}

fn loss_for_grad_accumulation(loss: &Tensor, grad_accumulation_steps: usize) -> Result<Tensor> {
    if grad_accumulation_steps <= 1 {
        return Ok(loss.clone());
    }
    let scale =
        Tensor::scalar(1.0 / grad_accumulation_steps as f32, false)?.to_device(loss.device())?;
    loss.mul(&scale)
}

fn start_cuda_compute_timer(device: Device) -> Result<Option<cuda::CudaEventTimer>> {
    match device {
        Device::Cuda(device_id) => Ok(Some(
            cuda::CudaEventTimer::start_current_compute_stream(device_id as i32)
                .map_err(cuda_error)?,
        )),
        Device::Cpu => Ok(None),
    }
}

fn start_nccl_timer(communicator: &cuda::NcclCommunicator) -> Result<Option<cuda::CudaEventTimer>> {
    Ok(Some(communicator.start_event_timer().map_err(cuda_error)?))
}

fn stop_cuda_timer(timer: Option<cuda::CudaEventTimer>) -> Result<Option<f64>> {
    let Some(mut timer) = timer else {
        return Ok(None);
    };
    timer.stop_elapsed_ms().map(Some).map_err(cuda_error)
}

struct TrainingPerformanceInput {
    tokens_seen: usize,
    train_elapsed: Duration,
    timings: TrainingTimings,
    dense_flops_per_token: f64,
    device_count: usize,
    micro_batch_size: usize,
    grad_accumulation_steps: usize,
    data_parallel_world_size: usize,
}

fn training_performance_report(input: TrainingPerformanceInput) -> serde_json::Value {
    let elapsed_secs = input.train_elapsed.as_secs_f64();
    let tokens_per_second = if elapsed_secs > 0.0 {
        input.tokens_seen as f64 / elapsed_secs
    } else {
        0.0
    };
    let achieved_dense_flops = input.tokens_seen as f64 * input.dense_flops_per_token;
    let dense_core_elapsed_secs = if input.timings.forward_backward_cuda_ms > 0.0 {
        input.timings.forward_backward_cuda_ms / 1000.0
    } else {
        elapsed_secs
    };
    let dense_core_timing_source = if input.timings.forward_backward_cuda_ms > 0.0 {
        "cuda_event_forward_backward_elapsed_ms"
    } else {
        "host_train_elapsed_ms_fallback"
    };
    let dense_core_mfu = if dense_core_elapsed_secs > 0.0 && input.device_count > 0 {
        achieved_dense_flops
            / (dense_core_elapsed_secs * input.device_count as f64 * A100_SXM_BF16_PEAK_FLOPS)
    } else {
        0.0
    };
    let end_to_end_mfu = if elapsed_secs > 0.0 && input.device_count > 0 {
        achieved_dense_flops / (elapsed_secs * input.device_count as f64 * A100_SXM_BF16_PEAK_FLOPS)
    } else {
        0.0
    };
    let effective_batch_size = input
        .micro_batch_size
        .saturating_mul(input.grad_accumulation_steps);
    let global_micro_batch_size = input
        .micro_batch_size
        .saturating_mul(input.data_parallel_world_size);
    let global_effective_batch_size =
        effective_batch_size.saturating_mul(input.data_parallel_world_size);
    serde_json::json!({
        "tokens_seen": input.tokens_seen,
        "train_elapsed_ms": duration_millis(input.train_elapsed),
        "dataloader_elapsed_ms": duration_millis(input.timings.dataloader),
        "host_to_device_elapsed_ms": duration_millis(input.timings.host_to_device),
        "forward_backward_elapsed_ms": duration_millis(input.timings.forward_backward),
        "all_reduce_elapsed_ms": duration_millis(input.timings.all_reduce),
        "optimizer_elapsed_ms": duration_millis(input.timings.optimizer),
        "dataloader_host_elapsed_ms": duration_millis(input.timings.dataloader),
        "host_to_device_host_elapsed_ms": duration_millis(input.timings.host_to_device),
        "forward_backward_host_elapsed_ms": duration_millis(input.timings.forward_backward),
        "all_reduce_host_elapsed_ms": duration_millis(input.timings.all_reduce),
        "optimizer_host_elapsed_ms": duration_millis(input.timings.optimizer),
        "host_to_device_cuda_elapsed_ms": input.timings.host_to_device_cuda_ms,
        "forward_backward_cuda_elapsed_ms": input.timings.forward_backward_cuda_ms,
        "all_reduce_cuda_elapsed_ms": input.timings.all_reduce_cuda_ms,
        "optimizer_cuda_elapsed_ms": input.timings.optimizer_cuda_ms,
        "cuda_event_timing_available": input.timings.host_to_device_cuda_ms > 0.0
            || input.timings.forward_backward_cuda_ms > 0.0
            || input.timings.all_reduce_cuda_ms > 0.0
            || input.timings.optimizer_cuda_ms > 0.0,
        "tokens_per_second": tokens_per_second,
        "active_dense_flops_per_token_estimate": input.dense_flops_per_token,
        "dense_core_mfu_estimate": dense_core_mfu,
        "end_to_end_mfu_estimate": end_to_end_mfu,
        "mfu_timing_source": {
            "dense_core_mfu_estimate": dense_core_timing_source,
            "end_to_end_mfu_estimate": "host_train_elapsed_ms",
        },
        "micro_batch_size": input.micro_batch_size,
        "grad_accumulation_steps": input.grad_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "data_parallel_world_size": input.data_parallel_world_size,
        "global_micro_batch_size": global_micro_batch_size,
        "global_effective_batch_size": global_effective_batch_size,
        "mfu_denominator": {
            "hardware": "NVIDIA A100 SXM BF16 Tensor Core peak",
            "per_device_flops": A100_SXM_BF16_PEAK_FLOPS,
            "device_count": input.device_count,
        },
        "estimate_notes": [
            "MFU fields are first-pass estimates for training-path observability, not optimization claims.",
            "Sparse memory lookup/update work is reported separately and does not inflate active_dense_flops_per_token_estimate."
        ],
    })
}

fn duration_millis(duration: Duration) -> u128 {
    duration.as_millis()
}

fn device_count_for_training(device: Device) -> usize {
    match device {
        Device::Cuda(_) => 1,
        Device::Cpu => 0,
    }
}

#[derive(Clone, Debug)]
struct LauncherTestConfig {
    ranks: usize,
    fail_rank: Option<usize>,
    hang_rank: Option<usize>,
    timeout: Duration,
    rank_start_timeout: Duration,
    kill_grace: Duration,
    report: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LauncherTestRankConfig {
    rank: usize,
    report_path: PathBuf,
    stage_path: PathBuf,
    behavior: LauncherTestRankBehavior,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum LauncherTestRankBehavior {
    Success,
    Failure,
    Hang,
}

#[derive(Clone, Debug, Serialize)]
struct LauncherEnvVar {
    name: String,
    value: String,
}

#[derive(Clone, Debug, Serialize)]
struct LauncherRankSpec {
    rank: usize,
    device_id: usize,
    executable: PathBuf,
    args: Vec<String>,
    env: Vec<LauncherEnvVar>,
    config_path: PathBuf,
    report_path: PathBuf,
    stage_path: PathBuf,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
}

#[derive(Clone, Debug)]
struct DistributedLauncher {
    name: String,
    launcher_report_path: PathBuf,
    timeout: Option<Duration>,
    rank_start_timeout: Duration,
    nccl_init_timeout: Option<Duration>,
    kill_grace: Duration,
}

#[derive(Clone, Debug)]
struct LauncherOutcome {
    report_path: PathBuf,
}

struct LauncherRankRuntime {
    spec: LauncherRankSpec,
    child: Child,
    pid: u32,
    spawn_time_millis: u128,
    exit_status: Option<String>,
    completed: bool,
}

impl DistributedLauncher {
    fn run(&self, specs: Vec<LauncherRankSpec>) -> Result<LauncherOutcome> {
        if specs.is_empty() {
            return Err(TensorError::InvalidOperation(
                "distributed launcher requires at least one rank".to_string(),
            ));
        }
        let start = Instant::now();
        let mut nccl_init_seen_at = None;
        let mut runtimes = Vec::with_capacity(specs.len());

        for spec in specs {
            if let Some(parent) = spec.stdout_path.parent() {
                fs::create_dir_all(parent).map_err(|err| {
                    TensorError::Io(format!("failed to create {}: {err}", parent.display()))
                })?;
            }
            if let Some(parent) = spec.stderr_path.parent() {
                fs::create_dir_all(parent).map_err(|err| {
                    TensorError::Io(format!("failed to create {}: {err}", parent.display()))
                })?;
            }
            let stdout = File::create(&spec.stdout_path).map_err(|err| {
                TensorError::Io(format!(
                    "failed to create {}: {err}",
                    spec.stdout_path.display()
                ))
            })?;
            let stderr = File::create(&spec.stderr_path).map_err(|err| {
                TensorError::Io(format!(
                    "failed to create {}: {err}",
                    spec.stderr_path.display()
                ))
            })?;
            let mut command = Command::new(&spec.executable);
            command
                .args(&spec.args)
                .stdout(Stdio::from(stdout))
                .stderr(Stdio::from(stderr));
            for var in &spec.env {
                command.env(&var.name, &var.value);
            }
            write_rank_stage(
                &spec.stage_path,
                spec.rank,
                "spawned",
                Some("parent spawning rank worker"),
            )?;
            let child = command.spawn().map_err(|err| {
                TensorError::Io(format!(
                    "failed to spawn {} rank {} on cuda:{}: {err}",
                    self.name, spec.rank, spec.device_id
                ))
            })?;
            let pid = child.id();
            let spawn_time_millis = now_millis();
            eprintln!(
                "{} parent spawned rank={} device=cuda:{} pid={} config={} stdout={} stderr={}",
                self.name,
                spec.rank,
                spec.device_id,
                pid,
                spec.config_path.display(),
                spec.stdout_path.display(),
                spec.stderr_path.display()
            );
            runtimes.push(LauncherRankRuntime {
                spec,
                child,
                pid,
                spawn_time_millis,
                exit_status: None,
                completed: false,
            });
        }

        loop {
            let mut remaining = 0usize;
            let mut saw_progress = false;
            for index in 0..runtimes.len() {
                if runtimes[index].completed {
                    continue;
                }
                remaining += 1;
                let rank = runtimes[index].spec.rank;
                let status = runtimes[index].child.try_wait().map_err(|err| {
                    TensorError::Io(format!("failed polling {} rank {rank}: {err}", self.name))
                })?;
                let Some(status) = status else {
                    continue;
                };
                saw_progress = true;
                runtimes[index].completed = true;
                runtimes[index].exit_status = Some(status.to_string());
                remaining -= 1;
                if !status.success() {
                    let reason = format!("{} rank {rank} exited with status {status}", self.name);
                    self.terminate_rank_workers(&mut runtimes, Some(index));
                    self.write_report("failed", Some(&reason), &runtimes)?;
                    return Err(TensorError::InvalidOperation(reason));
                }
            }

            if remaining == 0 {
                self.write_report("passed", None, &runtimes)?;
                return Ok(LauncherOutcome {
                    report_path: self.launcher_report_path.clone(),
                });
            }

            let stage_values = runtimes
                .iter()
                .map(|runtime| read_rank_stage(&runtime.spec.stage_path))
                .collect::<Vec<_>>();

            if start.elapsed() > self.rank_start_timeout {
                let stuck_ranks = runtimes
                    .iter()
                    .zip(stage_values.iter())
                    .filter(|(runtime, stage)| {
                        !runtime.completed
                            && stage_name(stage)
                                .map(|name| name == "spawned" || name == "missing")
                                .unwrap_or(true)
                    })
                    .map(|(runtime, _)| runtime.spec.rank.to_string())
                    .collect::<Vec<_>>();
                if !stuck_ranks.is_empty() {
                    let reason = format!(
                        "{} rank start timeout after {}s for ranks [{}]",
                        self.name,
                        self.rank_start_timeout.as_secs(),
                        stuck_ranks.join(",")
                    );
                    self.terminate_rank_workers(&mut runtimes, None);
                    self.write_report("failed", Some(&reason), &runtimes)?;
                    return Err(TensorError::InvalidOperation(reason));
                }
            }

            if self.nccl_init_timeout.is_some()
                && stage_values
                    .iter()
                    .any(|stage| stage_name(stage) == Some("nccl_init_start"))
            {
                nccl_init_seen_at.get_or_insert_with(Instant::now);
            }
            if let (Some(timeout), Some(seen_at)) = (self.nccl_init_timeout, nccl_init_seen_at) {
                let all_ready_or_done = stage_values.iter().all(|stage| {
                    stage_name(stage)
                        .map(stage_is_after_nccl_init)
                        .unwrap_or(false)
                });
                if !all_ready_or_done && seen_at.elapsed() > timeout {
                    let reason = format!(
                        "{} NCCL init timeout after {}s",
                        self.name,
                        timeout.as_secs()
                    );
                    self.terminate_rank_workers(&mut runtimes, None);
                    self.write_report("timeout", Some(&reason), &runtimes)?;
                    return Err(TensorError::InvalidOperation(reason));
                }
            }

            if let Some(timeout) = self.timeout {
                if start.elapsed() > timeout {
                    let reason = format!("{} timeout after {}s", self.name, timeout.as_secs());
                    self.terminate_rank_workers(&mut runtimes, None);
                    self.write_report("timeout", Some(&reason), &runtimes)?;
                    return Err(TensorError::InvalidOperation(reason));
                }
            }

            if !saw_progress {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    fn terminate_rank_workers(
        &self,
        runtimes: &mut [LauncherRankRuntime],
        failed_index: Option<usize>,
    ) {
        for (index, runtime) in runtimes.iter_mut().enumerate() {
            if Some(index) == failed_index || runtime.completed {
                continue;
            }
            let _ = runtime.child.kill();
        }
        let deadline = Instant::now() + self.kill_grace;
        for (index, runtime) in runtimes.iter_mut().enumerate() {
            if Some(index) == failed_index || runtime.completed {
                continue;
            }
            while Instant::now() < deadline {
                match runtime.child.try_wait() {
                    Ok(Some(status)) => {
                        runtime.completed = true;
                        runtime.exit_status = Some(status.to_string());
                        break;
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(25)),
                    Err(err) => {
                        runtime.exit_status = Some(format!("poll error after kill: {err}"));
                        break;
                    }
                }
            }
            if !runtime.completed {
                match runtime.child.wait() {
                    Ok(status) => {
                        runtime.completed = true;
                        runtime.exit_status = Some(status.to_string());
                    }
                    Err(err) => {
                        runtime.exit_status = Some(format!("wait error after kill: {err}"));
                    }
                }
            }
        }
    }

    fn write_report(
        &self,
        status: &str,
        reason: Option<&str>,
        runtimes: &[LauncherRankRuntime],
    ) -> Result<()> {
        let ranks = runtimes
            .iter()
            .map(|runtime| {
                serde_json::json!({
                    "rank": runtime.spec.rank,
                    "device": format!("cuda:{}", runtime.spec.device_id),
                    "pid": runtime.pid,
                    "spawn_time_millis": runtime.spawn_time_millis,
                    "completed": runtime.completed,
                    "exit_status": runtime.exit_status,
                    "config_path": runtime.spec.config_path.display().to_string(),
                    "report_path": runtime.spec.report_path.display().to_string(),
                    "stage_path": runtime.spec.stage_path.display().to_string(),
                    "stdout_path": runtime.spec.stdout_path.display().to_string(),
                    "stderr_path": runtime.spec.stderr_path.display().to_string(),
                    "command": launcher_command_for_report(&runtime.spec),
                    "env": &runtime.spec.env,
                    "last_stage": read_rank_stage(&runtime.spec.stage_path),
                    "rank_report_exists": runtime.spec.report_path.exists(),
                    "stdout_exists": runtime.spec.stdout_path.exists(),
                    "stderr_exists": runtime.spec.stderr_path.exists(),
                })
            })
            .collect::<Vec<_>>();
        write_json_file(
            &self.launcher_report_path,
            serde_json::json!({
                "command": self.name,
                "status": status,
                "reason": reason,
                "timeout_secs": self.timeout.map(|duration| duration.as_secs()),
                "rank_start_timeout_secs": self.rank_start_timeout.as_secs(),
                "nccl_init_timeout_secs": self.nccl_init_timeout.map(|duration| duration.as_secs()),
                "kill_grace_secs": self.kill_grace.as_secs(),
                "ranks": ranks,
            }),
        )
    }
}

fn launcher_command_for_report(spec: &LauncherRankSpec) -> Vec<String> {
    let mut command = Vec::with_capacity(spec.args.len() + 1);
    command.push(spec.executable.display().to_string());
    command.extend(spec.args.clone());
    command
}

fn stage_is_after_nccl_init(stage: &str) -> bool {
    !matches!(
        stage,
        "missing"
            | "spawned"
            | "config_loaded"
            | "cuda_context_start"
            | "cuda_context_ready"
            | "nccl_library_loaded"
            | "nccl_init_start"
    )
}

fn stage_name(stage: &serde_json::Value) -> Option<&str> {
    stage.get("stage").and_then(serde_json::Value::as_str)
}

fn read_rank_stage(path: &Path) -> serde_json::Value {
    let Ok(json) = fs::read_to_string(path) else {
        return serde_json::json!({
            "stage": "missing",
            "path": path.display().to_string(),
        });
    };
    serde_json::from_str(&json).unwrap_or_else(|err| {
        serde_json::json!({
            "stage": "unparseable",
            "path": path.display().to_string(),
            "error": err.to_string(),
        })
    })
}

fn write_rank_stage(path: &Path, rank: usize, stage: &str, message: Option<&str>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            TensorError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }
    let tmp_path = path.with_extension("tmp");
    let mut file = File::create(&tmp_path).map_err(|err| {
        TensorError::Io(format!("failed to create {}: {err}", tmp_path.display()))
    })?;
    let json = serde_json::to_string_pretty(&serde_json::json!({
        "rank": rank,
        "stage": stage,
        "message": message,
        "timestamp_millis": now_millis(),
    }))
    .unwrap();
    file.write_all(json.as_bytes())
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", tmp_path.display())))?;
    file.write_all(b"\n")
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", tmp_path.display())))?;
    file.sync_all()
        .map_err(|err| TensorError::Io(format!("failed to sync {}: {err}", tmp_path.display())))?;
    fs::rename(&tmp_path, path).map_err(|err| {
        TensorError::Io(format!(
            "failed to rename {} to {}: {err}",
            tmp_path.display(),
            path.display()
        ))
    })
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn launcher_env_from_process() -> Vec<LauncherEnvVar> {
    launcher_env_from_lookup(|name| std::env::var(name).ok())
}

fn launcher_env_from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Vec<LauncherEnvVar> {
    [
        (
            "NCCL_DEBUG",
            lookup("HEIRLOOM_NCCL_DEBUG")
                .or_else(|| lookup("NCCL_DEBUG"))
                .or_else(|| Some("INFO".to_string())),
        ),
        (
            "NCCL_DEBUG_SUBSYS",
            lookup("HEIRLOOM_NCCL_DEBUG_SUBSYS")
                .or_else(|| lookup("NCCL_DEBUG_SUBSYS"))
                .or_else(|| Some("INIT,COLL,GRAPH".to_string())),
        ),
        (
            "NCCL_NET_PLUGIN",
            lookup("HEIRLOOM_NCCL_NET_PLUGIN").or_else(|| Some("none".to_string())),
        ),
        (
            "NCCL_SOCKET_IFNAME",
            lookup("HEIRLOOM_NCCL_SOCKET_IFNAME").or_else(|| Some("lo".to_string())),
        ),
        (
            "NCCL_IB_DISABLE",
            lookup("HEIRLOOM_NCCL_IB_DISABLE").or_else(|| Some("1".to_string())),
        ),
        (
            "NCCL_CUMEM_ENABLE",
            lookup("HEIRLOOM_NCCL_CUMEM_ENABLE").or_else(|| lookup("NCCL_CUMEM_ENABLE")),
        ),
        (
            "NCCL_CUMEM_HOST_ENABLE",
            lookup("HEIRLOOM_NCCL_CUMEM_HOST_ENABLE").or_else(|| lookup("NCCL_CUMEM_HOST_ENABLE")),
        ),
        (
            "NCCL_P2P_DISABLE",
            lookup("HEIRLOOM_NCCL_P2P_DISABLE").or_else(|| lookup("NCCL_P2P_DISABLE")),
        ),
        (
            "NCCL_P2P_LEVEL",
            lookup("HEIRLOOM_NCCL_P2P_LEVEL").or_else(|| lookup("NCCL_P2P_LEVEL")),
        ),
        ("HEIRLOOM_NCCL_TRACE", lookup("HEIRLOOM_NCCL_TRACE")),
    ]
    .into_iter()
    .filter_map(|(name, value)| {
        value.map(|value| LauncherEnvVar {
            name: name.to_string(),
            value,
        })
    })
    .collect()
}

fn launcher_work_paths(report: Option<&Path>, dirname: &str) -> (PathBuf, PathBuf) {
    if let Some(report) = report {
        let parent = report.parent().unwrap_or_else(|| Path::new("."));
        return (parent.join(dirname), parent.join("launcher-report.json"));
    }
    let work_dir = std::env::temp_dir().join(format!(
        "heirloom-{dirname}-{}-{}",
        std::process::id(),
        now_millis()
    ));
    let launcher_report = work_dir.join("launcher-report.json");
    (work_dir, launcher_report)
}

struct NcclUniqueIdHelper {
    child: Child,
    hex: String,
}

impl NcclUniqueIdHelper {
    fn hex(&self) -> &str {
        &self.hex
    }
}

impl Drop for NcclUniqueIdHelper {
    fn drop(&mut self) {
        stop_nccl_unique_id_child(&mut self.child);
    }
}

fn stop_nccl_unique_id_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn start_nccl_unique_id_helper(
    executable: &Path,
    env: &[LauncherEnvVar],
) -> Result<NcclUniqueIdHelper> {
    let mut command = Command::new(executable);
    command
        .arg("nccl-unique-id")
        .arg("--hold-secs")
        .arg(NCCL_UNIQUE_ID_HELPER_HOLD_SECS.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    for var in env {
        command.env(&var.name, &var.value);
    }
    let mut child = command.spawn().map_err(|err| {
        TensorError::Io(format!(
            "failed to spawn NCCL unique-id helper {}: {err}",
            executable.display()
        ))
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        TensorError::Io("failed to capture NCCL unique-id helper stdout".to_string())
    })?;
    let (sender, receiver) = mpsc::channel();
    let _reader = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let mut sent_hex = false;
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    if !sent_hex {
                        let _ =
                            sender
                                .send(Err("NCCL unique-id helper exited before publishing an id"
                                    .to_string()));
                    }
                    break;
                }
                Ok(_) => match parse_nccl_unique_id_helper_line(&line) {
                    Ok(Some(hex)) => {
                        if !sent_hex {
                            let _ = sender.send(Ok(hex));
                            sent_hex = true;
                        }
                    }
                    Ok(None) => {}
                    Err(err) => {
                        if !sent_hex {
                            let _ = sender.send(Err(err.to_string()));
                        }
                        break;
                    }
                },
                Err(err) => {
                    if !sent_hex {
                        let _ = sender.send(Err(format!(
                            "failed reading NCCL unique-id helper stdout: {err}"
                        )));
                    }
                    break;
                }
            }
        }
    });
    let hex = match receiver.recv_timeout(Duration::from_secs(
        NCCL_UNIQUE_ID_HELPER_START_TIMEOUT_SECS,
    )) {
        Ok(Ok(hex)) => hex,
        Ok(Err(err)) => {
            stop_nccl_unique_id_child(&mut child);
            return Err(TensorError::Device(err));
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            stop_nccl_unique_id_child(&mut child);
            return Err(TensorError::Device(format!(
                "timed out after {NCCL_UNIQUE_ID_HELPER_START_TIMEOUT_SECS}s waiting for NCCL unique-id helper"
            )));
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            stop_nccl_unique_id_child(&mut child);
            return Err(TensorError::Device(
                "NCCL unique-id helper stdout reader disconnected".to_string(),
            ));
        }
    };
    if let Some(status) = child.try_wait().map_err(|err| {
        TensorError::Io(format!(
            "failed polling NCCL unique-id helper after id publication: {err}"
        ))
    })? {
        return Err(TensorError::Device(format!(
            "NCCL unique-id helper exited immediately after publishing id with status {status}; bootstrap endpoint is not being held"
        )));
    }
    Ok(NcclUniqueIdHelper { child, hex })
}

#[cfg(test)]
fn parse_nccl_unique_id_helper_stdout(stdout: &str) -> Result<String> {
    let mut parsed = None;
    let mut invalid_marker_payload = None;
    for line in stdout.lines() {
        match parse_nccl_unique_id_helper_line(line) {
            Ok(Some(hex)) => {
                if parsed.replace(hex).is_some() {
                    return Err(TensorError::Device(
                        "NCCL unique-id helper stdout contained multiple candidate ids".to_string(),
                    ));
                }
            }
            Ok(None) => {}
            Err(err) => {
                invalid_marker_payload = Some(err.to_string());
            }
        }
    }
    if let Some(hex) = parsed {
        return Ok(hex);
    }
    if let Some(err) = invalid_marker_payload {
        return Err(TensorError::Device(err));
    }
    Err(TensorError::Device(format!(
        "NCCL unique-id helper stdout did not contain {NCCL_UNIQUE_ID_HELPER_MARKER}; stdout_lines={}",
        stdout.lines().count()
    )))
}

fn parse_nccl_unique_id_helper_line(line: &str) -> Result<Option<String>> {
    let trimmed = line.trim();
    let candidate = if let Some(hex) = trimmed.strip_prefix(NCCL_UNIQUE_ID_HELPER_MARKER) {
        Some(hex.trim())
    } else if looks_like_nccl_unique_id_hex(trimmed) {
        Some(trimmed)
    } else {
        None
    };
    let Some(hex) = candidate else {
        return Ok(None);
    };
    cuda::NcclUniqueId::from_hex(hex).map_err(|err| {
        TensorError::Device(format!("NCCL unique-id helper returned invalid id: {err}"))
    })?;
    Ok(Some(hex.to_string()))
}

fn looks_like_nccl_unique_id_hex(value: &str) -> bool {
    value.len() == 256 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn run_nccl_probe(config: NcclProbeConfig) -> Result<()> {
    let devices = parse_cuda_devices(&config.devices_value)?;
    if devices.len() < 2 {
        return Err(TensorError::InvalidOperation(format!(
            "gpu nccl-probe requires at least two CUDA devices, got {}",
            devices.len()
        )));
    }
    if config.len == 0 {
        return Err(TensorError::InvalidOperation(
            "gpu nccl-probe --len must be greater than zero".to_string(),
        ));
    }

    let exe = std::env::current_exe()
        .map_err(|err| TensorError::Io(format!("failed to locate current executable: {err}")))?;
    let env = launcher_env_from_process();
    let nccl_id_helper = if config.probe_kind.requires_nccl() {
        eprintln!(
            "NCCL probe parent requesting helper-created unique id world_size={} devices={:?} len={} probe_kind={}",
            devices.len(),
            devices,
            config.len,
            config.probe_kind.as_str()
        );
        Some(start_nccl_unique_id_helper(&exe, &env)?)
    } else {
        eprintln!(
            "NCCL probe parent skipping unique id world_size={} devices={:?} len={} probe_kind={}",
            devices.len(),
            devices,
            config.len,
            config.probe_kind.as_str()
        );
        None
    };
    let nccl_id_hex = nccl_id_helper
        .as_ref()
        .map(|helper| helper.hex().to_string())
        .unwrap_or_default();

    let (work_dir, launcher_report_path) =
        launcher_work_paths(config.report.as_deref(), "nccl-probe-ranks");
    fs::create_dir_all(&work_dir).map_err(|err| {
        TensorError::Io(format!("failed to create {}: {err}", work_dir.display()))
    })?;

    let mut specs = Vec::with_capacity(devices.len());
    for (rank, device_id) in devices.iter().copied().enumerate() {
        let rank_config_path = work_dir.join(format!("rank-{rank}.json"));
        let rank_report_path = work_dir.join(format!("rank-{rank}-report.json"));
        let rank_stage_path = work_dir.join(format!("rank-{rank}-stage.json"));
        let rank_config = NcclProbeRankConfig {
            rank,
            world_size: devices.len(),
            device_id,
            nccl_unique_id_hex: nccl_id_hex.clone(),
            len: config.len,
            probe_kind: config.probe_kind,
            report_path: rank_report_path,
            stage_path: rank_stage_path,
        };
        write_rank_config(&rank_config_path, &rank_config)?;
        specs.push(LauncherRankSpec {
            rank,
            device_id,
            executable: exe.clone(),
            args: vec![
                "nccl-probe-rank".to_string(),
                "--config".to_string(),
                rank_config_path.display().to_string(),
            ],
            env: env.clone(),
            config_path: rank_config_path,
            report_path: work_dir.join(format!("rank-{rank}-report.json")),
            stage_path: work_dir.join(format!("rank-{rank}-stage.json")),
            stdout_path: work_dir.join(format!("rank-{rank}.stdout.log")),
            stderr_path: work_dir.join(format!("rank-{rank}.stderr.log")),
        });
    }

    let launcher = DistributedLauncher {
        name: "nccl-probe".to_string(),
        launcher_report_path,
        timeout: Some(config.timeout),
        rank_start_timeout: config.rank_start_timeout,
        nccl_init_timeout: Some(config.timeout),
        kill_grace: config.kill_grace,
    };
    let outcome = launcher.run(specs)?;

    let rank_reports = (0..devices.len())
        .map(|rank| read_json_value(&work_dir.join(format!("rank-{rank}-report.json"))))
        .collect::<Result<Vec<_>>>()?;
    let max_abs_error = rank_reports
        .iter()
        .filter_map(|rank| {
            rank.get("max_abs_error")
                .and_then(serde_json::Value::as_f64)
        })
        .fold(0.0, f64::max);
    let all_reduce_calls = sum_rank_u64(&rank_reports, "all_reduce_calls");
    let all_reduce_bytes = sum_rank_u64(&rank_reports, "all_reduce_bytes");
    let nccl_version = rank_reports
        .first()
        .and_then(|rank| rank.get("nccl_version"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let expected_sum = nccl_probe_expected_sum(devices.len());
    let status =
        if !config.probe_kind.requires_all_reduce() || max_abs_error <= NCCL_PROBE_TOLERANCE {
            "passed"
        } else {
            "failed"
        };
    let json = serde_json::json!({
        "command": "gpu nccl-probe",
        "status": status,
        "world_size": devices.len(),
        "devices": devices.iter().map(|id| format!("cuda:{id}")).collect::<Vec<_>>(),
        "len": config.len,
        "probe_kind": config.probe_kind.as_str(),
        "expected_sum": expected_sum,
        "max_abs_error": max_abs_error,
        "tolerance": NCCL_PROBE_TOLERANCE,
        "all_reduce_calls": all_reduce_calls,
        "all_reduce_bytes": all_reduce_bytes,
        "nccl_version": nccl_version,
        "launcher_report": outcome.report_path.display().to_string(),
        "work_dir": work_dir.display().to_string(),
        "ranks": rank_reports,
    });
    if let Some(report_path) = config.report {
        write_json_file(&report_path, json)?;
    }
    println!(
        "nccl_probe status={} probe_kind={} world_size={} len={} expected_sum={} max_abs_error={} all_reduce_calls={} all_reduce_bytes={}",
        status,
        config.probe_kind.as_str(),
        devices.len(),
        config.len,
        expected_sum,
        max_abs_error,
        all_reduce_calls,
        all_reduce_bytes
    );
    if config.probe_kind.requires_all_reduce() && max_abs_error > NCCL_PROBE_TOLERANCE {
        return Err(TensorError::Device(format!(
            "NCCL probe all-reduce max_abs_error={max_abs_error:.6e} exceeded tolerance={NCCL_PROBE_TOLERANCE:.6e}"
        )));
    }
    Ok(())
}

fn run_nccl_probe_rank(config_path: &Path) -> Result<()> {
    let config: NcclProbeRankConfig = read_json_typed(config_path)?;
    let result = run_nccl_probe_rank_inner(&config);
    if let Err(err) = &result {
        let _ = write_rank_stage(
            &config.stage_path,
            config.rank,
            "failed",
            Some(&err.to_string()),
        );
    }
    result
}

fn run_nccl_probe_rank_inner(config: &NcclProbeRankConfig) -> Result<()> {
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "config_loaded",
        Some("NCCL probe rank config loaded"),
    )?;
    eprintln!(
        "NCCL probe rank={} device=cuda:{} initializing world_size={} len={} probe_kind={}",
        config.rank,
        config.device_id,
        config.world_size,
        config.len,
        config.probe_kind.as_str()
    );
    let input_value = (config.rank + 1) as f32;
    let mut samples = Vec::new();
    let mut max_abs_error = 0.0;
    let mut all_reduce_calls = 0usize;
    let mut all_reduce_bytes = 0usize;
    let mut nccl_version = None;
    let expected = nccl_probe_expected_sum(config.world_size);

    if config.probe_kind.requires_cuda_context() {
        write_rank_stage(
            &config.stage_path,
            config.rank,
            "cuda_context_start",
            Some("allocating and copying CUDA probe buffer"),
        )?;
        let buffer =
            cuda::CudaBuffer::from_f32(config.device_id as i32, &vec![input_value; config.len])
                .map_err(cuda_error)?;
        samples = buffer
            .to_f32()
            .map_err(cuda_error)?
            .iter()
            .copied()
            .take(8)
            .collect::<Vec<_>>();
        write_rank_stage(
            &config.stage_path,
            config.rank,
            "cuda_context_ready",
            Some("CUDA probe allocation/copy round trip completed"),
        )?;

        if config.probe_kind.requires_nccl() {
            write_rank_stage(
                &config.stage_path,
                config.rank,
                "nccl_library_loaded",
                Some("NCCL init prerequisites ready; communicator will load NCCL"),
            )?;
            let nccl_id = cuda::NcclUniqueId::from_hex(&config.nccl_unique_id_hex)
                .map_err(|err| TensorError::Device(format!("invalid NCCL unique id: {err}")))?;
            write_rank_stage(
                &config.stage_path,
                config.rank,
                "nccl_init_start",
                Some("entering ncclCommInitRank"),
            )?;
            let mut communicator = cuda::NcclCommunicator::init_rank(
                config.device_id as i32,
                config.rank as i32,
                config.world_size as i32,
                nccl_id,
            )
            .map_err(|err| {
                TensorError::Device(format!(
                    "failed to initialize NCCL probe rank {} on cuda:{}: {err}",
                    config.rank, config.device_id
                ))
            })?;
            let communicator_version = communicator.version().map_err(cuda_error)?;
            nccl_version = Some(communicator_version);
            write_rank_stage(
                &config.stage_path,
                config.rank,
                "nccl_ready",
                Some("NCCL communicator initialized"),
            )?;
            eprintln!(
                "NCCL probe rank={} device=cuda:{} communicator ready version={}",
                config.rank, config.device_id, communicator_version
            );

            if config.probe_kind.requires_all_reduce() {
                write_rank_stage(
                    &config.stage_path,
                    config.rank,
                    "all_reduce_start",
                    Some("entering NCCL f32 all-reduce"),
                )?;
                let stats = communicator
                    .all_reduce_sum_in_place_f32(&buffer)
                    .map_err(cuda_error)?;
                let reduced = buffer.to_f32().map_err(cuda_error)?;
                max_abs_error = reduced
                    .iter()
                    .map(|value| (*value as f64 - expected).abs())
                    .fold(0.0, f64::max);
                samples = reduced.iter().copied().take(8).collect::<Vec<_>>();
                all_reduce_calls = stats.calls;
                all_reduce_bytes = stats.bytes;
                write_rank_stage(
                    &config.stage_path,
                    config.rank,
                    "all_reduce_done",
                    Some("NCCL f32 all-reduce completed"),
                )?;
            }
        }
    }

    write_json_file(
        &config.report_path,
        serde_json::json!({
            "rank": config.rank,
            "world_size": config.world_size,
            "device": format!("cuda:{}", config.device_id),
            "probe_kind": config.probe_kind.as_str(),
            "device_pci_bus_id": if config.probe_kind.requires_cuda_context() {
                serde_json::Value::String(cuda_device_pci_bus_id(config.device_id)?)
            } else {
                serde_json::Value::Null
            },
            "nccl_version": nccl_version,
            "len": config.len,
            "input_value": input_value,
            "expected_sum": expected,
            "samples": samples,
            "max_abs_error": max_abs_error,
            "all_reduce_calls": all_reduce_calls,
            "all_reduce_bytes": all_reduce_bytes,
        }),
    )?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "report_written",
        Some("NCCL probe rank report written"),
    )?;
    Ok(())
}

fn nccl_probe_expected_sum(world_size: usize) -> f64 {
    (world_size * (world_size + 1) / 2) as f64
}

fn run_launcher_test(config: LauncherTestConfig) -> Result<()> {
    if config.ranks == 0 {
        return Err(TensorError::InvalidOperation(
            "launcher-test requires --ranks greater than zero".to_string(),
        ));
    }
    for (label, value) in [
        ("--fail-rank", config.fail_rank),
        ("--hang-rank", config.hang_rank),
    ] {
        if let Some(rank) = value {
            if rank >= config.ranks {
                return Err(TensorError::InvalidOperation(format!(
                    "launcher-test {label}={rank} is outside world size {}",
                    config.ranks
                )));
            }
        }
    }

    let parent = config.report.parent().unwrap_or_else(|| Path::new("."));
    let work_dir = parent.join("launcher-test-ranks");
    let launcher_report_path = parent.join("launcher-report.json");
    fs::create_dir_all(&work_dir).map_err(|err| {
        TensorError::Io(format!("failed to create {}: {err}", work_dir.display()))
    })?;

    let exe = std::env::current_exe()
        .map_err(|err| TensorError::Io(format!("failed to locate current executable: {err}")))?;
    let mut specs = Vec::with_capacity(config.ranks);
    for rank in 0..config.ranks {
        let behavior = if config.hang_rank == Some(rank) {
            LauncherTestRankBehavior::Hang
        } else if config.fail_rank == Some(rank) {
            LauncherTestRankBehavior::Failure
        } else {
            LauncherTestRankBehavior::Success
        };
        let rank_config_path = work_dir.join(format!("rank-{rank}.json"));
        let rank_report_path = work_dir.join(format!("rank-{rank}-report.json"));
        let rank_stage_path = work_dir.join(format!("rank-{rank}-stage.json"));
        let rank_config = LauncherTestRankConfig {
            rank,
            report_path: rank_report_path.clone(),
            stage_path: rank_stage_path.clone(),
            behavior,
        };
        write_rank_config(&rank_config_path, &rank_config)?;
        specs.push(LauncherRankSpec {
            rank,
            device_id: rank,
            executable: exe.clone(),
            args: vec![
                "launcher-test-rank".to_string(),
                "--config".to_string(),
                rank_config_path.display().to_string(),
            ],
            env: Vec::new(),
            config_path: rank_config_path,
            report_path: rank_report_path,
            stage_path: rank_stage_path,
            stdout_path: work_dir.join(format!("rank-{rank}.stdout.log")),
            stderr_path: work_dir.join(format!("rank-{rank}.stderr.log")),
        });
    }

    let launcher = DistributedLauncher {
        name: "launcher-test".to_string(),
        launcher_report_path: launcher_report_path.clone(),
        timeout: Some(config.timeout),
        rank_start_timeout: config.rank_start_timeout,
        nccl_init_timeout: None,
        kill_grace: config.kill_grace,
    };
    let result = launcher.run(specs);
    let (status, reason) = match &result {
        Ok(_) => ("passed", None),
        Err(err) => ("failed", Some(err.to_string())),
    };
    write_json_file(
        &config.report,
        serde_json::json!({
            "command": "launcher-test",
            "status": status,
            "reason": reason,
            "ranks": config.ranks,
            "launcher_report": launcher_report_path.display().to_string(),
            "work_dir": work_dir.display().to_string(),
        }),
    )?;
    result.map(|_| ())
}

fn run_launcher_test_rank(config_path: &Path) -> Result<()> {
    let config: LauncherTestRankConfig = read_json_typed(config_path)?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "config_loaded",
        Some("launcher test rank config loaded"),
    )?;
    match config.behavior {
        LauncherTestRankBehavior::Success => {
            println!("launcher-test rank={} success", config.rank);
            eprintln!("launcher-test rank={} stderr success", config.rank);
            write_json_file(
                &config.report_path,
                serde_json::json!({
                    "rank": config.rank,
                    "status": "passed",
                    "behavior": "success",
                }),
            )?;
            write_rank_stage(
                &config.stage_path,
                config.rank,
                "report_written",
                Some("launcher test rank report written"),
            )
        }
        LauncherTestRankBehavior::Failure => {
            write_rank_stage(
                &config.stage_path,
                config.rank,
                "failed",
                Some("launcher test rank intentional failure"),
            )?;
            Err(TensorError::InvalidOperation(format!(
                "launcher-test rank {} intentionally failed",
                config.rank
            )))
        }
        LauncherTestRankBehavior::Hang => loop {
            std::thread::sleep(Duration::from_secs(60));
        },
    }
}

fn run_train_memory_lm(config: TrainMemoryLmConfig) -> Result<()> {
    validate_grad_accumulation_steps(config.grad_accumulation_steps)?;
    let train_device = parse_device(&config.device)?;
    ensure_precision_supported_for_device(config.precision, train_device)?;
    let prepared = load_prepared_from_cli_or_checkpoint(
        &config.dataset_manifest,
        &config.checkpoint,
        config.resume,
    )?;
    let (model, mut optimizer, tokenizer, dataset_state, start_step) = if config.resume
        && config.checkpoint.exists()
    {
        let loaded = load_memory_lm_checkpoint_on_device(&config.checkpoint, train_device)?;
        if let Some(prepared) = &prepared {
            validate_checkpoint_tokenizer_matches_manifest(&loaded.tokenizer, prepared)?;
        }
        let dataset_state = loaded.metadata.dataset_state();
        let start_step = loaded.metadata.step;
        (
            loaded.model,
            loaded.optimizer,
            loaded.tokenizer,
            dataset_state,
            start_step,
        )
    } else {
        let tokenizer = if let Some(prepared) = &prepared {
            prepared.load_tokenizer()?
        } else {
            let tokenizer_path = require_path(config.tokenizer.as_ref(), "--tokenizer")?;
            BpeTokenizer::load(tokenizer_path)?
        };
        let model_config = MemoryTransformerConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: config.block_size,
            n_layers: config.n_layers,
            d_model: config.d_model,
            n_heads: config.n_heads,
            ff_hidden: config.ff_hidden,
            memory_layer_indices: config.memory_layer_indices.clone(),
            memory_slots: config.memory_slots,
            memory_key_dim: config.memory_key_dim,
            memory_value_dim: config.memory_value_dim,
            memory_top_k: config.memory_top_k,
            memory_heads: config.memory_heads,
            memory_lookup: config.memory_lookup.clone(),
            shared_memory: config.shared_memory,
            memory_plus: config.memory_plus,
            memory_update_policy: config.memory_update_policy.clone(),
            smft_mode: config.smft_mode.clone(),
        };
        let mut rng = HeirloomRng::new(config.seed);
        let model = MemoryTransformerLm::new(model_config, &mut rng)?.to_device(train_device)?;
        let optimizer = AdamW::new(model.parameters(), config.lr)?
            .with_weight_decay(config.weight_decay)?
            .with_clip_norm(Some(config.clip_norm))?;
        (
            model,
            optimizer,
            tokenizer,
            TokenDatasetState::from_seed(config.seed),
            0,
        )
    };

    let dense_flops_per_token = dense_training_flops_per_token_estimate(
        memory_dense_parameter_estimate(&model.config, tokenizer.vocab_size()),
    );
    let mut dataset = build_single_train_dataset(
        prepared.as_ref(),
        config.data.as_ref(),
        &tokenizer,
        model.config.block_size,
        dataset_state,
    )?;
    let smft_row_mask = config
        .smft_row_mask
        .as_deref()
        .map(|path| load_smft_row_mask(path, &model))
        .transpose()?;
    if config.smft_refresh_every > 0 && !memory_uses_sparse_rows(&model) {
        return Err(TensorError::InvalidOperation(
            "--smft-refresh-every was supplied, but memory sparse-row updates are not active; set --smft-mode or --memory-update-policy sparse-rows before enabling online SMFT refresh"
                .to_string(),
        ));
    }
    let mut active_smft_row_mask = smft_row_mask.clone();
    let mut active_smft_row_mask_source = config
        .smft_row_mask
        .as_ref()
        .map(|path| path.display().to_string());
    let smft_background_counts = config
        .smft_background_counts
        .as_deref()
        .map(|path| load_memory_access_counts(path, model.config.memory_slots))
        .transpose()?;
    let online_smft_refresh = config.smft_refresh_every > 0;
    let collect_smft_counts = config.smft_access_counts_out.is_some()
        || config.smft_mask_out.is_some()
        || online_smft_refresh;
    let mut accumulated_smft_counts =
        collect_smft_counts.then(|| MemoryAccessCounts::empty(model.config.memory_slots));
    if config.smft_mode == SmftMode::MaskedMemoryRows
        && active_smft_row_mask.is_none()
        && config.smft_refresh_every != 1
    {
        return Err(TensorError::InvalidOperation(
            "single-rank masked-memory-rows SMFT requires --smft-row-mask, or --smft-refresh-every 1 so a mask is materialized before the first optimizer step"
                .to_string(),
        ));
    }
    let mut smft_refresh_count = 0usize;
    let mut smft_last_refresh_step = None;
    let mut initial_loss = None;
    let mut final_loss = 0.0;
    let mut timings = TrainingTimings::default();
    let train_start = Instant::now();
    cuda::reset_tensor_core_counters();
    cuda::reset_cuda_runtime_counters();
    cuda::reset_memory_kernel_counters();
    reset_amp_bf16_tensor_core_coverage();

    let _amp_guard = (config.precision == Precision::AmpBf16).then(amp::enter_amp_bf16_training);
    for local_step in 0..config.steps {
        optimizer.zero_grad();
        let global_step = start_step + local_step + 1;
        let mut step_loss_sum = 0.0;
        for _micro_step in 0..config.grad_accumulation_steps {
            let dataloader_start = Instant::now();
            let (input, target) = dataset.next_batch(config.batch_size)?;
            timings.add_dataloader(dataloader_start.elapsed());
            let h2d_start = Instant::now();
            let h2d_timer = start_cuda_compute_timer(train_device)?;
            let input = input.to_device(train_device)?;
            let target = target.to_device(train_device)?;
            let h2d_cuda_ms = stop_cuda_timer(h2d_timer)?;
            timings.add_host_to_device(h2d_start.elapsed());
            timings.add_host_to_device_cuda_ms(h2d_cuda_ms);
            let forward_backward_start = Instant::now();
            let forward_backward_timer = start_cuda_compute_timer(train_device)?;
            let loss = match config.precision {
                Precision::F32 => model.loss(&input, &target)?,
                Precision::Bf16 => model.loss_bf16_activations(&input, &target)?,
                Precision::AmpBf16 => model.loss_amp_bf16(&input, &target)?,
            };
            let loss_value = training_loss_scalar(&loss, config.precision)?;
            step_loss_sum += loss_value;
            loss_for_grad_accumulation(&loss, config.grad_accumulation_steps)?.backward()?;
            let forward_backward_cuda_ms = stop_cuda_timer(forward_backward_timer)?;
            timings.add_forward_backward(forward_backward_start.elapsed());
            timings.add_forward_backward_cuda_ms(forward_backward_cuda_ms);
            if let Some(counts) = &mut accumulated_smft_counts {
                let latest_counts = collect_smft_access_counts(&model, config.precision)?;
                counts.merge_in(&latest_counts)?;
            }
        }
        let step_loss = step_loss_sum / config.grad_accumulation_steps as f64;
        initial_loss.get_or_insert(step_loss);
        final_loss = step_loss;
        if let Some(counts) = &mut accumulated_smft_counts {
            if online_smft_refresh && global_step % config.smft_refresh_every == 0 {
                let empty_background;
                let background = if let Some(background) = &smft_background_counts {
                    background
                } else {
                    empty_background = MemoryAccessCounts::empty(model.config.memory_slots);
                    &empty_background
                };
                let refreshed_mask = counts.smft_mask_against(
                    background,
                    config.smft_trainable_fraction,
                    config.smft_min_rows,
                )?;
                active_smft_row_mask = Some(refreshed_mask);
                active_smft_row_mask_source = Some(format!("online_refresh_step_{global_step}"));
                smft_refresh_count += 1;
                smft_last_refresh_step = Some(global_step);
            }
        }
        let optimizer_start = Instant::now();
        let optimizer_timer = start_cuda_compute_timer(train_device)?;
        step_memory_optimizer(&model, &mut optimizer, active_smft_row_mask.as_ref())?;
        let optimizer_cuda_ms = stop_cuda_timer(optimizer_timer)?;
        timings.add_optimizer(optimizer_start.elapsed());
        timings.add_optimizer_cuda_ms(optimizer_cuda_ms);
        if config.log_every > 0 && (local_step == 0 || global_step % config.log_every == 0) {
            println!("memory step={global_step} loss={step_loss:.6}");
        }
    }
    let train_elapsed = train_start.elapsed();

    let mut generated_smft_mask = None;
    if let Some(counts) = &accumulated_smft_counts {
        if let Some(path) = &config.smft_access_counts_out {
            write_typed_json(path, counts, "SMFT access counts")?;
        }
        if let Some(path) = &config.smft_mask_out {
            let empty_background;
            let background = if let Some(background) = &smft_background_counts {
                background
            } else {
                empty_background = MemoryAccessCounts::empty(model.config.memory_slots);
                &empty_background
            };
            let mask = counts.smft_mask_against(
                background,
                config.smft_trainable_fraction,
                config.smft_min_rows,
            )?;
            write_typed_json(path, &mask, "SMFT row mask")?;
            generated_smft_mask = Some(mask);
        }
    }

    let tensor_core_counters = cuda::tensor_core_counters();
    let cuda_runtime_counters = cuda::cuda_runtime_counters();
    let memory_kernel_counters = cuda::memory_kernel_counters();
    let tensor_core_coverage = amp_bf16_tensor_core_coverage();
    let final_training_step = start_step + config.steps;
    let loader_report = dataset.loader_report();
    let tokens_seen = accumulated_training_tokens_seen(
        config.steps,
        config.batch_size,
        model.config.block_size,
        config.grad_accumulation_steps,
    )?;
    let performance_report = training_performance_report(TrainingPerformanceInput {
        tokens_seen,
        train_elapsed,
        timings,
        dense_flops_per_token,
        device_count: device_count_for_training(train_device),
        micro_batch_size: config.batch_size,
        grad_accumulation_steps: config.grad_accumulation_steps,
        data_parallel_world_size: 1,
    });
    checkpoint_with_amp_staging_allowed(config.precision, || {
        save_memory_lm_checkpoint_with_dataset_state_and_step(
            &config.checkpoint,
            &model,
            &optimizer,
            &tokenizer,
            dataset.state(),
            prepared
                .as_ref()
                .map(|prepared| prepared.manifest_path.display().to_string()),
            final_training_step,
        )
    })?;
    if let Some(report_path) = &config.report {
        let memory_selection_report = if config.precision == Precision::AmpBf16 {
            amp::with_cuda_host_staging_allowed("amp-bf16 memory selection report", || {
                model.memory_selection_report()
            })
        } else {
            model.memory_selection_report()
        }?;
        let memory_selection_report =
            serde_json::to_value(memory_selection_report).map_err(|err| {
                TensorError::Io(format!(
                    "failed to serialize memory selection report: {err}"
                ))
            })?;
        let memory_optimizer_report = memory_optimizer_report(
            &model,
            active_smft_row_mask.as_ref(),
            active_smft_row_mask_source.as_deref(),
        )?;
        let smft_access_report = if config.precision == Precision::AmpBf16 {
            amp::with_cuda_host_staging_allowed("amp-bf16 SMFT access report", || {
                model.smft_access_report(16)
            })
        } else {
            model.smft_access_report(16)
        }?;
        let smft_access_report = serde_json::to_value(smft_access_report).map_err(|err| {
            TensorError::Io(format!("failed to serialize SMFT access report: {err}"))
        })?;
        let smft_artifacts_report = smft_artifacts_report(SmftArtifactsReportInput {
            config: &config,
            background_counts: smft_background_counts.as_ref(),
            accumulated_counts: accumulated_smft_counts.as_ref(),
            generated_mask: generated_smft_mask.as_ref(),
            active_mask: active_smft_row_mask.as_ref(),
            active_mask_source: active_smft_row_mask_source.as_deref(),
            refresh_count: smft_refresh_count,
            last_refresh_step: smft_last_refresh_step,
        });
        let (memory_table_checksum_sum, memory_table_checksum_sumsq) =
            if config.precision == Precision::AmpBf16 {
                amp::with_cuda_host_staging_allowed("amp-bf16 memory table checksum report", || {
                    memory_table_parameter_checksums(&model)
                })
            } else {
                memory_table_parameter_checksums(&model)
            }?;
        let checkpoint_tokenizer_path = config.checkpoint.join("tokenizer.json");
        write_memory_train_report(MemoryTrainReportInput {
            path: report_path,
            start_step,
            final_step: final_training_step,
            initial_loss: initial_loss.unwrap_or(final_loss),
            final_loss,
            checkpoint: config.checkpoint.display().to_string(),
            device: train_device,
            precision: config.precision,
            learning_rate: config.lr,
            micro_batch_size: config.batch_size,
            grad_accumulation_steps: config.grad_accumulation_steps,
            memory_config: &model.config,
            tensor_core_counters,
            cuda_runtime_counters,
            memory_kernel_counters,
            tensor_core_coverage,
            memory_selection_report,
            memory_optimizer_report,
            smft_access_report,
            smft_artifacts_report,
            memory_table_checksum_sum,
            memory_table_checksum_sumsq,
            loader_report,
            performance_report,
            tokenizer_report: tokenizer_metadata_report(
                &tokenizer,
                Some(&checkpoint_tokenizer_path),
            )?,
        })?;
    }
    println!(
        "saved memory checkpoint={} precision={} start_loss={:.6} final_loss={:.6}",
        config.checkpoint.display(),
        config.precision.as_str(),
        initial_loss.unwrap_or(final_loss),
        final_loss
    );
    Ok(())
}

fn step_memory_optimizer(
    model: &MemoryTransformerLm,
    optimizer: &mut AdamW,
    smft_row_mask: Option<&SmftRowMask>,
) -> Result<()> {
    if model.config.memory_update_policy == MemoryUpdatePolicy::Frozen {
        return Ok(());
    }
    if memory_uses_sparse_rows(model) {
        let updates = if let Some(mask) = smft_row_mask {
            model.memory_sparse_adamw_updates_with_mask(mask)?
        } else {
            model.memory_sparse_adamw_updates()?
        };
        if updates.is_empty() {
            return Err(TensorError::InvalidOperation(
                "memory sparse-row update requested, but no selected rows were captured; run a memory forward/backward before optimizer step".to_string(),
            ));
        }
        return optimizer.step_sparse_rows_mut(&updates);
    }
    match model.config.memory_update_policy {
        MemoryUpdatePolicy::Frozen | MemoryUpdatePolicy::SparseRows => {
            unreachable!("handled above")
        }
        MemoryUpdatePolicy::MemoryOnly => {
            let indices = model.memory_table_parameter_indices()?;
            if indices.is_empty() {
                return Err(TensorError::InvalidOperation(
                    "memory-only update requested, but the model has no memory table parameters"
                        .to_string(),
                ));
            }
            optimizer.step_parameter_indices_mut(&indices)
        }
        MemoryUpdatePolicy::Full => optimizer.step_mut(),
    }
}

fn memory_optimizer_report(
    model: &MemoryTransformerLm,
    smft_row_mask: Option<&SmftRowMask>,
    smft_row_mask_source: Option<&str>,
) -> Result<serde_json::Value> {
    let memory_table_parameter_indices = model.memory_table_parameter_indices()?;
    let memory_table_parameter_count = memory_table_parameter_indices.len();
    let uses_sparse_rows = memory_uses_sparse_rows(model);
    let sparse_updates = if uses_sparse_rows {
        if let Some(mask) = smft_row_mask {
            model.memory_sparse_adamw_updates_with_mask(mask)?
        } else {
            model.memory_sparse_adamw_updates()?
        }
    } else {
        Vec::new()
    };
    let row_mask_attached_sparse_update_count = sparse_updates
        .iter()
        .filter(|update| update.row_mask.is_some())
        .count();
    let sparse_update_selected_row_events = sparse_updates
        .iter()
        .map(|update| update.selected_rows.numel())
        .sum::<usize>();
    let applied_path = if model.config.memory_update_policy == MemoryUpdatePolicy::Frozen {
        "frozen_no_optimizer_step"
    } else if uses_sparse_rows {
        match model.device() {
            Device::Cpu => "cpu_sparse_rows_selected_memory_tables",
            Device::Cuda(_) => "cuda_sparse_rows_selected_memory_tables",
        }
    } else {
        match model.config.memory_update_policy {
            MemoryUpdatePolicy::Full => "dense_all_parameters",
            MemoryUpdatePolicy::MemoryOnly => "dense_memory_tables_only",
            MemoryUpdatePolicy::Frozen | MemoryUpdatePolicy::SparseRows => {
                unreachable!("handled above")
            }
        }
    };
    let product_key_projection = smft_row_mask
        .map(|mask| model.smft_product_key_mask_projection(mask))
        .transpose()?
        .flatten();
    let product_key_projection_json = product_key_projection.map(|projection| {
        serde_json::json!({
            "policy": projection.policy,
            "side": projection.side,
            "value_trainable_rows": projection.value_trainable_rows,
            "left_trainable_rows": projection.left_trainable_rows.len(),
            "right_trainable_rows": projection.right_trainable_rows.len(),
            "half_key_rows_are_conservative": projection.half_key_rows_are_conservative,
        })
    });
    let smft_row_mask_json = smft_row_mask.map(|mask| {
        serde_json::json!({
            "source": smft_row_mask_source,
            "applied": uses_sparse_rows,
            "memory_slots": mask.memory_slots,
            "trainable_rows": mask.trainable_rows.len(),
            "frozen_rows": mask.frozen_rows,
            "trainable_fraction": mask.trainable_fraction,
            "product_key_projection": product_key_projection_json,
        })
    });
    Ok(serde_json::json!({
        "memory_update_policy": model.config.memory_update_policy.clone(),
        "smft_mode": model.config.smft_mode.clone(),
        "applied_path": applied_path,
        "memory_table_parameter_indices": memory_table_parameter_indices,
        "memory_table_parameter_count": memory_table_parameter_count,
        "sparse_update_parameter_count": sparse_updates.len(),
        "row_mask_attached_sparse_update_count": row_mask_attached_sparse_update_count,
        "sparse_update_selected_row_events": sparse_update_selected_row_events,
        "smft_row_mask": smft_row_mask_json,
        "sparse_updates_accumulate_dense_gradient_buffers": uses_sparse_rows,
        "sparse_optimizer_updates_selected_rows": uses_sparse_rows,
        "sparse_optimizer_gathers_compact_gradient_rows": uses_sparse_rows && matches!(model.device(), Device::Cuda(_)),
        "compressed_sparse_gradient_transport": false,
        "note": "SparseRows and SMFT update only selected memory rows. Autograd still accumulates dense memory-table gradient buffers. CPU sparse AdamW updates the selected row ranges directly; CUDA sparse AdamW gathers compact selected gradient rows before the row-update kernel. Distributed sparse-gradient transport is reported by DDP rank and aggregate reports.",
    }))
}

struct SmftArtifactsReportInput<'a> {
    config: &'a TrainMemoryLmConfig,
    background_counts: Option<&'a MemoryAccessCounts>,
    accumulated_counts: Option<&'a MemoryAccessCounts>,
    generated_mask: Option<&'a SmftRowMask>,
    active_mask: Option<&'a SmftRowMask>,
    active_mask_source: Option<&'a str>,
    refresh_count: usize,
    last_refresh_step: Option<usize>,
}

fn smft_artifacts_report(input: SmftArtifactsReportInput<'_>) -> serde_json::Value {
    serde_json::json!({
        "access_counts_out": input.config
            .smft_access_counts_out
            .as_ref()
            .map(|path| path.display().to_string()),
        "mask_out": input.config
            .smft_mask_out
            .as_ref()
            .map(|path| path.display().to_string()),
        "background_counts_source": input.config
            .smft_background_counts
            .as_ref()
            .map(|path| path.display().to_string()),
        "background_counts": input.background_counts.map(|counts| {
            serde_json::json!({
                "memory_slots": counts.memory_slots,
                "total_events": counts.total_events,
                "unique_rows": counts.unique_rows,
            })
        }),
        "accumulated_counts": input.accumulated_counts.map(|counts| {
            serde_json::json!({
                "memory_slots": counts.memory_slots,
                "total_events": counts.total_events,
                "unique_rows": counts.unique_rows,
            })
        }),
        "generated_mask": input.generated_mask.map(|mask| {
            serde_json::json!({
                "memory_slots": mask.memory_slots,
                "trainable_rows": mask.trainable_rows.len(),
                "frozen_rows": mask.frozen_rows,
                "trainable_fraction": mask.trainable_fraction,
            })
        }),
        "online_refresh": {
            "enabled": input.config.smft_refresh_every > 0,
            "refresh_every": input.config.smft_refresh_every,
            "refresh_count": input.refresh_count,
            "last_refresh_step": input.last_refresh_step,
            "active_mask_source": input.active_mask_source,
            "active_mask": input.active_mask.map(|mask| {
                serde_json::json!({
                    "memory_slots": mask.memory_slots,
                    "trainable_rows": mask.trainable_rows.len(),
                    "frozen_rows": mask.frozen_rows,
                    "trainable_fraction": mask.trainable_fraction,
                })
            }),
        },
        "trainable_fraction": input.config.smft_trainable_fraction,
        "min_rows": input.config.smft_min_rows,
        "note": "These artifacts persist SMFT access counts and derived masks. Online refresh can update the active sparse-row mask from accumulated foreground counts during a run, but long-running background collection/windowing remains outside this path.",
    })
}

fn memory_table_parameter_checksums(model: &MemoryTransformerLm) -> Result<(f64, f64)> {
    no_grad(|| {
        let parameters = model.parameters();
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for index in model.memory_table_parameter_indices()? {
            let parameter = parameters.get(index).ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "memory table parameter index {index} is out of range for {} parameters",
                    parameters.len()
                ))
            })?;
            sum += parameter.sum()?.data()[0] as f64;
            sumsq += parameter.mul(parameter)?.sum()?.data()[0] as f64;
        }
        Ok((sum, sumsq))
    })
}

fn run_distributed_train_memory_lm(config: TrainMemoryLmConfig) -> Result<()> {
    validate_grad_accumulation_steps(config.grad_accumulation_steps)?;
    let Some(DistributedMode::Nccl) = config.distributed else {
        return Err(TensorError::InvalidOperation(
            "distributed memory training requires --distributed nccl".to_string(),
        ));
    };
    let devices_value = config.devices.as_ref().ok_or_else(|| {
        TensorError::InvalidOperation(
            "distributed memory training requires --devices cuda:<id>,...".to_string(),
        )
    })?;
    let devices = parse_cuda_devices(devices_value)?;
    if devices.len() < 2 {
        return Err(TensorError::InvalidOperation(format!(
            "distributed NCCL memory training requires at least two CUDA devices, got {}",
            devices.len()
        )));
    }
    for device_id in &devices {
        ensure_precision_supported_for_device(config.precision, Device::Cuda(*device_id))?;
    }
    if config.smft_background_counts.is_some()
        || config.smft_access_counts_out.is_some()
        || config.smft_mask_out.is_some()
        || config.smft_refresh_every > 0
        || config.smft_mode == SmftMode::FreezeDenseUpdateMemory
    {
        return Err(TensorError::InvalidOperation(
            "distributed train-memory-lm supports offline SMFT row masks only; background counts, generated mask artifacts, online refresh, and freeze-dense SMFT need distributed mask generation/refresh before they can run safely"
                .to_string(),
        ));
    }
    if config.smft_mode == SmftMode::MaskedMemoryRows && config.smft_row_mask.is_none() {
        return Err(TensorError::InvalidOperation(
            "distributed masked-memory-rows SMFT requires --smft-row-mask so every rank applies the same offline mask"
                .to_string(),
        ));
    }
    if config.smft_row_mask.is_some()
        && config.memory_update_policy != MemoryUpdatePolicy::SparseRows
    {
        return Err(TensorError::InvalidOperation(
            "distributed --smft-row-mask requires --memory-update-policy sparse-rows so row-union masks can be intersected with the offline SMFT mask"
                .to_string(),
        ));
    }
    if config.memory_update_policy == MemoryUpdatePolicy::Frozen {
        return Err(TensorError::InvalidOperation(
            "distributed train-memory-lm requires Full, MemoryOnly, or SparseRows updates so memory parameters participate in synchronized optimizer steps"
                .to_string(),
        ));
    }
    if config.precision != Precision::AmpBf16 {
        eprintln!(
            "warning: distributed memory NCCL training is intended for --precision amp-bf16; running {}",
            config.precision.as_str()
        );
    }

    eprintln!(
        "distributed memory parent requesting helper-created NCCL unique id world_size={} devices={:?}",
        devices.len(),
        devices
    );
    let exe = std::env::current_exe()
        .map_err(|err| TensorError::Io(format!("failed to locate current executable: {err}")))?;
    let env = launcher_env_from_process();
    let nccl_id_helper = start_nccl_unique_id_helper(&exe, &env)?;
    let nccl_id_hex = nccl_id_helper.hex().to_string();
    eprintln!(
        "distributed memory parent received helper-created NCCL unique id; spawning {} ranks",
        devices.len()
    );
    let (work_dir, launcher_report_path) =
        launcher_work_paths(config.report.as_deref(), "ddp-memory-ranks");
    fs::create_dir_all(&work_dir).map_err(|err| {
        TensorError::Io(format!("failed to create {}: {err}", work_dir.display()))
    })?;

    let mut specs = Vec::with_capacity(devices.len());
    for (rank, device_id) in devices.iter().copied().enumerate() {
        let rank_config_path = work_dir.join(format!("rank-{rank}.json"));
        let rank_report_path = work_dir.join(format!("rank-{rank}-report.json"));
        let rank_stage_path = work_dir.join(format!("rank-{rank}-stage.json"));
        let rank_config = DdpMemoryRankConfig {
            data: config.data.clone(),
            tokenizer: config.tokenizer.clone(),
            dataset_manifest: config.dataset_manifest.clone(),
            checkpoint: config.checkpoint.clone(),
            steps: config.steps,
            batch_size: config.batch_size,
            grad_accumulation_steps: config.grad_accumulation_steps,
            block_size: config.block_size,
            n_layers: config.n_layers,
            d_model: config.d_model,
            n_heads: config.n_heads,
            ff_hidden: config.ff_hidden,
            memory_layer_indices: config.memory_layer_indices.clone(),
            memory_slots: config.memory_slots,
            memory_key_dim: config.memory_key_dim,
            memory_value_dim: config.memory_value_dim,
            memory_top_k: config.memory_top_k,
            memory_heads: config.memory_heads,
            memory_lookup: config.memory_lookup.clone(),
            shared_memory: config.shared_memory,
            memory_plus: config.memory_plus,
            memory_update_policy: config.memory_update_policy.clone(),
            smft_mode: config.smft_mode.clone(),
            smft_row_mask: config.smft_row_mask.clone(),
            lr: config.lr,
            weight_decay: config.weight_decay,
            clip_norm: config.clip_norm,
            seed: config.seed,
            precision: config.precision,
            resume: config.resume,
            log_every: config.log_every,
            rank,
            world_size: devices.len(),
            device_id,
            nccl_unique_id_hex: nccl_id_hex.clone(),
            report_path: rank_report_path.clone(),
            stage_path: rank_stage_path.clone(),
            save_checkpoint: rank == 0,
            ddp_checksum_every: config.ddp_checksum_every,
        };
        write_rank_config(&rank_config_path, &rank_config)?;
        specs.push(LauncherRankSpec {
            rank,
            device_id,
            executable: exe.clone(),
            args: vec![
                "train-memory-lm-rank".to_string(),
                "--config".to_string(),
                rank_config_path.display().to_string(),
            ],
            env: env.clone(),
            config_path: rank_config_path,
            report_path: rank_report_path,
            stage_path: rank_stage_path,
            stdout_path: work_dir.join(format!("rank-{rank}.stdout.log")),
            stderr_path: work_dir.join(format!("rank-{rank}.stderr.log")),
        });
    }

    let launcher = DistributedLauncher {
        name: "ddp-train-memory-lm".to_string(),
        launcher_report_path,
        timeout: None,
        rank_start_timeout: Duration::from_secs(10),
        nccl_init_timeout: Some(config.ddp_init_timeout),
        kill_grace: Duration::from_secs(5),
    };
    let outcome = launcher.run(specs)?;

    let rank_reports = (0..devices.len())
        .map(|rank| read_json_value(&work_dir.join(format!("rank-{rank}-report.json"))))
        .collect::<Result<Vec<_>>>()?;
    let rank0 = &rank_reports[0];
    let all_reduce_calls = sum_rank_u64(&rank_reports, "all_reduce_calls");
    let all_reduce_bytes = sum_rank_u64(&rank_reports, "all_reduce_bytes");
    let compact_gradient_all_reduce_calls =
        sum_rank_u64(&rank_reports, "compact_gradient_all_reduce_calls");
    let compact_gradient_all_reduce_bytes =
        sum_rank_u64(&rank_reports, "compact_gradient_all_reduce_bytes");
    let row_union_all_reduce_calls = sum_rank_u64(&rank_reports, "row_union_all_reduce_calls");
    let row_union_all_reduce_bytes = sum_rank_u64(&rank_reports, "row_union_all_reduce_bytes");
    let row_union_candidate_rows = sum_rank_u64(&rank_reports, "row_union_candidate_rows");
    let memory_gradient_parameter_counts = rank_reports
        .iter()
        .map(|report| {
            report
                .get("memory_gradient_parameter_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let memory_table_checksum_sums = rank_f64_values(&rank_reports, "memory_table_checksum_sum");
    let memory_table_checksum_sumsq = rank_f64_values(&rank_reports, "memory_table_checksum_sumsq");
    let memory_table_checksum_sum_max_error = max_rank_drift(&memory_table_checksum_sums);
    let memory_table_checksum_sumsq_max_error = max_rank_drift(&memory_table_checksum_sumsq);
    let step_checksum_drift = ddp_memory_step_checksum_drifts(&rank_reports)?;
    let step_memory_sum_max_error =
        max_step_checksum_error(&step_checksum_drift, "memory_table_checksum_sum_max_error");
    let step_memory_sumsq_max_error = max_step_checksum_error(
        &step_checksum_drift,
        "memory_table_checksum_sumsq_max_error",
    );
    validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
        memory_update_policy: &config.memory_update_policy,
        all_reduce_calls,
        all_reduce_bytes,
        row_union_all_reduce_calls,
        row_union_all_reduce_bytes,
        row_union_candidate_rows,
        compact_gradient_all_reduce_calls,
        compact_gradient_all_reduce_bytes,
    })?;
    if memory_gradient_parameter_counts.contains(&0) {
        return Err(TensorError::Device(format!(
            "distributed memory NCCL training did not observe memory-table gradients on every rank: {:?}",
            memory_gradient_parameter_counts
        )));
    }
    if memory_table_checksum_sum_max_error > DDP_PARAMETER_CHECKSUM_TOLERANCE
        || memory_table_checksum_sumsq_max_error > DDP_PARAMETER_CHECKSUM_TOLERANCE
        || step_memory_sum_max_error > DDP_PARAMETER_CHECKSUM_TOLERANCE
        || step_memory_sumsq_max_error > DDP_PARAMETER_CHECKSUM_TOLERANCE
    {
        return Err(TensorError::Device(format!(
            "distributed memory NCCL checksum drift exceeded tolerance: \
             final_sum_error={memory_table_checksum_sum_max_error:.6e} \
             final_sumsq_error={memory_table_checksum_sumsq_max_error:.6e} \
             step_sum_error={step_memory_sum_max_error:.6e} \
             step_sumsq_error={step_memory_sumsq_max_error:.6e} \
             tolerance={DDP_PARAMETER_CHECKSUM_TOLERANCE:.6e}"
        )));
    }
    let loader_report = rank0_loader_report(&rank_reports);
    let performance_report = aggregate_rank_performance(&rank_reports);
    if let Some(report_path) = config.report {
        let json = serde_json::json!({
            "command": "train-memory-lm",
            "model_family": "memory_transformer",
            "distributed": "nccl",
            "status": "passed",
            "precision": config.precision.as_str(),
            "learning_rate": config.lr,
            "world_size": devices.len(),
            "devices": devices.iter().map(|id| format!("cuda:{id}")).collect::<Vec<_>>(),
            "per_rank_batch_size": config.batch_size,
            "global_batch_size": config.batch_size * devices.len(),
            "grad_accumulation_steps": config.grad_accumulation_steps,
            "per_rank_effective_batch_size": config.batch_size * config.grad_accumulation_steps,
            "global_micro_batch_size": config.batch_size * devices.len(),
            "global_effective_batch_size": config.batch_size * config.grad_accumulation_steps * devices.len(),
            "memory_update_policy": config.memory_update_policy.clone(),
            "smft_mode": config.smft_mode.clone(),
            "smft_row_mask": config
                .smft_row_mask
                .as_ref()
                .map(|path| path.display().to_string()),
            "smft_row_mask_trainable_rows_rank0": rank0.get("smft_row_mask_trainable_rows").cloned().unwrap_or(serde_json::Value::Null),
            "smft_row_mask_frozen_rows_rank0": rank0.get("smft_row_mask_frozen_rows").cloned().unwrap_or(serde_json::Value::Null),
            "memory_config": rank0.get("memory_config").cloned().unwrap_or(serde_json::Value::Null),
            "ddp_checksum_every": config.ddp_checksum_every,
            "launcher_report": outcome.report_path.display().to_string(),
            "work_dir": work_dir.display().to_string(),
            "checkpoint": config.checkpoint.display().to_string(),
            "start_step": rank0.get("start_step").cloned().unwrap_or(serde_json::Value::Null),
            "final_step": rank0.get("final_step").cloned().unwrap_or(serde_json::Value::Null),
            "initial_loss": rank0.get("initial_loss").cloned().unwrap_or(serde_json::Value::Null),
            "final_loss": rank0.get("final_loss").cloned().unwrap_or(serde_json::Value::Null),
            "loss_reduction": rank0.get("loss_reduction").cloned().unwrap_or(serde_json::Value::Null),
            "rank_loss_reductions": rank_f64_values(&rank_reports, "loss_reduction"),
            "all_reduce_calls": all_reduce_calls,
            "all_reduce_bytes": all_reduce_bytes,
            "loader": loader_report,
            "performance": performance_report,
            "compact_gradient_all_reduce_calls": compact_gradient_all_reduce_calls,
            "compact_gradient_all_reduce_bytes": compact_gradient_all_reduce_bytes,
            "row_union_all_reduce_calls": row_union_all_reduce_calls,
            "row_union_all_reduce_bytes": row_union_all_reduce_bytes,
            "row_union_candidate_rows": row_union_candidate_rows,
            "compressed_sparse_gradient_transport": config.memory_update_policy == MemoryUpdatePolicy::SparseRows,
            "memory_table_parameter_indices_rank0": rank0.get("memory_table_parameter_indices").cloned().unwrap_or(serde_json::Value::Null),
            "memory_table_parameter_count": rank0.get("memory_table_parameter_count").cloned().unwrap_or(serde_json::Value::Null),
            "memory_gradient_parameter_counts": memory_gradient_parameter_counts,
            "memory_table_checksum_sums": memory_table_checksum_sums,
            "memory_table_checksum_sumsq": memory_table_checksum_sumsq,
            "memory_table_checksum_sum_max_error": memory_table_checksum_sum_max_error,
            "memory_table_checksum_sumsq_max_error": memory_table_checksum_sumsq_max_error,
            "step_checksum_drift": step_checksum_drift,
            "step_memory_checksum_sum_max_error": step_memory_sum_max_error,
            "step_memory_checksum_sumsq_max_error": step_memory_sumsq_max_error,
            "parameter_checksum_tolerance": DDP_PARAMETER_CHECKSUM_TOLERANCE,
            "cuda_memory_kernels_rank0": rank0.get("cuda_memory_kernels").cloned().unwrap_or(serde_json::Value::Null),
            "tensor_core_rank0": rank0.get("tensor_core").cloned().unwrap_or(serde_json::Value::Null),
            "cuda_runtime_rank0": rank0.get("cuda_runtime").cloned().unwrap_or(serde_json::Value::Null),
            "amp_bf16_policy": rank0.get("amp_bf16_policy").cloned().unwrap_or(serde_json::Value::Null),
            "amp_bf16_validation": rank0.get("amp_bf16_validation").cloned().unwrap_or(serde_json::Value::Null),
            "ranks": rank_reports,
        });
        write_json_file(&report_path, json)?;
    }
    println!(
        "distributed memory training complete mode=nccl precision={} world_size={} checkpoint={}",
        config.precision.as_str(),
        devices.len(),
        config.checkpoint.display()
    );
    Ok(())
}

fn run_distributed_train_lm(config: DistributedTrainLmConfig) -> Result<()> {
    validate_grad_accumulation_steps(config.grad_accumulation_steps)?;
    let Some(DistributedMode::Nccl) = config.distributed else {
        return Err(TensorError::InvalidOperation(
            "distributed training requires --distributed nccl".to_string(),
        ));
    };
    let devices_value = config.devices.as_ref().ok_or_else(|| {
        TensorError::InvalidOperation(
            "distributed training requires --devices cuda:<id>,...".to_string(),
        )
    })?;
    let devices = parse_cuda_devices(devices_value)?;
    if devices.len() < 2 {
        return Err(TensorError::InvalidOperation(format!(
            "distributed NCCL training requires at least two CUDA devices, got {}",
            devices.len()
        )));
    }
    for device_id in &devices {
        ensure_precision_supported_for_device(config.precision, Device::Cuda(*device_id))?;
    }
    if config.precision != Precision::AmpBf16 {
        eprintln!(
            "warning: distributed NCCL training is intended for --precision amp-bf16; running {}",
            config.precision.as_str()
        );
    }

    eprintln!(
        "distributed parent requesting helper-created NCCL unique id world_size={} devices={:?}",
        devices.len(),
        devices
    );
    let exe = std::env::current_exe()
        .map_err(|err| TensorError::Io(format!("failed to locate current executable: {err}")))?;
    let env = launcher_env_from_process();
    let nccl_id_helper = start_nccl_unique_id_helper(&exe, &env)?;
    let nccl_id_hex = nccl_id_helper.hex().to_string();
    eprintln!(
        "distributed parent received helper-created NCCL unique id; spawning {} ranks",
        devices.len()
    );
    let (work_dir, launcher_report_path) =
        launcher_work_paths(config.report.as_deref(), "ddp-ranks");
    fs::create_dir_all(&work_dir).map_err(|err| {
        TensorError::Io(format!("failed to create {}: {err}", work_dir.display()))
    })?;

    let mut specs = Vec::with_capacity(devices.len());
    for (rank, device_id) in devices.iter().copied().enumerate() {
        let rank_config_path = work_dir.join(format!("rank-{rank}.json"));
        let rank_report_path = work_dir.join(format!("rank-{rank}-report.json"));
        let rank_stage_path = work_dir.join(format!("rank-{rank}-stage.json"));
        let rank_config = DdpRankConfig {
            data: config.data.clone(),
            tokenizer: config.tokenizer.clone(),
            dataset_manifest: config.dataset_manifest.clone(),
            checkpoint: config.checkpoint.clone(),
            steps: config.steps,
            batch_size: config.batch_size,
            grad_accumulation_steps: config.grad_accumulation_steps,
            block_size: config.block_size,
            d_model: config.d_model,
            n_heads: config.n_heads,
            ff_hidden: config.ff_hidden,
            lr: config.lr,
            weight_decay: config.weight_decay,
            clip_norm: config.clip_norm,
            seed: config.seed,
            precision: config.precision,
            resume: config.resume,
            log_every: config.log_every,
            rank,
            world_size: devices.len(),
            device_id,
            nccl_unique_id_hex: nccl_id_hex.clone(),
            report_path: rank_report_path.clone(),
            stage_path: rank_stage_path.clone(),
            save_checkpoint: rank == 0,
            ddp_checksum_every: config.ddp_checksum_every,
        };
        write_rank_config(&rank_config_path, &rank_config)?;
        specs.push(LauncherRankSpec {
            rank,
            device_id,
            executable: exe.clone(),
            args: vec![
                "train-lm-rank".to_string(),
                "--config".to_string(),
                rank_config_path.display().to_string(),
            ],
            env: env.clone(),
            config_path: rank_config_path,
            report_path: rank_report_path,
            stage_path: rank_stage_path,
            stdout_path: work_dir.join(format!("rank-{rank}.stdout.log")),
            stderr_path: work_dir.join(format!("rank-{rank}.stderr.log")),
        });
    }

    let launcher = DistributedLauncher {
        name: "ddp-train-lm".to_string(),
        launcher_report_path,
        timeout: None,
        rank_start_timeout: Duration::from_secs(10),
        nccl_init_timeout: Some(config.ddp_init_timeout),
        kill_grace: Duration::from_secs(5),
    };
    let outcome = launcher.run(specs)?;

    let rank_reports = (0..devices.len())
        .map(|rank| read_json_value(&work_dir.join(format!("rank-{rank}-report.json"))))
        .collect::<Result<Vec<_>>>()?;
    let rank0 = &rank_reports[0];
    let all_reduce_calls = sum_rank_u64(&rank_reports, "all_reduce_calls");
    let all_reduce_bytes = sum_rank_u64(&rank_reports, "all_reduce_bytes");
    let tensor_core_probe_calls = sum_rank_u64(&rank_reports, "bf16_mma_probe_calls");
    let tensor_core_matmul_calls = sum_rank_u64(&rank_reports, "bf16_tensor_core_matmul_calls");
    let tensor_core_matmul_forward_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_matmul_forward_calls");
    let tensor_core_matmul_backward_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_matmul_backward_calls");
    let tensor_core_attention_forward_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_attention_forward_calls");
    let tensor_core_attention_qk_matmul_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_attention_qk_matmul_calls");
    let tensor_core_attention_av_matmul_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_attention_av_matmul_calls");
    let tensor_core_attention_backward_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_attention_backward_calls");
    let tensor_core_attention_score_grad_matmul_calls = sum_rank_u64(
        &rank_reports,
        "bf16_tensor_core_attention_score_grad_matmul_calls",
    );
    let tensor_core_attention_dq_matmul_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_attention_dq_matmul_calls");
    let tensor_core_attention_dk_matmul_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_attention_dk_matmul_calls");
    let tensor_core_attention_dv_matmul_calls =
        sum_rank_u64(&rank_reports, "bf16_tensor_core_attention_dv_matmul_calls");
    let scalar_matmul_fallback_calls =
        sum_rank_u64(&rank_reports, "bf16_scalar_matmul_fallback_calls");
    let cuda_runtime = aggregate_rank_cuda_runtime(&rank_reports);
    let tensor_core_counters_json = serde_json::json!({
        "bf16_mma_probe_calls": tensor_core_probe_calls,
        "bf16_tensor_core_matmul_calls": tensor_core_matmul_calls,
        "bf16_tensor_core_matmul_forward_calls": tensor_core_matmul_forward_calls,
        "bf16_tensor_core_matmul_backward_calls": tensor_core_matmul_backward_calls,
        "bf16_tensor_core_attention_forward_calls": tensor_core_attention_forward_calls,
        "bf16_tensor_core_attention_qk_matmul_calls": tensor_core_attention_qk_matmul_calls,
        "bf16_tensor_core_attention_av_matmul_calls": tensor_core_attention_av_matmul_calls,
        "bf16_tensor_core_attention_backward_calls": tensor_core_attention_backward_calls,
        "bf16_tensor_core_attention_score_grad_matmul_calls": tensor_core_attention_score_grad_matmul_calls,
        "bf16_tensor_core_attention_dq_matmul_calls": tensor_core_attention_dq_matmul_calls,
        "bf16_tensor_core_attention_dk_matmul_calls": tensor_core_attention_dk_matmul_calls,
        "bf16_tensor_core_attention_dv_matmul_calls": tensor_core_attention_dv_matmul_calls,
        "bf16_scalar_matmul_fallback_calls": scalar_matmul_fallback_calls,
    });
    let tensor_core_coverage = rank0
        .get("tensor_core_coverage")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let tensor_core_pad_crop = tensor_core_pad_crop_report_from_json(
        &cuda_runtime,
        &tensor_core_counters_json,
        &tensor_core_coverage,
        "distributed_rank0_shapes_summed_runtime_counters",
    );
    let parameter_checksum_sums = rank_f64_values(&rank_reports, "parameter_checksum_sum");
    let parameter_checksum_sumsq = rank_f64_values(&rank_reports, "parameter_checksum_sumsq");
    let parameter_checksum_sum_max_error = max_rank_drift(&parameter_checksum_sums);
    let parameter_checksum_sumsq_max_error = max_rank_drift(&parameter_checksum_sumsq);
    let step_checksum_drift = ddp_step_checksum_drifts(&rank_reports)?;
    let step_checksum_sum_max_error =
        max_step_checksum_error(&step_checksum_drift, "parameter_checksum_sum_max_error");
    let step_checksum_sumsq_max_error =
        max_step_checksum_error(&step_checksum_drift, "parameter_checksum_sumsq_max_error");
    if all_reduce_calls == 0 || all_reduce_bytes == 0 {
        return Err(TensorError::Device(format!(
            "distributed NCCL training completed without recorded gradient all-reduces: \
             all_reduce_calls={all_reduce_calls} all_reduce_bytes={all_reduce_bytes}"
        )));
    }
    if parameter_checksum_sum_max_error > DDP_PARAMETER_CHECKSUM_TOLERANCE
        || parameter_checksum_sumsq_max_error > DDP_PARAMETER_CHECKSUM_TOLERANCE
        || step_checksum_sum_max_error > DDP_PARAMETER_CHECKSUM_TOLERANCE
        || step_checksum_sumsq_max_error > DDP_PARAMETER_CHECKSUM_TOLERANCE
    {
        return Err(TensorError::Device(format!(
            "distributed NCCL parameter checksum drift exceeded tolerance: \
             final_sum_error={parameter_checksum_sum_max_error:.6e} \
             final_sumsq_error={parameter_checksum_sumsq_max_error:.6e} \
             step_sum_error={step_checksum_sum_max_error:.6e} \
             step_sumsq_error={step_checksum_sumsq_max_error:.6e} \
             tolerance={DDP_PARAMETER_CHECKSUM_TOLERANCE:.6e}"
        )));
    }
    let loader_report = rank0_loader_report(&rank_reports);
    let performance_report = aggregate_rank_performance(&rank_reports);
    if let Some(report_path) = config.report {
        let json = serde_json::json!({
            "command": "train-lm",
            "model_family": "tiny_transformer",
            "distributed": "nccl",
            "status": "passed",
            "precision": config.precision.as_str(),
            "learning_rate": config.lr,
            "world_size": devices.len(),
            "devices": devices.iter().map(|id| format!("cuda:{id}")).collect::<Vec<_>>(),
            "per_rank_batch_size": config.batch_size,
            "global_batch_size": config.batch_size * devices.len(),
            "grad_accumulation_steps": config.grad_accumulation_steps,
            "per_rank_effective_batch_size": config.batch_size * config.grad_accumulation_steps,
            "global_micro_batch_size": config.batch_size * devices.len(),
            "global_effective_batch_size": config.batch_size * config.grad_accumulation_steps * devices.len(),
            "ddp_checksum_every": config.ddp_checksum_every,
            "launcher_report": outcome.report_path.display().to_string(),
            "work_dir": work_dir.display().to_string(),
            "checkpoint": config.checkpoint.display().to_string(),
            "start_step": rank0.get("start_step").cloned().unwrap_or(serde_json::Value::Null),
            "final_step": rank0.get("final_step").cloned().unwrap_or(serde_json::Value::Null),
            "initial_loss": rank0.get("initial_loss").cloned().unwrap_or(serde_json::Value::Null),
            "final_loss": rank0.get("final_loss").cloned().unwrap_or(serde_json::Value::Null),
            "loss_reduction": rank0.get("loss_reduction").cloned().unwrap_or(serde_json::Value::Null),
            "rank_loss_reductions": rank_f64_values(&rank_reports, "loss_reduction"),
            "all_reduce_calls": all_reduce_calls,
            "all_reduce_bytes": all_reduce_bytes,
            "loader": loader_report,
            "performance": performance_report,
            "tensor_core": tensor_core_counters_json,
            "cuda_runtime": cuda_runtime,
            "tensor_core_pad_crop": tensor_core_pad_crop,
            "amp_bf16_policy": rank0.get("amp_bf16_policy").cloned().unwrap_or(serde_json::Value::Null),
            "amp_bf16_op_decisions_rank0": rank0.get("amp_bf16_op_decisions").cloned().unwrap_or(serde_json::Value::Null),
            "amp_bf16_finite_checks_rank0": rank0.get("amp_bf16_finite_checks").cloned().unwrap_or(serde_json::Value::Null),
            "amp_bf16_validation": rank0.get("amp_bf16_validation").cloned().unwrap_or(serde_json::Value::Null),
            "amp_bf16_validation_by_rank": rank_reports
                .iter()
                .map(|rank| rank.get("amp_bf16_validation").cloned().unwrap_or(serde_json::Value::Null))
                .collect::<Vec<_>>(),
            "cuda_host_staging_rank0": rank0.get("cuda_host_staging").cloned().unwrap_or(serde_json::Value::Null),
            "tensor_core_coverage": tensor_core_coverage,
            "parameter_checksum_sums": parameter_checksum_sums,
            "parameter_checksum_sumsq": parameter_checksum_sumsq,
            "parameter_checksum_sum_max_error": parameter_checksum_sum_max_error,
            "parameter_checksum_sumsq_max_error": parameter_checksum_sumsq_max_error,
            "step_checksum_drift": step_checksum_drift,
            "step_checksum_sum_max_error": step_checksum_sum_max_error,
            "step_checksum_sumsq_max_error": step_checksum_sumsq_max_error,
            "parameter_checksum_tolerance": DDP_PARAMETER_CHECKSUM_TOLERANCE,
            "ranks": rank_reports,
        });
        write_json_file(&report_path, json)?;
    }
    println!(
        "distributed training complete mode=nccl precision={} world_size={} checkpoint={}",
        config.precision.as_str(),
        devices.len(),
        config.checkpoint.display()
    );
    Ok(())
}

fn run_train_lm_rank(config_path: &Path) -> Result<()> {
    let config: DdpRankConfig = read_json_typed(config_path)?;
    let result = run_train_lm_rank_inner(&config);
    if let Err(err) = &result {
        let _ = write_rank_stage(
            &config.stage_path,
            config.rank,
            "failed",
            Some(&err.to_string()),
        );
    }
    result
}

fn run_train_lm_rank_inner(config: &DdpRankConfig) -> Result<()> {
    validate_grad_accumulation_steps(config.grad_accumulation_steps)?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "config_loaded",
        Some("DDP train rank config loaded"),
    )?;
    let train_device = Device::Cuda(config.device_id);
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "cuda_context_start",
        Some("allocating and copying CUDA DDP context probe buffer"),
    )?;
    let context_probe =
        cuda::CudaBuffer::from_f32(config.device_id as i32, &[0.0]).map_err(cuda_error)?;
    let _ = context_probe.to_f32().map_err(cuda_error)?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "cuda_context_ready",
        Some("CUDA DDP context allocation/copy round trip completed"),
    )?;
    ensure_precision_supported_for_device(config.precision, train_device)?;
    eprintln!(
        "rank={} device=cuda:{} initializing NCCL world_size={}",
        config.rank, config.device_id, config.world_size
    );
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "nccl_library_loaded",
        Some("NCCL init prerequisites ready; communicator will load NCCL"),
    )?;
    let nccl_id = cuda::NcclUniqueId::from_hex(&config.nccl_unique_id_hex)
        .map_err(|err| TensorError::Device(format!("invalid NCCL unique id: {err}")))?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "nccl_init_start",
        Some("entering ncclCommInitRank"),
    )?;
    let mut communicator = cuda::NcclCommunicator::init_rank(
        config.device_id as i32,
        config.rank as i32,
        config.world_size as i32,
        nccl_id,
    )
    .map_err(|err| {
        TensorError::Device(format!(
            "failed to initialize NCCL rank {} on cuda:{}: {err}",
            config.rank, config.device_id
        ))
    })?;
    let nccl_version = communicator.version().map_err(cuda_error)?;
    eprintln!(
        "rank={} device=cuda:{} NCCL communicator ready version={}",
        config.rank, config.device_id, nccl_version
    );
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "nccl_ready",
        Some("NCCL communicator initialized"),
    )?;
    let prepared = load_prepared_from_cli_or_checkpoint(
        &config.dataset_manifest,
        &config.checkpoint,
        config.resume,
    )?;
    let (model, mut optimizer, tokenizer, start_step) =
        if config.resume && config.checkpoint.exists() {
            let loaded = load_lm_checkpoint_on_device(&config.checkpoint, train_device)?;
            if let Some(prepared) = &prepared {
                validate_checkpoint_tokenizer_matches_manifest(&loaded.tokenizer, prepared)?;
            }
            (
                loaded.model,
                loaded.optimizer,
                loaded.tokenizer,
                loaded.metadata.step,
            )
        } else {
            let tokenizer = if let Some(prepared) = &prepared {
                prepared.load_tokenizer()?
            } else {
                let tokenizer_path = require_path(config.tokenizer.as_ref(), "--tokenizer")?;
                BpeTokenizer::load(tokenizer_path)?
            };
            let model_config = TinyTransformerConfig {
                vocab_size: tokenizer.vocab_size(),
                block_size: config.block_size,
                d_model: config.d_model,
                n_heads: config.n_heads,
                ff_hidden: config.ff_hidden,
            };
            let mut rng = HeirloomRng::new(config.seed);
            let model = TinyTransformerLm::new(model_config, &mut rng)?.to_device(train_device)?;
            let optimizer = AdamW::new(model.parameters(), config.lr)?
                .with_weight_decay(config.weight_decay)?
                .with_clip_norm(Some(config.clip_norm))?;
            (model, optimizer, tokenizer, 0)
        };

    let dense_flops_per_token =
        dense_training_flops_per_token_estimate(tiny_dense_parameter_estimate(&model.config));
    let mut dataset = build_sharded_train_dataset(
        prepared.as_ref(),
        config.data.as_ref(),
        &tokenizer,
        model.config.block_size,
        config.seed,
    )?;

    let mut initial_loss = None;
    let mut final_loss = 0.0;
    let mut all_reduce_calls = 0usize;
    let mut all_reduce_bytes = 0usize;
    let mut step_checksums = Vec::new();
    let mut timings = TrainingTimings::default();
    let train_start = Instant::now();
    cuda::reset_tensor_core_counters();
    cuda::reset_cuda_runtime_counters();
    reset_amp_bf16_tensor_core_coverage();
    let _amp_guard = (config.precision == Precision::AmpBf16).then(amp::enter_amp_bf16_training);
    for local_step in 0..config.steps {
        optimizer.zero_grad();
        let global_step = start_step + local_step;
        let mut step_loss_sum = 0.0;
        for micro_step in 0..config.grad_accumulation_steps {
            let sample_step = ddp_sample_step(
                start_step,
                local_step,
                micro_step,
                config.grad_accumulation_steps,
            )?;
            let dataloader_start = Instant::now();
            let (input, target) = dataset.deterministic_sharded_batch(
                config.batch_size,
                config.seed,
                sample_step,
                config.rank,
                config.world_size,
            )?;
            timings.add_dataloader(dataloader_start.elapsed());
            let h2d_start = Instant::now();
            let h2d_timer = start_cuda_compute_timer(train_device)?;
            let input = input.to_device(train_device)?;
            let target = target.to_device(train_device)?;
            let h2d_cuda_ms = stop_cuda_timer(h2d_timer)?;
            timings.add_host_to_device(h2d_start.elapsed());
            timings.add_host_to_device_cuda_ms(h2d_cuda_ms);
            let forward_backward_start = Instant::now();
            let forward_backward_timer = start_cuda_compute_timer(train_device)?;
            let loss = match config.precision {
                Precision::F32 => model.loss(&input, &target)?,
                Precision::Bf16 => model.loss_bf16_activations(&input, &target)?,
                Precision::AmpBf16 => model.loss_amp_bf16(&input, &target)?,
            };
            let loss_value = training_loss_scalar(&loss, config.precision)?;
            step_loss_sum += loss_value;
            loss_for_grad_accumulation(&loss, config.grad_accumulation_steps)?.backward()?;
            let forward_backward_cuda_ms = stop_cuda_timer(forward_backward_timer)?;
            timings.add_forward_backward(forward_backward_start.elapsed());
            timings.add_forward_backward_cuda_ms(forward_backward_cuda_ms);
        }
        let step_loss = step_loss_sum / config.grad_accumulation_steps as f64;
        initial_loss.get_or_insert(step_loss);
        final_loss = step_loss;
        write_rank_stage(
            &config.stage_path,
            config.rank,
            "all_reduce_start",
            Some(&format!(
                "entering gradient NCCL all-reduce step={}",
                global_step + 1
            )),
        )?;
        let all_reduce_start = Instant::now();
        let all_reduce_timer = start_nccl_timer(&communicator)?;
        let stats =
            optimizer.all_reduce_cuda_gradients_nccl(&mut communicator, config.world_size)?;
        let all_reduce_cuda_ms = stop_cuda_timer(all_reduce_timer)?;
        timings.add_all_reduce(all_reduce_start.elapsed());
        timings.add_all_reduce_cuda_ms(all_reduce_cuda_ms);
        write_rank_stage(
            &config.stage_path,
            config.rank,
            "all_reduce_done",
            Some(&format!(
                "completed gradient NCCL all-reduce step={}",
                global_step + 1
            )),
        )?;
        all_reduce_calls += stats.all_reduce_calls;
        all_reduce_bytes += stats.all_reduce_bytes;
        let optimizer_start = Instant::now();
        let optimizer_timer = start_cuda_compute_timer(train_device)?;
        optimizer.step_mut()?;
        let optimizer_cuda_ms = stop_cuda_timer(optimizer_timer)?;
        timings.add_optimizer(optimizer_start.elapsed());
        timings.add_optimizer_cuda_ms(optimizer_cuda_ms);

        let printed_step = global_step + 1;
        if config.ddp_checksum_every > 0
            && (printed_step % config.ddp_checksum_every == 0 || local_step + 1 == config.steps)
        {
            let (parameter_checksum_sum, parameter_checksum_sumsq) =
                amp::with_cuda_host_staging_allowed(
                    "amp-bf16 ddp checksum scalar logging",
                    || model_parameter_checksums(&model),
                )?;
            step_checksums.push(serde_json::json!({
                "step": printed_step,
                "parameter_checksum_sum": parameter_checksum_sum,
                "parameter_checksum_sumsq": parameter_checksum_sumsq,
            }));
        }
        if config.log_every > 0 && (local_step == 0 || printed_step % config.log_every == 0) {
            println!(
                "rank={} step={} loss={:.6} all_reduce_calls={} all_reduce_bytes={}",
                config.rank, printed_step, step_loss, all_reduce_calls, all_reduce_bytes
            );
        }
    }
    let train_elapsed = train_start.elapsed();

    if config.save_checkpoint {
        checkpoint_with_amp_staging_allowed(config.precision, || {
            save_lm_checkpoint_with_dataset_state(
                &config.checkpoint,
                &model,
                &optimizer,
                &tokenizer,
                TokenDatasetState {
                    rng_state: config.seed,
                    batches_seen: (start_step + config.steps) * config.grad_accumulation_steps,
                },
                prepared
                    .as_ref()
                    .map(|prepared| prepared.manifest_path.display().to_string()),
            )
        })?;
    }
    let initial_loss = initial_loss.unwrap_or(final_loss);
    let loss_reduction = if initial_loss == 0.0 {
        0.0
    } else {
        (initial_loss - final_loss) / initial_loss
    };
    let tensor_core_counters = cuda::tensor_core_counters();
    let cuda_runtime_counters = cuda::cuda_runtime_counters();
    let tensor_core_coverage = amp_bf16_tensor_core_coverage();
    let loader_report = dataset.loader_report();
    let tokens_seen = accumulated_training_tokens_seen(
        config.steps,
        config.batch_size,
        model.config.block_size,
        config.grad_accumulation_steps,
    )?;
    let performance_report = training_performance_report(TrainingPerformanceInput {
        tokens_seen,
        train_elapsed,
        timings,
        dense_flops_per_token,
        device_count: 1,
        micro_batch_size: config.batch_size,
        grad_accumulation_steps: config.grad_accumulation_steps,
        data_parallel_world_size: 1,
    });
    let (parameter_checksum_sum, parameter_checksum_sumsq) =
        amp::with_cuda_host_staging_allowed("amp-bf16 final ddp checksum scalar logging", || {
            model_parameter_checksums(&model)
        })?;
    write_json_file(
        &config.report_path,
        serde_json::json!({
            "rank": config.rank,
            "world_size": config.world_size,
            "device": format!("cuda:{}", config.device_id),
            "device_pci_bus_id": cuda_device_pci_bus_id(config.device_id)?,
            "distributed": "nccl",
            "nccl_version": nccl_version,
            "model_family": "tiny_transformer",
            "precision": config.precision.as_str(),
            "learning_rate": config.lr,
            "per_rank_batch_size": config.batch_size,
            "global_batch_size": config.batch_size * config.world_size,
            "grad_accumulation_steps": config.grad_accumulation_steps,
            "per_rank_effective_batch_size": config.batch_size * config.grad_accumulation_steps,
            "global_micro_batch_size": config.batch_size * config.world_size,
            "global_effective_batch_size": config.batch_size * config.grad_accumulation_steps * config.world_size,
            "start_step": start_step,
            "final_step": optimizer.step_index(),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": loss_reduction,
            "all_reduce_calls": all_reduce_calls,
            "all_reduce_bytes": all_reduce_bytes,
            "loader": loader_report,
            "performance": performance_report,
            "ddp_checksum_every": config.ddp_checksum_every,
            "step_checksums": step_checksums,
            "parameter_checksum_sum": parameter_checksum_sum,
            "parameter_checksum_sumsq": parameter_checksum_sumsq,
            "bf16_mma_probe_calls": tensor_core_counters.bf16_mma_probe_calls,
            "bf16_tensor_core_matmul_calls": tensor_core_counters.bf16_tensor_core_matmul_calls,
            "bf16_tensor_core_matmul_forward_calls": tensor_core_counters.bf16_tensor_core_matmul_forward_calls,
            "bf16_tensor_core_matmul_backward_calls": tensor_core_counters.bf16_tensor_core_matmul_backward_calls,
            "bf16_tensor_core_attention_forward_calls": tensor_core_counters.bf16_tensor_core_attention_forward_calls,
            "bf16_tensor_core_attention_qk_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_qk_matmul_calls,
            "bf16_tensor_core_attention_av_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_av_matmul_calls,
            "bf16_tensor_core_attention_backward_calls": tensor_core_counters.bf16_tensor_core_attention_backward_calls,
            "bf16_tensor_core_attention_score_grad_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_score_grad_matmul_calls,
            "bf16_tensor_core_attention_dq_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_dq_matmul_calls,
            "bf16_tensor_core_attention_dk_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_dk_matmul_calls,
            "bf16_tensor_core_attention_dv_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_dv_matmul_calls,
            "bf16_scalar_matmul_fallback_calls": tensor_core_counters.bf16_scalar_matmul_fallback_calls,
            "cuda_runtime": cuda_runtime_counters_json(cuda_runtime_counters),
            "amp_bf16_policy": amp_bf16_policy(),
            "amp_bf16_op_decisions": amp_bf16_op_decisions(),
            "amp_bf16_finite_checks": amp::amp_bf16_finite_check_events(),
            "amp_bf16_validation": amp::amp_bf16_validation_report(&amp_bf16_policy()),
            "cuda_host_staging": amp_bf16_cuda_host_staging_events(),
            "tensor_core_coverage": tensor_core_coverage,
            "checkpoint_saved": config.save_checkpoint,
            "checkpoint": config.checkpoint.display().to_string(),
        }),
    )?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "report_written",
        Some("DDP train rank report written"),
    )?;
    Ok(())
}

fn run_train_memory_lm_rank(config_path: &Path) -> Result<()> {
    let config: DdpMemoryRankConfig = read_json_typed(config_path)?;
    let result = run_train_memory_lm_rank_inner(&config);
    if let Err(err) = &result {
        let _ = write_rank_stage(
            &config.stage_path,
            config.rank,
            "failed",
            Some(&err.to_string()),
        );
    }
    result
}

fn run_train_memory_lm_rank_inner(config: &DdpMemoryRankConfig) -> Result<()> {
    validate_grad_accumulation_steps(config.grad_accumulation_steps)?;
    if config.smft_mode == SmftMode::FreezeDenseUpdateMemory {
        return Err(TensorError::InvalidOperation(
            "DDP memory rank does not support freeze-dense SMFT yet".to_string(),
        ));
    }
    if config.smft_mode == SmftMode::MaskedMemoryRows && config.smft_row_mask.is_none() {
        return Err(TensorError::InvalidOperation(
            "DDP masked-memory-rows SMFT requires smft_row_mask in rank config".to_string(),
        ));
    }
    if config.smft_row_mask.is_some()
        && config.memory_update_policy != MemoryUpdatePolicy::SparseRows
    {
        return Err(TensorError::InvalidOperation(
            "DDP smft_row_mask requires SparseRows memory updates".to_string(),
        ));
    }
    if config.memory_update_policy == MemoryUpdatePolicy::Frozen {
        return Err(TensorError::InvalidOperation(
            "DDP memory rank requires Full, MemoryOnly, or SparseRows updates".to_string(),
        ));
    }
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "config_loaded",
        Some("DDP memory train rank config loaded"),
    )?;
    let train_device = Device::Cuda(config.device_id);
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "cuda_context_start",
        Some("allocating and copying CUDA memory DDP context probe buffer"),
    )?;
    let context_probe =
        cuda::CudaBuffer::from_f32(config.device_id as i32, &[0.0]).map_err(cuda_error)?;
    let _ = context_probe.to_f32().map_err(cuda_error)?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "cuda_context_ready",
        Some("CUDA memory DDP context allocation/copy round trip completed"),
    )?;
    ensure_precision_supported_for_device(config.precision, train_device)?;
    eprintln!(
        "memory rank={} device=cuda:{} initializing NCCL world_size={}",
        config.rank, config.device_id, config.world_size
    );
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "nccl_library_loaded",
        Some("NCCL init prerequisites ready; communicator will load NCCL"),
    )?;
    let nccl_id = cuda::NcclUniqueId::from_hex(&config.nccl_unique_id_hex)
        .map_err(|err| TensorError::Device(format!("invalid NCCL unique id: {err}")))?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "nccl_init_start",
        Some("entering ncclCommInitRank for memory rank"),
    )?;
    let mut communicator = cuda::NcclCommunicator::init_rank(
        config.device_id as i32,
        config.rank as i32,
        config.world_size as i32,
        nccl_id,
    )
    .map_err(|err| {
        TensorError::Device(format!(
            "failed to initialize memory NCCL rank {} on cuda:{}: {err}",
            config.rank, config.device_id
        ))
    })?;
    let nccl_version = communicator.version().map_err(cuda_error)?;
    eprintln!(
        "memory rank={} device=cuda:{} NCCL communicator ready version={}",
        config.rank, config.device_id, nccl_version
    );
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "nccl_ready",
        Some("NCCL communicator initialized for memory rank"),
    )?;

    let prepared = load_prepared_from_cli_or_checkpoint(
        &config.dataset_manifest,
        &config.checkpoint,
        config.resume,
    )?;
    let (model, mut optimizer, tokenizer, start_step) = if config.resume
        && config.checkpoint.exists()
    {
        let loaded = load_memory_lm_checkpoint_on_device(&config.checkpoint, train_device)?;
        if let Some(prepared) = &prepared {
            validate_checkpoint_tokenizer_matches_manifest(&loaded.tokenizer, prepared)?;
        }
        (
            loaded.model,
            loaded.optimizer,
            loaded.tokenizer,
            loaded.metadata.step,
        )
    } else {
        let tokenizer = if let Some(prepared) = &prepared {
            prepared.load_tokenizer()?
        } else {
            let tokenizer_path = require_path(config.tokenizer.as_ref(), "--tokenizer")?;
            BpeTokenizer::load(tokenizer_path)?
        };
        let model_config = MemoryTransformerConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: config.block_size,
            n_layers: config.n_layers,
            d_model: config.d_model,
            n_heads: config.n_heads,
            ff_hidden: config.ff_hidden,
            memory_layer_indices: config.memory_layer_indices.clone(),
            memory_slots: config.memory_slots,
            memory_key_dim: config.memory_key_dim,
            memory_value_dim: config.memory_value_dim,
            memory_top_k: config.memory_top_k,
            memory_heads: config.memory_heads,
            memory_lookup: config.memory_lookup.clone(),
            shared_memory: config.shared_memory,
            memory_plus: config.memory_plus,
            memory_update_policy: config.memory_update_policy.clone(),
            smft_mode: config.smft_mode.clone(),
        };
        let mut rng = HeirloomRng::new(config.seed);
        let model = MemoryTransformerLm::new(model_config, &mut rng)?.to_device(train_device)?;
        let optimizer = AdamW::new(model.parameters(), config.lr)?
            .with_weight_decay(config.weight_decay)?
            .with_clip_norm(Some(config.clip_norm))?;
        (model, optimizer, tokenizer, 0)
    };

    let dense_flops_per_token = dense_training_flops_per_token_estimate(
        memory_dense_parameter_estimate(&model.config, tokenizer.vocab_size()),
    );
    let mut dataset = build_sharded_train_dataset(
        prepared.as_ref(),
        config.data.as_ref(),
        &tokenizer,
        model.config.block_size,
        config.seed,
    )?;
    let ddp_smft_row_mask = config
        .smft_row_mask
        .as_deref()
        .map(|path| load_smft_row_mask(path, &model))
        .transpose()?;
    let ddp_smft_row_mask_source = config
        .smft_row_mask
        .as_ref()
        .map(|path| path.display().to_string());

    let memory_table_parameter_indices = model.memory_table_parameter_indices()?;
    let mut initial_loss = None;
    let mut final_loss = 0.0;
    let mut all_reduce_calls = 0usize;
    let mut all_reduce_bytes = 0usize;
    let mut compact_gradient_all_reduce_calls = 0usize;
    let mut compact_gradient_all_reduce_bytes = 0usize;
    let mut row_union_all_reduce_calls = 0usize;
    let mut row_union_all_reduce_bytes = 0usize;
    let mut row_union_candidate_rows = 0usize;
    let mut step_checksums = Vec::new();
    let mut memory_gradient_parameter_count = 0usize;
    let mut timings = TrainingTimings::default();
    let train_start = Instant::now();
    cuda::reset_tensor_core_counters();
    cuda::reset_cuda_runtime_counters();
    cuda::reset_memory_kernel_counters();
    reset_amp_bf16_tensor_core_coverage();
    let _amp_guard = (config.precision == Precision::AmpBf16).then(amp::enter_amp_bf16_training);
    for local_step in 0..config.steps {
        optimizer.zero_grad();
        let global_step = start_step + local_step;
        let mut step_loss_sum = 0.0;
        for micro_step in 0..config.grad_accumulation_steps {
            let sample_step = ddp_sample_step(
                start_step,
                local_step,
                micro_step,
                config.grad_accumulation_steps,
            )?;
            let dataloader_start = Instant::now();
            let (input, target) = dataset.deterministic_sharded_batch(
                config.batch_size,
                config.seed,
                sample_step,
                config.rank,
                config.world_size,
            )?;
            timings.add_dataloader(dataloader_start.elapsed());
            let h2d_start = Instant::now();
            let h2d_timer = start_cuda_compute_timer(train_device)?;
            let input = input.to_device(train_device)?;
            let target = target.to_device(train_device)?;
            let h2d_cuda_ms = stop_cuda_timer(h2d_timer)?;
            timings.add_host_to_device(h2d_start.elapsed());
            timings.add_host_to_device_cuda_ms(h2d_cuda_ms);
            let forward_backward_start = Instant::now();
            let forward_backward_timer = start_cuda_compute_timer(train_device)?;
            let loss = match config.precision {
                Precision::F32 => model.loss(&input, &target)?,
                Precision::Bf16 => model.loss_bf16_activations(&input, &target)?,
                Precision::AmpBf16 => model.loss_amp_bf16(&input, &target)?,
            };
            let loss_value = training_loss_scalar(&loss, config.precision)?;
            step_loss_sum += loss_value;
            loss_for_grad_accumulation(&loss, config.grad_accumulation_steps)?.backward()?;
            let forward_backward_cuda_ms = stop_cuda_timer(forward_backward_timer)?;
            timings.add_forward_backward(forward_backward_start.elapsed());
            timings.add_forward_backward_cuda_ms(forward_backward_cuda_ms);
            memory_gradient_parameter_count = memory_gradient_parameter_count.max(
                memory_parameter_gradient_count(&model, &memory_table_parameter_indices)?,
            );
        }
        let step_loss = step_loss_sum / config.grad_accumulation_steps as f64;
        initial_loss.get_or_insert(step_loss);
        final_loss = step_loss;
        match config.memory_update_policy {
            MemoryUpdatePolicy::Full => {
                write_rank_stage(
                    &config.stage_path,
                    config.rank,
                    "all_reduce_start",
                    Some(&format!(
                        "entering dense memory gradient NCCL all-reduce step={}",
                        global_step + 1
                    )),
                )?;
                let all_reduce_start = Instant::now();
                let all_reduce_timer = start_nccl_timer(&communicator)?;
                let stats = optimizer
                    .all_reduce_cuda_gradients_nccl(&mut communicator, config.world_size)?;
                let all_reduce_cuda_ms = stop_cuda_timer(all_reduce_timer)?;
                timings.add_all_reduce(all_reduce_start.elapsed());
                timings.add_all_reduce_cuda_ms(all_reduce_cuda_ms);
                write_rank_stage(
                    &config.stage_path,
                    config.rank,
                    "all_reduce_done",
                    Some(&format!(
                        "completed dense memory gradient NCCL all-reduce step={}",
                        global_step + 1
                    )),
                )?;
                all_reduce_calls += stats.all_reduce_calls;
                all_reduce_bytes += stats.all_reduce_bytes;
                let optimizer_start = Instant::now();
                let optimizer_timer = start_cuda_compute_timer(train_device)?;
                optimizer.step_mut()?;
                let optimizer_cuda_ms = stop_cuda_timer(optimizer_timer)?;
                timings.add_optimizer(optimizer_start.elapsed());
                timings.add_optimizer_cuda_ms(optimizer_cuda_ms);
            }
            MemoryUpdatePolicy::MemoryOnly => {
                write_rank_stage(
                    &config.stage_path,
                    config.rank,
                    "all_reduce_start",
                    Some(&format!(
                        "entering dense memory-only gradient NCCL all-reduce step={}",
                        global_step + 1
                    )),
                )?;
                let all_reduce_start = Instant::now();
                let all_reduce_timer = start_nccl_timer(&communicator)?;
                let stats = optimizer
                    .all_reduce_cuda_gradients_nccl(&mut communicator, config.world_size)?;
                let all_reduce_cuda_ms = stop_cuda_timer(all_reduce_timer)?;
                timings.add_all_reduce(all_reduce_start.elapsed());
                timings.add_all_reduce_cuda_ms(all_reduce_cuda_ms);
                write_rank_stage(
                    &config.stage_path,
                    config.rank,
                    "all_reduce_done",
                    Some(&format!(
                        "completed dense memory-only gradient NCCL all-reduce step={}",
                        global_step + 1
                    )),
                )?;
                all_reduce_calls += stats.all_reduce_calls;
                all_reduce_bytes += stats.all_reduce_bytes;
                let optimizer_start = Instant::now();
                let optimizer_timer = start_cuda_compute_timer(train_device)?;
                optimizer.step_parameter_indices_mut(&memory_table_parameter_indices)?;
                let optimizer_cuda_ms = stop_cuda_timer(optimizer_timer)?;
                timings.add_optimizer(optimizer_start.elapsed());
                timings.add_optimizer_cuda_ms(optimizer_cuda_ms);
            }
            MemoryUpdatePolicy::SparseRows => {
                let updates = if let Some(mask) = ddp_smft_row_mask.as_ref() {
                    model.memory_sparse_adamw_updates_with_mask(mask)?
                } else {
                    model.memory_sparse_adamw_updates()?
                };
                if updates.is_empty() {
                    return Err(TensorError::InvalidOperation(
                        "memory sparse-row DDP update requested, but no selected rows were captured; run a memory forward/backward before optimizer step"
                            .to_string(),
                    ));
                }
                write_rank_stage(
                    &config.stage_path,
                    config.rank,
                    "all_reduce_start",
                    Some(&format!(
                        "entering compact sparse memory gradient NCCL all-reduce step={}",
                        global_step + 1
                    )),
                )?;
                let all_reduce_start = Instant::now();
                let all_reduce_timer = start_nccl_timer(&communicator)?;
                let (updates, row_stats, gradient_stats, candidate_rows) =
                    ddp_compact_sparse_gradient_updates(
                        &model,
                        updates,
                        &mut communicator,
                        config.world_size,
                    )?;
                let all_reduce_cuda_ms = stop_cuda_timer(all_reduce_timer)?;
                timings.add_all_reduce(all_reduce_start.elapsed());
                timings.add_all_reduce_cuda_ms(all_reduce_cuda_ms);
                write_rank_stage(
                    &config.stage_path,
                    config.rank,
                    "all_reduce_done",
                    Some(&format!(
                        "completed compact sparse memory gradient NCCL all-reduce step={}",
                        global_step + 1
                    )),
                )?;
                row_union_all_reduce_calls += row_stats.all_reduce_calls;
                row_union_all_reduce_bytes += row_stats.all_reduce_bytes;
                compact_gradient_all_reduce_calls += gradient_stats.all_reduce_calls;
                compact_gradient_all_reduce_bytes += gradient_stats.all_reduce_bytes;
                all_reduce_calls += gradient_stats.all_reduce_calls;
                all_reduce_bytes += gradient_stats.all_reduce_bytes;
                row_union_candidate_rows += candidate_rows;
                let optimizer_start = Instant::now();
                let optimizer_timer = start_cuda_compute_timer(train_device)?;
                optimizer.step_cuda_sparse_compact_rows_mut(&updates)?;
                let optimizer_cuda_ms = stop_cuda_timer(optimizer_timer)?;
                timings.add_optimizer(optimizer_start.elapsed());
                timings.add_optimizer_cuda_ms(optimizer_cuda_ms);
            }
            MemoryUpdatePolicy::Frozen => {
                unreachable!("rejected before rank training loop")
            }
        }

        let printed_step = global_step + 1;
        if config.ddp_checksum_every > 0
            && (printed_step % config.ddp_checksum_every == 0 || local_step + 1 == config.steps)
        {
            let (memory_table_checksum_sum, memory_table_checksum_sumsq) =
                amp::with_cuda_host_staging_allowed(
                    "amp-bf16 memory ddp checksum scalar logging",
                    || memory_table_parameter_checksums(&model),
                )?;
            step_checksums.push(serde_json::json!({
                "step": printed_step,
                "memory_table_checksum_sum": memory_table_checksum_sum,
                "memory_table_checksum_sumsq": memory_table_checksum_sumsq,
            }));
        }
        if config.log_every > 0 && (local_step == 0 || printed_step % config.log_every == 0) {
            println!(
                "memory rank={} step={} loss={:.6} all_reduce_calls={} all_reduce_bytes={} row_union_all_reduce_calls={} compact_gradient_all_reduce_calls={}",
                config.rank, printed_step, step_loss, all_reduce_calls, all_reduce_bytes, row_union_all_reduce_calls, compact_gradient_all_reduce_calls
            );
        }
    }
    let train_elapsed = train_start.elapsed();

    if config.save_checkpoint {
        let final_training_step = start_step + config.steps;
        checkpoint_with_amp_staging_allowed(config.precision, || {
            save_memory_lm_checkpoint_with_dataset_state_and_step(
                &config.checkpoint,
                &model,
                &optimizer,
                &tokenizer,
                TokenDatasetState {
                    rng_state: config.seed,
                    batches_seen: (start_step + config.steps) * config.grad_accumulation_steps,
                },
                prepared
                    .as_ref()
                    .map(|prepared| prepared.manifest_path.display().to_string()),
                final_training_step,
            )
        })?;
    }
    let initial_loss = initial_loss.unwrap_or(final_loss);
    let loss_reduction = if initial_loss == 0.0 {
        0.0
    } else {
        (initial_loss - final_loss) / initial_loss
    };
    let tensor_core_counters = cuda::tensor_core_counters();
    let cuda_runtime_counters = cuda::cuda_runtime_counters();
    let memory_kernel_counters = cuda::memory_kernel_counters();
    let tensor_core_coverage = amp_bf16_tensor_core_coverage();
    let loader_report = dataset.loader_report();
    let tokens_seen = accumulated_training_tokens_seen(
        config.steps,
        config.batch_size,
        model.config.block_size,
        config.grad_accumulation_steps,
    )?;
    let performance_report = training_performance_report(TrainingPerformanceInput {
        tokens_seen,
        train_elapsed,
        timings,
        dense_flops_per_token,
        device_count: 1,
        micro_batch_size: config.batch_size,
        grad_accumulation_steps: config.grad_accumulation_steps,
        data_parallel_world_size: 1,
    });
    let tensor_core_coverage_json = serde_json::to_value(&tensor_core_coverage).map_err(|err| {
        TensorError::Io(format!("failed to serialize Tensor Core coverage: {err}"))
    })?;
    let memory_selection_report = if config.precision == Precision::AmpBf16 {
        amp::with_cuda_host_staging_allowed("amp-bf16 memory ddp selection report", || {
            model.memory_selection_report()
        })
    } else {
        model.memory_selection_report()
    }?;
    let memory_selection_report = serde_json::to_value(memory_selection_report).map_err(|err| {
        TensorError::Io(format!(
            "failed to serialize memory DDP selection report: {err}"
        ))
    })?;
    let smft_access_report = if config.precision == Precision::AmpBf16 {
        amp::with_cuda_host_staging_allowed("amp-bf16 memory ddp SMFT access report", || {
            model.smft_access_report(16)
        })
    } else {
        model.smft_access_report(16)
    }?;
    let smft_access_report = serde_json::to_value(smft_access_report).map_err(|err| {
        TensorError::Io(format!(
            "failed to serialize memory DDP SMFT access report: {err}"
        ))
    })?;
    let (memory_table_checksum_sum, memory_table_checksum_sumsq) =
        amp::with_cuda_host_staging_allowed(
            "amp-bf16 final memory ddp checksum scalar logging",
            || memory_table_parameter_checksums(&model),
        )?;
    write_json_file(
        &config.report_path,
        serde_json::json!({
            "rank": config.rank,
            "world_size": config.world_size,
            "device": format!("cuda:{}", config.device_id),
            "device_pci_bus_id": cuda_device_pci_bus_id(config.device_id)?,
            "distributed": "nccl",
            "nccl_version": nccl_version,
            "model_family": "memory_transformer",
            "memory_config": &model.config,
            "precision": config.precision.as_str(),
            "learning_rate": config.lr,
            "per_rank_batch_size": config.batch_size,
            "global_batch_size": config.batch_size * config.world_size,
            "grad_accumulation_steps": config.grad_accumulation_steps,
            "per_rank_effective_batch_size": config.batch_size * config.grad_accumulation_steps,
            "global_micro_batch_size": config.batch_size * config.world_size,
            "global_effective_batch_size": config.batch_size * config.grad_accumulation_steps * config.world_size,
            "start_step": start_step,
            "final_step": start_step + config.steps,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": loss_reduction,
            "memory_update_policy": config.memory_update_policy.clone(),
            "smft_mode": config.smft_mode.clone(),
            "smft_row_mask_source": ddp_smft_row_mask_source,
            "smft_row_mask_trainable_rows": ddp_smft_row_mask
                .as_ref()
                .map(|mask| mask.trainable_rows.len()),
            "smft_row_mask_frozen_rows": ddp_smft_row_mask
                .as_ref()
                .map(|mask| mask.frozen_rows),
            "all_reduce_calls": all_reduce_calls,
            "all_reduce_bytes": all_reduce_bytes,
            "loader": loader_report,
            "performance": performance_report,
            "compact_gradient_all_reduce_calls": compact_gradient_all_reduce_calls,
            "compact_gradient_all_reduce_bytes": compact_gradient_all_reduce_bytes,
            "row_union_all_reduce_calls": row_union_all_reduce_calls,
            "row_union_all_reduce_bytes": row_union_all_reduce_bytes,
            "row_union_candidate_rows": row_union_candidate_rows,
            "compressed_sparse_gradient_transport": config.memory_update_policy == MemoryUpdatePolicy::SparseRows,
            "ddp_checksum_every": config.ddp_checksum_every,
            "step_checksums": step_checksums,
            "memory_table_parameter_indices": memory_table_parameter_indices,
            "memory_table_parameter_count": model.memory_table_parameter_indices()?.len(),
            "memory_gradient_parameter_count": memory_gradient_parameter_count,
            "memory_table_checksum_sum": memory_table_checksum_sum,
            "memory_table_checksum_sumsq": memory_table_checksum_sumsq,
            "memory_selection": memory_selection_report,
            "smft_access": smft_access_report,
            "tensor_core": {
                "bf16_mma_probe_calls": tensor_core_counters.bf16_mma_probe_calls,
                "bf16_tensor_core_matmul_calls": tensor_core_counters.bf16_tensor_core_matmul_calls,
                "bf16_tensor_core_matmul_forward_calls": tensor_core_counters.bf16_tensor_core_matmul_forward_calls,
                "bf16_tensor_core_matmul_backward_calls": tensor_core_counters.bf16_tensor_core_matmul_backward_calls,
                "bf16_tensor_core_attention_forward_calls": tensor_core_counters.bf16_tensor_core_attention_forward_calls,
                "bf16_tensor_core_attention_qk_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_qk_matmul_calls,
                "bf16_tensor_core_attention_av_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_av_matmul_calls,
                "bf16_tensor_core_attention_backward_calls": tensor_core_counters.bf16_tensor_core_attention_backward_calls,
                "bf16_tensor_core_attention_score_grad_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_score_grad_matmul_calls,
                "bf16_tensor_core_attention_dq_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_dq_matmul_calls,
                "bf16_tensor_core_attention_dk_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_dk_matmul_calls,
                "bf16_tensor_core_attention_dv_matmul_calls": tensor_core_counters.bf16_tensor_core_attention_dv_matmul_calls,
                "bf16_scalar_matmul_fallback_calls": tensor_core_counters.bf16_scalar_matmul_fallback_calls,
            },
            "cuda_runtime": cuda_runtime_counters_json(cuda_runtime_counters),
            "cuda_memory_kernels": {
                "implemented": true,
                "forward_cpu_fallback_for_cuda": false,
                "counters": {
                    "lookup_rejected_calls": memory_kernel_counters.lookup_rejected_calls,
                    "query_key_score_calls": memory_kernel_counters.query_key_score_calls,
                    "topk_calls": memory_kernel_counters.topk_calls,
                    "product_key_calls": memory_kernel_counters.product_key_calls,
                    "softmax_topk_calls": memory_kernel_counters.softmax_topk_calls,
                    "weighted_value_forward_calls": memory_kernel_counters.weighted_value_forward_calls,
                    "weighted_value_backward_calls": memory_kernel_counters.weighted_value_backward_calls,
                    "selected_key_backward_calls": memory_kernel_counters.selected_key_backward_calls,
                    "scatter_add_rows_calls": memory_kernel_counters.scatter_add_rows_calls,
                    "sparse_adamw_rows_calls": memory_kernel_counters.sparse_adamw_rows_calls,
                    "sparse_adamw_compact_rows_calls": memory_kernel_counters.sparse_adamw_compact_rows_calls,
                    "access_count_calls": memory_kernel_counters.access_count_calls,
                    "gather_selected_rows_calls": memory_kernel_counters.gather_selected_rows_calls,
                    "bool_mask_to_indices_calls": memory_kernel_counters.bool_mask_to_indices_calls,
                    "selected_tokens": memory_kernel_counters.selected_tokens,
                    "selected_rows": memory_kernel_counters.selected_rows,
                },
            },
            "amp_bf16_policy": amp_bf16_policy(),
            "amp_bf16_op_decisions": amp_bf16_op_decisions(),
            "amp_bf16_finite_checks": amp::amp_bf16_finite_check_events(),
            "amp_bf16_validation": amp::amp_bf16_validation_report(&amp_bf16_policy()),
            "cuda_host_staging": amp_bf16_cuda_host_staging_events(),
            "tensor_core_coverage": tensor_core_coverage_json,
            "checkpoint_saved": config.save_checkpoint,
            "checkpoint": config.checkpoint.display().to_string(),
        }),
    )?;
    write_rank_stage(
        &config.stage_path,
        config.rank,
        "report_written",
        Some("DDP memory train rank report written"),
    )?;
    Ok(())
}

fn memory_parameter_gradient_count(
    model: &MemoryTransformerLm,
    memory_table_parameter_indices: &[usize],
) -> Result<usize> {
    let parameters = model.parameters();
    let mut count = 0usize;
    for &index in memory_table_parameter_indices {
        let parameter = parameters.get(index).ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "memory table parameter index {index} is out of range for {} parameters",
                parameters.len()
            ))
        })?;
        if parameter.grad_tensor().is_some() {
            count += 1;
        }
    }
    Ok(count)
}

fn ddp_compact_sparse_gradient_updates(
    model: &MemoryTransformerLm,
    updates: Vec<SparseAdamWRowsUpdate>,
    communicator: &mut cuda::NcclCommunicator,
    world_size: usize,
) -> Result<(
    Vec<SparseAdamWCompactRowsUpdate>,
    DistributedGradientStats,
    DistributedGradientStats,
    usize,
)> {
    if world_size == 0 {
        return Err(TensorError::InvalidOperation(
            "world_size must be positive for compact sparse memory gradient all-reduce".to_string(),
        ));
    }
    if communicator.world_size() != world_size as i32 {
        return Err(TensorError::Device(format!(
            "NCCL communicator world_size={} does not match requested world_size={world_size}",
            communicator.world_size()
        )));
    }
    let parameters = model.parameters();
    let mut synchronized = Vec::with_capacity(updates.len());
    let mut row_union_stats = DistributedGradientStats::default();
    let mut gradient_stats = DistributedGradientStats::default();
    let mut candidate_rows = 0usize;
    for update in updates {
        let (union_mask, row_stats) = update
            .selected_rows
            .cuda_row_union_mask_nccl(update.rows, communicator)?;
        let row_mask = if let Some(smft_mask) = update.row_mask.as_ref() {
            union_mask.cuda_bool_and(smft_mask)?
        } else {
            union_mask
        };
        row_union_stats.all_reduce_calls += row_stats.calls;
        row_union_stats.all_reduce_bytes += row_stats.bytes;
        let compact_rows = row_mask.cuda_bool_mask_to_i64_indices()?;
        candidate_rows += compact_rows.numel();
        let parameter = parameters.get(update.parameter_index).ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "compact sparse memory gradient update parameter index {} is out of range for {} parameters",
                update.parameter_index,
                parameters.len()
            ))
        })?;
        let compact_grad_rows = if let Some(grad_rows) =
            parameter.cuda_memory_gather_selected_grad_rows_f32_i64(&compact_rows)?
        {
            grad_rows
        } else {
            let Device::Cuda(device_id) = compact_rows.device() else {
                return Err(TensorError::Device(format!(
                    "compact sparse memory gradient rows must be CUDA selected rows, got {:?}",
                    compact_rows.device()
                )));
            };
            Tensor::cuda_f32_zeros_on_device(&[compact_rows.numel(), update.row_dim], device_id)?
        };
        let reduced = compact_grad_rows
            .cuda_all_reduce_sum_in_place_f32_nccl(communicator, 1.0f32 / world_size as f32)?;
        gradient_stats.all_reduce_calls += reduced.calls;
        gradient_stats.all_reduce_bytes += reduced.bytes;
        synchronized.push(SparseAdamWCompactRowsUpdate {
            parameter_index: update.parameter_index,
            selected_rows: compact_rows,
            compact_grad_rows,
            rows: update.rows,
            row_dim: update.row_dim,
        });
    }
    Ok((
        synchronized,
        row_union_stats,
        gradient_stats,
        candidate_rows,
    ))
}

fn model_parameter_checksums(model: &TinyTransformerLm) -> Result<(f64, f64)> {
    no_grad(|| {
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for parameter in model.parameters() {
            sum += parameter.sum()?.data()[0] as f64;
            sumsq += parameter.mul(&parameter)?.sum()?.data()[0] as f64;
        }
        Ok((sum, sumsq))
    })
}

fn write_rank_config<T: Serialize>(path: &Path, config: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            TensorError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }
    let json = serde_json::to_string_pretty(config)
        .map_err(|err| TensorError::Io(format!("failed to serialize {}: {err}", path.display())))?;
    fs::write(path, json + "\n")
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", path.display())))
}

fn parse_cuda_devices(value: &str) -> Result<Vec<usize>> {
    let mut devices = Vec::new();
    for item in value.split(',') {
        let device = parse_device(item.trim())?;
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "distributed --devices accepts only cuda:<id> entries, got {item:?}"
            )));
        };
        if devices.contains(&device_id) {
            return Err(TensorError::Device(format!(
                "duplicate CUDA device cuda:{device_id} in --devices"
            )));
        }
        devices.push(device_id);
    }
    if devices.is_empty() {
        return Err(TensorError::Device(
            "--devices must include at least one cuda:<id> entry".to_string(),
        ));
    }
    Ok(devices)
}

fn sum_rank_u64(rank_reports: &[serde_json::Value], field: &str) -> u64 {
    rank_reports
        .iter()
        .filter_map(|report| report.get(field).and_then(serde_json::Value::as_u64))
        .sum()
}

struct DistributedMemoryAllReduceEvidence<'a> {
    memory_update_policy: &'a MemoryUpdatePolicy,
    all_reduce_calls: u64,
    all_reduce_bytes: u64,
    row_union_all_reduce_calls: u64,
    row_union_all_reduce_bytes: u64,
    row_union_candidate_rows: u64,
    compact_gradient_all_reduce_calls: u64,
    compact_gradient_all_reduce_bytes: u64,
}

fn validate_distributed_memory_all_reduce_evidence(
    evidence: DistributedMemoryAllReduceEvidence<'_>,
) -> Result<()> {
    if evidence.all_reduce_calls == 0 || evidence.all_reduce_bytes == 0 {
        return Err(TensorError::Device(format!(
            "distributed memory NCCL training completed without recorded gradient all-reduces: \
             all_reduce_calls={} all_reduce_bytes={}",
            evidence.all_reduce_calls, evidence.all_reduce_bytes
        )));
    }
    if evidence.memory_update_policy == &MemoryUpdatePolicy::SparseRows
        && (evidence.row_union_all_reduce_calls == 0
            || evidence.row_union_all_reduce_bytes == 0
            || evidence.row_union_candidate_rows == 0
            || evidence.compact_gradient_all_reduce_calls == 0
            || evidence.compact_gradient_all_reduce_bytes == 0)
    {
        return Err(TensorError::Device(format!(
            "distributed sparse-row memory training completed without recorded compact sparse all-reduces: \
             row_union_all_reduce_calls={} \
             row_union_all_reduce_bytes={} \
             row_union_candidate_rows={} \
             compact_gradient_all_reduce_calls={} \
             compact_gradient_all_reduce_bytes={}",
            evidence.row_union_all_reduce_calls,
            evidence.row_union_all_reduce_bytes,
            evidence.row_union_candidate_rows,
            evidence.compact_gradient_all_reduce_calls,
            evidence.compact_gradient_all_reduce_bytes
        )));
    }
    Ok(())
}

fn aggregate_rank_cuda_runtime(rank_reports: &[serde_json::Value]) -> serde_json::Value {
    let fields = [
        "kernel_launch_calls",
        "kernel_launch_elements",
        "kernel_launch_family_elapsed_us",
        "host_sync_calls",
        "stream_create_calls",
        "stream_sync_calls",
        "event_create_calls",
        "event_record_calls",
        "event_query_calls",
        "event_elapsed_calls",
        "h2d_bytes",
        "d2h_bytes",
        "d2d_bytes",
        "allocation_active_bytes",
        "allocation_reserved_bytes",
        "allocation_high_water_bytes",
        "allocation_calls",
        "allocation_cache_hits",
        "allocation_frees",
        "allocation_deferred_frees",
        "allocation_pending_reclaims",
        "module_load_calls",
        "module_cache_hits",
        "tensor_core_padded_tiles",
        "tensor_core_remainder_tiles",
        "tensor_core_cta_gemm_calls",
        "tensor_core_cta_tiles",
        "tensor_core_cta_warps_launched",
        "tensor_core_mma_warp_tiles",
        "tensor_core_staged_cta_gemm_calls",
        "tensor_core_staged_cta_gemm_elapsed_us",
        "tensor_core_shared_stage_tiles",
        "tensor_core_shared_stage_bytes",
        "tensor_core_wide_swizzled_cta_gemm_calls",
        "tensor_core_wide_swizzled_cta_gemm_elapsed_us",
        "tensor_core_swizzled_stage_tiles",
        "tensor_core_swizzled_stage_bytes",
        "tensor_core_ldmatrix_gemm_requested_calls",
        "tensor_core_ldmatrix_gemm_executed_calls",
        "tensor_core_ldmatrix_gemm_staged_fallback_calls",
        "tensor_core_ldmatrix_gemm_hard_require_failures",
        "tensor_core_ldmatrix_gemm_instructions",
        "tensor_core_ldmatrix_gemm_elapsed_us",
        "tensor_core_cp_async_gemm_requested_calls",
        "tensor_core_cp_async_gemm_executed_calls",
        "tensor_core_cp_async_gemm_staged_fallback_calls",
        "tensor_core_cp_async_gemm_hard_require_failures",
        "tensor_core_cp_async_gemm_instructions",
        "tensor_core_cp_async_gemm_elapsed_us",
        "tensor_core_global_cta_gemm_calls",
        "tensor_core_global_cta_gemm_elapsed_us",
        "tensor_core_legacy_warp_gemm_calls",
        "tensor_core_legacy_warp_gemm_elapsed_us",
        "tensor_core_attention_padded_tiles",
        "tensor_core_attention_remainder_tiles",
        "tensor_core_attention_scalar_fallbacks",
        "tensor_core_attention_ldmatrix_requested_calls",
        "tensor_core_attention_ldmatrix_global_fallback_calls",
        "tensor_core_attention_cp_async_requested_calls",
        "tensor_core_attention_cp_async_global_fallback_calls",
        "bf16_attention_materialized_reference_calls",
        "flash_bf16_attention_requested_calls",
        "flash_bf16_attention_executed_calls",
        "flash_bf16_attention_fallback_calls",
        "flash_bf16_attention_scalar_fallback_calls",
        "flash_bf16_attention_qk_tile_calls",
        "flash_bf16_attention_av_tile_calls",
        "flash_bf16_attention_ragged_tile_count",
        "flash_bf16_attention_causal_masked_tile_count",
        "flash_bf16_attention_elapsed_us",
        "flash_bf16_attention_hard_require_failures",
        "flash_bf16_scalar_streaming_requested_calls",
        "flash_bf16_scalar_streaming_executed_calls",
        "flash_bf16_scalar_streaming_qk_tile_calls",
        "flash_bf16_scalar_streaming_av_tile_calls",
        "flash_bf16_scalar_streaming_elapsed_us",
        "flash_bf16_tensor_core_requested_calls",
        "flash_bf16_tensor_core_executed_calls",
        "flash_bf16_tensor_core_fallback_calls",
        "flash_bf16_tensor_core_qk_mma_tile_calls",
        "flash_bf16_tensor_core_av_mma_tile_calls",
        "flash_bf16_tensor_core_ragged_tile_count",
        "flash_bf16_tensor_core_causal_masked_tile_count",
        "flash_bf16_tensor_core_elapsed_us",
        "flash_bf16_tensor_core_backward_requested_calls",
        "flash_bf16_tensor_core_backward_executed_calls",
        "flash_bf16_tensor_core_backward_fallback_calls",
        "flash_bf16_tensor_core_backward_row_dot_calls",
        "flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls",
        "flash_bf16_tensor_core_backward_dp_mma_tile_calls",
        "flash_bf16_tensor_core_backward_dq_mma_tile_calls",
        "flash_bf16_tensor_core_backward_dk_mma_tile_calls",
        "flash_bf16_tensor_core_backward_dv_mma_tile_calls",
        "flash_bf16_tensor_core_backward_scalar_tile_calls",
        "flash_bf16_tensor_core_backward_ragged_tile_count",
        "flash_bf16_tensor_core_backward_causal_masked_tile_count",
        "flash_bf16_tensor_core_backward_elapsed_us",
    ];
    let mut object = serde_json::Map::new();
    for field in fields {
        let value = rank_reports
            .iter()
            .filter_map(|report| {
                report
                    .get("cuda_runtime")
                    .and_then(|runtime| runtime.get(field))
                    .and_then(serde_json::Value::as_u64)
            })
            .sum::<u64>();
        object.insert(field.to_string(), serde_json::Value::from(value));
    }
    let mut family_totals: std::collections::BTreeMap<String, (u64, u64, u64)> =
        std::collections::BTreeMap::new();
    for report in rank_reports {
        let Some(families) = report
            .get("cuda_runtime")
            .and_then(|runtime| runtime.get("kernel_launch_families"))
            .and_then(serde_json::Value::as_object)
        else {
            continue;
        };
        for (label, stats) in families {
            let calls = stats
                .get("calls")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let elements = stats
                .get("elements")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let elapsed_us = stats
                .get("elapsed_us")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let entry = family_totals.entry(label.clone()).or_insert((0, 0, 0));
            entry.0 = entry.0.saturating_add(calls);
            entry.1 = entry.1.saturating_add(elements);
            entry.2 = entry.2.saturating_add(elapsed_us);
        }
    }
    let mut family_values = family_totals
        .into_iter()
        .collect::<Vec<(String, (u64, u64, u64))>>();
    let has_elapsed_timing = family_values.iter().any(|(_, stats)| stats.2 > 0);
    family_values.sort_by(|left, right| {
        let (left_label, left_stats) = left;
        let (right_label, right_stats) = right;
        if has_elapsed_timing {
            right_stats
                .2
                .cmp(&left_stats.2)
                .then_with(|| right_stats.0.cmp(&left_stats.0))
                .then_with(|| right_stats.1.cmp(&left_stats.1))
                .then_with(|| left_label.cmp(right_label))
        } else {
            right_stats
                .0
                .cmp(&left_stats.0)
                .then_with(|| right_stats.1.cmp(&left_stats.1))
                .then_with(|| left_label.cmp(right_label))
        }
    });
    let mut families = serde_json::Map::new();
    for (label, (calls, elements, elapsed_us)) in family_values {
        families.insert(
            label,
            serde_json::json!({
                "calls": calls,
                "elements": elements,
                "elapsed_us": elapsed_us,
            }),
        );
    }
    object.insert(
        "kernel_launch_families".to_string(),
        serde_json::Value::Object(families),
    );
    serde_json::Value::Object(object)
}

fn aggregate_rank_performance(rank_reports: &[serde_json::Value]) -> serde_json::Value {
    if rank_reports.is_empty() {
        return serde_json::Value::Null;
    }
    let max_performance_u64 = |field: &str| {
        rank_reports
            .iter()
            .filter_map(|report| {
                report
                    .get("performance")
                    .and_then(|performance| performance.get(field))
                    .and_then(serde_json::Value::as_u64)
            })
            .max()
            .unwrap_or(0)
    };
    let max_performance_f64 = |field: &str| {
        rank_reports
            .iter()
            .filter_map(|report| {
                report
                    .get("performance")
                    .and_then(|performance| performance.get(field))
                    .and_then(serde_json::Value::as_f64)
            })
            .fold(0.0, f64::max)
    };
    let train_elapsed_ms = max_performance_u64("train_elapsed_ms");
    let dataloader_elapsed_ms = max_performance_u64("dataloader_elapsed_ms");
    let host_to_device_elapsed_ms = max_performance_u64("host_to_device_elapsed_ms");
    let forward_backward_elapsed_ms = max_performance_u64("forward_backward_elapsed_ms");
    let all_reduce_elapsed_ms = max_performance_u64("all_reduce_elapsed_ms");
    let optimizer_elapsed_ms = max_performance_u64("optimizer_elapsed_ms");
    let host_to_device_cuda_elapsed_ms = max_performance_f64("host_to_device_cuda_elapsed_ms");
    let forward_backward_cuda_elapsed_ms = max_performance_f64("forward_backward_cuda_elapsed_ms");
    let all_reduce_cuda_elapsed_ms = max_performance_f64("all_reduce_cuda_elapsed_ms");
    let optimizer_cuda_elapsed_ms = max_performance_f64("optimizer_cuda_elapsed_ms");
    let tokens_seen = rank_reports
        .iter()
        .filter_map(|report| {
            report
                .get("performance")
                .and_then(|performance| performance.get("tokens_seen"))
                .and_then(serde_json::Value::as_u64)
        })
        .sum::<u64>();
    let dense_flops_per_token = rank_reports
        .iter()
        .find_map(|report| {
            report
                .get("performance")
                .and_then(|performance| performance.get("active_dense_flops_per_token_estimate"))
                .and_then(serde_json::Value::as_f64)
        })
        .unwrap_or(0.0);
    let first_performance_u64 = |field: &str, default: u64| {
        rank_reports
            .iter()
            .find_map(|report| {
                report
                    .get("performance")
                    .and_then(|performance| performance.get(field))
                    .and_then(serde_json::Value::as_u64)
            })
            .unwrap_or(default)
    };
    let micro_batch_size = first_performance_u64("micro_batch_size", 0);
    let grad_accumulation_steps = first_performance_u64("grad_accumulation_steps", 1);
    let per_rank_effective_batch_size = micro_batch_size.saturating_mul(grad_accumulation_steps);
    let global_micro_batch_size = micro_batch_size.saturating_mul(rank_reports.len() as u64);
    let global_effective_batch_size =
        per_rank_effective_batch_size.saturating_mul(rank_reports.len() as u64);
    let elapsed_secs = train_elapsed_ms as f64 / 1000.0;
    let tokens_per_second = if elapsed_secs > 0.0 {
        tokens_seen as f64 / elapsed_secs
    } else {
        0.0
    };
    let dense_core_elapsed_secs = if forward_backward_cuda_elapsed_ms > 0.0 {
        forward_backward_cuda_elapsed_ms / 1000.0
    } else {
        elapsed_secs
    };
    let dense_core_timing_source = if forward_backward_cuda_elapsed_ms > 0.0 {
        "cuda_event_forward_backward_elapsed_ms"
    } else {
        "host_train_elapsed_ms_fallback"
    };
    let dense_core_mfu = if dense_core_elapsed_secs > 0.0 {
        (tokens_seen as f64 * dense_flops_per_token)
            / (dense_core_elapsed_secs * rank_reports.len() as f64 * A100_SXM_BF16_PEAK_FLOPS)
    } else {
        0.0
    };
    let end_to_end_mfu = if elapsed_secs > 0.0 {
        (tokens_seen as f64 * dense_flops_per_token)
            / (elapsed_secs * rank_reports.len() as f64 * A100_SXM_BF16_PEAK_FLOPS)
    } else {
        0.0
    };
    serde_json::json!({
        "tokens_seen": tokens_seen,
        "train_elapsed_ms": train_elapsed_ms,
        "dataloader_elapsed_ms": dataloader_elapsed_ms,
        "host_to_device_elapsed_ms": host_to_device_elapsed_ms,
        "forward_backward_elapsed_ms": forward_backward_elapsed_ms,
        "all_reduce_elapsed_ms": all_reduce_elapsed_ms,
        "optimizer_elapsed_ms": optimizer_elapsed_ms,
        "dataloader_host_elapsed_ms": dataloader_elapsed_ms,
        "host_to_device_host_elapsed_ms": host_to_device_elapsed_ms,
        "forward_backward_host_elapsed_ms": forward_backward_elapsed_ms,
        "all_reduce_host_elapsed_ms": all_reduce_elapsed_ms,
        "optimizer_host_elapsed_ms": optimizer_elapsed_ms,
        "host_to_device_cuda_elapsed_ms": host_to_device_cuda_elapsed_ms,
        "forward_backward_cuda_elapsed_ms": forward_backward_cuda_elapsed_ms,
        "all_reduce_cuda_elapsed_ms": all_reduce_cuda_elapsed_ms,
        "optimizer_cuda_elapsed_ms": optimizer_cuda_elapsed_ms,
        "cuda_event_timing_available": host_to_device_cuda_elapsed_ms > 0.0
            || forward_backward_cuda_elapsed_ms > 0.0
            || all_reduce_cuda_elapsed_ms > 0.0
            || optimizer_cuda_elapsed_ms > 0.0,
        "tokens_per_second": tokens_per_second,
        "active_dense_flops_per_token_estimate": dense_flops_per_token,
        "dense_core_mfu_estimate": dense_core_mfu,
        "end_to_end_mfu_estimate": end_to_end_mfu,
        "mfu_timing_source": {
            "dense_core_mfu_estimate": dense_core_timing_source,
            "end_to_end_mfu_estimate": "host_train_elapsed_ms",
        },
        "micro_batch_size": micro_batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "effective_batch_size": per_rank_effective_batch_size,
        "data_parallel_world_size": rank_reports.len(),
        "global_micro_batch_size": global_micro_batch_size,
        "global_effective_batch_size": global_effective_batch_size,
        "mfu_denominator": {
            "hardware": "NVIDIA A100 SXM BF16 Tensor Core peak",
            "per_device_flops": A100_SXM_BF16_PEAK_FLOPS,
            "device_count": rank_reports.len(),
        },
        "estimate_notes": [
            "MFU fields are first-pass estimates for training-path observability, not optimization claims.",
            "Sparse memory lookup/update work is reported separately and does not inflate active_dense_flops_per_token_estimate.",
            "Distributed aggregate timing buckets use the maximum per-rank elapsed milliseconds for each bucket."
        ],
        "aggregation": "tokens summed across ranks; elapsed and timing buckets are max rank milliseconds",
        "rank_performance": rank_reports
            .iter()
            .map(|report| report.get("performance").cloned().unwrap_or(serde_json::Value::Null))
            .collect::<Vec<_>>(),
    })
}

fn rank0_loader_report(rank_reports: &[serde_json::Value]) -> serde_json::Value {
    rank_reports
        .first()
        .and_then(|report| report.get("loader"))
        .cloned()
        .unwrap_or(serde_json::Value::Null)
}

fn rank_f64_values(rank_reports: &[serde_json::Value], field: &str) -> Vec<f64> {
    rank_reports
        .iter()
        .filter_map(|report| report.get(field).and_then(serde_json::Value::as_f64))
        .collect()
}

fn max_rank_drift(values: &[f64]) -> f64 {
    let Some(reference) = values.first().copied() else {
        return 0.0;
    };
    values
        .iter()
        .map(|value| (value - reference).abs())
        .fold(0.0, f64::max)
}

fn ddp_step_checksum_drifts(rank_reports: &[serde_json::Value]) -> Result<Vec<serde_json::Value>> {
    if rank_reports.is_empty() {
        return Ok(Vec::new());
    }
    let Some(first_steps) = rank_reports[0]
        .get("step_checksums")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(Vec::new());
    };
    if first_steps.is_empty() {
        return Ok(Vec::new());
    }
    let mut drifts = Vec::with_capacity(first_steps.len());
    for step_index in 0..first_steps.len() {
        let reference_step = first_steps[step_index]
            .get("step")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "rank 0 step_checksums[{step_index}] is missing numeric step"
                ))
            })?;
        let mut sums = Vec::with_capacity(rank_reports.len());
        let mut sumsq = Vec::with_capacity(rank_reports.len());
        for (rank, report) in rank_reports.iter().enumerate() {
            let steps = report
                .get("step_checksums")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rank {rank} report is missing step_checksums"
                    ))
                })?;
            if steps.len() != first_steps.len() {
                return Err(TensorError::InvalidOperation(format!(
                    "rank {rank} step_checksums length {} does not match rank 0 length {}",
                    steps.len(),
                    first_steps.len()
                )));
            }
            let entry = &steps[step_index];
            let step = entry
                .get("step")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rank {rank} step_checksums[{step_index}] is missing numeric step"
                    ))
                })?;
            if step != reference_step {
                return Err(TensorError::InvalidOperation(format!(
                    "rank {rank} step_checksums[{step_index}] step {step} does not match rank 0 step {reference_step}"
                )));
            }
            let checksum_sum = entry
                .get("parameter_checksum_sum")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rank {rank} step_checksums[{step_index}] is missing parameter_checksum_sum"
                    ))
                })?;
            let checksum_sumsq = entry
                .get("parameter_checksum_sumsq")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rank {rank} step_checksums[{step_index}] is missing parameter_checksum_sumsq"
                    ))
                })?;
            sums.push(checksum_sum);
            sumsq.push(checksum_sumsq);
        }
        let sum_max_error = max_rank_drift(&sums);
        let sumsq_max_error = max_rank_drift(&sumsq);
        drifts.push(serde_json::json!({
            "step": reference_step,
            "parameter_checksum_sums": sums,
            "parameter_checksum_sumsq": sumsq,
            "parameter_checksum_sum_max_error": sum_max_error,
            "parameter_checksum_sumsq_max_error": sumsq_max_error,
        }));
    }
    Ok(drifts)
}

fn ddp_memory_step_checksum_drifts(
    rank_reports: &[serde_json::Value],
) -> Result<Vec<serde_json::Value>> {
    if rank_reports.is_empty() {
        return Ok(Vec::new());
    }
    let Some(first_steps) = rank_reports[0]
        .get("step_checksums")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(Vec::new());
    };
    if first_steps.is_empty() {
        return Ok(Vec::new());
    }
    let mut drifts = Vec::with_capacity(first_steps.len());
    for step_index in 0..first_steps.len() {
        let reference_step = first_steps[step_index]
            .get("step")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "rank 0 memory step_checksums[{step_index}] is missing numeric step"
                ))
            })?;
        let mut sums = Vec::with_capacity(rank_reports.len());
        let mut sumsq = Vec::with_capacity(rank_reports.len());
        for (rank, report) in rank_reports.iter().enumerate() {
            let steps = report
                .get("step_checksums")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rank {rank} memory report is missing step_checksums"
                    ))
                })?;
            if steps.len() != first_steps.len() {
                return Err(TensorError::InvalidOperation(format!(
                    "rank {rank} memory step_checksums length {} does not match rank 0 length {}",
                    steps.len(),
                    first_steps.len()
                )));
            }
            let entry = &steps[step_index];
            let step = entry
                .get("step")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rank {rank} memory step_checksums[{step_index}] is missing numeric step"
                    ))
                })?;
            if step != reference_step {
                return Err(TensorError::InvalidOperation(format!(
                    "rank {rank} memory step_checksums[{step_index}] step {step} does not match rank 0 step {reference_step}"
                )));
            }
            let checksum_sum = entry
                .get("memory_table_checksum_sum")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rank {rank} memory step_checksums[{step_index}] is missing memory_table_checksum_sum"
                    ))
                })?;
            let checksum_sumsq = entry
                .get("memory_table_checksum_sumsq")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rank {rank} memory step_checksums[{step_index}] is missing memory_table_checksum_sumsq"
                    ))
                })?;
            sums.push(checksum_sum);
            sumsq.push(checksum_sumsq);
        }
        let sum_max_error = max_rank_drift(&sums);
        let sumsq_max_error = max_rank_drift(&sumsq);
        drifts.push(serde_json::json!({
            "step": reference_step,
            "memory_table_checksum_sums": sums,
            "memory_table_checksum_sumsq": sumsq,
            "memory_table_checksum_sum_max_error": sum_max_error,
            "memory_table_checksum_sumsq_max_error": sumsq_max_error,
        }));
    }
    Ok(drifts)
}

fn max_step_checksum_error(step_drifts: &[serde_json::Value], field: &str) -> f64 {
    step_drifts
        .iter()
        .filter_map(|drift| drift.get(field).and_then(serde_json::Value::as_f64))
        .fold(0.0, f64::max)
}

fn ensure_precision_supported_for_device(precision: Precision, device: Device) -> Result<()> {
    if precision != Precision::AmpBf16 {
        return Ok(());
    }
    let Device::Cuda(device_id) = device else {
        return Err(TensorError::Device(
            "amp-bf16 precision requires a CUDA device".to_string(),
        ));
    };
    let supported = cuda::device_supports_bf16_tensor_cores(device_id as i32).map_err(|err| {
        TensorError::Device(format!(
            "failed to query BF16 Tensor Core support for cuda:{device_id}: {err}"
        ))
    })?;
    if !supported {
        return Err(TensorError::Device(format!(
            "amp-bf16 requires compute capability >= 8.0 on cuda:{device_id}"
        )));
    }
    Ok(())
}

fn cuda_device_pci_bus_id(device_id: usize) -> Result<String> {
    let info = cuda::system_info()
        .map_err(|err| TensorError::Device(format!("failed to query CUDA devices: {err}")))?;
    info.devices
        .into_iter()
        .find(|device| device.ordinal == device_id as i32)
        .map(|device| device.pci_bus_id)
        .ok_or_else(|| TensorError::Device(format!("cuda:{device_id} was not reported by CUDA")))
}

fn elapsed_ms_u64(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn rate_per_second(count: f64, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if seconds > 0.0 && count.is_finite() {
        count / seconds
    } else {
        0.0
    }
}

fn source_rejection_count(stats: &SourceCurationStats) -> usize {
    stats.rejections.values().sum()
}

fn materialize_blend(config: MaterializeBlendConfig) -> Result<MaterializeBlendOutcome> {
    let total_start = Instant::now();
    if config.target_tokens == 0 {
        return Err(TensorError::InvalidOperation(
            "--target-tokens must be greater than zero".to_string(),
        ));
    }
    if config.valid_fraction <= 0.0 || config.valid_fraction >= 1.0 {
        return Err(TensorError::InvalidOperation(format!(
            "--valid-fraction must be in (0, 1), got {}",
            config.valid_fraction
        )));
    }
    if config.shard_tokens == 0 {
        return Err(TensorError::InvalidOperation(
            "--shard-tokens must be greater than zero".to_string(),
        ));
    }
    if config.text_shard_bytes == 0 {
        return Err(TensorError::InvalidOperation(
            "--text-shard-bytes must be greater than zero".to_string(),
        ));
    }
    if config.max_tokens_per_byte <= 0.0 || !config.max_tokens_per_byte.is_finite() {
        return Err(TensorError::InvalidOperation(format!(
            "--max-tokens-per-byte must be finite and positive, got {}",
            config.max_tokens_per_byte
        )));
    }
    if config.candidate_retention_token_multiplier < 0.0
        || !config.candidate_retention_token_multiplier.is_finite()
    {
        return Err(TensorError::InvalidOperation(format!(
            "--candidate-retention-token-multiplier must be finite and non-negative, got {}",
            config.candidate_retention_token_multiplier
        )));
    }
    if config.candidate_retention_token_multiplier > 0.0 && config.candidate_prune_every == 0 {
        return Err(TensorError::InvalidOperation(
            "--candidate-prune-every must be greater than zero when candidate retention is enabled"
                .to_string(),
        ));
    }
    if config.resume_checkpoint && config.checkpoint_dir.is_none() {
        return Err(TensorError::InvalidOperation(
            "--resume-checkpoint requires --checkpoint-dir".to_string(),
        ));
    }
    if (config.checkpoint_every_records > 0 || config.checkpoint_every_bytes > 0)
        && config.checkpoint_dir.is_none()
    {
        return Err(TensorError::InvalidOperation(
            "--checkpoint-every-records/--checkpoint-every-bytes require --checkpoint-dir"
                .to_string(),
        ));
    }
    if config.checkpoint_stop_after_records.is_some() && config.checkpoint_dir.is_none() {
        return Err(TensorError::InvalidOperation(
            "--checkpoint-stop-after-records requires --checkpoint-dir".to_string(),
        ));
    }

    let load_start = Instant::now();
    fs::create_dir_all(&config.out_dir).map_err(|err| {
        TensorError::Io(format!(
            "failed to create materializer out dir {}: {err}",
            config.out_dir.display()
        ))
    })?;
    let manifest_bytes = fs::read(&config.corpus_blend).map_err(|err| {
        TensorError::Io(format!(
            "failed to read corpus blend {}: {err}",
            config.corpus_blend.display()
        ))
    })?;
    let manifest: CorpusBlendManifest = serde_json::from_slice(&manifest_bytes).map_err(|err| {
        TensorError::Io(format!(
            "failed to parse corpus blend {}: {err}",
            config.corpus_blend.display()
        ))
    })?;
    let tokenizer = BpeTokenizer::load(&config.tokenizer)?;
    let tokenizer_hash = tokenizer.fingerprint()?;
    let manifest_hash = stable_hash_bytes_local(&manifest_bytes);
    let allowed_license_statuses = allow_license_statuses(config.allow_license_status.clone());
    let manifest_root = config.corpus_blend.parent();
    let sources = materializer_sources(&manifest, &allowed_license_statuses, manifest_root)?;
    let quotas = materializer_source_quotas(&sources, config.target_tokens)?;
    let checkpoint_control = if let Some(dir) = &config.checkpoint_dir {
        fs::create_dir_all(dir).map_err(|err| {
            TensorError::Io(format!(
                "failed to create materializer checkpoint dir {}: {err}",
                dir.display()
            ))
        })?;
        Some(MaterializerCheckpointControl {
            dir: dir.clone(),
            manifest_hash: manifest_hash.clone(),
            tokenizer_hash: tokenizer_hash.clone(),
            resume: config.resume_checkpoint,
            every_records: config.checkpoint_every_records,
            every_bytes: config.checkpoint_every_bytes,
            stop_after_records: config.checkpoint_stop_after_records,
        })
    } else {
        None
    };
    if let Some(control) = &checkpoint_control {
        write_materializer_checkpoint_manifest(&config, &manifest, &sources, &quotas, control)?;
    }
    let load_elapsed = load_start.elapsed();

    eprintln!(
        "[materialize-blend] mode={} sources={} target_tokens={} out_dir={}",
        materialize_mode_label(config.mode),
        sources.len(),
        config.target_tokens,
        config.out_dir.display()
    );
    let scan_start = Instant::now();
    let mut global_hashes = BTreeSet::new();
    let mut source_candidates = Vec::new();
    for source in &sources {
        let token_quota = *quotas.get(source.source_id.as_str()).unwrap_or(&0);
        let scanned = scan_materializer_source(
            source,
            token_quota,
            &tokenizer,
            &config,
            manifest_root,
            &mut global_hashes,
            checkpoint_control.as_ref(),
        )?;
        source_candidates.push(scanned);
    }
    let scan_elapsed = scan_start.elapsed();

    let selection_start = Instant::now();
    let mut selected_hashes = BTreeSet::new();
    let mut source_reports = Vec::new();
    let mut selected_docs = 0usize;
    let mut selected_tokens = 0u64;
    for source in source_candidates.iter_mut() {
        let source_selection_start = Instant::now();
        source.candidates.sort_by(materialized_doc_selection_order);
        let mut source_tokens = 0u64;
        let mut source_docs = 0usize;
        let mut source_bytes = 0u64;
        for candidate in &source.candidates {
            if source_tokens >= source.stats.token_quota {
                break;
            }
            selected_hashes.insert(candidate.text_hash.clone());
            source_tokens += candidate.token_count as u64;
            source_docs += 1;
            source_bytes += candidate.text_bytes as u64;
        }
        source.stats.selected_docs = source_docs;
        source.stats.selected_tokens = source_tokens;
        source.stats.selected_bytes = source_bytes;
        source.stats.exhausted = source_tokens < source.stats.token_quota;
        source.stats.selection_elapsed_ms = elapsed_ms_u64(source_selection_start.elapsed());
        selected_docs += source_docs;
        selected_tokens += source_tokens;
        source_reports.push(source.stats.clone());
    }
    let selection_elapsed = selection_start.elapsed();
    if selected_docs == 0 {
        return Err(TensorError::InvalidOperation(
            "materialize-blend selected zero documents after filtering".to_string(),
        ));
    }
    if config.mode == MaterializeBlendMode::Full
        && selected_tokens < (config.target_tokens as f64 * 0.95).round() as u64
    {
        return Err(TensorError::InvalidOperation(format!(
            "full materialization selected {} tokens, below 95% of target {}",
            selected_tokens, config.target_tokens
        )));
    }
    let sizing_estimate = materialized_output_sizing_estimate(
        &config,
        &tokenizer,
        &source_candidates,
        &selected_hashes,
    );

    let source_index_path = config.out_dir.join("source-index.json");
    let selected_docs_path = config.out_dir.join("selected-docs.jsonl");
    let curation_report_path = config.out_dir.join("curation-report.json");
    let tokenizer_sample_manifest_path = config.out_dir.join("tokenizer-sample-manifest.json");
    let output_write_start = Instant::now();
    let (prepared_manifest_path, output_stats) = if config.mode == MaterializeBlendMode::DryRun {
        write_selected_docs_dry_run(
            &selected_docs_path,
            &source_candidates,
            &selected_hashes,
            config.valid_fraction,
        )?;
        (None, MaterializedOutputStats::default())
    } else {
        let output = match config.candidate_text_mode {
            MaterializerCandidateTextMode::Retain => write_materialized_outputs(
                &config,
                &tokenizer,
                &source_candidates,
                &selected_hashes,
                &selected_docs_path,
            )?,
            MaterializerCandidateTextMode::Rescan => write_materialized_outputs_rescan(
                &config,
                &tokenizer,
                &sources,
                &selected_hashes,
                manifest_root,
                &selected_docs_path,
            )?,
        };
        (Some(output.manifest_path), output.stats)
    };
    let output_write_elapsed = output_write_start.elapsed();
    for source_report in &mut source_reports {
        if let Some(write_stats) = output_stats.per_source.get(&source_report.source_id) {
            source_report.written_docs = write_stats.docs;
            source_report.written_tokens = write_stats.tokens;
            source_report.written_bytes = write_stats.bytes;
            source_report.write_elapsed_ms = write_stats.elapsed_ms;
            source_report.write_bytes_per_second = write_stats.bytes_per_second;
            source_report.write_tokens_per_second = write_stats.tokens_per_second;
        }
    }
    let total_elapsed = total_start.elapsed();
    let total_scanned_docs = source_reports
        .iter()
        .map(|source| source.scanned_docs as u64)
        .sum::<u64>();
    let total_candidate_docs = source_reports
        .iter()
        .map(|source| source.candidate_docs as u64)
        .sum::<u64>();
    let total_candidate_tokens = source_reports
        .iter()
        .map(|source| source.candidate_tokens)
        .sum::<u64>();
    let total_tokenizer_encode_elapsed_ms = source_reports
        .iter()
        .map(|source| source.tokenizer_encode_elapsed_ms)
        .sum::<u64>();
    let total_score_elapsed_ms = source_reports
        .iter()
        .map(|source| source.score_elapsed_ms)
        .sum::<u64>();
    let total_hash_elapsed_ms = source_reports
        .iter()
        .map(|source| source.hash_elapsed_ms)
        .sum::<u64>();
    let total_scanned_bytes = source_reports
        .iter()
        .map(|source| source.scanned_bytes)
        .sum::<u64>();
    let total_selected_bytes = source_reports
        .iter()
        .map(|source| source.selected_bytes)
        .sum::<u64>();
    let selected_tokens_per_second = rate_per_second(selected_tokens as f64, total_elapsed);

    let source_index = serde_json::json!({
        "format": "heirloom.materialized_source_index",
        "version": 1,
        "blend_id": &manifest.blend_id,
        "mode": materialize_mode_label(config.mode),
        "target_tokens": config.target_tokens,
        "selected_docs": selected_docs,
        "selected_tokens": selected_tokens,
        "timing": {
            "load_elapsed_ms": elapsed_ms_u64(load_elapsed),
            "scan_elapsed_ms": elapsed_ms_u64(scan_elapsed),
            "selection_elapsed_ms": elapsed_ms_u64(selection_elapsed),
            "write_outputs_elapsed_ms": elapsed_ms_u64(output_write_elapsed),
            "score_elapsed_ms": total_score_elapsed_ms,
            "tokenizer_encode_elapsed_ms": total_tokenizer_encode_elapsed_ms,
            "hash_elapsed_ms": total_hash_elapsed_ms,
            "total_elapsed_ms": elapsed_ms_u64(total_elapsed),
        },
        "throughput": {
            "scan_bytes_per_second": rate_per_second(total_scanned_bytes as f64, scan_elapsed),
            "scan_docs_per_second": rate_per_second(total_scanned_docs as f64, scan_elapsed),
            "candidate_docs_per_second": rate_per_second(total_candidate_docs as f64, scan_elapsed),
            "candidate_tokens_per_second": rate_per_second(total_candidate_tokens as f64, scan_elapsed),
            "tokenizer_encode_tokens_per_second": rate_per_second(
                total_candidate_tokens as f64,
                Duration::from_millis(total_tokenizer_encode_elapsed_ms),
            ),
            "selected_tokens_per_second_end_to_end": selected_tokens_per_second,
            "selected_bytes_per_second_end_to_end": rate_per_second(total_selected_bytes as f64, total_elapsed),
            "written_tokens_per_second": rate_per_second(output_stats.written_tokens as f64, output_write_elapsed),
            "written_bytes_per_second": rate_per_second(output_stats.written_bytes as f64, output_write_elapsed),
        },
        "candidate_retention": {
            "enabled": config.candidate_retention_token_multiplier > 0.0,
            "token_multiplier": config.candidate_retention_token_multiplier,
            "min_docs": config.candidate_retention_min_docs,
            "prune_every": config.candidate_prune_every,
            "exact_for_quota_when_multiplier_at_least_one": config.candidate_retention_token_multiplier >= 1.0,
        },
        "candidate_text": {
            "mode": candidate_text_mode_label(config.candidate_text_mode),
            "retained_in_scan": config.candidate_text_mode == MaterializerCandidateTextMode::Retain,
        },
        "checkpoint": {
            "enabled": checkpoint_control.is_some(),
            "checkpoint_dir": checkpoint_control.as_ref().map(|control| control.dir.display().to_string()),
            "resume_checkpoint": config.resume_checkpoint,
            "checkpoint_every_records": config.checkpoint_every_records,
            "checkpoint_every_bytes": config.checkpoint_every_bytes,
            "source_checkpoint_count": source_reports.iter().filter(|source| source.checkpoint_path.is_some()).count(),
        },
        "sizing": sizing_estimate.clone(),
        "sources": source_reports,
    });
    write_json_file(&source_index_path, source_index.clone())?;
    let curation_report = serde_json::json!({
        "format": "heirloom.blend_curation_report",
        "version": 1,
        "status": "passed",
        "mode": materialize_mode_label(config.mode),
        "corpus_blend": config.corpus_blend.display().to_string(),
        "tokenizer": tokenizer_metadata_report(&tokenizer, Some(&config.tokenizer))?,
        "target_tokens": config.target_tokens,
        "selected_tokens": selected_tokens,
        "selected_docs": selected_docs,
        "valid_fraction": config.valid_fraction,
        "source_index": source_index_path.display().to_string(),
        "selected_docs_path": selected_docs_path.display().to_string(),
        "prepared_manifest": prepared_manifest_path.as_ref().map(|path| path.display().to_string()),
        "timing": source_index["timing"].clone(),
        "throughput": source_index["throughput"].clone(),
        "candidate_retention": source_index["candidate_retention"].clone(),
        "candidate_text": source_index["candidate_text"].clone(),
        "checkpoint": source_index["checkpoint"].clone(),
        "sizing": source_index["sizing"].clone(),
        "output": {
            "written_docs": output_stats.written_docs,
            "written_tokens": output_stats.written_tokens,
            "written_bytes": output_stats.written_bytes,
            "train_tokens": output_stats.train_tokens,
            "valid_tokens": output_stats.valid_tokens,
            "text_shards": output_stats.text_shards,
            "train_token_shards": output_stats.train_token_shards,
            "valid_token_shards": output_stats.valid_token_shards,
        },
        "filters": {
            "license_statuses": allowed_license_statuses,
            "min_doc_bytes": config.min_doc_bytes,
            "max_doc_bytes": config.max_doc_bytes,
            "max_tokens_per_byte": config.max_tokens_per_byte,
            "max_source_bytes": config.max_source_bytes,
            "max_docs_per_source": config.max_docs_per_source,
            "candidate_text_mode": candidate_text_mode_label(config.candidate_text_mode),
        },
        "quota_policy": {
            "kind": "source_sampling_weight_token_quota",
            "selection": "score_desc_then_hash_key",
            "seed": config.seed,
        },
        "sources": source_index["sources"].clone(),
    });
    write_json_file(&curation_report_path, curation_report)?;
    let tokenizer_sample_manifest = serde_json::json!({
        "format": "heirloom.tokenizer_sample_manifest",
        "version": 1,
        "blend_id": &manifest.blend_id,
        "source_blend_hash": manifest_hash,
        "seed": config.seed,
        "selected_tokens": selected_tokens,
        "selected_docs": selected_docs,
        "sources": source_index["sources"].clone(),
        "note": "This manifest records the materialized document selection that can be reused for tokenizer sampling; tokenizer training may still use a smaller byte quota.",
    });
    write_json_file(&tokenizer_sample_manifest_path, tokenizer_sample_manifest)?;

    Ok(MaterializeBlendOutcome {
        mode: config.mode,
        selected_docs,
        selected_tokens,
        total_elapsed_ms: elapsed_ms_u64(total_elapsed),
        selected_tokens_per_second,
        prepared_manifest_path,
    })
}

fn materializer_sources<'a>(
    manifest: &'a CorpusBlendManifest,
    allowed_license_statuses: &[String],
    manifest_root: Option<&Path>,
) -> Result<Vec<&'a CorpusBlendSource>> {
    let sources = manifest
        .sources
        .iter()
        .filter(|source| source.include_in_pretraining && source.sampling_weight > 0.0)
        .collect::<Vec<_>>();
    if sources.is_empty() {
        return Err(TensorError::InvalidOperation(
            "corpus blend has no positive-weight pretraining sources".to_string(),
        ));
    }
    for source in &sources {
        if !allowed_license_statuses.contains(&source.license_status) {
            return Err(TensorError::InvalidOperation(format!(
                "source {} has unapproved license_status {}; allowed={:?}",
                source.source_id, source.license_status, allowed_license_statuses
            )));
        }
        if source.path.is_none() {
            return Err(TensorError::InvalidOperation(format!(
                "source {} is included in pretraining but has no materialized local path",
                source.source_id
            )));
        }
        let path = source.path.as_deref().unwrap();
        if path.ends_with(".gz") || path.ends_with(".zst") || path.ends_with(".zstd") {
            return Err(TensorError::InvalidOperation(format!(
                "source {} path {} appears compressed; materialize/decompress it before data materialize-blend",
                source.source_id, path
            )));
        }
        let files = resolve_source_files(path, manifest_root)?;
        if files.is_empty() {
            return Err(TensorError::InvalidOperation(format!(
                "source {} path {} resolved to zero readable source files",
                source.source_id, path
            )));
        }
    }
    Ok(sources)
}

fn materializer_source_quotas(
    sources: &[&CorpusBlendSource],
    target_tokens: u64,
) -> Result<BTreeMap<String, u64>> {
    let total_weight = sources
        .iter()
        .map(|source| source.sampling_weight)
        .sum::<f64>();
    if total_weight <= 0.0 || !total_weight.is_finite() {
        return Err(TensorError::InvalidOperation(
            "materializer sources have invalid total sampling weight".to_string(),
        ));
    }
    let mut quotas = BTreeMap::new();
    let mut assigned = 0u64;
    for (index, source) in sources.iter().enumerate() {
        let quota = if index + 1 == sources.len() {
            target_tokens.saturating_sub(assigned)
        } else {
            ((target_tokens as f64) * source.sampling_weight / total_weight).round() as u64
        };
        quotas.insert(source.source_id.clone(), quota);
        assigned = assigned.saturating_add(quota);
    }
    Ok(quotas)
}

fn materialized_doc_selection_order(
    a: &MaterializedDoc,
    b: &MaterializedDoc,
) -> std::cmp::Ordering {
    b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.sample_key.cmp(&b.sample_key))
        .then_with(|| a.text_hash.cmp(&b.text_hash))
}

fn candidate_retention_token_limit(
    config: &MaterializeBlendConfig,
    token_quota: u64,
) -> Option<u64> {
    if config.candidate_retention_token_multiplier <= 0.0 {
        return None;
    }
    let limit = (token_quota as f64 * config.candidate_retention_token_multiplier).ceil();
    if !limit.is_finite() || limit >= u64::MAX as f64 {
        Some(u64::MAX)
    } else {
        Some((limit as u64).max(token_quota))
    }
}

fn prune_materializer_candidates(
    candidates: &mut Vec<MaterializedDoc>,
    token_limit: Option<u64>,
    min_docs: usize,
) -> CandidatePruneStats {
    let Some(token_limit) = token_limit else {
        return CandidatePruneStats::default();
    };
    if candidates.len() <= min_docs {
        return CandidatePruneStats::default();
    }
    candidates.sort_by(materialized_doc_selection_order);
    let mut keep_docs = 0usize;
    let mut keep_tokens = 0u64;
    for candidate in candidates.iter() {
        keep_docs += 1;
        keep_tokens = keep_tokens.saturating_add(candidate.token_count as u64);
        if keep_docs >= min_docs && keep_tokens >= token_limit {
            break;
        }
    }
    if keep_docs >= candidates.len() {
        return CandidatePruneStats::default();
    }
    let pruned = candidates[keep_docs..].iter().fold(
        CandidatePruneStats::default(),
        |mut stats, candidate| {
            stats.docs += 1;
            stats.tokens = stats.tokens.saturating_add(candidate.token_count as u64);
            stats.bytes = stats.bytes.saturating_add(candidate.text_bytes as u64);
            stats
        },
    );
    candidates.truncate(keep_docs);
    pruned
}

fn candidate_totals(candidates: &[MaterializedDoc]) -> (u64, u64) {
    candidates
        .iter()
        .fold((0u64, 0u64), |mut totals, candidate| {
            totals.0 = totals.0.saturating_add(candidate.token_count as u64);
            totals.1 = totals.1.saturating_add(candidate.text_bytes as u64);
            totals
        })
}

fn div_ceil_u64(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        return 0;
    }
    value
        .checked_add(divisor.saturating_sub(1))
        .map(|sum| sum / divisor)
        .unwrap_or(u64::MAX / divisor)
}

fn token_shard_dtype_for_vocab(vocab_size: usize) -> TokenShardDType {
    if vocab_size <= u16::MAX as usize + 1 {
        TokenShardDType::U16
    } else {
        TokenShardDType::U32
    }
}

fn materialized_output_sizing_estimate(
    config: &MaterializeBlendConfig,
    tokenizer: &BpeTokenizer,
    sources: &[SourceCandidates],
    selected_hashes: &BTreeSet<String>,
) -> serde_json::Value {
    let dtype = token_shard_dtype_for_vocab(tokenizer.vocab_size());
    let bytes_per_token = dtype.bytes_per_token() as u64;
    let split_assignments =
        build_materialized_split_assignments(selected_hashes, config.valid_fraction)
            .unwrap_or_else(|_| {
                selected_hashes
                    .iter()
                    .map(|hash| (hash.clone(), "train"))
                    .collect()
            });
    let mut selected_docs = 0u64;
    let mut selected_tokens = 0u64;
    let mut selected_text_bytes = 0u64;
    let mut train_tokens = 0u64;
    let mut valid_tokens = 0u64;
    let mut source_estimates = Vec::new();
    for source in sources {
        let mut source_docs = 0u64;
        let mut source_tokens = 0u64;
        let mut source_text_bytes = 0u64;
        for doc in &source.candidates {
            if !selected_hashes.contains(&doc.text_hash) {
                continue;
            }
            selected_docs += 1;
            selected_tokens = selected_tokens.saturating_add(doc.token_count as u64);
            selected_text_bytes = selected_text_bytes.saturating_add(doc.text_bytes as u64);
            source_docs += 1;
            source_tokens = source_tokens.saturating_add(doc.token_count as u64);
            source_text_bytes = source_text_bytes.saturating_add(doc.text_bytes as u64);
            if split_assignments
                .get(&doc.text_hash)
                .copied()
                .unwrap_or("train")
                == "valid"
            {
                valid_tokens = valid_tokens.saturating_add(doc.token_count as u64);
            } else {
                train_tokens = train_tokens.saturating_add(doc.token_count as u64);
            }
        }
        source_estimates.push(serde_json::json!({
            "source_id": source.stats.source_id,
            "selected_docs": source_docs,
            "selected_tokens": source_tokens,
            "selected_text_bytes": source_text_bytes,
            "estimated_token_payload_bytes": source_tokens.saturating_mul(bytes_per_token),
        }));
    }
    let estimated_text_payload_bytes = selected_text_bytes.saturating_add(selected_docs);
    let estimated_token_payload_bytes = selected_tokens.saturating_mul(bytes_per_token);
    let train_shards = div_ceil_u64(train_tokens, config.shard_tokens as u64);
    let valid_shards = div_ceil_u64(valid_tokens, config.shard_tokens as u64);
    let text_shards = div_ceil_u64(estimated_text_payload_bytes, config.text_shard_bytes as u64);
    serde_json::json!({
        "kind": "selected_document_output_estimate",
        "tokenizer_vocab_size": tokenizer.vocab_size(),
        "estimated_token_dtype": dtype,
        "estimated_bytes_per_token": bytes_per_token,
        "target_tokens": config.target_tokens,
        "selected_tokens": selected_tokens,
        "selected_docs": selected_docs,
        "target_token_coverage": selected_tokens as f64 / config.target_tokens.max(1) as f64,
        "selected_text_bytes": selected_text_bytes,
        "estimated_text_payload_bytes": estimated_text_payload_bytes,
        "estimated_token_payload_bytes": estimated_token_payload_bytes,
        "estimated_total_payload_bytes": estimated_text_payload_bytes.saturating_add(estimated_token_payload_bytes),
        "train_tokens_estimate": train_tokens,
        "valid_tokens_estimate": valid_tokens,
        "train_token_shards_estimate": train_shards,
        "valid_token_shards_estimate": valid_shards,
        "text_shards_estimate": text_shards,
        "shard_tokens": config.shard_tokens,
        "text_shard_bytes": config.text_shard_bytes,
        "valid_fraction": config.valid_fraction,
        "source_estimates": source_estimates,
    })
}

fn materializer_checkpoint_source_path(dir: &Path, source_id: &str) -> PathBuf {
    let mut sanitized = String::with_capacity(source_id.len());
    for ch in source_id.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    let hash = stable_hash_bytes_local(source_id.as_bytes());
    dir.join(format!("{sanitized}-{hash}.scan-checkpoint.json"))
}

fn materializer_source_config_hash(
    source: &CorpusBlendSource,
    token_quota: u64,
    source_files: &[PathBuf],
    config: &MaterializeBlendConfig,
    control: &MaterializerCheckpointControl,
) -> Result<String> {
    let files = source_files
        .iter()
        .map(|path| {
            let metadata = fs::metadata(path).map_err(|err| {
                TensorError::Io(format!("failed to stat {}: {err}", path.display()))
            })?;
            Ok(serde_json::json!({
                "path": path.display().to_string(),
                "bytes": metadata.len(),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let value = serde_json::json!({
        "format": "heirloom.materializer_scan_config",
        "version": 1,
        "manifest_hash": control.manifest_hash,
        "tokenizer_hash": control.tokenizer_hash,
        "source_id": source.source_id,
        "source_path": source.path,
        "source_files": files,
        "target_tokens": config.target_tokens,
        "token_quota": token_quota,
        "seed": config.seed,
        "max_source_bytes": config.max_source_bytes,
        "max_docs_per_source": config.max_docs_per_source,
        "min_doc_bytes": config.min_doc_bytes,
        "max_doc_bytes": config.max_doc_bytes,
        "max_tokens_per_byte": config.max_tokens_per_byte,
        "candidate_text_mode": candidate_text_mode_label(config.candidate_text_mode),
        "candidate_retention_token_multiplier": config.candidate_retention_token_multiplier,
        "candidate_retention_min_docs": config.candidate_retention_min_docs,
        "candidate_prune_every": config.candidate_prune_every,
    });
    let bytes = serde_json::to_vec(&value)
        .map_err(|err| TensorError::Io(format!("failed to serialize checkpoint config: {err}")))?;
    Ok(stable_hash_bytes_local(&bytes))
}

fn write_materializer_checkpoint_manifest(
    config: &MaterializeBlendConfig,
    manifest: &CorpusBlendManifest,
    sources: &[&CorpusBlendSource],
    quotas: &BTreeMap<String, u64>,
    control: &MaterializerCheckpointControl,
) -> Result<()> {
    let path = control.dir.join("manifest.json");
    let value = serde_json::json!({
        "format": "heirloom.materializer_checkpoint_manifest",
        "version": 1,
        "blend_id": manifest.blend_id,
        "mode": materialize_mode_label(config.mode),
        "target_tokens": config.target_tokens,
        "checkpoint_dir": control.dir.display().to_string(),
        "resume_checkpoint": control.resume,
        "checkpoint_every_records": control.every_records,
        "checkpoint_every_bytes": control.every_bytes,
        "manifest_hash": control.manifest_hash,
        "tokenizer_hash": control.tokenizer_hash,
        "saved_unix_ms": unix_epoch_millis(),
        "sources": sources
            .iter()
            .map(|source| serde_json::json!({
                "source_id": source.source_id,
                "token_quota": quotas.get(source.source_id.as_str()).copied().unwrap_or(0),
                "checkpoint_path": materializer_checkpoint_source_path(&control.dir, &source.source_id).display().to_string(),
            }))
            .collect::<Vec<_>>(),
    });
    write_json_file(&path, value)
}

fn unix_epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn checkpoint_progress_threshold_usize(current: usize, every: usize) -> usize {
    if every == 0 {
        return 0;
    }
    current
        .checked_div(every)
        .and_then(|quotient| quotient.checked_add(1))
        .and_then(|next| next.checked_mul(every))
        .unwrap_or(usize::MAX)
}

fn checkpoint_progress_threshold_u64(current: u64, every: u64) -> u64 {
    if every == 0 {
        return 0;
    }
    current
        .checked_div(every)
        .and_then(|quotient| quotient.checked_add(1))
        .and_then(|next| next.checked_mul(every))
        .unwrap_or(u64::MAX)
}

fn load_materializer_source_checkpoint(
    path: &Path,
    source_id: &str,
    config_hash: &str,
) -> Result<Option<MaterializerSourceCheckpoint>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)
        .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
    let checkpoint: MaterializerSourceCheckpoint = serde_json::from_slice(&bytes)
        .map_err(|err| TensorError::Io(format!("failed to parse {}: {err}", path.display())))?;
    if checkpoint.format != "heirloom.materializer_source_checkpoint" || checkpoint.version != 1 {
        return Err(TensorError::InvalidOperation(format!(
            "materializer checkpoint {} has unsupported format/version",
            path.display()
        )));
    }
    if checkpoint.source_id != source_id {
        return Err(TensorError::InvalidOperation(format!(
            "materializer checkpoint {} source mismatch: expected {} got {}",
            path.display(),
            source_id,
            checkpoint.source_id
        )));
    }
    if checkpoint.config_hash != config_hash {
        return Err(TensorError::InvalidOperation(format!(
            "materializer checkpoint {} config hash mismatch; rerun without --resume-checkpoint or delete stale checkpoint",
            path.display()
        )));
    }
    Ok(Some(checkpoint))
}

struct MaterializerCheckpointWriteInput<'a> {
    path: &'a Path,
    source_id: &'a str,
    config_hash: &'a str,
    source_files: &'a [PathBuf],
    cursor: &'a MaterializerScanCursor,
    stats: &'a SourceCurationStats,
    candidates: &'a [MaterializedDoc],
    accepted_hashes: &'a BTreeSet<String>,
    content_hash_state: u64,
}

struct MaterializerCheckpointSaveInput<'a> {
    control: Option<&'a MaterializerCheckpointControl>,
    checkpoint_path: Option<&'a Path>,
    config_hash: Option<&'a str>,
    source_files: &'a [PathBuf],
    cursor: &'a MaterializerScanCursor,
    stats: &'a mut SourceCurationStats,
    candidates: &'a [MaterializedDoc],
    accepted_hashes: &'a BTreeSet<String>,
    content_hash_state: u64,
    next_checkpoint_records: &'a mut usize,
    next_checkpoint_bytes: &'a mut u64,
    force: bool,
}

fn write_materializer_source_checkpoint(input: MaterializerCheckpointWriteInput<'_>) -> Result<()> {
    if let Some(parent) = input.path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            TensorError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }
    let checkpoint = MaterializerSourceCheckpoint {
        format: "heirloom.materializer_source_checkpoint".to_string(),
        version: 1,
        source_id: input.source_id.to_string(),
        config_hash: input.config_hash.to_string(),
        saved_unix_ms: unix_epoch_millis(),
        cursor: input.cursor.clone(),
        stats: input.stats.clone(),
        candidates: input.candidates.to_vec(),
        accepted_hashes: input.accepted_hashes.iter().cloned().collect(),
        content_hash_state: input.content_hash_state,
        source_files: input
            .source_files
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
    };
    let json = serde_json::to_string_pretty(&checkpoint)
        .map_err(|err| TensorError::Io(format!("failed to serialize checkpoint: {err}")))?
        + "\n";
    let tmp_path = input.path.with_extension("json.tmp");
    fs::write(&tmp_path, json)
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", tmp_path.display())))?;
    fs::rename(&tmp_path, input.path).map_err(|err| {
        TensorError::Io(format!(
            "failed to move {} to {}: {err}",
            tmp_path.display(),
            input.path.display()
        ))
    })
}

fn maybe_save_materializer_source_checkpoint(
    input: MaterializerCheckpointSaveInput<'_>,
) -> Result<()> {
    let Some(control) = input.control else {
        return Ok(());
    };
    let Some(path) = input.checkpoint_path else {
        return Ok(());
    };
    let Some(config_hash) = input.config_hash else {
        return Ok(());
    };
    let mut should_save = input.force;
    if control.every_records > 0
        && *input.next_checkpoint_records > 0
        && input.stats.scanned_docs >= *input.next_checkpoint_records
    {
        should_save = true;
        *input.next_checkpoint_records =
            checkpoint_progress_threshold_usize(input.stats.scanned_docs, control.every_records);
    }
    if control.every_bytes > 0
        && *input.next_checkpoint_bytes > 0
        && input.stats.scanned_bytes >= *input.next_checkpoint_bytes
    {
        should_save = true;
        *input.next_checkpoint_bytes =
            checkpoint_progress_threshold_u64(input.stats.scanned_bytes, control.every_bytes);
    }
    if !should_save {
        return Ok(());
    }
    input.stats.checkpoint_saved_count = input.stats.checkpoint_saved_count.saturating_add(1);
    input.stats.checkpoint_cursor_file_index = input.cursor.file_index;
    input.stats.checkpoint_cursor_byte_offset = input.cursor.byte_offset;
    input.stats.checkpoint_completed = input.cursor.completed;
    input.stats.checkpoint_completion_reason = input.cursor.completion_reason.clone();
    input.stats.content_hash = format!("{:016x}", input.content_hash_state);
    write_materializer_source_checkpoint(MaterializerCheckpointWriteInput {
        path,
        source_id: &input.stats.source_id,
        config_hash,
        source_files: input.source_files,
        cursor: input.cursor,
        stats: input.stats,
        candidates: input.candidates,
        accepted_hashes: input.accepted_hashes,
        content_hash_state: input.content_hash_state,
    })
}

fn scan_materializer_source(
    source: &CorpusBlendSource,
    token_quota: u64,
    tokenizer: &BpeTokenizer,
    config: &MaterializeBlendConfig,
    manifest_root: Option<&Path>,
    global_hashes: &mut BTreeSet<String>,
    checkpoint_control: Option<&MaterializerCheckpointControl>,
) -> Result<SourceCandidates> {
    let source_files = resolve_source_files(source.path.as_deref().unwrap(), manifest_root)?;
    let source_file_count = source_files.len();
    let source_path = resolve_source_path(source.path.as_deref().unwrap(), manifest_root);
    let candidate_retention_token_limit = candidate_retention_token_limit(config, token_quota);
    let checkpoint_path = checkpoint_control
        .map(|control| materializer_checkpoint_source_path(&control.dir, &source.source_id));
    let checkpoint_config_hash = if let Some(control) = checkpoint_control {
        Some(materializer_source_config_hash(
            source,
            token_quota,
            &source_files,
            config,
            control,
        )?)
    } else {
        None
    };
    let mut stats = SourceCurationStats {
        source_id: source.source_id.clone(),
        display_name: source.display_name.clone(),
        path: Some(source_path.display().to_string()),
        license_status: source.license_status.clone(),
        sampling_weight: source.sampling_weight,
        token_quota,
        source_file_count,
        candidate_retention_token_multiplier: config.candidate_retention_token_multiplier,
        candidate_retention_min_docs: config.candidate_retention_min_docs,
        candidate_prune_every: config.candidate_prune_every,
        candidate_retention_enabled: candidate_retention_token_limit.is_some(),
        candidate_retention_token_limit,
        candidate_retention_exact_for_quota: config.candidate_retention_token_multiplier >= 1.0,
        candidate_text_mode: candidate_text_mode_label(config.candidate_text_mode).to_string(),
        candidate_text_retained: config.candidate_text_mode
            == MaterializerCandidateTextMode::Retain,
        checkpoint_path: checkpoint_path
            .as_ref()
            .map(|path| path.display().to_string()),
        ..SourceCurationStats::default()
    };
    let mut candidates = Vec::new();
    let mut accepted_hashes = BTreeSet::new();
    let mut content_hasher = StableFnv64::new();
    let mut cursor = MaterializerScanCursor::default();
    if let (Some(control), Some(path), Some(config_hash)) = (
        checkpoint_control,
        checkpoint_path.as_ref(),
        checkpoint_config_hash.as_deref(),
    ) {
        if control.resume {
            if let Some(checkpoint) =
                load_materializer_source_checkpoint(path, &source.source_id, config_hash)?
            {
                stats = checkpoint.stats;
                stats.checkpoint_path = Some(path.display().to_string());
                stats.checkpoint_loaded = true;
                stats.checkpoint_completed = checkpoint.cursor.completed;
                stats.checkpoint_cursor_file_index = checkpoint.cursor.file_index;
                stats.checkpoint_cursor_byte_offset = checkpoint.cursor.byte_offset;
                stats.checkpoint_completion_reason = checkpoint.cursor.completion_reason.clone();
                candidates = checkpoint.candidates;
                accepted_hashes = checkpoint.accepted_hashes.into_iter().collect();
                global_hashes.extend(accepted_hashes.iter().cloned());
                content_hasher = StableFnv64::from_value(checkpoint.content_hash_state);
                cursor = checkpoint.cursor;
                eprintln!(
                    "[materialize-blend] checkpoint loaded source={} completed={} scanned_docs={} retained_candidates={} path={}",
                    stats.source_id,
                    cursor.completed,
                    stats.scanned_docs,
                    candidates.len(),
                    path.display()
                );
                if cursor.completed {
                    return Ok(SourceCandidates { stats, candidates });
                }
            }
        }
    }
    eprintln!(
        "[materialize-blend] scan start source={} files={} quota_tokens={}",
        stats.source_id, stats.source_file_count, stats.token_quota
    );
    let scan_start = Instant::now();
    let previous_scan_elapsed = Duration::from_millis(stats.scan_elapsed_ms);
    let mut next_progress_records =
        checkpoint_progress_threshold_usize(stats.scanned_docs, config.progress_every_records);
    let mut next_progress_bytes =
        checkpoint_progress_threshold_u64(stats.scanned_bytes, config.progress_every_bytes);
    let mut next_checkpoint_records = if let Some(control) = checkpoint_control {
        checkpoint_progress_threshold_usize(stats.scanned_docs, control.every_records)
    } else {
        0
    };
    let mut next_checkpoint_bytes = if let Some(control) = checkpoint_control {
        checkpoint_progress_threshold_u64(stats.scanned_bytes, control.every_bytes)
    } else {
        0
    };
    macro_rules! maybe_save_checkpoint {
        ($force:expr) => {
            maybe_save_materializer_source_checkpoint(MaterializerCheckpointSaveInput {
                control: checkpoint_control,
                checkpoint_path: checkpoint_path.as_deref(),
                config_hash: checkpoint_config_hash.as_deref(),
                source_files: &source_files,
                cursor: &cursor,
                stats: &mut stats,
                candidates: &candidates,
                accepted_hashes: &accepted_hashes,
                content_hash_state: content_hasher.value(),
                next_checkpoint_records: &mut next_checkpoint_records,
                next_checkpoint_bytes: &mut next_checkpoint_bytes,
                force: $force,
            })
        };
    }
    let mut score_elapsed = Duration::from_millis(stats.score_elapsed_ms);
    let mut tokenizer_encode_elapsed = Duration::from_millis(stats.tokenizer_encode_elapsed_ms);
    let mut hash_elapsed = Duration::from_millis(stats.hash_elapsed_ms);
    let mut line = String::new();
    let mut ordinal = cursor.ordinal;
    let mut completion_reason = Some("end_of_source".to_string());
    'files: for (file_index, path) in source_files.iter().enumerate().skip(cursor.file_index) {
        let file = File::open(path)
            .map_err(|err| TensorError::Io(format!("failed to open {}: {err}", path.display())))?;
        let mut reader = BufReader::new(file);
        if file_index == cursor.file_index && cursor.byte_offset > 0 {
            reader
                .seek(SeekFrom::Start(cursor.byte_offset))
                .map_err(|err| {
                    TensorError::Io(format!(
                        "failed to seek {} to checkpoint offset {}: {err}",
                        path.display(),
                        cursor.byte_offset
                    ))
                })?;
        }
        loop {
            line.clear();
            let read = reader.read_line(&mut line).map_err(|err| {
                TensorError::Io(format!("failed to read {}: {err}", path.display()))
            })?;
            if read == 0 {
                cursor = MaterializerScanCursor {
                    file_index: file_index.saturating_add(1),
                    byte_offset: 0,
                    ordinal,
                    completed: false,
                    completion_reason: None,
                };
                break;
            }
            let next_offset = reader.stream_position().map_err(|err| {
                TensorError::Io(format!(
                    "failed to get stream position for {}: {err}",
                    path.display()
                ))
            })?;
            content_hasher.update(line.as_bytes());
            stats.scanned_bytes += read as u64;
            cursor = MaterializerScanCursor {
                file_index,
                byte_offset: next_offset,
                ordinal,
                completed: false,
                completion_reason: None,
            };
            maybe_log_materializer_scan_progress(
                &stats,
                config,
                &mut next_progress_records,
                &mut next_progress_bytes,
                scan_start,
            );
            if config
                .max_source_bytes
                .is_some_and(|limit| stats.scanned_bytes as usize > limit)
            {
                stats.limited_by_max_source_bytes = true;
                completion_reason = Some("max_source_bytes".to_string());
                cursor.completed = true;
                cursor.completion_reason = completion_reason.clone();
                break 'files;
            }
            if config
                .max_docs_per_source
                .is_some_and(|limit| stats.scanned_docs >= limit)
            {
                stats.limited_by_max_docs_per_source = true;
                completion_reason = Some("max_docs_per_source".to_string());
                cursor.completed = true;
                cursor.completion_reason = completion_reason.clone();
                break 'files;
            }
            if line.trim().is_empty() {
                increment_rejection(&mut stats, "empty");
                maybe_save_checkpoint!(false)?;
                continue;
            }
            stats.scanned_docs += 1;
            maybe_log_materializer_scan_progress(
                &stats,
                config,
                &mut next_progress_records,
                &mut next_progress_bytes,
                scan_start,
            );
            let Some(rendered) = render_materializer_record(source, line.trim_end()) else {
                increment_rejection(&mut stats, "malformed");
                ordinal += 1;
                cursor.ordinal = ordinal;
                maybe_save_checkpoint!(false)?;
                if checkpoint_control
                    .and_then(|control| control.stop_after_records)
                    .is_some_and(|limit| stats.scanned_docs >= limit)
                {
                    cursor.completed = false;
                    cursor.completion_reason = Some("checkpoint_stop_after_records".to_string());
                    maybe_save_checkpoint!(true)?;
                    return Err(TensorError::InvalidOperation(format!(
                        "materializer checkpoint stop requested after {} scanned docs for source {}",
                        stats.scanned_docs, stats.source_id
                    )));
                }
                continue;
            };
            let score_start = Instant::now();
            let doc = score_materializer_doc(
                source,
                path,
                ordinal,
                rendered,
                tokenizer,
                config,
                config.candidate_text_mode == MaterializerCandidateTextMode::Retain,
            );
            score_elapsed += score_start.elapsed();
            ordinal += 1;
            cursor.ordinal = ordinal;
            let doc = match doc {
                Ok((doc, timing)) => {
                    tokenizer_encode_elapsed += timing.tokenize_elapsed;
                    hash_elapsed += timing.hash_elapsed;
                    doc
                }
                Err(rejection) => {
                    tokenizer_encode_elapsed += rejection.timing.tokenize_elapsed;
                    hash_elapsed += rejection.timing.hash_elapsed;
                    increment_rejection(&mut stats, rejection.reason.as_str());
                    continue;
                }
            };
            if !global_hashes.insert(doc.text_hash.clone()) {
                increment_rejection(&mut stats, "duplicate_exact");
                maybe_save_checkpoint!(false)?;
                if checkpoint_control
                    .and_then(|control| control.stop_after_records)
                    .is_some_and(|limit| stats.scanned_docs >= limit)
                {
                    cursor.completed = false;
                    cursor.completion_reason = Some("checkpoint_stop_after_records".to_string());
                    maybe_save_checkpoint!(true)?;
                    return Err(TensorError::InvalidOperation(format!(
                        "materializer checkpoint stop requested after {} scanned docs for source {}",
                        stats.scanned_docs, stats.source_id
                    )));
                }
                continue;
            }
            accepted_hashes.insert(doc.text_hash.clone());
            stats.candidate_docs += 1;
            stats.candidate_tokens += doc.token_count as u64;
            stats.candidate_bytes += doc.text_bytes as u64;
            candidates.push(doc);
            if stats.candidate_retention_enabled
                && stats
                    .candidate_docs
                    .is_multiple_of(config.candidate_prune_every)
            {
                let pruned = prune_materializer_candidates(
                    &mut candidates,
                    candidate_retention_token_limit,
                    config.candidate_retention_min_docs,
                );
                stats.pruned_candidate_docs += pruned.docs;
                stats.pruned_candidate_tokens =
                    stats.pruned_candidate_tokens.saturating_add(pruned.tokens);
                stats.pruned_candidate_bytes =
                    stats.pruned_candidate_bytes.saturating_add(pruned.bytes);
            }
            maybe_save_checkpoint!(false)?;
            if checkpoint_control
                .and_then(|control| control.stop_after_records)
                .is_some_and(|limit| stats.scanned_docs >= limit)
            {
                cursor.completed = false;
                cursor.completion_reason = Some("checkpoint_stop_after_records".to_string());
                maybe_save_checkpoint!(true)?;
                return Err(TensorError::InvalidOperation(format!(
                    "materializer checkpoint stop requested after {} scanned docs for source {}",
                    stats.scanned_docs, stats.source_id
                )));
            }
        }
    }
    if !cursor.completed {
        cursor.file_index = source_files.len();
        cursor.byte_offset = 0;
        cursor.ordinal = ordinal;
        cursor.completed = true;
        cursor.completion_reason = completion_reason.clone();
    }
    if stats.candidate_retention_enabled {
        let pruned = prune_materializer_candidates(
            &mut candidates,
            candidate_retention_token_limit,
            config.candidate_retention_min_docs,
        );
        stats.pruned_candidate_docs += pruned.docs;
        stats.pruned_candidate_tokens = stats.pruned_candidate_tokens.saturating_add(pruned.tokens);
        stats.pruned_candidate_bytes = stats.pruned_candidate_bytes.saturating_add(pruned.bytes);
    }
    stats.content_hash = content_hasher.finish_hex();
    let scan_elapsed = previous_scan_elapsed + scan_start.elapsed();
    let (retained_tokens, retained_bytes) = candidate_totals(&candidates);
    stats.retained_candidate_docs = candidates.len();
    stats.retained_candidate_tokens = retained_tokens;
    stats.retained_candidate_bytes = retained_bytes;
    stats.retained_candidate_text_bytes = candidates
        .iter()
        .filter_map(|candidate| candidate.text.as_ref())
        .map(|text| text.len() as u64)
        .sum::<u64>();
    stats.scan_elapsed_ms = elapsed_ms_u64(scan_elapsed);
    stats.score_elapsed_ms = elapsed_ms_u64(score_elapsed);
    stats.tokenizer_encode_elapsed_ms = elapsed_ms_u64(tokenizer_encode_elapsed);
    stats.hash_elapsed_ms = elapsed_ms_u64(hash_elapsed);
    stats.scan_bytes_per_second = rate_per_second(stats.scanned_bytes as f64, scan_elapsed);
    stats.scan_docs_per_second = rate_per_second(stats.scanned_docs as f64, scan_elapsed);
    stats.score_docs_per_second = rate_per_second(stats.scanned_docs as f64, score_elapsed);
    stats.candidate_docs_per_second = rate_per_second(stats.candidate_docs as f64, scan_elapsed);
    stats.candidate_tokens_per_second =
        rate_per_second(stats.candidate_tokens as f64, scan_elapsed);
    stats.tokenizer_encode_tokens_per_second =
        rate_per_second(stats.candidate_tokens as f64, tokenizer_encode_elapsed);
    stats.tokenizer_encode_bytes_per_second =
        rate_per_second(stats.candidate_bytes as f64, tokenizer_encode_elapsed);
    stats.checkpoint_cursor_file_index = cursor.file_index;
    stats.checkpoint_cursor_byte_offset = cursor.byte_offset;
    stats.checkpoint_completed = cursor.completed;
    stats.checkpoint_completion_reason = cursor.completion_reason.clone();
    maybe_save_checkpoint!(true)?;
    eprintln!(
        "[materialize-blend] scan done source={} scanned_docs={} candidates={} retained_candidates={} selected_quota_tokens={} rejections={} elapsed_ms={} scan_docs_per_second={:.2}",
        stats.source_id,
        stats.scanned_docs,
        stats.candidate_docs,
        stats.retained_candidate_docs,
        stats.token_quota,
        source_rejection_count(&stats),
        stats.scan_elapsed_ms,
        stats.scan_docs_per_second
    );
    if let Some(expected_hash) = &source.content_hash {
        if stats.content_hash != *expected_hash && config.max_source_bytes.is_none() {
            return Err(TensorError::InvalidOperation(format!(
                "source {} hash mismatch: manifest={} actual={}",
                source.source_id, expected_hash, stats.content_hash
            )));
        }
    }
    Ok(SourceCandidates { stats, candidates })
}

fn score_materializer_doc(
    source: &CorpusBlendSource,
    path: &Path,
    ordinal: usize,
    text: String,
    tokenizer: &BpeTokenizer,
    config: &MaterializeBlendConfig,
    retain_text: bool,
) -> std::result::Result<(MaterializedDoc, MaterializerDocTiming), MaterializerDocReject> {
    let mut timing = MaterializerDocTiming::default();
    let text_bytes = text.len();
    if text_bytes < config.min_doc_bytes {
        return Err(MaterializerDocReject {
            reason: "too_small".to_string(),
            timing,
        });
    }
    if config.max_doc_bytes.is_some_and(|limit| text_bytes > limit) {
        return Err(MaterializerDocReject {
            reason: "too_large".to_string(),
            timing,
        });
    }
    if contains_secret_like_text(&text) {
        return Err(MaterializerDocReject {
            reason: "secret_like".to_string(),
            timing,
        });
    }
    if has_pathological_repetition(&text) {
        return Err(MaterializerDocReject {
            reason: "repetition".to_string(),
            timing,
        });
    }
    let tokenize_start = Instant::now();
    let tokens = tokenizer.encode(&text, true, true);
    timing.tokenize_elapsed = tokenize_start.elapsed();
    let token_count = tokens.len();
    if token_count < 2 {
        return Err(MaterializerDocReject {
            reason: "too_few_tokens".to_string(),
            timing,
        });
    }
    let tokens_per_byte = token_count as f64 / text_bytes.max(1) as f64;
    if tokens_per_byte > config.max_tokens_per_byte {
        return Err(MaterializerDocReject {
            reason: "fertility".to_string(),
            timing,
        });
    }
    let hash_start = Instant::now();
    let text_hash = stable_hash_bytes_local(text.as_bytes());
    timing.hash_elapsed = hash_start.elapsed();
    let sample_key = stable_hash_u64(
        format!(
            "{}:{}:{}:{}",
            source.source_id,
            path.display(),
            ordinal,
            config.seed
        )
        .as_bytes(),
    );
    let score = materializer_doc_score(source, &text, token_count, tokens_per_byte);
    let text = if retain_text { Some(text) } else { None };
    Ok((
        MaterializedDoc {
            source_id: source.source_id.clone(),
            source_path: path.display().to_string(),
            ordinal,
            text,
            text_hash,
            text_bytes,
            token_count,
            tokens_per_byte,
            score,
            sample_key,
        },
        timing,
    ))
}

fn materializer_doc_score(
    source: &CorpusBlendSource,
    text: &str,
    token_count: usize,
    tokens_per_byte: f64,
) -> f64 {
    let mut score = 10.0 * source.sampling_weight;
    if source.source_id.contains("vecl_qb") {
        score += 4.0;
    }
    if source.source_id.contains("math") || source.role.contains("math") {
        score += 2.5;
    }
    if source.source_id.contains("nemotron") {
        score += 1.5;
    }
    if source.source_id.contains("olmo") {
        score += 1.5;
    }
    if text.contains("<|tool_call|>") || text.contains("<|trace|>") {
        score += 1.0;
    }
    if text.contains('{') && text.contains('}') {
        score += 0.5;
    }
    score += (token_count as f64 / 512.0).min(2.0);
    score -= tokens_per_byte.max(0.0) * 0.25;
    score
}

fn render_materializer_record(source: &CorpusBlendSource, line: &str) -> Option<String> {
    if source.data_format.contains("jsonl") {
        let value = serde_json::from_str::<serde_json::Value>(line).ok()?;
        if let Some(text) = value.get("text").and_then(serde_json::Value::as_str) {
            return Some(format!(
                "<|source_id|>{}\n<|document|>{text}\n<|record_end|>",
                source.source_id
            ));
        }
        if value.get("prompt").is_some() || value.get("target_text").is_some() {
            return Some(render_jsonl_record_for_tokenizer(source, line));
        }
        return Some(format!(
            "<|source_id|>{}\n<|document|>{}\n<|record_end|>",
            source.source_id, value
        ));
    }
    Some(format!(
        "<|source_id|>{}\n<|document|>{line}\n<|record_end|>",
        source.source_id
    ))
}

fn increment_rejection(stats: &mut SourceCurationStats, reason: &str) {
    *stats.rejections.entry(reason.to_string()).or_default() += 1;
}

fn maybe_log_materializer_scan_progress(
    stats: &SourceCurationStats,
    config: &MaterializeBlendConfig,
    next_progress_records: &mut usize,
    next_progress_bytes: &mut u64,
    scan_start: Instant,
) {
    let mut should_log = false;
    if config.progress_every_records > 0
        && *next_progress_records > 0
        && stats.scanned_docs >= *next_progress_records
    {
        should_log = true;
        while stats.scanned_docs >= *next_progress_records && *next_progress_records > 0 {
            *next_progress_records =
                match next_progress_records.checked_add(config.progress_every_records) {
                    Some(next) => next,
                    None => usize::MAX,
                };
            if *next_progress_records == usize::MAX {
                break;
            }
        }
    }
    if config.progress_every_bytes > 0
        && *next_progress_bytes > 0
        && stats.scanned_bytes >= *next_progress_bytes
    {
        should_log = true;
        while stats.scanned_bytes >= *next_progress_bytes && *next_progress_bytes > 0 {
            *next_progress_bytes = next_progress_bytes
                .checked_add(config.progress_every_bytes)
                .unwrap_or(u64::MAX);
            if *next_progress_bytes == u64::MAX {
                break;
            }
        }
    }
    if should_log {
        let elapsed = scan_start.elapsed();
        eprintln!(
            "[materialize-blend] scan progress source={} scanned_docs={} scanned_bytes={} candidates={} rejections={} elapsed_ms={} scan_docs_per_second={:.2}",
            stats.source_id,
            stats.scanned_docs,
            stats.scanned_bytes,
            stats.candidate_docs,
            source_rejection_count(stats),
            elapsed_ms_u64(elapsed),
            rate_per_second(stats.scanned_docs as f64, elapsed)
        );
    }
}

fn contains_secret_like_text(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("begin rsa private key")
        || lower.contains("begin openssh private key")
        || lower.contains("password=")
        || lower.contains("api_key=")
        || lower.contains("secret_key=")
        || text.contains("AKIA")
}

fn has_pathological_repetition(text: &str) -> bool {
    if text.len() < 64 {
        return false;
    }
    let mut longest_run = 1usize;
    let mut current_run = 1usize;
    let mut previous = '\0';
    for ch in text.chars() {
        if ch == previous {
            current_run += 1;
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 1;
            previous = ch;
        }
    }
    if longest_run >= 64 {
        return true;
    }
    let lines = text.lines().filter(|line| !line.trim().is_empty());
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut total = 0usize;
    for line in lines {
        total += 1;
        *counts.entry(line.trim()).or_default() += 1;
    }
    total >= 4 && counts.values().copied().max().unwrap_or(0) * 2 > total
}

fn write_selected_docs_dry_run(
    selected_docs_path: &Path,
    sources: &[SourceCandidates],
    selected_hashes: &BTreeSet<String>,
    valid_fraction: f64,
) -> Result<()> {
    let mut file = File::create(selected_docs_path).map_err(|err| {
        TensorError::Io(format!(
            "failed to create {}: {err}",
            selected_docs_path.display()
        ))
    })?;
    let split_assignments = build_materialized_split_assignments(selected_hashes, valid_fraction)
        .unwrap_or_else(|_| {
            selected_hashes
                .iter()
                .map(|hash| (hash.clone(), "train"))
                .collect()
        });
    for source in sources {
        for doc in &source.candidates {
            if !selected_hashes.contains(&doc.text_hash) {
                continue;
            }
            let record = MaterializedDocRecord {
                source_id: doc.source_id.clone(),
                source_path: doc.source_path.clone(),
                ordinal: doc.ordinal,
                text_hash: doc.text_hash.clone(),
                text_bytes: doc.text_bytes,
                token_count: doc.token_count,
                tokens_per_byte: doc.tokens_per_byte,
                score: doc.score,
                sample_key: doc.sample_key,
                split: split_assignments
                    .get(&doc.text_hash)
                    .copied()
                    .unwrap_or("train")
                    .to_string(),
                text_shard_path: None,
            };
            let line = serde_json::to_string(&record).map_err(|err| {
                TensorError::Io(format!("failed to serialize selected document: {err}"))
            })?;
            writeln!(file, "{line}").map_err(|err| {
                TensorError::Io(format!(
                    "failed to write {}: {err}",
                    selected_docs_path.display()
                ))
            })?;
        }
    }
    Ok(())
}

fn write_materialized_outputs(
    config: &MaterializeBlendConfig,
    tokenizer: &BpeTokenizer,
    sources: &[SourceCandidates],
    selected_hashes: &BTreeSet<String>,
    selected_docs_path: &Path,
) -> Result<MaterializedOutputReport> {
    let text_shards_dir = config.out_dir.join("text-shards");
    let prepared_dir = config.out_dir.join("prepared");
    let token_shards_dir = prepared_dir.join("shards");
    fs::create_dir_all(&text_shards_dir).map_err(|err| {
        TensorError::Io(format!(
            "failed to create {}: {err}",
            text_shards_dir.display()
        ))
    })?;
    fs::create_dir_all(&token_shards_dir).map_err(|err| {
        TensorError::Io(format!(
            "failed to create {}: {err}",
            token_shards_dir.display()
        ))
    })?;
    let mut selected_docs_file = File::create(selected_docs_path).map_err(|err| {
        TensorError::Io(format!(
            "failed to create {}: {err}",
            selected_docs_path.display()
        ))
    })?;
    let mut text_writer =
        MaterializedTextShardWriter::new(&text_shards_dir, config.text_shard_bytes)?;
    let mut train_writer = MaterializedTokenShardWriter::new(
        "train",
        &token_shards_dir,
        &prepared_dir,
        config.shard_tokens,
    );
    let mut valid_writer = MaterializedTokenShardWriter::new(
        "valid",
        &token_shards_dir,
        &prepared_dir,
        config.shard_tokens,
    );
    let mut prepared_sources = BTreeMap::<String, PreparedDataSource>::new();
    let mut source_hash_input = Vec::new();
    let mut source_bytes = 0usize;
    let mut output_stats = MaterializedOutputStats::default();
    let mut written_hashes = BTreeSet::new();
    let split_assignments =
        build_materialized_split_assignments(selected_hashes, config.valid_fraction)?;

    for source in sources {
        if written_hashes.len() == selected_hashes.len() {
            break;
        }
        let source_write_start = Instant::now();
        let mut selected_docs = source
            .candidates
            .iter()
            .filter(|doc| selected_hashes.contains(&doc.text_hash))
            .collect::<Vec<_>>();
        selected_docs.sort_by(|a, b| {
            a.ordinal
                .cmp(&b.ordinal)
                .then_with(|| a.source_path.cmp(&b.source_path))
                .then_with(|| a.text_hash.cmp(&b.text_hash))
        });
        let mut source_write_stats = MaterializedSourceWriteStats::default();
        for doc in selected_docs {
            if !written_hashes.insert(doc.text_hash.clone()) {
                continue;
            }
            let doc_text = doc.text.as_ref().ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "candidate {} was selected without retained text; use --candidate-text-mode retain or the rescan writer",
                    doc.text_hash
                ))
            })?;
            let split = split_assignments
                .get(&doc.text_hash)
                .copied()
                .unwrap_or("train");
            let text_shard_path = text_writer.write_doc(doc_text)?;
            let tokens = tokenizer.encode(doc_text, true, true);
            if split == "train" {
                train_writer.push_tokens(&tokens)?;
            } else {
                valid_writer.push_tokens(&tokens)?;
            }
            let source_entry = prepared_sources
                .entry(source.stats.source_id.clone())
                .or_insert_with(|| PreparedDataSource {
                    source_id: source.stats.source_id.clone(),
                    path: source
                        .stats
                        .path
                        .clone()
                        .unwrap_or_else(|| doc.source_path.clone()),
                    bytes: 0,
                    hash: String::new(),
                    tokens: 0,
                });
            source_entry.bytes += doc.text_bytes;
            source_entry.tokens += tokens.len();
            source_hash_input.extend_from_slice(source.stats.source_id.as_bytes());
            source_hash_input.push(0);
            source_hash_input.extend_from_slice(doc.text_hash.as_bytes());
            source_hash_input.push(0);
            source_bytes += doc.text_bytes;
            source_write_stats.docs += 1;
            source_write_stats.tokens += tokens.len() as u64;
            source_write_stats.bytes += doc.text_bytes as u64;
            output_stats.written_docs += 1;
            output_stats.written_tokens += tokens.len() as u64;
            output_stats.written_bytes += doc.text_bytes as u64;
            let record = MaterializedDocRecord {
                source_id: doc.source_id.clone(),
                source_path: doc.source_path.clone(),
                ordinal: doc.ordinal,
                text_hash: doc.text_hash.clone(),
                text_bytes: doc.text_bytes,
                token_count: doc.token_count,
                tokens_per_byte: doc.tokens_per_byte,
                score: doc.score,
                sample_key: doc.sample_key,
                split: split.to_string(),
                text_shard_path: Some(path_for_manifest_local(&text_shard_path, &config.out_dir)),
            };
            let line = serde_json::to_string(&record).map_err(|err| {
                TensorError::Io(format!("failed to serialize selected document: {err}"))
            })?;
            writeln!(selected_docs_file, "{line}").map_err(|err| {
                TensorError::Io(format!(
                    "failed to write {}: {err}",
                    selected_docs_path.display()
                ))
            })?;
        }
        let source_write_elapsed = source_write_start.elapsed();
        source_write_stats.elapsed_ms = elapsed_ms_u64(source_write_elapsed);
        source_write_stats.bytes_per_second =
            rate_per_second(source_write_stats.bytes as f64, source_write_elapsed);
        source_write_stats.tokens_per_second =
            rate_per_second(source_write_stats.tokens as f64, source_write_elapsed);
        if source_write_stats.docs > 0 {
            output_stats
                .per_source
                .insert(source.stats.source_id.clone(), source_write_stats);
        }
    }
    if written_hashes.len() != selected_hashes.len() {
        return Err(TensorError::InvalidOperation(format!(
            "materialized output writer saw {} selected hashes but wrote {}; this indicates an internal selection/write mismatch",
            selected_hashes.len(),
            written_hashes.len()
        )));
    }

    let train_finished = train_writer.finish()?;
    let valid_finished = valid_writer.finish()?;
    if train_finished.shards.is_empty() || valid_finished.shards.is_empty() {
        return Err(TensorError::InvalidOperation(format!(
            "materialized blend requires non-empty train and valid shards, got train={} valid={}; increase docs or valid_fraction",
        train_finished.total_tokens,
        valid_finished.total_tokens
    )));
    }
    output_stats.train_tokens = train_finished.total_tokens;
    output_stats.valid_tokens = valid_finished.total_tokens;
    output_stats.text_shards = text_writer.shards_written();
    output_stats.train_token_shards = train_finished.shards.len();
    output_stats.valid_token_shards = valid_finished.shards.len();
    let mut sources_vec = prepared_sources.into_values().collect::<Vec<_>>();
    sources_vec.sort_by(|a, b| a.source_id.cmp(&b.source_id));
    for source in &mut sources_vec {
        source.hash = stable_hash_bytes_local(
            format!("{}:{}:{}", source.source_id, source.bytes, source.tokens).as_bytes(),
        );
    }
    let manifest_path = prepared_dir.join("manifest.json");
    let prepared_manifest = PreparedDataManifest {
        format: DATASET_MANIFEST_FORMAT.to_string(),
        version: DATASET_MANIFEST_VERSION_V2,
        storage: DATASET_STORAGE_BINARY_SHARDS.to_string(),
        source_path: selected_docs_path.display().to_string(),
        source_bytes,
        source_hash: stable_hash_bytes_local(&source_hash_input),
        tokenizer_path: path_for_manifest_local(&config.tokenizer, &prepared_dir),
        tokenizer_hash: tokenizer.fingerprint()?,
        train_tokens_path: String::new(),
        valid_tokens_path: String::new(),
        train_tokens: train_finished.total_tokens,
        valid_tokens: valid_finished.total_tokens,
        train_hash: train_finished.token_hash,
        valid_hash: valid_finished.token_hash,
        split: DataSplit {
            kind: "deterministic_hash_valid".to_string(),
            valid_fraction: config.valid_fraction,
        },
        sources: sources_vec,
        train_shards: train_finished.shards,
        valid_shards: valid_finished.shards,
    };
    prepared_manifest.validate()?;
    write_json_file(
        &manifest_path,
        serde_json::to_value(&prepared_manifest)
            .map_err(|err| TensorError::Io(format!("failed to serialize manifest: {err}")))?,
    )?;
    let reloaded = PreparedTokenData::load(&manifest_path)?;
    drop(reloaded);
    Ok(MaterializedOutputReport {
        manifest_path,
        stats: output_stats,
    })
}

fn write_materialized_outputs_rescan(
    config: &MaterializeBlendConfig,
    tokenizer: &BpeTokenizer,
    sources: &[&CorpusBlendSource],
    selected_hashes: &BTreeSet<String>,
    manifest_root: Option<&Path>,
    selected_docs_path: &Path,
) -> Result<MaterializedOutputReport> {
    let text_shards_dir = config.out_dir.join("text-shards");
    let prepared_dir = config.out_dir.join("prepared");
    let token_shards_dir = prepared_dir.join("shards");
    fs::create_dir_all(&text_shards_dir).map_err(|err| {
        TensorError::Io(format!(
            "failed to create {}: {err}",
            text_shards_dir.display()
        ))
    })?;
    fs::create_dir_all(&token_shards_dir).map_err(|err| {
        TensorError::Io(format!(
            "failed to create {}: {err}",
            token_shards_dir.display()
        ))
    })?;
    let mut selected_docs_file = File::create(selected_docs_path).map_err(|err| {
        TensorError::Io(format!(
            "failed to create {}: {err}",
            selected_docs_path.display()
        ))
    })?;
    let mut text_writer =
        MaterializedTextShardWriter::new(&text_shards_dir, config.text_shard_bytes)?;
    let mut train_writer = MaterializedTokenShardWriter::new(
        "train",
        &token_shards_dir,
        &prepared_dir,
        config.shard_tokens,
    );
    let mut valid_writer = MaterializedTokenShardWriter::new(
        "valid",
        &token_shards_dir,
        &prepared_dir,
        config.shard_tokens,
    );
    let mut prepared_sources = BTreeMap::<String, PreparedDataSource>::new();
    let mut source_hash_input = Vec::new();
    let mut source_bytes = 0usize;
    let mut output_stats = MaterializedOutputStats::default();
    let mut written_hashes = BTreeSet::new();
    let split_assignments =
        build_materialized_split_assignments(selected_hashes, config.valid_fraction)?;

    for source in sources {
        if written_hashes.len() == selected_hashes.len() {
            break;
        }
        let source_write_start = Instant::now();
        let source_path = resolve_source_path(source.path.as_deref().unwrap(), manifest_root);
        let source_files = resolve_source_files(source.path.as_deref().unwrap(), manifest_root)?;
        let mut source_write_stats = MaterializedSourceWriteStats::default();
        let mut line = String::new();
        let mut ordinal = 0usize;
        let mut scanned_bytes = 0usize;
        let mut scanned_docs = 0usize;
        'files: for path in &source_files {
            let file = File::open(path).map_err(|err| {
                TensorError::Io(format!("failed to open {}: {err}", path.display()))
            })?;
            let mut reader = BufReader::new(file);
            loop {
                if written_hashes.len() == selected_hashes.len() {
                    break 'files;
                }
                line.clear();
                let read = reader.read_line(&mut line).map_err(|err| {
                    TensorError::Io(format!("failed to read {}: {err}", path.display()))
                })?;
                if read == 0 {
                    break;
                }
                scanned_bytes = scanned_bytes.saturating_add(read);
                if config
                    .max_source_bytes
                    .is_some_and(|limit| scanned_bytes > limit)
                {
                    break 'files;
                }
                if config
                    .max_docs_per_source
                    .is_some_and(|limit| scanned_docs >= limit)
                {
                    break 'files;
                }
                if line.trim().is_empty() {
                    continue;
                }
                scanned_docs = scanned_docs.saturating_add(1);
                let Some(rendered) = render_materializer_record(source, line.trim_end()) else {
                    ordinal += 1;
                    continue;
                };
                let rendered_hash = stable_hash_bytes_local(rendered.as_bytes());
                if !selected_hashes.contains(&rendered_hash) {
                    ordinal += 1;
                    continue;
                }
                let doc = match score_materializer_doc(
                    source, path, ordinal, rendered, tokenizer, config, true,
                ) {
                    Ok((doc, _timing)) => doc,
                    Err(_) => {
                        ordinal += 1;
                        continue;
                    }
                };
                ordinal += 1;
                if !written_hashes.insert(doc.text_hash.clone()) {
                    continue;
                }
                let doc_text = doc.text.as_ref().ok_or_else(|| {
                    TensorError::InvalidOperation(format!(
                        "rescan writer produced selected candidate {} without text",
                        doc.text_hash
                    ))
                })?;
                let split = split_assignments
                    .get(&doc.text_hash)
                    .copied()
                    .unwrap_or("train");
                let text_shard_path = text_writer.write_doc(doc_text)?;
                let tokens = tokenizer.encode(doc_text, true, true);
                if split == "train" {
                    train_writer.push_tokens(&tokens)?;
                } else {
                    valid_writer.push_tokens(&tokens)?;
                }
                let source_entry = prepared_sources
                    .entry(source.source_id.clone())
                    .or_insert_with(|| PreparedDataSource {
                        source_id: source.source_id.clone(),
                        path: source_path.display().to_string(),
                        bytes: 0,
                        hash: String::new(),
                        tokens: 0,
                    });
                source_entry.bytes += doc.text_bytes;
                source_entry.tokens += tokens.len();
                source_hash_input.extend_from_slice(source.source_id.as_bytes());
                source_hash_input.push(0);
                source_hash_input.extend_from_slice(doc.text_hash.as_bytes());
                source_hash_input.push(0);
                source_bytes += doc.text_bytes;
                source_write_stats.docs += 1;
                source_write_stats.tokens += tokens.len() as u64;
                source_write_stats.bytes += doc.text_bytes as u64;
                output_stats.written_docs += 1;
                output_stats.written_tokens += tokens.len() as u64;
                output_stats.written_bytes += doc.text_bytes as u64;
                let record = MaterializedDocRecord {
                    source_id: doc.source_id,
                    source_path: doc.source_path,
                    ordinal: doc.ordinal,
                    text_hash: doc.text_hash,
                    text_bytes: doc.text_bytes,
                    token_count: doc.token_count,
                    tokens_per_byte: doc.tokens_per_byte,
                    score: doc.score,
                    sample_key: doc.sample_key,
                    split: split.to_string(),
                    text_shard_path: Some(path_for_manifest_local(
                        &text_shard_path,
                        &config.out_dir,
                    )),
                };
                let line = serde_json::to_string(&record).map_err(|err| {
                    TensorError::Io(format!("failed to serialize selected document: {err}"))
                })?;
                writeln!(selected_docs_file, "{line}").map_err(|err| {
                    TensorError::Io(format!(
                        "failed to write {}: {err}",
                        selected_docs_path.display()
                    ))
                })?;
            }
        }
        let source_write_elapsed = source_write_start.elapsed();
        source_write_stats.elapsed_ms = elapsed_ms_u64(source_write_elapsed);
        source_write_stats.bytes_per_second =
            rate_per_second(source_write_stats.bytes as f64, source_write_elapsed);
        source_write_stats.tokens_per_second =
            rate_per_second(source_write_stats.tokens as f64, source_write_elapsed);
        if source_write_stats.docs > 0 {
            output_stats
                .per_source
                .insert(source.source_id.clone(), source_write_stats);
        }
    }
    if written_hashes.len() != selected_hashes.len() {
        return Err(TensorError::InvalidOperation(format!(
            "rescan output writer saw {} selected hashes but wrote {}; this indicates an internal selection/write mismatch",
            selected_hashes.len(),
            written_hashes.len()
        )));
    }

    let train_finished = train_writer.finish()?;
    let valid_finished = valid_writer.finish()?;
    if train_finished.shards.is_empty() || valid_finished.shards.is_empty() {
        return Err(TensorError::InvalidOperation(format!(
            "materialized blend requires non-empty train and valid shards, got train={} valid={}; increase docs or valid_fraction",
            train_finished.total_tokens,
            valid_finished.total_tokens
        )));
    }
    output_stats.train_tokens = train_finished.total_tokens;
    output_stats.valid_tokens = valid_finished.total_tokens;
    output_stats.text_shards = text_writer.shards_written();
    output_stats.train_token_shards = train_finished.shards.len();
    output_stats.valid_token_shards = valid_finished.shards.len();
    let mut sources_vec = prepared_sources.into_values().collect::<Vec<_>>();
    sources_vec.sort_by(|a, b| a.source_id.cmp(&b.source_id));
    for source in &mut sources_vec {
        source.hash = stable_hash_bytes_local(
            format!("{}:{}:{}", source.source_id, source.bytes, source.tokens).as_bytes(),
        );
    }
    let manifest_path = prepared_dir.join("manifest.json");
    let prepared_manifest = PreparedDataManifest {
        format: DATASET_MANIFEST_FORMAT.to_string(),
        version: DATASET_MANIFEST_VERSION_V2,
        storage: DATASET_STORAGE_BINARY_SHARDS.to_string(),
        source_path: selected_docs_path.display().to_string(),
        source_bytes,
        source_hash: stable_hash_bytes_local(&source_hash_input),
        tokenizer_path: path_for_manifest_local(&config.tokenizer, &prepared_dir),
        tokenizer_hash: tokenizer.fingerprint()?,
        train_tokens_path: String::new(),
        valid_tokens_path: String::new(),
        train_tokens: train_finished.total_tokens,
        valid_tokens: valid_finished.total_tokens,
        train_hash: train_finished.token_hash,
        valid_hash: valid_finished.token_hash,
        split: DataSplit {
            kind: "deterministic_hash_valid".to_string(),
            valid_fraction: config.valid_fraction,
        },
        sources: sources_vec,
        train_shards: train_finished.shards,
        valid_shards: valid_finished.shards,
    };
    prepared_manifest.validate()?;
    write_json_file(
        &manifest_path,
        serde_json::to_value(&prepared_manifest)
            .map_err(|err| TensorError::Io(format!("failed to serialize manifest: {err}")))?,
    )?;
    let reloaded = PreparedTokenData::load(&manifest_path)?;
    drop(reloaded);
    Ok(MaterializedOutputReport {
        manifest_path,
        stats: output_stats,
    })
}

struct MaterializedTextShardWriter {
    dir: PathBuf,
    max_bytes: usize,
    index: usize,
    current_bytes: usize,
    current_path: PathBuf,
    file: File,
}

impl MaterializedTextShardWriter {
    fn new(dir: &Path, max_bytes: usize) -> Result<Self> {
        fs::create_dir_all(dir)
            .map_err(|err| TensorError::Io(format!("failed to create {}: {err}", dir.display())))?;
        let current_path = dir.join("text-00000.txt");
        let file = File::create(&current_path).map_err(|err| {
            TensorError::Io(format!(
                "failed to create {}: {err}",
                current_path.display()
            ))
        })?;
        Ok(Self {
            dir: dir.to_path_buf(),
            max_bytes,
            index: 0,
            current_bytes: 0,
            current_path,
            file,
        })
    }

    fn write_doc(&mut self, text: &str) -> Result<PathBuf> {
        let bytes = text.len() + 1;
        if self.current_bytes > 0 && self.current_bytes + bytes > self.max_bytes {
            self.index += 1;
            self.current_bytes = 0;
            self.current_path = self.dir.join(format!("text-{:05}.txt", self.index));
            self.file = File::create(&self.current_path).map_err(|err| {
                TensorError::Io(format!(
                    "failed to create {}: {err}",
                    self.current_path.display()
                ))
            })?;
        }
        writeln!(self.file, "{text}").map_err(|err| {
            TensorError::Io(format!(
                "failed to write {}: {err}",
                self.current_path.display()
            ))
        })?;
        self.current_bytes += bytes;
        Ok(self.current_path.clone())
    }

    fn shards_written(&self) -> usize {
        self.index + usize::from(self.current_bytes > 0)
    }
}

struct MaterializedTokenShardWriter {
    split: String,
    shards_dir: PathBuf,
    manifest_root: PathBuf,
    shard_tokens: usize,
    index: usize,
    buffer: Vec<usize>,
    total_tokens: usize,
    token_hash: StableFnv64,
    shards: Vec<PreparedDataShard>,
}

struct FinishedTokenShards {
    shards: Vec<PreparedDataShard>,
    total_tokens: usize,
    token_hash: String,
}

impl MaterializedTokenShardWriter {
    fn new(split: &str, shards_dir: &Path, manifest_root: &Path, shard_tokens: usize) -> Self {
        Self {
            split: split.to_string(),
            shards_dir: shards_dir.to_path_buf(),
            manifest_root: manifest_root.to_path_buf(),
            shard_tokens,
            index: 0,
            buffer: Vec::with_capacity(shard_tokens.min(1_000_000)),
            total_tokens: 0,
            token_hash: StableFnv64::new(),
            shards: Vec::new(),
        }
    }

    fn push_tokens(&mut self, tokens: &[usize]) -> Result<()> {
        for &token in tokens {
            self.buffer.push(token);
            self.token_hash.update_token(token);
            self.total_tokens += 1;
            if self.buffer.len() >= self.shard_tokens {
                self.flush()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let payload_path = self
            .shards_dir
            .join(format!("{}-{:05}.tokens.bin", self.split, self.index));
        let metadata_path = self
            .shards_dir
            .join(format!("{}-{:05}.tokens.json", self.split, self.index));
        let metadata = write_token_shard(&payload_path, &metadata_path, &self.buffer)?;
        self.shards.push(PreparedDataShard {
            split: self.split.clone(),
            metadata_path: path_for_manifest_local(&metadata_path, &self.manifest_root),
            tokens: metadata.token_count,
            token_hash: metadata.token_hash,
        });
        self.buffer.clear();
        self.index += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<FinishedTokenShards> {
        self.flush()?;
        Ok(FinishedTokenShards {
            shards: self.shards,
            total_tokens: self.total_tokens,
            token_hash: self.token_hash.finish_hex(),
        })
    }
}

fn split_for_doc_hash(hash: &str, valid_fraction: f64) -> &'static str {
    if valid_fraction <= 0.0 {
        return "train";
    }
    let key = stable_hash_u64(hash.as_bytes()) as f64 / u64::MAX as f64;
    if key < valid_fraction {
        "valid"
    } else {
        "train"
    }
}

fn build_materialized_split_assignments(
    selected_hashes: &BTreeSet<String>,
    valid_fraction: f64,
) -> Result<BTreeMap<String, &'static str>> {
    if selected_hashes.len() < 2 {
        return Err(TensorError::InvalidOperation(
            "materialized blend requires at least two selected documents for train/valid splits"
                .to_string(),
        ));
    }
    let mut assignments = BTreeMap::new();
    let mut train_count = 0usize;
    let mut valid_count = 0usize;
    for hash in selected_hashes {
        let split = split_for_doc_hash(hash, valid_fraction);
        if split == "valid" {
            valid_count += 1;
        } else {
            train_count += 1;
        }
        assignments.insert(hash.clone(), split);
    }
    if valid_count == 0 {
        if let Some(hash) = selected_hashes.iter().next() {
            assignments.insert(hash.clone(), "valid");
        }
    } else if train_count == 0 {
        if let Some(hash) = selected_hashes.iter().next() {
            assignments.insert(hash.clone(), "train");
        }
    }
    Ok(assignments)
}

fn path_for_manifest_local(path: &Path, manifest_root: &Path) -> String {
    let absolute_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let absolute_root =
        fs::canonicalize(manifest_root).unwrap_or_else(|_| manifest_root.to_path_buf());
    absolute_path
        .strip_prefix(absolute_root)
        .unwrap_or(&absolute_path)
        .display()
        .to_string()
}

fn materialize_mode_label(mode: MaterializeBlendMode) -> &'static str {
    match mode {
        MaterializeBlendMode::DryRun => "dry-run",
        MaterializeBlendMode::Sample => "sample",
        MaterializeBlendMode::Full => "full",
    }
}

fn tokenizer_encode_bench_report(config: TokenizerEncodeBenchConfig) -> Result<serde_json::Value> {
    if config.input.is_empty() {
        return Err(TensorError::InvalidOperation(
            "tokenizer bench-encode requires at least one --input".to_string(),
        ));
    }
    if config.max_bytes == 0 {
        return Err(TensorError::InvalidOperation(
            "--max-bytes must be greater than zero".to_string(),
        ));
    }
    if config.iterations == 0 {
        return Err(TensorError::InvalidOperation(
            "--iterations must be greater than zero".to_string(),
        ));
    }
    let tokenizer = BpeTokenizer::load(&config.tokenizer_path)?;
    let input_files = tokenizer_bench_input_files(&config.input)?;
    let samples = collect_tokenizer_bench_samples(&input_files, config.max_bytes)?;
    if samples.is_empty() {
        return Err(TensorError::InvalidOperation(
            "tokenizer bench-encode collected zero non-empty samples".to_string(),
        ));
    }
    let sampled_bytes = samples.iter().map(|sample| sample.len()).sum::<usize>();

    let warmup_start = Instant::now();
    let mut warmup_tokens = 0usize;
    for sample in &samples {
        warmup_tokens += tokenizer
            .encode_bytes(sample, config.add_bos, config.add_eos)
            .len();
    }
    let warmup_elapsed = warmup_start.elapsed();

    let encode_start = Instant::now();
    let mut total_tokens = 0usize;
    for _ in 0..config.iterations {
        for sample in &samples {
            let encoded = tokenizer.encode_bytes(sample, config.add_bos, config.add_eos);
            total_tokens += std::hint::black_box(encoded.len());
        }
    }
    let encode_elapsed = encode_start.elapsed();
    let total_bytes = sampled_bytes.saturating_mul(config.iterations);
    let total_docs = samples.len().saturating_mul(config.iterations);
    let elapsed_ms = elapsed_ms_u64(encode_elapsed);
    let elapsed_secs = encode_elapsed.as_secs_f64();
    let tokenizer_report = tokenizer_metadata_report(&tokenizer, Some(&config.tokenizer_path))?;
    Ok(serde_json::json!({
        "command": "tokenizer bench-encode",
        "status": "passed",
        "tokenizer": tokenizer_report,
        "input_files": input_files.iter().map(|path| path.display().to_string()).collect::<Vec<_>>(),
        "sample": {
            "documents": samples.len(),
            "bytes": sampled_bytes,
            "max_bytes": config.max_bytes,
            "average_bytes_per_document": sampled_bytes as f64 / samples.len().max(1) as f64,
        },
        "settings": {
            "iterations": config.iterations,
            "add_bos": config.add_bos,
            "add_eos": config.add_eos,
            "digit_isolation": tokenizer.digit_isolation_enabled(),
            "reserved_tokens": tokenizer.reserved_tokens().len(),
        },
        "timing": {
            "warmup_elapsed_ms": elapsed_ms_u64(warmup_elapsed),
            "encode_elapsed_ms": elapsed_ms,
        },
        "totals": {
            "warmup_tokens": warmup_tokens,
            "encoded_documents": total_docs,
            "encoded_bytes": total_bytes,
            "encoded_tokens": total_tokens,
        },
        "throughput": {
            "documents_per_second": rate_per_second(total_docs as f64, encode_elapsed),
            "bytes_per_second": rate_per_second(total_bytes as f64, encode_elapsed),
            "tokens_per_second": rate_per_second(total_tokens as f64, encode_elapsed),
            "tokens_per_byte": total_tokens as f64 / total_bytes.max(1) as f64,
            "average_tokens_per_document": total_tokens as f64 / total_docs.max(1) as f64,
            "elapsed_seconds": elapsed_secs,
        },
    }))
}

fn tokenizer_bench_input_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        let Some(path_text) = input.to_str() else {
            return Err(TensorError::InvalidOperation(format!(
                "input path is not valid UTF-8: {}",
                input.display()
            )));
        };
        files.extend(resolve_source_files(path_text, None)?);
    }
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(TensorError::InvalidOperation(
            "tokenizer bench-encode resolved zero input files".to_string(),
        ));
    }
    Ok(files)
}

fn collect_tokenizer_bench_samples(paths: &[PathBuf], max_bytes: usize) -> Result<Vec<Vec<u8>>> {
    let mut samples = Vec::new();
    let mut sampled_bytes = 0usize;
    let mut line = Vec::new();
    'files: for path in paths {
        let file = File::open(path)
            .map_err(|err| TensorError::Io(format!("failed to open {}: {err}", path.display())))?;
        let mut reader = BufReader::new(file);
        loop {
            line.clear();
            let read = reader.read_until(b'\n', &mut line).map_err(|err| {
                TensorError::Io(format!("failed to read {}: {err}", path.display()))
            })?;
            if read == 0 {
                break;
            }
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }
            if line.is_empty() {
                continue;
            }
            let remaining = max_bytes.saturating_sub(sampled_bytes);
            if remaining == 0 {
                break 'files;
            }
            let take = remaining.min(line.len());
            samples.push(line[..take].to_vec());
            sampled_bytes = sampled_bytes.saturating_add(take);
            if sampled_bytes >= max_bytes {
                break 'files;
            }
        }
    }
    Ok(samples)
}

fn train_tokenizer_from_corpus_blend(
    config: TokenizerCorpusTrainConfig,
) -> Result<TokenizerCorpusTrainOutcome> {
    let total_start = Instant::now();
    fs::create_dir_all(&config.work_dir).map_err(|err| {
        TensorError::Io(format!(
            "failed to create tokenizer work dir {}: {err}",
            config.work_dir.display()
        ))
    })?;
    let manifest_bytes = fs::read(&config.corpus_blend).map_err(|err| {
        TensorError::Io(format!(
            "failed to read corpus blend {}: {err}",
            config.corpus_blend.display()
        ))
    })?;
    let manifest: CorpusBlendManifest = serde_json::from_slice(&manifest_bytes).map_err(|err| {
        TensorError::Io(format!(
            "failed to parse corpus blend {}: {err}",
            config.corpus_blend.display()
        ))
    })?;
    let reserved_tokens = load_reserved_token_registry(config.reserved_tokens.as_deref())?;
    let manifest_root = config.corpus_blend.parent();
    let sample_start = Instant::now();
    let materialized = materialize_tokenizer_samples(
        &manifest,
        &allow_license_statuses(config.allow_license_status),
        config.sample_bytes,
        config.seed,
        Some(&config.work_dir),
        manifest_root,
    )?;
    let sample_elapsed_ms = sample_start.elapsed().as_millis() as u64;
    let train_start = Instant::now();
    let tokenizer = BpeTokenizer::train_v2(
        &materialized.samples,
        reserved_tokens,
        BpeTokenizerV2Options {
            tokenizer_id: format!("heirloom-byte-bpe-{}-v2", config.vocab_size),
            vocab_size: config.vocab_size,
            sample_bytes: config.sample_bytes,
            seed: config.seed,
            memory_limit_bytes: config.memory_limit_bytes,
            source_blend_hash: stable_hash_bytes_local(&manifest_bytes),
            sample_manifest_hash: materialized.sample_manifest_hash.clone(),
            digit_isolation: true,
            require_exact_vocab: config.require_exact_vocab,
        },
    )?;
    let tokenizer_train_elapsed_ms = train_start.elapsed().as_millis() as u64;
    let save_start = Instant::now();
    tokenizer.save(&config.out)?;
    let tokenizer_hash = tokenizer.fingerprint()?;
    let save_hash_elapsed_ms = save_start.elapsed().as_millis() as u64;
    let total_elapsed_ms = total_start.elapsed().as_millis() as u64;
    let report_value = serde_json::json!({
        "command": "tokenizer train-corpus",
        "status": "passed",
        "corpus_blend": config.corpus_blend.display().to_string(),
        "work_dir": config.work_dir.display().to_string(),
        "sample_manifest": materialized.sample_manifest_path.as_ref().map(|path| path.display().to_string()),
        "sample_manifest_hash": materialized.sample_manifest_hash,
        "sampled_bytes": materialized.sampled_bytes,
        "tokenizer": tokenizer_metadata_report(&tokenizer, Some(&config.out))?,
        "token_length_histogram": token_length_histogram(&tokenizer),
        "source_reports": materialized.source_reports,
        "timing": {
            "sample_materialization_elapsed_ms": sample_elapsed_ms,
            "tokenizer_train_elapsed_ms": tokenizer_train_elapsed_ms,
            "save_hash_elapsed_ms": save_hash_elapsed_ms,
            "total_elapsed_ms": total_elapsed_ms,
        },
        "hard_path": {
            "native_trainer": true,
            "external_tokenizer_dependency": false,
            "tiny_stories_allowed": false,
        },
    });
    if let Some(report_path) = &config.report {
        write_json_file(report_path, report_value)?;
    }
    Ok(TokenizerCorpusTrainOutcome {
        tokenizer_path: config.out,
        tokenizer_hash,
        version: tokenizer.metadata().version,
        vocab_size: tokenizer.vocab_size(),
        reserved_tokens: tokenizer.reserved_tokens().len(),
    })
}

fn load_reserved_token_registry(path: Option<&Path>) -> Result<Vec<ReservedToken>> {
    let tokens = if let Some(path) = path {
        let json = fs::read_to_string(path).map_err(|err| {
            TensorError::Io(format!(
                "failed to read reserved token registry {}: {err}",
                path.display()
            ))
        })?;
        reserved_tokens_from_json_str(&json)?
    } else {
        default_reserved_tokens()
    };
    validate_reserved_tokens(&tokens)?;
    Ok(tokens)
}

fn allow_license_statuses(extra: Vec<String>) -> Vec<String> {
    let mut statuses = vec![
        "approved".to_string(),
        "source_terms_verified".to_string(),
        "redistribution_allowed".to_string(),
        "odc_by_verified".to_string(),
        "odc_by_internal_attribution".to_string(),
        "nvidia_data_agreement_internal_training".to_string(),
        "internal_synthetic".to_string(),
    ];
    for status in extra {
        if !statuses.contains(&status) {
            statuses.push(status);
        }
    }
    statuses
}

fn materialize_tokenizer_samples(
    manifest: &CorpusBlendManifest,
    allowed_license_statuses: &[String],
    sample_bytes: u64,
    seed: u64,
    work_dir: Option<&Path>,
    manifest_root: Option<&Path>,
) -> Result<MaterializedTokenizerSamples> {
    let sources = manifest
        .sources
        .iter()
        .filter(|source| source.include_in_tokenizer_training && source.sampling_weight > 0.0)
        .collect::<Vec<_>>();
    if sources.is_empty() {
        return Err(TensorError::InvalidOperation(
            "corpus blend has no positive-weight tokenizer training sources".to_string(),
        ));
    }
    for source in &sources {
        if !allowed_license_statuses.contains(&source.license_status) {
            return Err(TensorError::InvalidOperation(format!(
                "source {} has unapproved license_status {}; allowed={:?}",
                source.source_id, source.license_status, allowed_license_statuses
            )));
        }
        if source.path.is_none() {
            return Err(TensorError::InvalidOperation(format!(
                "source {} is included in tokenizer training but has no materialized local path",
                source.source_id
            )));
        }
    }
    let total_weight = sources
        .iter()
        .map(|source| source.sampling_weight)
        .sum::<f64>();
    if total_weight <= 0.0 || !total_weight.is_finite() {
        return Err(TensorError::InvalidOperation(
            "tokenizer corpus blend has invalid total sampling weight".to_string(),
        ));
    }

    let mut samples = Vec::new();
    let mut source_reports = Vec::new();
    let mut sampled_bytes = 0u64;
    for (index, source) in sources.iter().enumerate() {
        let mut quota =
            ((sample_bytes as f64) * source.sampling_weight / total_weight).round() as u64;
        if index + 1 == sources.len() {
            let assigned = source_reports
                .iter()
                .map(|report: &TokenizerSampleSourceReport| report.quota_bytes)
                .sum::<u64>();
            quota = sample_bytes.saturating_sub(assigned);
        }
        if source.source_id == "vecl_qb.synthetic.v1-hard" {
            quota = quota.max(source.local_bytes.unwrap_or(0) as u64);
        }
        let source_path = resolve_source_path(source.path.as_deref().unwrap(), manifest_root);
        let source_files = resolve_source_files(source.path.as_deref().unwrap(), manifest_root)?;
        let bytes = read_source_files_bytes(&source_files)?;
        if let Some(expected_hash) = &source.content_hash {
            let actual_hash = stable_hash_bytes_local(&bytes);
            if &actual_hash != expected_hash && source_files.len() == 1 {
                return Err(TensorError::InvalidOperation(format!(
                    "source {} hash mismatch: manifest={} actual={}",
                    source.source_id, expected_hash, actual_hash
                )));
            }
        }
        let content_hash = stable_hash_bytes_local(&bytes);
        let (sample_bytes_vec, records, exhausted) =
            sample_source_bytes(source, &bytes, quota, seed)?;
        let sampled_len = sample_bytes_vec.len() as u64;
        let effective_weight = if sampled_len > 0 && sampled_len < quota {
            quota.div_ceil(sampled_len).max(1)
        } else {
            1
        };
        sampled_bytes += sampled_len;
        samples.push(BpeTrainingSample {
            source_id: source.source_id.clone(),
            bytes: sample_bytes_vec,
            weight: effective_weight,
        });
        source_reports.push(TokenizerSampleSourceReport {
            source_id: source.source_id.clone(),
            display_name: source.display_name.clone(),
            path: Some(source_path.display().to_string()),
            source_url: source.source_url.clone(),
            license_status: source.license_status.clone(),
            sampling_weight: source.sampling_weight,
            quota_bytes: quota,
            sampled_bytes: sampled_len,
            effective_weight,
            exhausted,
            records,
            content_hash: Some(content_hash),
        });
    }

    let sample_manifest = serde_json::json!({
        "format": "heirloom.tokenizer_sample_manifest",
        "version": 1,
        "blend_id": &manifest.blend_id,
        "tokenizer_target_vocab_size": manifest.tokenizer_target_vocab_size,
        "seed": seed,
        "requested_sample_bytes": sample_bytes,
        "sampled_bytes": sampled_bytes,
        "sources": &source_reports,
    });
    let sample_manifest_bytes = serde_json::to_vec(&sample_manifest).map_err(|err| {
        TensorError::Io(format!(
            "failed to serialize tokenizer sample manifest: {err}"
        ))
    })?;
    let sample_manifest_hash = stable_hash_bytes_local(&sample_manifest_bytes);
    let sample_manifest_path = if let Some(work_dir) = work_dir {
        let path = work_dir.join("tokenizer-sample-manifest.json");
        write_json_file(&path, sample_manifest)?;
        Some(path)
    } else {
        None
    };
    Ok(MaterializedTokenizerSamples {
        samples,
        source_reports,
        sample_manifest_path,
        sample_manifest_hash,
        sampled_bytes,
    })
}

fn resolve_source_path(path: &str, manifest_root: Option<&Path>) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else if let Some(root) = manifest_root {
        root.join(path)
    } else {
        path
    }
}

fn resolve_source_files(path: &str, manifest_root: Option<&Path>) -> Result<Vec<PathBuf>> {
    let resolved = resolve_source_path(path, manifest_root);
    if resolved.is_file() {
        reject_compressed_source_file(&resolved)?;
        return Ok(vec![resolved]);
    }
    if !resolved.is_dir() {
        return Err(TensorError::InvalidOperation(format!(
            "source path {} is neither a file nor a directory",
            resolved.display()
        )));
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(&resolved).map_err(|err| {
        TensorError::Io(format!(
            "failed to read source directory {}: {err}",
            resolved.display()
        ))
    })? {
        let entry = entry.map_err(|err| {
            TensorError::Io(format!(
                "failed to read source directory entry {}: {err}",
                resolved.display()
            ))
        })?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            return Err(TensorError::InvalidOperation(format!(
                "source directory {} contains nested directory {}; source slice directories must be flat",
                resolved.display(),
                path.display()
            )));
        }
        if !path.is_file() {
            continue;
        }
        if name.ends_with(".report.json") || name == "metadata.json" {
            continue;
        }
        reject_compressed_source_file(&path)?;
        files.push(path);
    }
    files.sort();
    if files.is_empty() {
        return Err(TensorError::InvalidOperation(format!(
            "source directory {} contains no source slice files",
            resolved.display()
        )));
    }
    Ok(files)
}

fn reject_compressed_source_file(path: &Path) -> Result<()> {
    let path_text = path.display().to_string();
    if path_text.ends_with(".gz") || path_text.ends_with(".zst") || path_text.ends_with(".zstd") {
        return Err(TensorError::InvalidOperation(format!(
            "source file {} appears compressed; materialize/decompress it before use",
            path.display()
        )));
    }
    Ok(())
}

fn read_source_files_bytes(paths: &[PathBuf]) -> Result<Vec<u8>> {
    let total_bytes = paths
        .iter()
        .map(|path| path.metadata().map(|metadata| metadata.len()).unwrap_or(0))
        .sum::<u64>();
    let mut bytes = Vec::with_capacity(total_bytes.min(usize::MAX as u64) as usize);
    for path in paths {
        let mut file_bytes = fs::read(path).map_err(|err| {
            TensorError::Io(format!("failed to read source {}: {err}", path.display()))
        })?;
        bytes.append(&mut file_bytes);
        if !bytes.ends_with(b"\n") {
            bytes.push(b'\n');
        }
    }
    Ok(bytes)
}

fn sample_source_bytes(
    source: &CorpusBlendSource,
    bytes: &[u8],
    quota: u64,
    seed: u64,
) -> Result<(Vec<u8>, usize, bool)> {
    if quota == 0 || bytes.is_empty() {
        return Ok((Vec::new(), 0, bytes.is_empty()));
    }
    if source.source_id == "vecl_qb.synthetic.v1-hard" || quota as usize >= bytes.len() {
        if source.data_format.contains("jsonl") {
            let text = jsonl_training_text(source, bytes, usize::MAX, seed)?;
            return Ok((text.into_bytes(), count_nonempty_lines(bytes), true));
        }
        return Ok((bytes.to_vec(), 1, true));
    }
    if source.data_format.contains("jsonl") {
        let text = jsonl_training_text(source, bytes, quota as usize, seed)?;
        let exhausted = text.len() < quota as usize;
        let records = count_nonempty_lines(text.as_bytes());
        return Ok((text.into_bytes(), records, exhausted));
    }
    let start = deterministic_offset(source, bytes.len(), seed);
    let mut out = Vec::with_capacity(quota.min(bytes.len() as u64) as usize);
    let mut index = start;
    while out.len() < quota as usize && out.len() < bytes.len() {
        out.push(bytes[index]);
        index = (index + 1) % bytes.len();
    }
    let exhausted = out.len() == bytes.len();
    Ok((out, 1, exhausted))
}

fn jsonl_training_text(
    source: &CorpusBlendSource,
    bytes: &[u8],
    quota: usize,
    seed: u64,
) -> Result<String> {
    let text = String::from_utf8_lossy(bytes);
    let lines = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return Ok(String::new());
    }
    let start = deterministic_offset(source, lines.len(), seed);
    let mut out = String::new();
    for step in 0..lines.len() {
        if out.len() >= quota {
            break;
        }
        let line = lines[(start + step) % lines.len()];
        let rendered = render_jsonl_record_for_tokenizer(source, line);
        out.push_str(&rendered);
        out.push('\n');
    }
    Ok(out)
}

fn render_jsonl_record_for_tokenizer(source: &CorpusBlendSource, line: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        return line.to_string();
    };
    if let Some(text) = value.get("text").and_then(serde_json::Value::as_str) {
        return format!(
            "<|source_id|>{}\n<|document|>{text}\n<|record_end|>",
            source.source_id
        );
    }
    let prompt = value
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let target = value
        .get("target_text")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let source_id = value
        .get("source_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(source.source_id.as_str());
    let task_kind = value
        .get("task_kind")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if prompt.is_empty() && target.is_empty() && task_kind.is_empty() {
        return format!(
            "<|source_id|>{}\n<|document|>{value}\n<|record_end|>",
            source.source_id
        );
    }
    format!(
        "<|source_id|>{source_id}\n<|trace|>{task_kind}\n<|user|>{prompt}<|message_end|>\n<|assistant|>{target}<|message_end|>\n<|record_end|>"
    )
}

fn count_nonempty_lines(bytes: &[u8]) -> usize {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count()
}

fn deterministic_offset(source: &CorpusBlendSource, len: usize, seed: u64) -> usize {
    if len == 0 {
        return 0;
    }
    let mut input = Vec::new();
    input.extend_from_slice(source.source_id.as_bytes());
    input.extend_from_slice(&seed.to_le_bytes());
    (stable_hash_u64(&input) as usize) % len
}

fn tokenizer_fertility_report(
    tokenizer: &BpeTokenizer,
    tokenizer_path: Option<&Path>,
    manifest: &CorpusBlendManifest,
    materialized: MaterializedTokenizerSamples,
) -> Result<serde_json::Value> {
    let mut source_reports = Vec::new();
    let reserved_ids = tokenizer
        .reserved_tokens()
        .iter()
        .map(|token| token.id)
        .collect::<Vec<_>>();
    for (sample, source_report) in materialized
        .samples
        .iter()
        .zip(materialized.source_reports.iter())
    {
        let tokens = tokenizer.encode_bytes(&sample.bytes, false, false);
        let reserved_hits = tokens
            .iter()
            .filter(|token| reserved_ids.contains(token))
            .count();
        let byte_token_hits = tokens
            .iter()
            .filter(|token| **token >= BYTE_OFFSET && **token < BYTE_OFFSET + BYTE_VOCAB)
            .count();
        source_reports.push(serde_json::json!({
            "source_id": source_report.source_id,
            "sampled_bytes": sample.bytes.len(),
            "tokens": tokens.len(),
            "tokens_per_byte": if sample.bytes.is_empty() { 0.0 } else { tokens.len() as f64 / sample.bytes.len() as f64 },
            "bytes_per_token": if tokens.is_empty() { 0.0 } else { sample.bytes.len() as f64 / tokens.len() as f64 },
            "byte_token_share": if tokens.is_empty() { 0.0 } else { byte_token_hits as f64 / tokens.len() as f64 },
            "reserved_token_hits": reserved_hits,
            "records": source_report.records,
            "tokens_per_record": if source_report.records == 0 { 0.0 } else { tokens.len() as f64 / source_report.records as f64 },
        }));
    }
    Ok(serde_json::json!({
        "command": "tokenizer fertility",
        "status": "passed",
        "blend_id": manifest.blend_id,
        "sample_manifest_hash": materialized.sample_manifest_hash,
        "sampled_bytes": materialized.sampled_bytes,
        "tokenizer": tokenizer_metadata_report(tokenizer, tokenizer_path)?,
        "token_length_histogram": token_length_histogram(tokenizer),
        "sources": source_reports,
    }))
}

const PADAWAN_REF_PREFIX: &str = "artifact://padawan/";
const DEFAULT_PADAWAN_ALLOWED_PATH_PREFIXES: &[&str] = &[
    "padawan/",
    "src/bin/heirloom.rs",
    "PADAWAN_LOOP_DESIGN.md",
    "CODEX_GOAL_LOOP.md",
];

const LEARNING_SANITY_MANIFEST_FORMAT: &str = "heirloom.learning_sanity_ladder";
const LEARNING_SANITY_VALIDATION_FORMAT: &str = "heirloom.learning_sanity_ladder_validation";
const LEARNING_SANITY_LR_GRAD_SWEEP_FORMAT: &str =
    "heirloom.learning_sanity_lr_grad_accumulation_sweep";

fn validate_learning_sanity_manifest(manifest_path: &Path) -> Result<serde_json::Value> {
    let manifest = read_json_value(manifest_path)?;
    let manifest_object = manifest.as_object().ok_or_else(|| {
        TensorError::InvalidOperation("learning sanity manifest must be an object".to_string())
    })?;
    let format = json_field_str(manifest_object, "format")?;
    if format != LEARNING_SANITY_MANIFEST_FORMAT {
        return Err(TensorError::InvalidOperation(format!(
            "learning sanity manifest format must be {LEARNING_SANITY_MANIFEST_FORMAT}, got {format}"
        )));
    }
    let version = json_field_i64(manifest_object, "version")?;
    if version != 0 {
        return Err(TensorError::InvalidOperation(format!(
            "learning sanity manifest version must be 0, got {version}"
        )));
    }
    let stages = manifest_object
        .get("stages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            TensorError::InvalidOperation(
                "learning sanity manifest stages must be an array".to_string(),
            )
        })?;
    if stages.is_empty() {
        return Err(TensorError::InvalidOperation(
            "learning sanity manifest must contain at least one stage".to_string(),
        ));
    }
    let manifest_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut stage_reports = Vec::new();
    for (index, stage) in stages.iter().enumerate() {
        let report = validate_learning_sanity_stage(index, stage, manifest_root).unwrap_or_else(|err| {
            serde_json::json!({
                "stage_index": index,
                "stage_id": stage.get("stage_id").and_then(serde_json::Value::as_str).unwrap_or("<invalid>"),
                "status": "failed",
                "reason": err.to_string(),
            })
        });
        stage_reports.push(report);
    }
    let passed = stage_reports
        .iter()
        .filter(|stage| stage["status"].as_str() == Some("passed"))
        .count();
    let failed = stage_reports.len() - passed;
    let min_loss_reduction = stage_reports
        .iter()
        .filter(|stage| stage["status"].as_str() == Some("passed"))
        .filter_map(|stage| stage["loss_reduction"].as_f64())
        .fold(None, |acc: Option<f64>, value| {
            Some(acc.map_or(value, |current| current.min(value)))
        });
    Ok(serde_json::json!({
        "format": LEARNING_SANITY_VALIDATION_FORMAT,
        "version": 0,
        "status": if failed == 0 { "passed" } else { "failed" },
        "manifest": manifest_path.display().to_string(),
        "passed": passed,
        "failed": failed,
        "min_loss_reduction_observed": min_loss_reduction.unwrap_or(0.0),
        "stages": stage_reports,
    }))
}

fn validate_learning_sanity_stage(
    index: usize,
    stage: &serde_json::Value,
    manifest_root: &Path,
) -> Result<serde_json::Value> {
    let stage_object = stage.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!("learning sanity stage {index} must be an object"))
    })?;
    let stage_id = json_field_str(stage_object, "stage_id")?;
    validate_learning_sanity_stage_id(stage_id)?;
    let report_value = stage_object
        .get("report")
        .or_else(|| stage_object.get("report_path"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "learning sanity stage {stage_id} requires report or report_path"
            ))
        })?;
    let report_path = resolve_manifest_relative_path(manifest_root, report_value);
    let report = read_json_value(&report_path)?;
    let report_object = report.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!("{report_value} must contain a JSON object"))
    })?;
    if stage_id == "lr_grad_accumulation_sweep" {
        return validate_learning_sanity_lr_grad_accumulation_sweep(
            index,
            stage_id,
            stage_object,
            &report_path,
            report_object,
        );
    }
    if stage_id == "longer_32k_blend" {
        return validate_learning_sanity_longer_32k_blend(
            index,
            stage_id,
            stage_object,
            &report_path,
            report_object,
        );
    }
    validate_learning_sanity_report_expectations(stage_object, report_object, stage_id)?;
    if stage_id == "memory_layers_disabled" {
        require_learning_sanity_memory_layers_disabled(report_object, stage_id)?;
    }
    if stage_id == "memory_exact_smft_enabled" {
        require_learning_sanity_smft_enabled(report_object, stage_id)?;
    }
    if stage_id == "product_key_parity" {
        require_learning_sanity_product_key_parity(report_object, stage_id)?;
    }
    let min_loss_reduction = stage_object
        .get("min_loss_reduction")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let require_loss_improvement = stage_object
        .get("require_loss_improvement")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let loss = validate_learning_sanity_report_loss(
        stage_id,
        report_object,
        min_loss_reduction,
        require_loss_improvement,
    )?;
    Ok(serde_json::json!({
        "stage_index": index,
        "stage_id": stage_id,
        "status": "passed",
        "report": report_path.display().to_string(),
        "command": report_object.get("command").cloned().unwrap_or(serde_json::Value::Null),
        "model_family": report_object.get("model_family").cloned().unwrap_or(serde_json::Value::Null),
        "initial_loss": loss.initial_loss,
        "final_loss": loss.final_loss,
        "loss_reduction": loss.loss_reduction,
        "min_loss_reduction": min_loss_reduction,
    }))
}

#[derive(Clone, Copy, Debug)]
struct LearningSanityLossSummary {
    initial_loss: f64,
    final_loss: f64,
    loss_reduction: f64,
}

fn validate_learning_sanity_report_expectations(
    expectations: &serde_json::Map<String, serde_json::Value>,
    report_object: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
) -> Result<()> {
    if let Some(expected) = optional_json_str(expectations, "expected_command") {
        require_json_str_equals(report_object, "command", expected)?;
    }
    if let Some(expected) = optional_json_str(expectations, "expected_model_family") {
        require_json_str_equals(report_object, "model_family", expected)?;
    }
    if let Some(expected) = optional_json_str(expectations, "expected_loader_kind") {
        let loader = report_object
            .get("loader")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "stage {stage_id} expected loader kind {expected}, but report has no loader object"
                ))
            })?;
        require_json_str_equals(loader, "kind", expected)?;
    }
    if let Some(expected) = optional_json_str(expectations, "expected_memory_lookup") {
        let memory_config = report_object
            .get("memory_config")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "stage {stage_id} expected memory lookup {expected}, but report has no memory_config object"
                ))
            })?;
        require_json_str_equals(memory_config, "memory_lookup", expected)?;
    }
    if let Some(expected) = optional_json_str(expectations, "expected_smft_mode") {
        let memory_config = report_object
            .get("memory_config")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "stage {stage_id} expected SMFT mode {expected}, but report has no memory_config object"
                ))
            })?;
        require_json_str_equals(memory_config, "smft_mode", expected)?;
    }
    if let Some(expected) = optional_json_str(expectations, "expected_memory_update_policy") {
        let memory_config = report_object
            .get("memory_config")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "stage {stage_id} expected memory update policy {expected}, but report has no memory_config object"
                ))
            })?;
        require_json_str_equals(memory_config, "memory_update_policy", expected)?;
    }
    Ok(())
}

fn validate_learning_sanity_report_loss(
    stage_id: &str,
    report_object: &serde_json::Map<String, serde_json::Value>,
    min_loss_reduction: f64,
    require_loss_improvement: bool,
) -> Result<LearningSanityLossSummary> {
    let initial_loss = json_field_f64(report_object, "initial_loss")?;
    let final_loss = json_field_f64(report_object, "final_loss")?;
    if !initial_loss.is_finite() || !final_loss.is_finite() {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} has non-finite loss values"
        )));
    }
    let computed_loss_reduction = if initial_loss == 0.0 {
        0.0
    } else {
        (initial_loss - final_loss) / initial_loss
    };
    let reported_loss_reduction = report_object
        .get("loss_reduction")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(computed_loss_reduction);
    if (reported_loss_reduction - computed_loss_reduction).abs() > 1.0e-6 {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} reported loss_reduction {} does not match computed {}",
            reported_loss_reduction, computed_loss_reduction
        )));
    }
    if require_loss_improvement && final_loss >= initial_loss {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} did not improve loss: initial={initial_loss} final={final_loss}"
        )));
    }
    if reported_loss_reduction < min_loss_reduction {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} loss_reduction {reported_loss_reduction} < required {min_loss_reduction}"
        )));
    }
    Ok(LearningSanityLossSummary {
        initial_loss,
        final_loss,
        loss_reduction: reported_loss_reduction,
    })
}

fn validate_learning_sanity_lr_grad_accumulation_sweep(
    index: usize,
    stage_id: &str,
    stage_object: &serde_json::Map<String, serde_json::Value>,
    sweep_report_path: &Path,
    sweep_object: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Value> {
    let format = json_field_str(sweep_object, "format")?;
    if format != LEARNING_SANITY_LR_GRAD_SWEEP_FORMAT {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} sweep report format must be {LEARNING_SANITY_LR_GRAD_SWEEP_FORMAT}, got {format}"
        )));
    }
    let version = json_field_i64(sweep_object, "version")?;
    if version != 0 {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} sweep report version must be 0, got {version}"
        )));
    }
    let runs = sweep_object
        .get("runs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep report requires runs array"
            ))
        })?;
    let min_runs = optional_json_usize(stage_object, "min_runs")
        .unwrap_or(4)
        .max(4);
    if runs.len() < min_runs {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} requires at least {min_runs} LR/grad-accumulation runs, got {}",
            runs.len()
        )));
    }

    let min_world_size = optional_json_usize(stage_object, "min_data_parallel_world_size")
        .unwrap_or(4)
        .max(4);
    let min_distinct_learning_rates =
        optional_json_usize(stage_object, "min_distinct_learning_rates")
            .unwrap_or(2)
            .max(2);
    let min_distinct_grad_accumulation_steps =
        optional_json_usize(stage_object, "min_distinct_grad_accumulation_steps")
            .unwrap_or(2)
            .max(2);
    let expected_distributed =
        optional_json_str(stage_object, "expected_distributed").unwrap_or("nccl");
    let expected_precision =
        optional_json_str(stage_object, "expected_precision").unwrap_or("amp-bf16");
    let require_rank_loss_improvement = stage_object
        .get("require_rank_loss_improvement")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let stage_min_loss_reduction = stage_object
        .get("min_loss_reduction")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let stage_min_best_loss_reduction = stage_object
        .get("min_best_loss_reduction")
        .and_then(serde_json::Value::as_f64);
    let stage_require_loss_improvement = stage_object
        .get("require_loss_improvement")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let sweep_root = sweep_report_path.parent().unwrap_or_else(|| Path::new("."));
    let mut learning_rates = Vec::new();
    let mut grad_accumulation_steps = BTreeSet::new();
    let mut min_loss_reduction = f64::INFINITY;
    let mut best_loss_reduction = f64::NEG_INFINITY;
    let mut best_run_summary: Option<serde_json::Value> = None;
    let mut min_observed_world_size = usize::MAX;
    let mut run_reports = Vec::with_capacity(runs.len());

    for (run_index, run) in runs.iter().enumerate() {
        let run_object = run.as_object().ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_index} must be an object"
            ))
        })?;
        let run_label = optional_json_str(run_object, "label")
            .map(str::to_string)
            .unwrap_or_else(|| format!("run-{run_index}"));
        let run_report_value = run_object
            .get("report")
            .or_else(|| run_object.get("report_path"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                TensorError::InvalidOperation(format!(
                    "stage {stage_id} sweep run {run_label} requires report or report_path"
                ))
            })?;
        let run_report_path = resolve_manifest_relative_path(sweep_root, run_report_value);
        let run_report = read_json_value(&run_report_path)?;
        let run_report_object = run_report.as_object().ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} report must be a JSON object"
            ))
        })?;
        validate_learning_sanity_report_expectations(stage_object, run_report_object, stage_id)?;
        validate_learning_sanity_report_expectations(run_object, run_report_object, stage_id)?;
        require_json_str_equals(run_report_object, "distributed", expected_distributed)?;
        require_json_str_equals(run_report_object, "precision", expected_precision)?;

        let run_min_loss_reduction = run_object
            .get("min_loss_reduction")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(stage_min_loss_reduction);
        let run_require_loss_improvement = run_object
            .get("require_loss_improvement")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(stage_require_loss_improvement);
        let loss = validate_learning_sanity_report_loss(
            stage_id,
            run_report_object,
            run_min_loss_reduction,
            run_require_loss_improvement,
        )?;
        min_loss_reduction = min_loss_reduction.min(loss.loss_reduction);

        let learning_rate = json_field_f64(run_report_object, "learning_rate")?;
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} learning_rate must be positive and finite"
            )));
        }
        if let Some(expected_learning_rate) = run_object
            .get("expected_learning_rate")
            .and_then(serde_json::Value::as_f64)
        {
            let tolerance = 1.0e-12 * expected_learning_rate.abs().max(1.0);
            if (learning_rate - expected_learning_rate).abs() > tolerance {
                return Err(TensorError::InvalidOperation(format!(
                    "stage {stage_id} sweep run {run_label} learning_rate {learning_rate} does not match expected {expected_learning_rate}"
                )));
            }
        }
        let grad_accumulation = json_field_usize(run_report_object, "grad_accumulation_steps")?;
        if grad_accumulation == 0 {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} grad_accumulation_steps must be greater than zero"
            )));
        }
        if let Some(expected_grad_accumulation) =
            optional_json_usize(run_object, "expected_grad_accumulation_steps")
        {
            if grad_accumulation != expected_grad_accumulation {
                return Err(TensorError::InvalidOperation(format!(
                    "stage {stage_id} sweep run {run_label} grad_accumulation_steps {grad_accumulation} does not match expected {expected_grad_accumulation}"
                )));
            }
        }

        let world_size = learning_sanity_report_world_size(run_report_object, stage_id)?;
        if world_size < min_world_size {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} data_parallel_world_size {world_size} < required {min_world_size}"
            )));
        }
        require_learning_sanity_cuda_devices(
            run_report_object,
            stage_id,
            &run_label,
            min_world_size,
            world_size,
        )?;
        require_json_usize_at_least(run_report_object, "all_reduce_calls", 1)?;
        require_json_usize_at_least(run_report_object, "all_reduce_bytes", 1)?;
        require_learning_sanity_sweep_batch_geometry(
            run_report_object,
            stage_id,
            &run_label,
            world_size,
            grad_accumulation,
        )?;
        if require_rank_loss_improvement {
            require_learning_sanity_rank_loss_improvement(
                run_report_object,
                stage_id,
                &run_label,
                world_size,
            )?;
        }

        learning_rates.push(learning_rate);
        grad_accumulation_steps.insert(grad_accumulation);
        min_observed_world_size = min_observed_world_size.min(world_size);
        let run_summary = serde_json::json!({
            "label": run_label,
            "report": run_report_path.display().to_string(),
            "command": run_report_object.get("command").cloned().unwrap_or(serde_json::Value::Null),
            "model_family": run_report_object.get("model_family").cloned().unwrap_or(serde_json::Value::Null),
            "learning_rate": learning_rate,
            "grad_accumulation_steps": grad_accumulation,
            "data_parallel_world_size": world_size,
            "initial_loss": loss.initial_loss,
            "final_loss": loss.final_loss,
            "loss_reduction": loss.loss_reduction,
        });
        if loss.loss_reduction > best_loss_reduction {
            best_loss_reduction = loss.loss_reduction;
            best_run_summary = Some(run_summary.clone());
        }
        run_reports.push(run_summary);
    }

    let distinct_learning_rates = distinct_f64_values(&learning_rates);
    if distinct_learning_rates.len() < min_distinct_learning_rates {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} requires at least {min_distinct_learning_rates} distinct learning rates, got {}",
            distinct_learning_rates.len()
        )));
    }
    if grad_accumulation_steps.len() < min_distinct_grad_accumulation_steps {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} requires at least {min_distinct_grad_accumulation_steps} distinct grad_accumulation_steps values, got {}",
            grad_accumulation_steps.len()
        )));
    }
    if let Some(required_best_loss_reduction) = stage_min_best_loss_reduction {
        if best_loss_reduction < required_best_loss_reduction {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} best loss_reduction {best_loss_reduction} < required {required_best_loss_reduction}"
            )));
        }
    }
    let best_run_summary = best_run_summary.ok_or_else(|| {
        TensorError::InvalidOperation(format!("stage {stage_id} sweep has no best run"))
    })?;
    let recommended_learning_rate = best_run_summary
        .get("learning_rate")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let recommended_grad_accumulation_steps = best_run_summary
        .get("grad_accumulation_steps")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    Ok(serde_json::json!({
        "stage_index": index,
        "stage_id": stage_id,
        "status": "passed",
        "report": sweep_report_path.display().to_string(),
        "runs": run_reports,
        "learning_rates": distinct_learning_rates,
        "grad_accumulation_steps": grad_accumulation_steps.into_iter().collect::<Vec<_>>(),
        "min_data_parallel_world_size": min_world_size,
        "min_observed_data_parallel_world_size": min_observed_world_size,
        "min_loss_reduction": stage_min_loss_reduction,
        "min_best_loss_reduction": stage_min_best_loss_reduction,
        "loss_reduction": min_loss_reduction,
        "best_loss_reduction": best_loss_reduction,
        "recommended_learning_rate": recommended_learning_rate,
        "recommended_grad_accumulation_steps": recommended_grad_accumulation_steps,
        "best_run": best_run_summary,
    }))
}

fn learning_sanity_report_world_size(
    report_object: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
) -> Result<usize> {
    let top_level_world = report_object
        .get("world_size")
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as usize);
    let performance_world = report_object
        .get("performance")
        .and_then(serde_json::Value::as_object)
        .and_then(|performance| performance.get("data_parallel_world_size"))
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as usize);
    if let (Some(top_level), Some(performance)) = (top_level_world, performance_world) {
        if top_level != performance {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} world_size {top_level} does not match performance.data_parallel_world_size {performance}"
            )));
        }
    }
    let devices_len = report_object
        .get("devices")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len);
    let world_size = top_level_world
        .or(performance_world)
        .or(devices_len)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} report requires world_size or performance.data_parallel_world_size"
            ))
        })?;
    if let Some(devices_len) = devices_len {
        if devices_len != world_size {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} devices length {devices_len} does not match world_size {world_size}"
            )));
        }
    }
    Ok(world_size)
}

fn require_learning_sanity_cuda_devices(
    report_object: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
    run_label: &str,
    min_world_size: usize,
    world_size: usize,
) -> Result<()> {
    let devices = report_object
        .get("devices")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} requires devices array"
            ))
        })?;
    if devices.len() < min_world_size || devices.len() != world_size {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} sweep run {run_label} devices length {} does not prove world_size {world_size} and required minimum {min_world_size}",
            devices.len()
        )));
    }
    for device in devices {
        let Some(label) = device.as_str() else {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} devices entries must be strings"
            )));
        };
        if !label.starts_with("cuda:") {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} expected cuda device labels, got {label}"
            )));
        }
    }
    Ok(())
}

fn require_learning_sanity_sweep_batch_geometry(
    report_object: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
    run_label: &str,
    world_size: usize,
    grad_accumulation: usize,
) -> Result<()> {
    let per_rank_batch = json_field_usize(report_object, "per_rank_batch_size")?;
    let expected_global_effective = per_rank_batch
        .saturating_mul(grad_accumulation)
        .saturating_mul(world_size);
    let global_effective = json_field_usize(report_object, "global_effective_batch_size")?;
    if global_effective != expected_global_effective {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} sweep run {run_label} global_effective_batch_size {global_effective} does not match per_rank_batch_size*grad_accumulation_steps*world_size {expected_global_effective}"
        )));
    }
    let performance = report_object
        .get("performance")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} requires performance evidence"
            ))
        })?;
    require_json_usize_at_least(performance, "tokens_seen", 1)?;
    let performance_grad_accumulation = json_field_usize(performance, "grad_accumulation_steps")?;
    if performance_grad_accumulation != grad_accumulation {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} sweep run {run_label} performance.grad_accumulation_steps {performance_grad_accumulation} does not match report grad_accumulation_steps {grad_accumulation}"
        )));
    }
    let performance_world_size = json_field_usize(performance, "data_parallel_world_size")?;
    if performance_world_size != world_size {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} sweep run {run_label} performance.data_parallel_world_size {performance_world_size} does not match world_size {world_size}"
        )));
    }
    let performance_global_effective =
        json_field_usize(performance, "global_effective_batch_size")?;
    if performance_global_effective != expected_global_effective {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} sweep run {run_label} performance.global_effective_batch_size {performance_global_effective} does not match expected {expected_global_effective}"
        )));
    }
    Ok(())
}

fn require_learning_sanity_rank_loss_improvement(
    report_object: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
    run_label: &str,
    world_size: usize,
) -> Result<()> {
    let rank_losses = report_object
        .get("rank_loss_reductions")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} requires rank_loss_reductions"
            ))
        })?;
    if rank_losses.len() < world_size {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} sweep run {run_label} rank_loss_reductions length {} < world_size {world_size}",
            rank_losses.len()
        )));
    }
    for (rank, value) in rank_losses.iter().enumerate().take(world_size) {
        let Some(reduction) = value.as_f64() else {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} rank {rank} loss_reduction must be numeric"
            )));
        };
        if !reduction.is_finite() || reduction <= 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} sweep run {run_label} rank {rank} did not improve loss: loss_reduction={reduction}"
            )));
        }
    }
    Ok(())
}

fn distinct_f64_values(values: &[f64]) -> Vec<f64> {
    let mut distinct = Vec::new();
    for value in values {
        let tolerance = 1.0e-12 * value.abs().max(1.0);
        if !distinct
            .iter()
            .any(|existing: &f64| (*existing - *value).abs() <= tolerance)
        {
            distinct.push(*value);
        }
    }
    distinct.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    distinct
}

fn validate_learning_sanity_longer_32k_blend(
    index: usize,
    stage_id: &str,
    stage_object: &serde_json::Map<String, serde_json::Value>,
    summary_path: &Path,
    summary_object: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Value> {
    require_json_str_equals(summary_object, "status", "passed")?;
    let expected_vocab = optional_json_usize(stage_object, "expected_tokenizer_vocab_size")
        .unwrap_or(PRODUCTION_TOKENIZER_VOCAB_SIZE);
    let min_reserved_tokens =
        optional_json_usize(stage_object, "min_reserved_tokens").unwrap_or(128);
    let min_selected_tokens = optional_json_usize(stage_object, "min_selected_tokens").unwrap_or(1);
    let min_selected_docs = optional_json_usize(stage_object, "min_selected_docs").unwrap_or(1);
    let min_target_tokens =
        optional_json_usize(stage_object, "min_target_tokens").unwrap_or(min_selected_tokens);
    let min_blend_sources = optional_json_usize(stage_object, "min_blend_sources").unwrap_or(1);
    let min_final_step = optional_json_usize(stage_object, "min_final_step").unwrap_or(1);
    let require_production_gate = stage_object
        .get("require_production_gate")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    require_json_usize_equals(
        summary_object,
        "tokenizer_version",
        TOKENIZER_VERSION_V2 as usize,
    )?;
    require_json_usize_equals(summary_object, "tokenizer_vocab_size", expected_vocab)?;
    require_json_usize_at_least(summary_object, "reserved_tokens", min_reserved_tokens)?;
    require_json_usize_equals(
        summary_object,
        "manifest_version",
        DATASET_MANIFEST_VERSION_V2 as usize,
    )?;
    require_json_str_equals(
        summary_object,
        "manifest_storage",
        DATASET_STORAGE_BINARY_SHARDS,
    )?;
    require_json_str_equals(summary_object, "loader_kind", "binary_shard_streaming")?;
    require_json_bool_equals(summary_object, "tokens_materialized", false)?;
    require_json_usize_at_least(summary_object, "target_tokens", min_target_tokens)?;
    require_json_usize_at_least(summary_object, "selected_tokens", min_selected_tokens)?;
    require_json_usize_at_least(summary_object, "selected_docs", min_selected_docs)?;
    if require_production_gate {
        require_json_bool_equals(summary_object, "production_gate", true)?;
        require_json_str_equals(summary_object, "materialize_mode", "full")?;
    }

    let artifacts = summary_object
        .get("artifacts")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires summary artifacts object"
            ))
        })?;
    let summary_root = summary_path.parent().unwrap_or_else(|| Path::new("."));
    let corpus_blend_path =
        learning_sanity_artifact_path(artifacts, "corpus_blend", summary_root, stage_id)?;
    let tokenizer_report_path =
        learning_sanity_artifact_path(artifacts, "tokenizer_report", summary_root, stage_id)?;
    let fertility_report_path =
        learning_sanity_artifact_path(artifacts, "fertility_report", summary_root, stage_id)?;
    let curation_report_path =
        learning_sanity_artifact_path(artifacts, "curation_report", summary_root, stage_id)?;
    let dataset_manifest_path =
        learning_sanity_artifact_path(artifacts, "dataset_manifest", summary_root, stage_id)?;
    let train_report_path =
        learning_sanity_artifact_path(artifacts, "train_report", summary_root, stage_id)?;
    let resume_report_path =
        learning_sanity_artifact_path(artifacts, "resume_report", summary_root, stage_id)?;
    let eval_report_path =
        learning_sanity_artifact_path(artifacts, "eval_report", summary_root, stage_id)?;
    let generation_report_path =
        learning_sanity_artifact_path(artifacts, "generation_report", summary_root, stage_id)?;

    let corpus_blend = read_json_value(&corpus_blend_path)?;
    let corpus_blend = corpus_blend.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} corpus_blend artifact must be a JSON object"
        ))
    })?;
    require_json_str_equals(corpus_blend, "format", CORPUS_BLEND_MANIFEST_FORMAT)?;
    require_json_usize_equals(
        corpus_blend,
        "version",
        CORPUS_BLEND_MANIFEST_VERSION as usize,
    )?;
    require_json_usize_equals(corpus_blend, "tokenizer_target_vocab_size", expected_vocab)?;
    require_learning_sanity_corpus_blend_sources(corpus_blend, stage_id, min_blend_sources)?;

    let tokenizer_report = read_json_value(&tokenizer_report_path)?;
    let tokenizer_report = tokenizer_report.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} tokenizer_report artifact must be a JSON object"
        ))
    })?;
    require_json_str_equals(tokenizer_report, "command", "tokenizer train-corpus")?;
    require_json_str_equals(tokenizer_report, "status", "passed")?;
    let tokenizer = tokenizer_report_object(tokenizer_report, stage_id, "tokenizer_report")?;
    require_learning_sanity_tokenizer_metadata(
        tokenizer,
        stage_id,
        expected_vocab,
        min_reserved_tokens,
    )?;
    let hard_path = tokenizer_report
        .get("hard_path")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} tokenizer_report requires hard_path object"
            ))
        })?;
    require_json_bool_equals(hard_path, "native_trainer", true)?;
    require_json_bool_equals(hard_path, "external_tokenizer_dependency", false)?;
    require_json_bool_equals(hard_path, "tiny_stories_allowed", false)?;

    let fertility_report = read_json_value(&fertility_report_path)?;
    let fertility_report = fertility_report.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} fertility_report artifact must be a JSON object"
        ))
    })?;
    require_json_str_equals(fertility_report, "command", "tokenizer fertility")?;
    require_json_str_equals(fertility_report, "status", "passed")?;
    require_learning_sanity_tokenizer_metadata(
        tokenizer_report_object(fertility_report, stage_id, "fertility_report")?,
        stage_id,
        expected_vocab,
        min_reserved_tokens,
    )?;
    require_learning_sanity_fertility_sources(fertility_report, stage_id)?;

    let curation_report = read_json_value(&curation_report_path)?;
    let curation_report = curation_report.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} curation_report artifact must be a JSON object"
        ))
    })?;
    require_json_str_equals(curation_report, "format", "heirloom.blend_curation_report")?;
    require_json_str_equals(curation_report, "status", "passed")?;
    require_json_usize_at_least(curation_report, "selected_tokens", min_selected_tokens)?;
    require_json_usize_at_least(curation_report, "selected_docs", min_selected_docs)?;
    require_json_usize_at_least(curation_report, "target_tokens", min_target_tokens)?;
    require_learning_sanity_tokenizer_metadata(
        tokenizer_report_object(curation_report, stage_id, "curation_report")?,
        stage_id,
        expected_vocab,
        min_reserved_tokens,
    )?;

    let dataset_manifest = read_json_value(&dataset_manifest_path)?;
    let dataset_manifest = dataset_manifest.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} dataset_manifest artifact must be a JSON object"
        ))
    })?;
    require_json_usize_equals(
        dataset_manifest,
        "version",
        DATASET_MANIFEST_VERSION_V2 as usize,
    )?;
    require_json_str_equals(dataset_manifest, "storage", DATASET_STORAGE_BINARY_SHARDS)?;
    require_json_usize_at_least(dataset_manifest, "train_tokens", 1)?;
    require_json_usize_at_least(dataset_manifest, "valid_tokens", 1)?;

    let train_report = read_json_value(&train_report_path)?;
    let train_report = train_report.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} train_report artifact must be a JSON object"
        ))
    })?;
    require_json_str_equals(train_report, "command", "train-memory-lm")?;
    require_json_str_equals(train_report, "model_family", "memory_transformer")?;
    require_learning_sanity_training_loader(train_report, stage_id, "train_report")?;
    if let Some(tokenizer) = optional_tokenizer_report_object(train_report) {
        require_learning_sanity_tokenizer_metadata(
            tokenizer,
            stage_id,
            expected_vocab,
            min_reserved_tokens,
        )?;
    }
    require_json_usize_at_least(train_report, "final_step", min_final_step)?;
    validate_learning_sanity_report_expectations(stage_object, train_report, stage_id)?;
    let train_loss = validate_learning_sanity_report_loss(stage_id, train_report, 0.0, false)?;
    let min_loss_reduction = stage_object
        .get("min_loss_reduction")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let require_loss_improvement = stage_object
        .get("require_loss_improvement")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);

    let resume_report = read_json_value(&resume_report_path)?;
    let resume_report = resume_report.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} resume_report artifact must be a JSON object"
        ))
    })?;
    require_json_str_equals(resume_report, "command", "train-memory-lm")?;
    require_json_str_equals(resume_report, "model_family", "memory_transformer")?;
    require_learning_sanity_training_loader(resume_report, stage_id, "resume_report")?;
    if let Some(tokenizer) = optional_tokenizer_report_object(resume_report) {
        require_learning_sanity_tokenizer_metadata(
            tokenizer,
            stage_id,
            expected_vocab,
            min_reserved_tokens,
        )?;
    }
    let resume_final_step = json_field_usize(resume_report, "final_step")?;
    if resume_final_step <= json_field_usize(train_report, "final_step")? {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} resume_report final_step {resume_final_step} must advance beyond train_report final_step"
        )));
    }
    let resume_loss = validate_learning_sanity_report_loss(stage_id, resume_report, 0.0, false)?;
    let combined_loss_reduction = if train_loss.initial_loss == 0.0 {
        0.0
    } else {
        (train_loss.initial_loss - resume_loss.final_loss) / train_loss.initial_loss
    };
    if require_loss_improvement && resume_loss.final_loss >= train_loss.initial_loss {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} train+resume did not improve loss: initial={} final={}",
            train_loss.initial_loss, resume_loss.final_loss
        )));
    }
    if combined_loss_reduction < min_loss_reduction {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} train+resume loss_reduction {combined_loss_reduction} < required {min_loss_reduction}"
        )));
    }
    let loss = LearningSanityLossSummary {
        initial_loss: train_loss.initial_loss,
        final_loss: resume_loss.final_loss,
        loss_reduction: combined_loss_reduction,
    };

    let eval_report = read_json_value(&eval_report_path)?;
    let eval_report = eval_report.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} eval_report artifact must be a JSON object"
        ))
    })?;
    require_json_str_equals(eval_report, "command", "eval-memory-lm")?;
    require_json_str_equals(eval_report, "model_family", "memory_transformer")?;
    require_learning_sanity_training_loader(eval_report, stage_id, "eval_report")?;
    require_learning_sanity_tokenizer_metadata(
        tokenizer_report_object(eval_report, stage_id, "eval_report")?,
        stage_id,
        expected_vocab,
        min_reserved_tokens,
    )?;
    let metrics = eval_report
        .get("metrics")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} eval_report requires metrics object"
            ))
        })?;
    require_json_f64_positive(metrics, "loss")?;
    require_json_f64_positive(metrics, "perplexity")?;

    let generation_report = read_json_value(&generation_report_path)?;
    let generation_report = generation_report.as_object().ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} generation_report artifact must be a JSON object"
        ))
    })?;
    require_json_str_equals(generation_report, "command", "generate-memory-lm")?;
    require_json_str_equals(generation_report, "model_family", "memory_transformer")?;
    require_json_usize_at_least(generation_report, "generated_tokens", 1)?;
    require_json_usize_at_least(generation_report, "total_tokens", 1)?;
    require_learning_sanity_tokenizer_metadata(
        tokenizer_report_object(generation_report, stage_id, "generation_report")?,
        stage_id,
        expected_vocab,
        min_reserved_tokens,
    )?;

    Ok(serde_json::json!({
        "stage_index": index,
        "stage_id": stage_id,
        "status": "passed",
        "report": summary_path.display().to_string(),
        "tokenizer_vocab_size": expected_vocab,
        "reserved_tokens": summary_object.get("reserved_tokens").cloned().unwrap_or(serde_json::Value::Null),
        "selected_tokens": summary_object.get("selected_tokens").cloned().unwrap_or(serde_json::Value::Null),
        "selected_docs": summary_object.get("selected_docs").cloned().unwrap_or(serde_json::Value::Null),
        "manifest_storage": summary_object.get("manifest_storage").cloned().unwrap_or(serde_json::Value::Null),
        "loader_kind": summary_object.get("loader_kind").cloned().unwrap_or(serde_json::Value::Null),
        "initial_loss": loss.initial_loss,
        "final_loss": loss.final_loss,
        "loss_reduction": loss.loss_reduction,
        "min_loss_reduction": min_loss_reduction,
        "artifacts": {
            "corpus_blend": corpus_blend_path.display().to_string(),
            "tokenizer_report": tokenizer_report_path.display().to_string(),
            "fertility_report": fertility_report_path.display().to_string(),
            "curation_report": curation_report_path.display().to_string(),
            "dataset_manifest": dataset_manifest_path.display().to_string(),
            "train_report": train_report_path.display().to_string(),
            "resume_report": resume_report_path.display().to_string(),
            "eval_report": eval_report_path.display().to_string(),
            "generation_report": generation_report_path.display().to_string(),
        }
    }))
}

fn learning_sanity_artifact_path(
    artifacts: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    root: &Path,
    stage_id: &str,
) -> Result<PathBuf> {
    let value = artifacts
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!("stage {stage_id} requires artifacts.{field}"))
        })?;
    let raw_path = Path::new(value);
    if raw_path.is_absolute() {
        return Ok(raw_path.to_path_buf());
    }
    let root_relative = root.join(raw_path);
    if root_relative.exists() {
        return Ok(root_relative);
    }
    if raw_path.exists() {
        return Ok(raw_path.to_path_buf());
    }
    Ok(root_relative)
}

fn tokenizer_report_object<'a>(
    report: &'a serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
    artifact_name: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>> {
    optional_tokenizer_report_object(report).ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "stage {stage_id} {artifact_name} requires tokenizer or tokenizer_report object"
        ))
    })
}

fn optional_tokenizer_report_object(
    report: &serde_json::Map<String, serde_json::Value>,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    report
        .get("tokenizer")
        .or_else(|| report.get("tokenizer_report"))
        .and_then(serde_json::Value::as_object)
}

fn require_learning_sanity_tokenizer_metadata(
    tokenizer: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
    expected_vocab: usize,
    min_reserved_tokens: usize,
) -> Result<()> {
    require_json_usize_equals(tokenizer, "version", TOKENIZER_VERSION_V2 as usize)?;
    require_json_usize_equals(tokenizer, "vocab_size", expected_vocab)?;
    require_json_usize_at_least(tokenizer, "reserved_tokens", min_reserved_tokens)?;
    require_json_non_empty_str(tokenizer, "tokenizer_hash")?;
    require_json_non_empty_str(tokenizer, "source_blend_hash")?;
    require_json_non_empty_str(tokenizer, "sample_manifest_hash")?;
    require_json_non_empty_str(tokenizer, "reserved_registry_hash")?;
    let training_config = tokenizer
        .get("training_config")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} tokenizer requires training_config object"
            ))
        })?;
    require_json_usize_equals(training_config, "target_vocab_size", expected_vocab)?;
    require_json_bool_equals(training_config, "digit_isolation", true)?;
    let validation = tokenizer
        .get("validation")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} tokenizer requires validation object"
            ))
        })?;
    require_json_usize_at_least(validation, "reserved_tokens", min_reserved_tokens)?;
    require_json_usize_at_least(validation, "byte_token_count", BYTE_VOCAB)?;
    let byte_coverage = json_field_f64(validation, "byte_coverage")?;
    if !byte_coverage.is_finite() || byte_coverage < 1.0 {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} tokenizer validation byte_coverage must be 1.0, got {byte_coverage}"
        )));
    }
    Ok(())
}

fn require_learning_sanity_corpus_blend_sources(
    corpus_blend: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
    min_sources: usize,
) -> Result<()> {
    let sources = corpus_blend
        .get("sources")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} corpus_blend requires sources array"
            ))
        })?;
    if sources.len() < min_sources {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} corpus_blend requires at least {min_sources} sources, got {}",
            sources.len()
        )));
    }
    let mut pretraining_sources = 0usize;
    for source in sources {
        let source = source.as_object().ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} corpus_blend sources must be objects"
            ))
        })?;
        let source_id = json_field_str(source, "source_id")?;
        if source_id.to_ascii_lowercase().contains("tinystories") {
            return Err(TensorError::InvalidOperation(format!(
                "stage {stage_id} corpus_blend must not include TinyStories source {source_id}"
            )));
        }
        if source
            .get("include_in_pretraining")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            pretraining_sources += 1;
        }
    }
    if pretraining_sources == 0 {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} corpus_blend has no pretraining sources"
        )));
    }
    Ok(())
}

fn require_learning_sanity_fertility_sources(
    fertility_report: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
) -> Result<()> {
    let sources = fertility_report
        .get("sources")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} fertility_report requires sources array"
            ))
        })?;
    if sources.is_empty() {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} fertility_report requires at least one source"
        )));
    }
    for source in sources {
        let source = source.as_object().ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} fertility_report sources must be objects"
            ))
        })?;
        require_json_f64_positive(source, "tokens_per_byte")?;
        require_json_f64_positive(source, "bytes_per_token")?;
    }
    Ok(())
}

fn require_learning_sanity_training_loader(
    report: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
    artifact_name: &str,
) -> Result<()> {
    let loader = report
        .get("loader")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} {artifact_name} requires loader object"
            ))
        })?;
    require_json_str_equals(loader, "kind", "binary_shard_streaming")?;
    require_json_bool_equals(loader, "tokens_materialized", false)?;
    Ok(())
}

fn require_learning_sanity_memory_layers_disabled(
    report_object: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
) -> Result<()> {
    let memory_config = report_object
        .get("memory_config")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires memory_config to prove memory layers are disabled"
            ))
        })?;
    let n_layers = json_field_usize(memory_config, "n_layers")?;
    let layer_values = memory_config
        .get("memory_layer_indices")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires memory_config.memory_layer_indices"
            ))
        })?;
    let mut active_layers = Vec::new();
    for value in layer_values {
        let index = value.as_u64().ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} memory_layer_indices must contain unsigned integers"
            ))
        })? as usize;
        if index < n_layers {
            active_layers.push(index);
        }
    }
    if active_layers.is_empty() {
        return Ok(());
    }
    Err(TensorError::InvalidOperation(format!(
        "stage {stage_id} expected memory layers disabled, but active memory layer indices {:?} are below n_layers={n_layers}",
        active_layers
    )))
}

fn require_learning_sanity_smft_enabled(
    report_object: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
) -> Result<()> {
    let memory_config = report_object
        .get("memory_config")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires memory_config to prove SMFT is enabled"
            ))
        })?;
    require_json_str_equals(memory_config, "memory_lookup", "exact")?;
    require_json_str_equals(memory_config, "memory_update_policy", "sparse_rows")?;
    require_json_str_equals(memory_config, "smft_mode", "masked_memory_rows")?;

    let memory_optimizer = report_object
        .get("memory_optimizer")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires memory_optimizer evidence"
            ))
        })?;
    require_json_bool_equals(
        memory_optimizer,
        "sparse_optimizer_updates_selected_rows",
        true,
    )?;
    require_json_usize_at_least(memory_optimizer, "sparse_update_parameter_count", 1)?;
    require_json_usize_at_least(memory_optimizer, "row_mask_attached_sparse_update_count", 1)?;
    require_json_usize_at_least(memory_optimizer, "sparse_update_selected_row_events", 1)?;
    let row_mask = memory_optimizer
        .get("smft_row_mask")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires memory_optimizer.smft_row_mask"
            ))
        })?;
    require_json_bool_equals(row_mask, "applied", true)?;
    require_json_usize_at_least(row_mask, "trainable_rows", 1)?;

    let artifacts = report_object
        .get("smft_artifacts")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires smft_artifacts evidence"
            ))
        })?;
    let accumulated_counts = artifacts
        .get("accumulated_counts")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires accumulated SMFT access counts"
            ))
        })?;
    require_json_usize_at_least(accumulated_counts, "total_events", 1)?;
    require_json_usize_at_least(accumulated_counts, "unique_rows", 1)?;
    let generated_mask = artifacts
        .get("generated_mask")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires generated SMFT mask evidence"
            ))
        })?;
    require_json_usize_at_least(generated_mask, "trainable_rows", 1)?;
    let online_refresh = artifacts
        .get("online_refresh")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires online SMFT refresh evidence"
            ))
        })?;
    require_json_bool_equals(online_refresh, "enabled", true)?;
    require_json_usize_at_least(online_refresh, "refresh_count", 1)?;
    let active_mask = online_refresh
        .get("active_mask")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires online refresh active_mask evidence"
            ))
        })?;
    require_json_usize_at_least(active_mask, "trainable_rows", 1)?;
    Ok(())
}

fn require_learning_sanity_product_key_parity(
    report_object: &serde_json::Map<String, serde_json::Value>,
    stage_id: &str,
) -> Result<()> {
    let memory_config = report_object
        .get("memory_config")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires memory_config to prove product-key parity"
            ))
        })?;
    require_json_str_equals(memory_config, "memory_lookup", "product_key")?;
    require_json_str_equals(memory_config, "smft_mode", "disabled")?;
    let memory_slots = json_field_usize(memory_config, "memory_slots")?;
    let side = (memory_slots as f64).sqrt() as usize;
    if side == 0 || side * side != memory_slots {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} product-key memory_slots must be square, got {memory_slots}"
        )));
    }
    let memory_key_dim = json_field_usize(memory_config, "memory_key_dim")?;
    if memory_key_dim % 2 != 0 {
        return Err(TensorError::InvalidOperation(format!(
            "stage {stage_id} product-key memory_key_dim must be even, got {memory_key_dim}"
        )));
    }

    let memory_selection = report_object
        .get("memory_selection")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires memory_selection evidence"
            ))
        })?;
    require_json_str_equals(memory_selection, "memory_lookup", "product_key")?;
    require_json_usize_at_least(memory_selection, "configured_memory_layers", 1)?;
    require_json_usize_at_least(memory_selection, "captured_memory_layers", 1)?;
    require_json_usize_at_least(memory_selection, "selected_row_events", 1)?;
    require_json_usize_at_least(memory_selection, "unique_selected_rows", 1)?;

    let memory_optimizer = report_object
        .get("memory_optimizer")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires memory_optimizer evidence"
            ))
        })?;
    require_json_usize_at_least(memory_optimizer, "memory_table_parameter_count", 3)?;

    let op_decisions = report_object
        .get("amp_bf16_op_decisions")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "stage {stage_id} requires amp_bf16_op_decisions evidence"
            ))
        })?;
    require_json_array_str_field_contains(
        op_decisions,
        "kernel_path",
        "product_key_candidate_topk_memory_lookup",
    )?;
    Ok(())
}

fn validate_learning_sanity_stage_id(stage_id: &str) -> Result<()> {
    const KNOWN_STAGES: &[&str] = &[
        "dense_fixed_shard",
        "memory_layers_disabled",
        "memory_exact_smft_disabled",
        "memory_exact_smft_enabled",
        "product_key_parity",
        "lr_grad_accumulation_sweep",
        "longer_32k_blend",
    ];
    if KNOWN_STAGES.contains(&stage_id) {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(format!(
            "unknown learning sanity stage_id {stage_id}"
        )))
    }
}

fn resolve_manifest_relative_path(root: &Path, value: &str) -> PathBuf {
    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn json_field_str<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'a str> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be a non-empty string")))
}

fn optional_json_str<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Option<&'a str> {
    object.get(field).and_then(serde_json::Value::as_str)
}

fn optional_json_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Option<usize> {
    object
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as usize)
}

fn json_field_i64(object: &serde_json::Map<String, serde_json::Value>, field: &str) -> Result<i64> {
    object
        .get(field)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be an integer")))
}

fn json_field_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<usize> {
    object
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!("{field} must be an unsigned integer"))
        })
}

fn require_json_bool_equals(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    expected: bool,
) -> Result<()> {
    let actual = object
        .get(field)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be boolean")))?;
    if actual == expected {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(format!(
            "{field} expected {expected}, got {actual}"
        )))
    }
}

fn require_json_usize_at_least(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    min_value: usize,
) -> Result<()> {
    let actual = json_field_usize(object, field)?;
    if actual >= min_value {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(format!(
            "{field} expected at least {min_value}, got {actual}"
        )))
    }
}

fn require_json_usize_equals(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    expected: usize,
) -> Result<()> {
    let actual = json_field_usize(object, field)?;
    if actual == expected {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(format!(
            "{field} expected {expected}, got {actual}"
        )))
    }
}

fn require_json_non_empty_str(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<()> {
    let _ = json_field_str(object, field)?;
    Ok(())
}

fn require_json_f64_positive(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<()> {
    let actual = json_field_f64(object, field)?;
    if actual.is_finite() && actual > 0.0 {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(format!(
            "{field} expected positive finite value, got {actual}"
        )))
    }
}

fn require_json_array_str_field_contains(
    array: &[serde_json::Value],
    field: &str,
    needle: &str,
) -> Result<()> {
    if array.iter().any(|item| {
        item.get(field)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|value| value.contains(needle))
    }) {
        return Ok(());
    }
    Err(TensorError::InvalidOperation(format!(
        "{field} did not contain {needle}"
    )))
}

fn json_field_f64(object: &serde_json::Map<String, serde_json::Value>, field: &str) -> Result<f64> {
    object
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be numeric")))
}

fn require_json_str_equals(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    expected: &str,
) -> Result<()> {
    let actual = json_field_str(object, field)?;
    if actual == expected {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(format!(
            "{field} mismatch: expected {expected} got {actual}"
        )))
    }
}

fn padawan_validate_report(
    episode_paths: &[PathBuf],
    artifact_root: Option<&Path>,
) -> Result<serde_json::Value> {
    let mut summaries = Vec::new();
    for episode_path in episode_paths {
        let root = artifact_root
            .unwrap_or_else(|| episode_path.parent().unwrap_or_else(|| Path::new(".")));
        for episode in padawan_load_jsonl(episode_path)? {
            summaries.push(padawan_validate_episode(&episode, root)?);
        }
    }
    let smft_count = summaries
        .iter()
        .filter(|item| item["eligible_for_smft"].as_bool() == Some(true))
        .count();
    let sft_count = summaries
        .iter()
        .filter(|item| item["eligible_for_sft"].as_bool() == Some(true))
        .count();
    let max_guidance = summaries
        .iter()
        .filter_map(|item| item["guidance_level"].as_u64())
        .max()
        .unwrap_or(0);
    Ok(serde_json::json!({
        "format": "heirloom.padawan_validation_report",
        "version": 0,
        "status": "passed",
        "episodes": summaries.len(),
        "sft_eligible": sft_count,
        "smft_eligible": smft_count,
        "max_guidance_level": max_guidance,
        "episode_summaries": summaries,
    }))
}

fn padawan_verify_report(
    episode_paths: &[PathBuf],
    artifact_root: Option<&Path>,
    requested_families: &[PadawanVerifierFamily],
    allowed_path_prefixes: &[String],
) -> Result<serde_json::Value> {
    let families = padawan_expand_verifier_families(requested_families);
    let allowed_prefixes = if allowed_path_prefixes.is_empty() {
        DEFAULT_PADAWAN_ALLOWED_PATH_PREFIXES
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
    } else {
        allowed_path_prefixes.to_vec()
    };
    let mut episode_reports = Vec::new();
    for episode_path in episode_paths {
        let root = artifact_root
            .unwrap_or_else(|| episode_path.parent().unwrap_or_else(|| Path::new(".")));
        for episode in padawan_load_jsonl(episode_path)? {
            padawan_validate_episode(&episode, root)?;
            episode_reports.push(padawan_verify_episode(
                &episode,
                root,
                &families,
                &allowed_prefixes,
            ));
        }
    }
    let mut passed = 0u64;
    let mut failed = 0u64;
    let mut skipped = 0u64;
    for report in &episode_reports {
        if let Some(family_reports) = report["families"].as_array() {
            for family in family_reports {
                match family["status"].as_str().unwrap_or("failed") {
                    "passed" => passed += 1,
                    "skipped" => skipped += 1,
                    _ => failed += 1,
                }
            }
        }
    }
    let status = if failed == 0 { "passed" } else { "failed" };
    Ok(serde_json::json!({
        "format": "heirloom.padawan_verifier_harness_report",
        "version": 0,
        "status": status,
        "episodes": episode_reports,
        "family_counts": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "families_requested": families.iter().map(|family| family.label()).collect::<Vec<_>>(),
        "code_patch_allowed_path_prefixes": allowed_prefixes,
    }))
}

fn padawan_expand_verifier_families(
    requested: &[PadawanVerifierFamily],
) -> Vec<PadawanVerifierFamily> {
    if requested.is_empty() || requested.contains(&PadawanVerifierFamily::All) {
        return vec![
            PadawanVerifierFamily::CodePatch,
            PadawanVerifierFamily::JsonToolCall,
            PadawanVerifierFamily::EvidenceCitation,
            PadawanVerifierFamily::MemorySmft,
        ];
    }
    let mut families = Vec::new();
    for family in requested {
        if *family != PadawanVerifierFamily::All && !families.contains(family) {
            families.push(*family);
        }
    }
    families
}

fn padawan_verify_episode(
    episode: &serde_json::Value,
    artifact_root: &Path,
    families: &[PadawanVerifierFamily],
    allowed_prefixes: &[String],
) -> serde_json::Value {
    let mut family_reports = Vec::new();
    for family in families {
        let result = match family {
            PadawanVerifierFamily::All => unreachable!("expanded verifier families exclude all"),
            PadawanVerifierFamily::CodePatch => {
                padawan_verify_code_patch(episode, artifact_root, allowed_prefixes)
            }
            PadawanVerifierFamily::JsonToolCall => {
                padawan_verify_json_tool_call(episode, artifact_root)
            }
            PadawanVerifierFamily::EvidenceCitation => {
                padawan_verify_evidence_citation(episode, artifact_root)
            }
            PadawanVerifierFamily::MemorySmft => padawan_verify_memory_smft(episode, artifact_root),
        };
        family_reports.push(result.unwrap_or_else(|err| {
            serde_json::json!({
                "family": family.label(),
                "status": "failed",
                "reason": err.to_string(),
            })
        }));
    }
    let status = if family_reports
        .iter()
        .all(|item| matches!(item["status"].as_str(), Some("passed" | "skipped")))
    {
        "passed"
    } else {
        "failed"
    };
    serde_json::json!({
        "episode_id": padawan_json_str(episode, "episode_id").unwrap_or("<invalid>").to_string(),
        "status": status,
        "families": family_reports,
    })
}

fn padawan_verify_code_patch(
    episode: &serde_json::Value,
    artifact_root: &Path,
    allowed_prefixes: &[String],
) -> Result<serde_json::Value> {
    let padawan = padawan_json_object(episode, "padawan")?;
    let verifier = padawan_json_object(episode, "verifier")?;
    let diff_text = String::from_utf8(padawan_read_ref_bytes(
        padawan_value_str(padawan, "artifact_ref")?,
        artifact_root,
    )?)
    .map_err(|err| TensorError::InvalidOperation(format!("patch artifact is not UTF-8: {err}")))?;
    padawan_require(
        !diff_text.trim().is_empty(),
        "patch artifact must not be empty",
    )?;
    padawan_require(
        diff_text.contains("diff --git "),
        "patch artifact must contain unified diff headers",
    )?;
    let unrelated = verifier
        .get("unrelated_changes")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    padawan_require(unrelated == 0, "episode verifier reports unrelated changes")?;

    let mut changed_paths = BTreeSet::new();
    let mut added_lines = 0u64;
    let mut removed_lines = 0u64;
    for line in diff_text.lines() {
        if let Some(rest) = line.strip_prefix("diff --git a/") {
            let Some((left, right)) = rest.split_once(" b/") else {
                continue;
            };
            if let Some(path) = padawan_clean_diff_path(left)? {
                changed_paths.insert(path);
            }
            if let Some(path) = padawan_clean_diff_path(right)? {
                changed_paths.insert(path);
            }
        } else if let Some(raw_path) = line.strip_prefix("+++ ") {
            if let Some(path) = padawan_clean_diff_path(raw_path.trim())? {
                changed_paths.insert(path);
            }
        } else if let Some(raw_path) = line.strip_prefix("--- ") {
            if let Some(path) = padawan_clean_diff_path(raw_path.trim())? {
                changed_paths.insert(path);
            }
        } else if line.starts_with('+') {
            added_lines += 1;
        } else if line.starts_with('-') {
            removed_lines += 1;
        }
    }
    padawan_require(
        !changed_paths.is_empty(),
        "patch artifact did not declare changed paths",
    )?;
    let blocked = changed_paths
        .iter()
        .filter(|path| {
            !allowed_prefixes
                .iter()
                .any(|prefix| *path == prefix || path.starts_with(prefix))
        })
        .cloned()
        .collect::<Vec<_>>();
    padawan_require(
        blocked.is_empty(),
        format!("patch touches paths outside Padawan sidecar allowlist: {blocked:?}"),
    )?;
    padawan_require(
        changed_paths.len() <= 24,
        format!(
            "patch changes too many files for a local verifier fixture: {}",
            changed_paths.len()
        ),
    )?;
    padawan_require(
        added_lines + removed_lines <= 5000,
        "patch is too large for the local P3 harness",
    )?;
    Ok(serde_json::json!({
        "family": "code_patch",
        "status": "passed",
        "changed_paths": changed_paths.into_iter().collect::<Vec<_>>(),
        "added_lines": added_lines,
        "removed_lines": removed_lines,
    }))
}

fn padawan_verify_json_tool_call(
    episode: &serde_json::Value,
    artifact_root: &Path,
) -> Result<serde_json::Value> {
    let bundle = padawan_load_bundle(episode, artifact_root)?;
    let artifacts = padawan_value_array(padawan_json_object(&bundle, "bundle")?, "artifacts")?;
    let mut json_artifacts = Vec::new();
    for artifact in artifacts {
        let artifact = padawan_value_object(artifact, "bundle artifact")?;
        if artifact
            .get("media_type")
            .and_then(serde_json::Value::as_str)
            == Some("application/json")
        {
            let name = padawan_value_str(artifact, "name")?.to_string();
            padawan_load_ref_json(padawan_value_str(artifact, "ref")?, artifact_root)?;
            json_artifacts.push(name);
        }
    }
    json_artifacts.sort();

    let padawan = padawan_json_object(episode, "padawan")?;
    let trace = padawan_load_ref_json(padawan_value_str(padawan, "trace_ref")?, artifact_root)?;
    let trace_object = padawan_json_object(&trace, "trace")?;
    let tool_calls = padawan_value_array(trace_object, "tool_calls")?;
    for (index, call) in tool_calls.iter().enumerate() {
        let call = padawan_value_object(call, format!("tool_calls[{index}]"))?;
        padawan_require(
            call.get("tool")
                .and_then(serde_json::Value::as_str)
                .is_some(),
            format!("tool call {index} missing tool"),
        )?;
        padawan_require(
            call.contains_key("arguments") || call.contains_key("input"),
            format!("tool call {index} must include arguments or input"),
        )?;
        padawan_require(
            !call.contains_key("result"),
            format!("tool call {index} should not inline tool results; use observations"),
        )?;
    }
    Ok(serde_json::json!({
        "family": "json_tool_call",
        "status": "passed",
        "json_artifacts": json_artifacts,
        "tool_call_count": tool_calls.len(),
    }))
}

fn padawan_verify_evidence_citation(
    episode: &serde_json::Value,
    artifact_root: &Path,
) -> Result<serde_json::Value> {
    let padawan = padawan_json_object(episode, "padawan")?;
    let trace = padawan_load_ref_json(padawan_value_str(padawan, "trace_ref")?, artifact_root)?;
    let trace_object = padawan_json_object(&trace, "trace")?;
    let observations = padawan_value_array(trace_object, "observations")?;
    let final_validation = padawan_value_object(
        trace_object.get("final_validation").ok_or_else(|| {
            TensorError::InvalidOperation("trace.final_validation missing".to_string())
        })?,
        "trace.final_validation",
    )?;
    let final_response = String::from_utf8(padawan_read_ref_bytes(
        padawan_value_str(padawan, "final_response_ref")?,
        artifact_root,
    )?)
    .map_err(|err| TensorError::InvalidOperation(format!("final response is not UTF-8: {err}")))?;

    padawan_require(
        !observations.is_empty(),
        "trace must include at least one observation or evidence note",
    )?;
    for (index, observation) in observations.iter().enumerate() {
        padawan_require(
            observation
                .as_str()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty()),
            format!("observation {index} must be non-empty text"),
        )?;
    }
    padawan_require(
        final_validation
            .get("validator")
            .and_then(serde_json::Value::as_str)
            .is_some(),
        "final_validation.validator must identify a verifier",
    )?;
    padawan_require(
        final_validation
            .get("expected")
            .and_then(serde_json::Value::as_str)
            .is_some(),
        "final_validation.expected must state expected outcome",
    )?;
    padawan_require(
        !final_response.trim().is_empty(),
        "final response must not be empty",
    )?;
    let evidence_refs = final_validation
        .get("evidence_refs")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for (index, ref_value) in evidence_refs.iter().enumerate() {
        let ref_text = ref_value.as_str().ok_or_else(|| {
            TensorError::InvalidOperation(format!("evidence_refs[{index}] must be a string ref"))
        })?;
        padawan_read_ref_bytes(ref_text, artifact_root)?;
    }
    let verifier = padawan_json_object(episode, "verifier")?;
    padawan_require(
        verifier
            .get("failure_class")
            .is_none_or(serde_json::Value::is_null),
        "failed episodes cannot pass evidence verifier",
    )?;
    Ok(serde_json::json!({
        "family": "evidence_citation",
        "status": "passed",
        "observations": observations.len(),
        "evidence_refs": evidence_refs.len(),
    }))
}

fn padawan_verify_memory_smft(
    episode: &serde_json::Value,
    artifact_root: &Path,
) -> Result<serde_json::Value> {
    let selection = padawan_json_object(episode, "selection")?;
    if selection
        .get("eligible_for_smft")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return Ok(serde_json::json!({
            "family": "memory_smft",
            "status": "skipped",
            "reason": "episode is not SMFT-eligible",
        }));
    }
    let memory = padawan_json_object(episode, "memory")?;
    let selection_report = padawan_load_ref_json(
        padawan_value_str(memory, "selection_report_ref")?,
        artifact_root,
    )?;
    let access_counts = padawan_load_ref_json(
        padawan_value_str(memory, "smft_access_counts_ref")?,
        artifact_root,
    )?;
    padawan_require(
        selection_report
            .get("format")
            .and_then(serde_json::Value::as_str)
            == Some("heirloom.padawan.memory_selection"),
        "bad memory selection format",
    )?;
    padawan_require(
        access_counts
            .get("format")
            .and_then(serde_json::Value::as_str)
            == Some("heirloom.padawan.smft_access_counts"),
        "bad SMFT access-count format",
    )?;
    let slots = padawan_json_usize(&selection_report, "memory_slots")?;
    padawan_require(slots > 0, "memory_slots must be positive")?;
    padawan_require(
        padawan_json_usize(&access_counts, "memory_slots")? == slots,
        "memory slot count mismatch",
    )?;
    let foreground_rows = padawan_row_list(&access_counts, "foreground_rows", slots)?;
    let background_rows = padawan_row_list(&access_counts, "background_rows", slots)?;
    padawan_require(
        !foreground_rows.is_empty(),
        "SMFT foreground rows must be non-empty",
    )?;
    padawan_require(
        !background_rows.is_empty(),
        "SMFT background rows must be non-empty",
    )?;

    let layers = padawan_value_object(
        selection_report.get("layers").ok_or_else(|| {
            TensorError::InvalidOperation("memory_selection.layers missing".to_string())
        })?,
        "memory_selection.layers",
    )?;
    let mut selected_total = 0usize;
    for (layer_name, layer_value) in layers {
        let layer =
            padawan_value_object(layer_value, format!("memory_selection.layers.{layer_name}"))?;
        let selected_rows = padawan_row_list_from_value(
            layer.get("selected_rows").ok_or_else(|| {
                TensorError::InvalidOperation(format!("{layer_name}.selected_rows missing"))
            })?,
            &format!("{layer_name}.selected_rows"),
            slots,
        )?;
        selected_total += selected_rows.len();
        padawan_require(
            selected_rows
                .iter()
                .all(|row| foreground_rows.contains(row)),
            format!("{layer_name} selected rows missing from foreground rows"),
        )?;
        let events = padawan_object_usize(layer, "selected_row_events")?;
        padawan_require(
            events >= selected_rows.len(),
            format!("{layer_name} selected_row_events must cover selected rows"),
        )?;
    }
    let lift = padawan_json_f64(&access_counts, "foreground_background_lift")?;
    let replay_jaccard = padawan_json_f64(&access_counts, "replay_jaccard_overlap")?;
    let per_episode_cap = padawan_json_usize(&access_counts, "per_episode_row_cap")?;
    let task_family_cap = padawan_json_usize(&access_counts, "task_family_row_cap")?;
    let design_episode_cap = (slots / 200).clamp(32, 4096);
    let design_family_cap = (slots / 50).clamp(128, 16384);
    padawan_require(
        lift >= 4.0,
        format!("foreground/background lift {lift} below 4.0"),
    )?;
    padawan_require(
        replay_jaccard >= 0.30,
        format!("replay Jaccard {replay_jaccard} below 0.30"),
    )?;
    padawan_require(
        selected_total <= per_episode_cap,
        "selected rows exceed per-episode cap",
    )?;
    padawan_require(
        per_episode_cap <= design_episode_cap,
        "per-episode cap exceeds initial design budget",
    )?;
    padawan_require(
        task_family_cap <= design_family_cap,
        "task-family cap exceeds initial design budget",
    )?;
    Ok(serde_json::json!({
        "family": "memory_smft",
        "status": "passed",
        "layers": layers.len(),
        "selected_rows": selected_total,
        "foreground_rows": foreground_rows.len(),
        "lift": lift,
        "replay_jaccard_overlap": replay_jaccard,
    }))
}

fn padawan_validate_episode(
    episode: &serde_json::Value,
    artifact_root: &Path,
) -> Result<serde_json::Value> {
    padawan_require(
        padawan_json_str(episode, "format")? == "heirloom.padawan_episode",
        "episode format must be heirloom.padawan_episode",
    )?;
    padawan_require(
        padawan_json_i64(episode, "version")? == 0,
        "episode version must be 0",
    )?;
    let episode_id = padawan_json_str(episode, "episode_id")?;
    padawan_require(
        episode_id.starts_with("padawan_"),
        format!("episode_id must start with padawan_: {episode_id}"),
    )?;
    let source_id = padawan_json_str(episode, "source_id")?;
    padawan_require(
        source_id.starts_with("heirloom.padawan.v0."),
        format!("bad source_id: {source_id}"),
    )?;
    padawan_validate_datetime(padawan_json_str(episode, "created_at")?, "created_at")?;
    padawan_json_str(episode, "checkpoint_ref")?;

    let task = padawan_json_object(episode, "task")?;
    for field in ["domain", "task_kind", "prompt", "artifact_type"] {
        padawan_value_str(task, field)?;
    }
    padawan_number_range(
        padawan_value_f64(task, "difficulty")?,
        "task.difficulty",
        0.0,
        1.0,
    )?;

    let teacher = padawan_json_object(episode, "teacher")?;
    let guidance_level = padawan_value_i64(teacher, "guidance_level")?;
    padawan_require(
        (0..=4).contains(&guidance_level),
        "teacher.guidance_level must be 0..4",
    )?;
    padawan_require(
        padawan_value_bool(teacher, "teacher_tokens_masked_from_loss")?,
        "teacher tokens must be masked from loss",
    )?;
    padawan_require(
        padawan_value_bool(teacher, "forbidden_outputs_checked")?,
        "teacher forbidden-output check must pass",
    )?;
    let policy = padawan_value_object(
        teacher.get("guidance_policy").ok_or_else(|| {
            TensorError::InvalidOperation("teacher.guidance_policy missing".to_string())
        })?,
        "teacher.guidance_policy",
    )?;
    padawan_value_str(policy, "difficulty_band")?;
    padawan_value_str(policy, "distance_proxy")?;
    padawan_number_range(
        policy
            .get("over_guidance_score")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        "teacher.guidance_policy.over_guidance_score",
        0.0,
        1.0,
    )?;
    if guidance_level > 0 {
        padawan_read_ref_bytes(padawan_value_str(teacher, "brief_ref")?, artifact_root)?;
    }

    let padawan = padawan_json_object(episode, "padawan")?;
    for field in [
        "model",
        "trace_ref",
        "compressed_rationale_ref",
        "artifact_ref",
        "artifact_bundle_ref",
        "final_response_ref",
    ] {
        padawan_value_str(padawan, field)?;
    }
    padawan_validate_trace(
        padawan_value_str(padawan, "trace_ref")?,
        artifact_root,
        episode_id,
    )?;
    padawan_validate_rationale(
        padawan_value_str(padawan, "compressed_rationale_ref")?,
        artifact_root,
    )?;
    padawan_read_ref_bytes(padawan_value_str(padawan, "artifact_ref")?, artifact_root)?;
    padawan_read_ref_bytes(
        padawan_value_str(padawan, "final_response_ref")?,
        artifact_root,
    )?;
    let bundle = padawan_validate_artifact_bundle(
        padawan_value_str(padawan, "artifact_bundle_ref")?,
        artifact_root,
        episode,
    )?;

    let verifier = padawan_json_object(episode, "verifier")?;
    padawan_value_str(verifier, "verifier_id")?;
    let verifier_pack_hash = padawan_require_hash(
        padawan_value_str(verifier, "verifier_pack_hash")?,
        "verifier.verifier_pack_hash",
    )?;
    padawan_value_bool(verifier, "tests_passed")?;
    padawan_value_bool(verifier, "schema_valid")?;
    if verifier.contains_key("hidden_tests_passed") {
        padawan_value_bool(verifier, "hidden_tests_passed")?;
    }
    padawan_value_f64(verifier, "reward")?;
    padawan_value_object(
        verifier.get("reward_components").ok_or_else(|| {
            TensorError::InvalidOperation("verifier.reward_components missing".to_string())
        })?,
        "verifier.reward_components",
    )?;
    padawan_require(
        verifier
            .get("failure_class")
            .is_some_and(|value| value.is_null() || value.is_string()),
        "verifier.failure_class must be null or string",
    )?;
    let bundle_pack_hash = bundle["restricted_audit"]["verifier_pack_hash"]
        .as_str()
        .ok_or_else(|| {
            TensorError::InvalidOperation("bundle verifier_pack_hash missing".to_string())
        })?;
    padawan_require(
        verifier_pack_hash == bundle_pack_hash,
        "episode verifier_pack_hash must match bundle audit",
    )?;

    let memory = padawan_json_object(episode, "memory")?;
    let selection = padawan_json_object(episode, "selection")?;
    for field in [
        "eligible_for_sft",
        "eligible_for_preference",
        "eligible_for_rlvr_replay",
        "eligible_for_smft",
        "requires_teacherless_replay",
    ] {
        padawan_value_bool(selection, field)?;
    }
    let sft_weight = padawan_value_f64(selection, "sft_weight")?;
    padawan_number_range(sft_weight, "selection.sft_weight", 0.0, 1.0)?;
    if sft_weight >= 0.75 && guidance_level > 0 {
        let replay_tracked = selection
            .get("requires_teacherless_replay")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
            || selection
                .get("teacherless_replay_episode_id")
                .and_then(serde_json::Value::as_str)
                .is_some();
        padawan_require(
            replay_tracked,
            "high-weight guided SFT examples require teacherless replay tracking",
        )?;
    }
    if selection
        .get("eligible_for_smft")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        let selection_report = padawan_validate_memory_ref(
            memory.get("selection_report_ref"),
            artifact_root,
            "memory.selection_report_ref",
        )?;
        let access_counts = padawan_validate_memory_ref(
            memory.get("smft_access_counts_ref"),
            artifact_root,
            "memory.smft_access_counts_ref",
        )?;
        padawan_require(
            selection_report.is_some(),
            "SMFT-eligible episodes require memory selection report",
        )?;
        let access_counts = access_counts.ok_or_else(|| {
            TensorError::InvalidOperation(
                "SMFT-eligible episodes require SMFT access counts".to_string(),
            )
        })?;
        padawan_require(
            padawan_json_f64(&access_counts, "foreground_background_lift")? >= 4.0,
            "smft_access_counts.foreground_background_lift must be at least 4.0",
        )?;
        padawan_require(
            padawan_json_f64(&access_counts, "replay_jaccard_overlap")? >= 0.30,
            "smft_access_counts.replay_jaccard_overlap must be at least 0.30",
        )?;
    }
    Ok(serde_json::json!({
        "episode_id": episode_id,
        "eligible_for_sft": selection.get("eligible_for_sft").and_then(serde_json::Value::as_bool).unwrap_or(false),
        "eligible_for_smft": selection.get("eligible_for_smft").and_then(serde_json::Value::as_bool).unwrap_or(false),
        "guidance_level": guidance_level,
    }))
}

fn padawan_validate_trace(ref_value: &str, artifact_root: &Path, episode_id: &str) -> Result<()> {
    let trace = padawan_load_ref_json(ref_value, artifact_root)?;
    padawan_require(
        padawan_json_str(&trace, "format")? == "heirloom.padawan_trace",
        "trace format must be heirloom.padawan_trace",
    )?;
    padawan_require(
        padawan_json_i64(&trace, "version")? == 0,
        "trace version must be 0",
    )?;
    padawan_require(
        padawan_json_str(&trace, "episode_id")? == episode_id,
        "trace episode_id mismatch",
    )?;
    for field in [
        "task_understanding",
        "assumptions",
        "compressed_rationale",
        "subgoals",
        "tool_calls",
        "observations",
        "failed_checks",
        "repairs",
        "final_validation",
        "final_artifact_ref",
    ] {
        padawan_require(
            trace.get(field).is_some(),
            format!("trace missing field {field}"),
        )?;
    }
    let rationale = padawan_json_str(&trace, "compressed_rationale")?;
    padawan_require(
        rationale.len() <= 1200,
        "trace compressed_rationale should stay compressed",
    )?;
    Ok(())
}

fn padawan_validate_rationale(ref_value: &str, artifact_root: &Path) -> Result<()> {
    let text =
        String::from_utf8(padawan_read_ref_bytes(ref_value, artifact_root)?).map_err(|err| {
            TensorError::InvalidOperation(format!("{ref_value} rationale is not UTF-8: {err}"))
        })?;
    padawan_require(
        !text.trim().is_empty(),
        format!("{ref_value} rationale is empty"),
    )?;
    padawan_require(
        text.len() <= 1200,
        format!("{ref_value} rationale is too long to be compressed"),
    )?;
    Ok(())
}

fn padawan_validate_artifact_bundle(
    ref_value: &str,
    artifact_root: &Path,
    episode: &serde_json::Value,
) -> Result<serde_json::Value> {
    let bundle = padawan_load_ref_json(ref_value, artifact_root)?;
    let episode_id = padawan_json_str(episode, "episode_id")?;
    padawan_require(
        padawan_json_str(&bundle, "format")? == "heirloom.padawan_artifact_bundle",
        "artifact bundle format must be heirloom.padawan_artifact_bundle",
    )?;
    padawan_require(
        padawan_json_i64(&bundle, "version")? == 0,
        "artifact bundle version must be 0",
    )?;
    padawan_require(
        padawan_json_str(&bundle, "episode_id")? == episode_id,
        "artifact bundle episode_id mismatch",
    )?;
    let artifacts = padawan_json_array(&bundle, "artifacts")?;
    padawan_require(
        !artifacts.is_empty(),
        "artifact bundle artifacts must be non-empty",
    )?;
    let mut seen = BTreeSet::new();
    let mut actual_hashes = BTreeMap::new();
    for (index, artifact) in artifacts.iter().enumerate() {
        let artifact = padawan_value_object(artifact, format!("artifacts[{index}]"))?;
        let name = padawan_value_str(artifact, "name")?;
        padawan_require(
            seen.insert(name.to_string()),
            format!("duplicate artifact name {name}"),
        )?;
        let artifact_ref = padawan_value_str(artifact, "ref")?;
        padawan_value_str(artifact, "media_type")?;
        let expected_hash =
            padawan_require_hash(padawan_value_str(artifact, "sha256")?, "artifact.sha256")?;
        let expected_bytes = padawan_object_usize(artifact, "bytes")?;
        let data = padawan_read_ref_bytes(artifact_ref, artifact_root)?;
        let actual_hash = sha256_prefixed_hex(&data);
        padawan_require(
            actual_hash == expected_hash,
            format!("{artifact_ref} hash mismatch: {actual_hash} != {expected_hash}"),
        )?;
        padawan_require(
            data.len() == expected_bytes,
            format!(
                "{artifact_ref} byte size mismatch: {} != {expected_bytes}",
                data.len()
            ),
        )?;
        actual_hashes.insert(name.to_string(), actual_hash);
    }
    let audit = padawan_value_object(
        bundle
            .get("restricted_audit")
            .ok_or_else(|| TensorError::InvalidOperation("restricted_audit missing".to_string()))?,
        "restricted_audit",
    )?;
    for field in [
        "verifier_pack_hash",
        "public_verifier_hash",
        "hidden_verifier_hash",
        "result_digest",
    ] {
        padawan_require_hash(padawan_value_str(audit, field)?, field)?;
    }
    let hidden_ids = padawan_value_array(audit, "hidden_case_ids_hmac")?;
    padawan_require(
        !hidden_ids.is_empty(),
        "hidden_case_ids_hmac must be non-empty",
    )?;
    for (index, hidden_id) in hidden_ids.iter().enumerate() {
        let value = hidden_id.as_str().ok_or_else(|| {
            TensorError::InvalidOperation(format!("hidden_case_ids_hmac[{index}] must be a string"))
        })?;
        padawan_require_hmac(value, &format!("hidden_case_ids_hmac[{index}]"))?;
    }
    padawan_value_str(audit, "restricted_bundle_ref")?;
    let declared_hashes = padawan_value_object(
        audit.get("artifact_hashes").ok_or_else(|| {
            TensorError::InvalidOperation("audit artifact_hashes missing".to_string())
        })?,
        "artifact_hashes",
    )?;
    for (name, digest) in actual_hashes {
        let declared = declared_hashes
            .get(&name)
            .and_then(serde_json::Value::as_str);
        padawan_require(
            declared == Some(digest.as_str()),
            format!("audit artifact hash mismatch for {name}"),
        )?;
    }
    Ok(bundle)
}

fn padawan_validate_memory_ref(
    ref_value: Option<&serde_json::Value>,
    artifact_root: &Path,
    label: &str,
) -> Result<Option<serde_json::Value>> {
    let Some(ref_value) = ref_value else {
        return Ok(None);
    };
    if ref_value.is_null() {
        return Ok(None);
    }
    let ref_text = ref_value.as_str().ok_or_else(|| {
        TensorError::InvalidOperation(format!("{label} must be a string or null"))
    })?;
    let value = padawan_load_ref_json(ref_text, artifact_root)?;
    let format = value
        .get("format")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    padawan_require(
        format.starts_with("heirloom.padawan."),
        format!("{label} has bad format"),
    )?;
    Ok(Some(value))
}

fn padawan_load_bundle(
    episode: &serde_json::Value,
    artifact_root: &Path,
) -> Result<serde_json::Value> {
    let padawan = padawan_json_object(episode, "padawan")?;
    padawan_load_ref_json(
        padawan_value_str(padawan, "artifact_bundle_ref")?,
        artifact_root,
    )
}

fn padawan_load_jsonl(path: &Path) -> Result<Vec<serde_json::Value>> {
    let file = File::open(path)
        .map_err(|err| TensorError::Io(format!("failed to open {}: {err}", path.display())))?;
    let mut records = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line
            .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(&line).map_err(|err| {
            TensorError::Io(format!(
                "{}:{} invalid JSONL: {err}",
                path.display(),
                index + 1
            ))
        })?;
        padawan_require(
            value.is_object(),
            format!("{}:{} must be a JSON object", path.display(), index + 1),
        )?;
        records.push(value);
    }
    padawan_require(
        !records.is_empty(),
        format!("{} contains no episode records", path.display()),
    )?;
    Ok(records)
}

fn padawan_load_ref_json(ref_value: &str, artifact_root: &Path) -> Result<serde_json::Value> {
    let path = padawan_ref_to_path(ref_value, artifact_root)?;
    read_json_value(&path)
}

fn padawan_read_ref_bytes(ref_value: &str, artifact_root: &Path) -> Result<Vec<u8>> {
    let path = padawan_ref_to_path(ref_value, artifact_root)?;
    fs::read(&path).map_err(|err| {
        TensorError::Io(format!("failed to read artifact {}: {err}", path.display()))
    })
}

fn padawan_ref_to_path(ref_value: &str, artifact_root: &Path) -> Result<PathBuf> {
    padawan_require(
        ref_value.starts_with(PADAWAN_REF_PREFIX),
        format!("Padawan artifact refs must start with {PADAWAN_REF_PREFIX}: {ref_value}"),
    )?;
    let suffix = &ref_value[PADAWAN_REF_PREFIX.len()..];
    padawan_require(!suffix.is_empty(), "empty Padawan artifact ref suffix")?;
    let suffix_path = Path::new(suffix);
    for component in suffix_path.components() {
        use std::path::Component;
        match component {
            Component::Normal(_) | Component::CurDir => {}
            _ => {
                return Err(TensorError::InvalidOperation(format!(
                    "unsafe Padawan artifact ref: {ref_value}"
                )));
            }
        }
    }
    Ok(artifact_root.join(suffix_path))
}

fn padawan_clean_diff_path(path: &str) -> Result<Option<String>> {
    if path == "/dev/null" {
        return Ok(None);
    }
    let path = path
        .strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path);
    padawan_require(!path.is_empty(), "diff path must be non-empty")?;
    padawan_require(
        !path.starts_with('/'),
        format!("diff path must be relative: {path}"),
    )?;
    let path_ref = Path::new(path);
    for component in path_ref.components() {
        use std::path::Component;
        match component {
            Component::Normal(value) => {
                padawan_require(
                    value != ".env",
                    format!("diff path may not touch env files: {path}"),
                )?;
            }
            Component::CurDir => {}
            _ => {
                return Err(TensorError::InvalidOperation(format!(
                    "diff path may not escape workspace: {path}"
                )))
            }
        }
    }
    Ok(Some(path.to_string()))
}

fn padawan_row_list(
    value: &serde_json::Value,
    field: &str,
    memory_slots: usize,
) -> Result<Vec<usize>> {
    let object = padawan_json_object(value, "row-list owner")?;
    let rows = object
        .get(field)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} missing")))?;
    padawan_row_list_from_value(rows, field, memory_slots)
}

fn padawan_row_list_from_value(
    value: &serde_json::Value,
    label: &str,
    memory_slots: usize,
) -> Result<Vec<usize>> {
    let rows = value
        .as_array()
        .ok_or_else(|| TensorError::InvalidOperation(format!("{label} must be a list")))?;
    let mut clean = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        let Some(row) = row.as_u64() else {
            return Err(TensorError::InvalidOperation(format!(
                "{label}[{index}] must be an integer row"
            )));
        };
        let row = usize::try_from(row).map_err(|_| {
            TensorError::InvalidOperation(format!("{label}[{index}] does not fit usize"))
        })?;
        padawan_require(
            row < memory_slots,
            format!("{label}[{index}]={row} is outside memory_slots={memory_slots}"),
        )?;
        padawan_require(seen.insert(row), format!("{label} contains duplicate rows"))?;
        clean.push(row);
    }
    Ok(clean)
}

fn padawan_validate_datetime(value: &str, label: &str) -> Result<()> {
    padawan_require(
        value.contains('T') && (value.ends_with('Z') || value.rsplit_once(['+', '-']).is_some()),
        format!("{label} is not RFC3339-like: {value}"),
    )
}

fn padawan_json_object(
    value: &serde_json::Value,
    label: impl AsRef<str>,
) -> Result<&serde_json::Map<String, serde_json::Value>> {
    let label = label.as_ref();
    let object_value = value.get(label).unwrap_or(value);
    object_value
        .as_object()
        .ok_or_else(|| TensorError::InvalidOperation(format!("{label} must be an object")))
}

fn padawan_json_array<'a>(
    value: &'a serde_json::Value,
    field: &str,
) -> Result<&'a Vec<serde_json::Value>> {
    value
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be an array")))
}

fn padawan_json_str<'a>(value: &'a serde_json::Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be a non-empty string")))
}

fn padawan_json_i64(value: &serde_json::Value, field: &str) -> Result<i64> {
    value
        .get(field)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be an integer")))
}

fn padawan_json_usize(value: &serde_json::Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!("{field} must be a non-negative integer"))
        })
        .and_then(|item| {
            usize::try_from(item)
                .map_err(|_| TensorError::InvalidOperation(format!("{field} does not fit usize")))
        })
}

fn padawan_json_f64(value: &serde_json::Value, field: &str) -> Result<f64> {
    value
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be numeric")))
}

fn padawan_value_object(
    value: &serde_json::Value,
    label: impl AsRef<str>,
) -> Result<&serde_json::Map<String, serde_json::Value>> {
    padawan_json_object(value, label)
}

fn padawan_value_array<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'a Vec<serde_json::Value>> {
    object
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be an array")))
}

fn padawan_value_str<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'a str> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be a non-empty string")))
}

fn padawan_value_bool(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<bool> {
    object
        .get(field)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be boolean")))
}

fn padawan_value_i64(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<i64> {
    object
        .get(field)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be an integer")))
}

fn padawan_value_f64(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<f64> {
    object
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| TensorError::InvalidOperation(format!("{field} must be numeric")))
}

fn padawan_object_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<usize> {
    object
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!("{field} must be a non-negative integer"))
        })
        .and_then(|item| {
            usize::try_from(item)
                .map_err(|_| TensorError::InvalidOperation(format!("{field} does not fit usize")))
        })
}

fn padawan_number_range(value: f64, label: &str, minimum: f64, maximum: f64) -> Result<()> {
    padawan_require(
        value >= minimum,
        format!("{label}={value} is below minimum {minimum}"),
    )?;
    padawan_require(
        value <= maximum,
        format!("{label}={value} is above maximum {maximum}"),
    )
}

fn padawan_require_hash<'a>(value: &'a str, label: &str) -> Result<&'a str> {
    padawan_require_prefixed_hex(value, "sha256:", 64, label)?;
    Ok(value)
}

fn padawan_require_hmac<'a>(value: &'a str, label: &str) -> Result<&'a str> {
    padawan_require_prefixed_hex(value, "hmac-sha256:", 64, label)?;
    Ok(value)
}

fn padawan_require_prefixed_hex(
    value: &str,
    prefix: &str,
    hex_len: usize,
    label: &str,
) -> Result<()> {
    let Some(hex) = value.strip_prefix(prefix) else {
        return Err(TensorError::InvalidOperation(format!(
            "{label} must start with {prefix}"
        )));
    };
    padawan_require(
        hex.len() == hex_len,
        format!("{label} must contain {hex_len} hex chars"),
    )?;
    padawan_require(
        hex.as_bytes().iter().all(u8::is_ascii_hexdigit),
        format!("{label} contains non-hex characters"),
    )
}

fn padawan_require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(message.into()))
    }
}

fn sha256_prefixed_hex(bytes: &[u8]) -> String {
    let digest = sha256_digest(bytes);
    let mut text = String::with_capacity("sha256:".len() + 64);
    text.push_str("sha256:");
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut message = bytes.to_vec();
    let bit_len = (message.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = H0;
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in w.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut digest = [0u8; 32];
    for (index, word) in h.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn stable_hash_bytes_local(bytes: &[u8]) -> String {
    format!("{:016x}", stable_hash_u64(bytes))
}

fn stable_hash_u64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn read_json_value(path: &Path) -> Result<serde_json::Value> {
    let json = fs::read_to_string(path)
        .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
    serde_json::from_str(&json)
        .map_err(|err| TensorError::Io(format!("failed to parse {}: {err}", path.display())))
}

fn read_json_typed<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let json = fs::read_to_string(path)
        .map_err(|err| TensorError::Io(format!("failed to read {}: {err}", path.display())))?;
    serde_json::from_str(&json)
        .map_err(|err| TensorError::Io(format!("failed to parse {}: {err}", path.display())))
}

fn cuda_error(error: cuda::CudaError) -> TensorError {
    TensorError::Device(error.to_string())
}

fn maybe_truncate(text: String, max_bytes: Option<usize>) -> String {
    let Some(limit) = max_bytes else {
        return text;
    };
    if text.len() <= limit {
        text
    } else {
        String::from_utf8_lossy(&text.as_bytes()[..limit]).into_owned()
    }
}

fn load_prepared_from_cli_or_checkpoint(
    dataset_manifest: &Option<PathBuf>,
    checkpoint: &Path,
    resume: bool,
) -> Result<Option<PreparedTokenData>> {
    if let Some(path) = dataset_manifest {
        return PreparedTokenData::load(path).map(Some);
    }
    if !resume || !checkpoint.exists() {
        return Ok(None);
    }
    let metadata_path = checkpoint.join("metadata.json");
    if !metadata_path.exists() {
        return Ok(None);
    }
    let json = std::fs::read_to_string(&metadata_path).map_err(|err| {
        TensorError::Io(format!("failed to read {}: {err}", metadata_path.display()))
    })?;
    let metadata: serde_json::Value = serde_json::from_str(&json).map_err(|err| {
        TensorError::Io(format!(
            "failed to parse {}: {err}",
            metadata_path.display()
        ))
    })?;
    metadata
        .get("dataset_manifest_path")
        .and_then(serde_json::Value::as_str)
        .map(PreparedTokenData::load)
        .transpose()
}

fn validate_checkpoint_tokenizer_matches_manifest(
    tokenizer: &BpeTokenizer,
    prepared: &PreparedTokenData,
) -> Result<()> {
    let actual_hash = tokenizer.fingerprint()?;
    if actual_hash != prepared.manifest.tokenizer_hash {
        return Err(TensorError::InvalidOperation(format!(
            "checkpoint tokenizer hash mismatch for prepared data: manifest={} checkpoint={}",
            prepared.manifest.tokenizer_hash, actual_hash
        )));
    }
    Ok(())
}

fn require_path<'a>(path: Option<&'a PathBuf>, flag: &str) -> Result<&'a PathBuf> {
    path.ok_or_else(|| {
        TensorError::InvalidOperation(format!(
            "missing {flag}; pass {flag} with the legacy raw text path or use --dataset-manifest"
        ))
    })
}

fn parse_device(value: &str) -> Result<Device> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("cpu") {
        return Ok(Device::Cpu);
    }
    let Some(id) = value.strip_prefix("cuda:") else {
        return Err(TensorError::Device(format!(
            "unsupported device {value:?}; expected cpu or cuda:<id>"
        )));
    };
    let device_id = id.parse::<usize>().map_err(|err| {
        TensorError::Device(format!(
            "failed to parse CUDA device id from {value:?}: {err}"
        ))
    })?;
    Ok(Device::Cuda(device_id))
}

fn load_eval_tokens(
    prepared: Option<&PreparedTokenData>,
    data: Option<&PathBuf>,
    tokenizer: &BpeTokenizer,
    split: EvalSplit,
) -> Result<(Vec<usize>, String)> {
    if let Some(prepared) = prepared {
        let tokens = match split {
            EvalSplit::Train => prepared.train_tokens()?,
            EvalSplit::Valid => prepared.valid_tokens()?,
        };
        return Ok((
            tokens,
            format!("{}:{}", prepared.manifest_path.display(), split.as_str()),
        ));
    }

    let data_path = require_path(data, "--data")?;
    let text = read_text(data_path)?;
    Ok((
        tokenizer.encode(&text, true, true),
        data_path.display().to_string(),
    ))
}

fn evaluate_streaming_lm(
    model: &TinyTransformerLm,
    prepared: &PreparedTokenData,
    split: EvalSplit,
    batch_size: usize,
    max_batches: Option<usize>,
    precision: Precision,
) -> Result<(LmEvalMetrics, serde_json::Value)> {
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
    let split = match split {
        EvalSplit::Train => TokenDataSplit::Train,
        EvalSplit::Valid => TokenDataSplit::Valid,
    };
    let mut dataset = prepared.streaming_dataset(
        split,
        model.config.block_size,
        TokenDatasetState::from_seed(0),
    )?;
    let mut offset = 0usize;
    let mut batches = 0usize;
    let mut examples = 0usize;
    let mut predicted_tokens = 0usize;
    let mut loss_sum = 0.0;
    while offset + model.config.block_size < dataset.len() {
        if max_batches.is_some_and(|limit| batches >= limit) {
            break;
        }
        let Some((input, target, actual_batch)) =
            dataset.sequential_batch_at(offset, batch_size)?
        else {
            break;
        };
        offset += actual_batch * model.config.block_size;
        let input = input.to_device(model.device())?;
        let target = target.to_device(model.device())?;
        let loss = no_grad(|| match precision {
            Precision::F32 => model.loss(&input, &target),
            Precision::Bf16 => model.loss_bf16_activations(&input, &target),
            Precision::AmpBf16 => model.loss_amp_bf16(&input, &target),
        })?;
        let weight = actual_batch * model.config.block_size;
        let loss_value = eval_loss_scalar(&loss, precision)?;
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
    let metrics = LmEvalMetrics {
        batches,
        examples,
        tokens: predicted_tokens,
        loss,
        perplexity: loss.exp(),
    };
    let loader_report = serde_json::json!({
        "kind": "binary_shard_streaming",
        "source": format!("{}:{}", prepared.manifest_path.display(), split.as_str()),
        "tokens_materialized": false,
        "total_tokens": dataset.len(),
        "block_size": dataset.block_size(),
        "stats": dataset.loader_stats(),
    });
    Ok((metrics, loader_report))
}

fn evaluate_streaming_memory_lm(
    model: &MemoryTransformerLm,
    prepared: &PreparedTokenData,
    split: EvalSplit,
    batch_size: usize,
    max_batches: Option<usize>,
    precision: Precision,
) -> Result<(LmEvalMetrics, serde_json::Value)> {
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
    let split = match split {
        EvalSplit::Train => TokenDataSplit::Train,
        EvalSplit::Valid => TokenDataSplit::Valid,
    };
    let mut dataset = prepared.streaming_dataset(
        split,
        model.config.block_size,
        TokenDatasetState::from_seed(0),
    )?;
    let mut offset = 0usize;
    let mut batches = 0usize;
    let mut examples = 0usize;
    let mut predicted_tokens = 0usize;
    let mut loss_sum = 0.0;
    while offset + model.config.block_size < dataset.len() {
        if max_batches.is_some_and(|limit| batches >= limit) {
            break;
        }
        let Some((input, target, actual_batch)) =
            dataset.sequential_batch_at(offset, batch_size)?
        else {
            break;
        };
        offset += actual_batch * model.config.block_size;
        let input = input.to_device(model.device())?;
        let target = target.to_device(model.device())?;
        let loss = no_grad(|| match precision {
            Precision::F32 => model.loss(&input, &target),
            Precision::Bf16 => model.loss_bf16_activations(&input, &target),
            Precision::AmpBf16 => model.loss_amp_bf16(&input, &target),
        })?;
        let weight = actual_batch * model.config.block_size;
        let loss_value = eval_loss_scalar(&loss, precision)?;
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
    let metrics = LmEvalMetrics {
        batches,
        examples,
        tokens: predicted_tokens,
        loss,
        perplexity: loss.exp(),
    };
    let loader_report = serde_json::json!({
        "kind": "binary_shard_streaming",
        "source": format!("{}:{}", prepared.manifest_path.display(), split.as_str()),
        "tokens_materialized": false,
        "total_tokens": dataset.len(),
        "block_size": dataset.block_size(),
        "stats": dataset.loader_stats(),
    });
    Ok((metrics, loader_report))
}

fn training_loss_scalar(loss: &Tensor, precision: Precision) -> Result<f64> {
    if precision == Precision::AmpBf16 {
        let value =
            amp::with_cuda_host_staging_allowed("amp-bf16 training loss scalar logging", || {
                loss.data()[0] as f64
            });
        amp::record_amp_bf16_finite_check(
            "training_loss",
            "loss",
            loss.dtype(),
            loss.device(),
            value,
        )?;
        Ok(value)
    } else {
        let value = loss.data()[0] as f64;
        if !value.is_finite() {
            return Err(TensorError::Autograd(format!(
                "non-finite training loss: {value}"
            )));
        }
        Ok(value)
    }
}

fn checkpoint_with_amp_staging_allowed<T>(
    precision: Precision,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if precision == Precision::AmpBf16 {
        amp::with_cuda_host_staging_allowed("amp-bf16 checkpoint serialization", f)
    } else {
        f()
    }
}

fn memory_uses_sparse_rows(model: &MemoryTransformerLm) -> bool {
    model.config.smft_mode != SmftMode::Disabled
        || model.config.memory_update_policy == MemoryUpdatePolicy::SparseRows
}

fn load_smft_row_mask(path: &Path, model: &MemoryTransformerLm) -> Result<SmftRowMask> {
    if !memory_uses_sparse_rows(model) {
        return Err(TensorError::InvalidOperation(format!(
            "--smft-row-mask was supplied, but memory sparse-row updates are not active; set --smft-mode or --memory-update-policy sparse-rows before using {}",
            path.display()
        )));
    }
    let mask: SmftRowMask = read_json_typed(path)?;
    if mask.memory_slots != model.config.memory_slots {
        return Err(TensorError::InvalidOperation(format!(
            "SMFT row mask memory_slots {} does not match model memory_slots {}",
            mask.memory_slots, model.config.memory_slots
        )));
    }
    mask.validate()?;
    Ok(mask)
}

fn load_memory_access_counts(path: &Path, memory_slots: usize) -> Result<MemoryAccessCounts> {
    let counts: MemoryAccessCounts = read_json_typed(path)?;
    counts.validate()?;
    if counts.memory_slots != memory_slots {
        return Err(TensorError::InvalidOperation(format!(
            "SMFT access counts memory_slots {} does not match model memory_slots {}",
            counts.memory_slots, memory_slots
        )));
    }
    Ok(counts)
}

fn collect_smft_access_counts(
    model: &MemoryTransformerLm,
    precision: Precision,
) -> Result<MemoryAccessCounts> {
    if precision == Precision::AmpBf16 {
        amp::with_cuda_host_staging_allowed("amp-bf16 SMFT access count collection", || {
            model.memory_access_counts()
        })
    } else {
        model.memory_access_counts()
    }
}

fn write_typed_json<T: Serialize>(path: &Path, value: &T, role: &str) -> Result<()> {
    let json = serde_json::to_value(value)
        .map_err(|err| TensorError::Io(format!("failed to serialize {role}: {err}")))?;
    write_json_file(path, json)
}

struct TrainReportInput<'a> {
    path: &'a Path,
    start_step: usize,
    final_step: usize,
    initial_loss: f64,
    final_loss: f64,
    checkpoint: String,
    device: Device,
    precision: Precision,
    learning_rate: f64,
    micro_batch_size: usize,
    grad_accumulation_steps: usize,
    tensor_core_counters: cuda::TensorCoreCounters,
    cuda_runtime_counters: cuda::CudaRuntimeCounters,
    tensor_core_coverage: AmpBf16TensorCoreCoverage,
    loader_report: serde_json::Value,
    performance_report: serde_json::Value,
    tokenizer_report: serde_json::Value,
}

struct TensorCoreMicrobenchConfig {
    device: i32,
    section: TensorCoreMicrobenchSection,
    iterations: usize,
    warmup: usize,
    m: usize,
    k: usize,
    n: usize,
    attention_time: usize,
    attention_head_dim: usize,
    attention_heads: usize,
    attention_batch: usize,
}

fn tensor_core_microbench_report(config: TensorCoreMicrobenchConfig) -> Result<serde_json::Value> {
    if config.iterations == 0 {
        return Err(TensorError::InvalidOperation(
            "gpu tensor-core-microbench --iterations must be greater than zero".to_string(),
        ));
    }
    if !cuda::device_supports_bf16_tensor_cores(config.device).map_err(cuda_error)? {
        return Err(TensorError::Device(format!(
            "gpu tensor-core-microbench requires compute capability >= 8.0 on cuda:{}",
            config.device
        )));
    }
    let gemm = if config.section.includes_gemm() {
        capture_tensor_core_microbench_section("gemm", || {
            tensor_core_gemm_microbench_report(&config)
        })
    } else {
        skipped_tensor_core_microbench_section("gemm")
    };
    let attention = if config.section.includes_attention() {
        capture_tensor_core_microbench_section("attention", || {
            tensor_core_attention_microbench_report(&config)
        })
    } else {
        skipped_tensor_core_microbench_section("attention")
    };
    let passed = json_bool(&gemm, "passed") && json_bool(&attention, "passed");
    let status = if passed { "passed" } else { "failed" };
    Ok(serde_json::json!({
        "command": "gpu tensor-core-microbench",
        "section": config.section.label(),
        "status": status,
        "passed": passed,
        "device": {
            "ordinal": config.device,
            "bf16_tensor_cores_supported": true,
        },
        "iterations": config.iterations,
        "warmup": config.warmup,
        "env_toggles": {
            "HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM": cuda::tensor_core_legacy_warp_gemm_enabled(),
            "HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM": cuda::tensor_core_global_cta_gemm_enabled(),
            "HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM": cuda::tensor_core_wide_swizzled_gemm_enabled(),
            "HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM": cuda::tensor_core_ldmatrix_gemm_enabled(),
            "HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_LDMATRIX_GEMM": cuda::require_tensor_core_ldmatrix_gemm_enabled(),
            "HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM": cuda::tensor_core_cp_async_gemm_enabled(),
            "HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM": cuda::require_tensor_core_cp_async_gemm_enabled(),
            "HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_ATTENTION": cuda::tensor_core_attention_ldmatrix_enabled(),
            "HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_ATTENTION": cuda::tensor_core_attention_cp_async_enabled(),
        "HEIRLOOM_CUDA_FLASH_BF16_ATTENTION": cuda::flash_bf16_attention_enabled(),
        "HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE": cuda::flash_bf16_tensor_core_attention_enabled(),
        "HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD": cuda::flash_bf16_tensor_core_attention_backward_enabled(),
        "HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION": cuda::require_flash_bf16_attention_enabled(),
        "HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING": cuda::flash_bf16_attention_timing_enabled(),
        },
        "gemm": gemm,
        "attention": attention,
        "note": "This is a kernel/runtime throughput microbench. It does not alter model weights, data loading, tokenization, or training semantics. The guarded ldmatrix and double-buffered cp.async GEMM tiers report their own executed counters; attention ldmatrix/cp.async toggles remain explicit global fallback probes until their real instruction-path kernels replace them.",
    }))
}

fn capture_tensor_core_microbench_section(
    section: &'static str,
    f: impl FnOnce() -> Result<serde_json::Value>,
) -> serde_json::Value {
    match f() {
        Ok(mut report) => {
            if let serde_json::Value::Object(map) = &mut report {
                map.entry("section".to_string())
                    .or_insert_with(|| serde_json::json!(section));
                map.entry("status".to_string())
                    .or_insert_with(|| serde_json::json!("ok"));
            }
            report
        }
        Err(err) => serde_json::json!({
            "section": section,
            "status": "error",
            "passed": false,
            "reason": err.to_string(),
            "note": "This section failed before a complete benchmark report could be produced; inspect the companion text log for CUDA/JIT/runtime output.",
        }),
    }
}

fn skipped_tensor_core_microbench_section(section: &'static str) -> serde_json::Value {
    serde_json::json!({
        "section": section,
        "status": "skipped",
        "passed": true,
    })
}

fn tensor_core_gemm_microbench_report(
    config: &TensorCoreMicrobenchConfig,
) -> Result<serde_json::Value> {
    if config.m == 0 || config.k == 0 || config.n == 0 {
        return Err(TensorError::InvalidOperation(
            "gpu tensor-core-microbench GEMM dimensions must be non-zero".to_string(),
        ));
    }
    let staged = with_env_overrides(
        &[
            ("HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM", None),
        ],
        || {
            capture_tensor_core_microbench_section("gemm.staged_cta_reference", || {
                tensor_core_gemm_microbench_report_once(config)
            })
        },
    );
    let ldmatrix = with_env_overrides(
        &[
            ("HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM", Some("1")),
            ("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM", None),
        ],
        || {
            capture_tensor_core_microbench_section("gemm.ldmatrix_requested", || {
                tensor_core_gemm_microbench_report_once(config)
            })
        },
    );
    let cp_async = with_env_overrides(
        &[
            ("HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM", None),
            ("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM", Some("1")),
        ],
        || {
            capture_tensor_core_microbench_section("gemm.cp_async_requested", || {
                tensor_core_gemm_microbench_report_once(config)
            })
        },
    );
    let (selected, selected_alias_of) = match cuda::tensor_core_gemm_tier_label() {
        "staged_cta" => (staged.clone(), Some("staged_cta_reference")),
        "ldmatrix_a_shared_mma" => (ldmatrix.clone(), Some("ldmatrix_requested")),
        "cp_async_double_buffered_ldmatrix_a_mma" => (cp_async.clone(), Some("cp_async_requested")),
        _ => (
            capture_tensor_core_microbench_section("gemm.selected", || {
                tensor_core_gemm_microbench_report_once(config)
            }),
            None,
        ),
    };
    let passed = json_bool(&selected, "passed")
        && json_bool(&staged, "passed")
        && json_bool(&ldmatrix, "passed")
        && json_bool(&cp_async, "passed");
    Ok(serde_json::json!({
        "status": if passed { "ok" } else { "failed" },
        "kind": selected["kind"].clone(),
        "tier": selected["tier"].clone(),
        "shape": selected["shape"].clone(),
        "iterations": selected["iterations"].clone(),
        "elapsed_ms": selected["elapsed_ms"].clone(),
        "flops_per_iteration": selected["flops_per_iteration"].clone(),
        "total_flops": selected["total_flops"].clone(),
        "tflops_per_second": selected["tflops_per_second"].clone(),
        "max_abs_error": selected["max_abs_error"].clone(),
        "tolerance": selected["tolerance"].clone(),
        "passed": passed,
        "selected_alias_of": selected_alias_of,
        "selected": selected,
        "staged_cta_reference": staged,
        "ldmatrix_requested": ldmatrix,
        "cp_async_requested": cp_async,
        "note": "ldmatrix_requested runs the guarded ldmatrix-A shared-memory MMA tier. cp_async_requested runs the guarded double-buffered cp.async global-to-shared pipeline feeding the same ldmatrix-A MMA path.",
    }))
}

fn tensor_core_gemm_microbench_report_once(
    config: &TensorCoreMicrobenchConfig,
) -> Result<serde_json::Value> {
    let left_values = patterned_f32_values(
        checked_product("microbench GEMM left", &[config.m, config.k])?,
        29,
        14.0,
        17.0,
    );
    let right_values = patterned_f32_values(
        checked_product("microbench GEMM right", &[config.n, config.k])?,
        31,
        15.0,
        19.0,
    );
    let left_bf16 =
        Tensor::from_f32(left_values, &[config.m, config.k], false)?.to_dtype(DType::BFloat16)?;
    let right_bf16 =
        Tensor::from_f32(right_values, &[config.n, config.k], false)?.to_dtype(DType::BFloat16)?;
    let expected = left_bf16
        .to_dtype(DType::F32)?
        .matmul(&right_bf16.to_dtype(DType::F32)?.transpose()?)?
        .data_f32()?;
    let left = cuda::CudaBuffer::from_u16(config.device, &left_bf16.data_bf16_bits()?)
        .map_err(cuda_error)?;
    let right = cuda::CudaBuffer::from_u16(config.device, &right_bf16.data_bf16_bits()?)
        .map_err(cuda_error)?;

    for _ in 0..config.warmup {
        let _ = cuda::matmul_bf16_tensor_core_rhs_t_f32_buffers(
            &left, &right, config.m, config.k, config.n,
        )
        .map_err(cuda_error)?;
    }
    cuda::reset_tensor_core_counters();
    cuda::reset_cuda_runtime_counters();
    let mut timer =
        cuda::CudaEventTimer::start_current_compute_stream(config.device).map_err(cuda_error)?;
    let mut output = cuda::matmul_bf16_tensor_core_rhs_t_f32_buffers(
        &left, &right, config.m, config.k, config.n,
    )
    .map_err(cuda_error)?;
    for _ in 1..config.iterations {
        output = cuda::matmul_bf16_tensor_core_rhs_t_f32_buffers(
            &left, &right, config.m, config.k, config.n,
        )
        .map_err(cuda_error)?;
    }
    let elapsed_ms = timer.stop_elapsed_ms().map_err(cuda_error)?;
    let actual = output.to_f32().map_err(cuda_error)?;
    let max_abs_error = max_abs_error_f32(&actual, &expected)?;
    let elapsed_secs = (elapsed_ms / 1000.0).max(f64::MIN_POSITIVE);
    let flops_per_iter = 2.0 * config.m as f64 * config.k as f64 * config.n as f64;
    let total_flops = flops_per_iter * config.iterations as f64;
    Ok(serde_json::json!({
        "kind": "bf16_rhs_t_gemm_f32_accum",
        "tier": cuda::tensor_core_gemm_tier_label(),
        "shape": {"m": config.m, "k": config.k, "n": config.n},
        "iterations": config.iterations,
        "elapsed_ms": elapsed_ms,
        "flops_per_iteration": flops_per_iter,
        "total_flops": total_flops,
        "tflops_per_second": total_flops / elapsed_secs / 1.0e12,
        "max_abs_error": max_abs_error,
        "tolerance": 8.0e-2,
        "passed": max_abs_error <= 8.0e-2,
        "tensor_core": tensor_core_counters_json(cuda::tensor_core_counters()),
        "cuda_runtime": cuda_runtime_counters_json(cuda::cuda_runtime_counters()),
    }))
}

struct EnvOverride {
    key: &'static str,
    old: Option<String>,
}

impl EnvOverride {
    fn apply(key: &'static str, value: Option<&'static str>) -> Self {
        let old = std::env::var(key).ok();
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
        Self { key, old }
    }
}

impl Drop for EnvOverride {
    fn drop(&mut self) {
        if let Some(value) = self.old.as_deref() {
            std::env::set_var(self.key, value);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn with_env_overrides<T>(
    overrides: &[(&'static str, Option<&'static str>)],
    f: impl FnOnce() -> T,
) -> T {
    let _guards = overrides
        .iter()
        .map(|(key, value)| EnvOverride::apply(key, *value))
        .collect::<Vec<_>>();
    f()
}

fn tensor_core_attention_microbench_report(
    config: &TensorCoreMicrobenchConfig,
) -> Result<serde_json::Value> {
    if config.attention_time == 0 || config.attention_head_dim == 0 || config.attention_heads == 0 {
        return Err(TensorError::InvalidOperation(
            "gpu tensor-core-microbench attention dimensions must be non-zero".to_string(),
        ));
    }
    let channels = config
        .attention_head_dim
        .checked_mul(config.attention_heads)
        .ok_or_else(|| TensorError::InvalidOperation("attention channels overflow".to_string()))?;
    let len = checked_product(
        "microbench attention qkv",
        &[config.attention_batch, config.attention_time, channels],
    )?;
    let query_values = patterned_f32_values(len, 37, 18.0, 23.0);
    let key_values = patterned_f32_values(len, 41, 20.0, 29.0);
    let value_values = patterned_f32_values(len, 43, 21.0, 31.0);

    let expected_query = Tensor::from_f32(
        query_values.clone(),
        &[config.attention_batch, config.attention_time, channels],
        false,
    )?
    .to_dtype(DType::BFloat16)?
    .to_dtype(DType::F32)?;
    let expected_key = Tensor::from_f32(
        key_values.clone(),
        &[config.attention_batch, config.attention_time, channels],
        false,
    )?
    .to_dtype(DType::BFloat16)?
    .to_dtype(DType::F32)?;
    let expected_value = Tensor::from_f32(
        value_values.clone(),
        &[config.attention_batch, config.attention_time, channels],
        false,
    )?
    .to_dtype(DType::BFloat16)?
    .to_dtype(DType::F32)?;
    let expected = expected_query
        .causal_self_attention(&expected_key, &expected_value, config.attention_heads)?
        .data_f32()?;

    let query = cuda::CudaBuffer::from_f32(config.device, &query_values).map_err(cuda_error)?;
    let key = cuda::CudaBuffer::from_f32(config.device, &key_values).map_err(cuda_error)?;
    let value = cuda::CudaBuffer::from_f32(config.device, &value_values).map_err(cuda_error)?;
    let dims = cuda::CausalAttentionDims {
        batch: config.attention_batch,
        time: config.attention_time,
        channels,
        n_heads: config.attention_heads,
    };

    let matmul_flops_per_iter = 4.0
        * config.attention_batch as f64
        * config.attention_heads as f64
        * config.attention_time as f64
        * config.attention_time as f64
        * config.attention_head_dim as f64;
    let total_matmul_flops = matmul_flops_per_iter * config.iterations as f64;
    let scalar_streaming_flops_per_iter = causal_attention_forward_causal_flops(
        config.attention_batch,
        config.attention_heads,
        config.attention_time,
        config.attention_head_dim,
    );
    let scalar_streaming_total_flops = scalar_streaming_flops_per_iter * config.iterations as f64;

    for _ in 0..config.warmup {
        let _ =
            cuda::causal_attention_bf16_tensor_core_forward_f32_buffers(&query, &key, &value, dims)
                .map_err(cuda_error)?;
    }
    cuda::reset_tensor_core_counters();
    cuda::reset_cuda_runtime_counters();
    let mut timer =
        cuda::CudaEventTimer::start_current_compute_stream(config.device).map_err(cuda_error)?;
    let mut output =
        cuda::causal_attention_bf16_tensor_core_forward_f32_buffers(&query, &key, &value, dims)
            .map_err(cuda_error)?
            .0;
    for _ in 1..config.iterations {
        output =
            cuda::causal_attention_bf16_tensor_core_forward_f32_buffers(&query, &key, &value, dims)
                .map_err(cuda_error)?
                .0;
    }
    let current_elapsed_ms = timer.stop_elapsed_ms().map_err(cuda_error)?;
    let actual = output.to_f32().map_err(cuda_error)?;
    let current_max_abs_error = max_abs_error_f32(&actual, &expected)?;
    let current_elapsed_secs = (current_elapsed_ms / 1000.0).max(f64::MIN_POSITIVE);
    let current_tensor_core = cuda::tensor_core_counters();
    let current_runtime = cuda::cuda_runtime_counters();

    for _ in 0..config.warmup {
        let _ = cuda::causal_attention_bf16_flash_forward_f32_buffers(&query, &key, &value, dims)
            .map_err(cuda_error)?;
    }
    cuda::reset_tensor_core_counters();
    cuda::reset_cuda_runtime_counters();
    let mut timer =
        cuda::CudaEventTimer::start_current_compute_stream(config.device).map_err(cuda_error)?;
    let mut flash_output =
        cuda::causal_attention_bf16_flash_forward_f32_buffers(&query, &key, &value, dims)
            .map_err(cuda_error)?;
    for _ in 1..config.iterations {
        flash_output =
            cuda::causal_attention_bf16_flash_forward_f32_buffers(&query, &key, &value, dims)
                .map_err(cuda_error)?;
    }
    let flash_elapsed_ms = timer.stop_elapsed_ms().map_err(cuda_error)?;
    let flash_actual = flash_output.to_f32().map_err(cuda_error)?;
    let flash_max_abs_error = max_abs_error_f32(&flash_actual, &expected)?;
    let flash_elapsed_secs = (flash_elapsed_ms / 1000.0).max(f64::MIN_POSITIVE);
    let flash_tensor_core = cuda::tensor_core_counters();
    let flash_runtime = cuda::cuda_runtime_counters();
    let tensor_core_flash_report = (|| -> Result<serde_json::Value> {
        for _ in 0..config.warmup {
            let _ = cuda::causal_attention_bf16_flash_tensor_core_forward_f32_buffers(
                &query, &key, &value, dims,
            )
            .map_err(cuda_error)?;
        }
        cuda::reset_tensor_core_counters();
        cuda::reset_cuda_runtime_counters();
        let mut timer =
            cuda::CudaEventTimer::start_current_compute_stream(config.device).map_err(cuda_error)?;
        let mut output = cuda::causal_attention_bf16_flash_tensor_core_forward_f32_buffers(
            &query, &key, &value, dims,
        )
        .map_err(cuda_error)?
        .0;
        for _ in 1..config.iterations {
            output = cuda::causal_attention_bf16_flash_tensor_core_forward_f32_buffers(
                &query, &key, &value, dims,
            )
            .map_err(cuda_error)?
            .0;
        }
        let elapsed_ms = timer.stop_elapsed_ms().map_err(cuda_error)?;
        let actual = output.to_f32().map_err(cuda_error)?;
        let max_abs_error = max_abs_error_f32(&actual, &expected)?;
        let elapsed_secs = (elapsed_ms / 1000.0).max(f64::MIN_POSITIVE);
        let executed_flops_per_iteration = flash_tensor_core_forward_mma_flops(
            config.attention_batch,
            config.attention_heads,
            config.attention_time,
            config.attention_head_dim,
        );
        let total_executed_flops = executed_flops_per_iteration * config.iterations as f64;
        Ok(serde_json::json!({
            "status": "ok",
            "kind": "bf16_causal_attention_tensor_core_tiled_flash_forward_f32_accum",
            "tier": "tensor_core_tiled_flash_forward_mma",
            "iterations": config.iterations,
            "elapsed_ms": elapsed_ms,
            "ideal_materialized_matmul_flops_per_iteration": matmul_flops_per_iter,
            "executed_mma_flops_per_iteration": executed_flops_per_iteration,
            "total_executed_mma_flops": total_executed_flops,
            "executed_mma_tflops_per_second": total_executed_flops / elapsed_secs / 1.0e12,
            "max_abs_error": max_abs_error,
            "tolerance": 2.5e-1,
            "passed": max_abs_error <= 2.5e-1,
            "speedup_ratio_vs_current_materialized": if elapsed_ms > 0.0 { current_elapsed_ms / elapsed_ms } else { 0.0 },
            "tensor_core": tensor_core_counters_json(cuda::tensor_core_counters()),
            "cuda_runtime": cuda_runtime_counters_json(cuda::cuda_runtime_counters()),
        }))
    })()
    .unwrap_or_else(|err| {
        serde_json::json!({
            "status": "error",
            "kind": "bf16_causal_attention_tensor_core_tiled_flash_forward_f32_accum",
            "tier": "tensor_core_tiled_flash_forward_mma",
            "reason": err.to_string(),
            "note": "The guarded Tensor Core tiled flash forward kernel is experimental and must pass A100 PTX JIT/parity before it is treated as production-ready.",
        })
    });
    let materialized_attention_elements = config
        .attention_batch
        .checked_mul(config.attention_heads)
        .and_then(|value| value.checked_mul(config.attention_time))
        .and_then(|value| value.checked_mul(config.attention_time))
        .ok_or_else(|| {
            TensorError::InvalidOperation("attention materialized elements overflow".to_string())
        })?;
    let materialized_attention_bytes = materialized_attention_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            TensorError::InvalidOperation("attention materialized bytes overflow".to_string())
        })?;
    let speedup_ratio = if flash_elapsed_ms > 0.0 {
        current_elapsed_ms / flash_elapsed_ms
    } else {
        0.0
    };
    let current_report = serde_json::json!({
        "kind": "bf16_causal_attention_materialized_forward_f32_accum",
        "tier": cuda::tensor_core_attention_tier_label(),
        "iterations": config.iterations,
        "elapsed_ms": current_elapsed_ms,
        "matmul_flops_per_iteration": matmul_flops_per_iter,
        "total_matmul_flops": total_matmul_flops,
        "matmul_tflops_per_second": total_matmul_flops / current_elapsed_secs / 1.0e12,
        "max_abs_error": current_max_abs_error,
        "tolerance": 2.5e-1,
        "passed": current_max_abs_error <= 2.5e-1,
        "tensor_core": tensor_core_counters_json(current_tensor_core),
        "cuda_runtime": cuda_runtime_counters_json(current_runtime.clone()),
    });
    let flash_report = serde_json::json!({
        "kind": "bf16_causal_attention_scalar_streaming_forward_f32_accum",
        "tier": "scalar_streaming_forward",
        "iterations": config.iterations,
        "elapsed_ms": flash_elapsed_ms,
        "ideal_materialized_matmul_flops_per_iteration": matmul_flops_per_iter,
        "executed_flops_per_iteration": scalar_streaming_flops_per_iter,
        "total_executed_flops": scalar_streaming_total_flops,
        "executed_tflops_per_second": scalar_streaming_total_flops / flash_elapsed_secs / 1.0e12,
        "max_abs_error": flash_max_abs_error,
        "tolerance": 2.5e-1,
        "passed": flash_max_abs_error <= 2.5e-1,
        "tensor_core": tensor_core_counters_json(flash_tensor_core),
        "cuda_runtime": cuda_runtime_counters_json(flash_runtime),
    });
    let tensor_core_flash_required = cuda::flash_bf16_tensor_core_attention_enabled();
    let attention_passed = json_bool(&current_report, "passed")
        && json_bool(&flash_report, "passed")
        && tensor_core_flash_microbench_passed(
            &tensor_core_flash_report,
            tensor_core_flash_required,
        );
    Ok(serde_json::json!({
        "kind": "bf16_causal_attention_forward_f32_accum",
        "tier": cuda::tensor_core_attention_tier_label(),
        "shape": {
            "batch": config.attention_batch,
            "time": config.attention_time,
            "channels": channels,
            "n_heads": config.attention_heads,
            "head_dim": config.attention_head_dim,
        },
        "iterations": config.iterations,
        "elapsed_ms": current_elapsed_ms,
        "matmul_flops_per_iteration": matmul_flops_per_iter,
        "total_matmul_flops": total_matmul_flops,
        "matmul_tflops_per_second": total_matmul_flops / current_elapsed_secs / 1.0e12,
        "max_abs_error": current_max_abs_error,
        "tolerance": 2.5e-1,
        "passed": attention_passed,
        "tensor_core_flash_forward_required": tensor_core_flash_required,
        "speedup_ratio_flash_vs_current": speedup_ratio,
        "materialized_attention_elements_avoided": materialized_attention_elements,
        "materialized_attention_bytes_avoided": materialized_attention_bytes,
        "current_materialized": current_report,
        "scalar_streaming_forward": flash_report,
        "flash_forward": flash_report,
        "tensor_core_flash_forward": tensor_core_flash_report,
        "tensor_core": tensor_core_counters_json(current_tensor_core),
        "cuda_runtime": cuda_runtime_counters_json(current_runtime),
    }))
}

fn json_bool(value: &serde_json::Value, key: &str) -> bool {
    value
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn tensor_core_flash_microbench_passed(report: &serde_json::Value, required: bool) -> bool {
    if !required {
        return true;
    }
    report.get("status").and_then(serde_json::Value::as_str) == Some("ok")
        && json_bool(report, "passed")
}

fn patterned_f32_values(len: usize, modulus: usize, center: f32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| ((index % modulus) as f32 - center) / scale)
        .collect()
}

fn checked_product(name: &str, values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value).ok_or_else(|| {
            TensorError::InvalidOperation(format!("{name} dimension product overflow"))
        })
    })
}

fn max_abs_error_f32(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err(TensorError::InvalidOperation(format!(
            "microbench output length {} did not match reference length {}",
            actual.len(),
            expected.len()
        )));
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max))
}

fn causal_attention_forward_causal_flops(
    batch: usize,
    heads: usize,
    time: usize,
    head_dim: usize,
) -> f64 {
    let causal_pairs = (time * (time + 1) / 2) as f64;
    4.0 * batch as f64 * heads as f64 * causal_pairs * head_dim as f64
}

fn flash_tensor_core_forward_mma_flops(
    batch: usize,
    heads: usize,
    time: usize,
    head_dim: usize,
) -> f64 {
    let query_blocks = time.div_ceil(16) as f64;
    let key_blocks = time.div_ceil(16) as f64;
    let output_dim_tiles = head_dim.div_ceil(8) as f64;
    let head_dim_chunks = head_dim.div_ceil(16) as f64;
    let base_tiles = batch as f64 * heads as f64 * query_blocks * key_blocks * output_dim_tiles;
    let mma_m16n8k16_flops = 16.0 * 8.0 * 16.0 * 2.0;
    base_tiles * ((2.0 * head_dim_chunks + 1.0) * mma_m16n8k16_flops)
}

fn tensor_core_counters_json(counters: cuda::TensorCoreCounters) -> serde_json::Value {
    serde_json::json!({
        "bf16_mma_probe_calls": counters.bf16_mma_probe_calls,
        "bf16_tensor_core_matmul_calls": counters.bf16_tensor_core_matmul_calls,
        "bf16_tensor_core_matmul_forward_calls": counters.bf16_tensor_core_matmul_forward_calls,
        "bf16_tensor_core_matmul_backward_calls": counters.bf16_tensor_core_matmul_backward_calls,
        "bf16_tensor_core_attention_forward_calls": counters.bf16_tensor_core_attention_forward_calls,
        "bf16_tensor_core_attention_qk_matmul_calls": counters.bf16_tensor_core_attention_qk_matmul_calls,
        "bf16_tensor_core_attention_av_matmul_calls": counters.bf16_tensor_core_attention_av_matmul_calls,
        "bf16_tensor_core_attention_backward_calls": counters.bf16_tensor_core_attention_backward_calls,
        "bf16_tensor_core_attention_score_grad_matmul_calls": counters.bf16_tensor_core_attention_score_grad_matmul_calls,
        "bf16_tensor_core_attention_dq_matmul_calls": counters.bf16_tensor_core_attention_dq_matmul_calls,
        "bf16_tensor_core_attention_dk_matmul_calls": counters.bf16_tensor_core_attention_dk_matmul_calls,
        "bf16_tensor_core_attention_dv_matmul_calls": counters.bf16_tensor_core_attention_dv_matmul_calls,
        "bf16_scalar_matmul_fallback_calls": counters.bf16_scalar_matmul_fallback_calls,
    })
}

fn write_train_report(input: TrainReportInput<'_>) -> Result<()> {
    let reduction = if input.initial_loss == 0.0 {
        0.0
    } else {
        (input.initial_loss - input.final_loss) / input.initial_loss
    };
    let tensor_core_json = serde_json::json!({
        "bf16_mma_probe_calls": input.tensor_core_counters.bf16_mma_probe_calls,
        "bf16_tensor_core_matmul_calls": input.tensor_core_counters.bf16_tensor_core_matmul_calls,
        "bf16_tensor_core_matmul_forward_calls": input.tensor_core_counters.bf16_tensor_core_matmul_forward_calls,
        "bf16_tensor_core_matmul_backward_calls": input.tensor_core_counters.bf16_tensor_core_matmul_backward_calls,
        "bf16_tensor_core_attention_forward_calls": input.tensor_core_counters.bf16_tensor_core_attention_forward_calls,
        "bf16_tensor_core_attention_qk_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_qk_matmul_calls,
        "bf16_tensor_core_attention_av_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_av_matmul_calls,
        "bf16_tensor_core_attention_backward_calls": input.tensor_core_counters.bf16_tensor_core_attention_backward_calls,
        "bf16_tensor_core_attention_score_grad_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_score_grad_matmul_calls,
        "bf16_tensor_core_attention_dq_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_dq_matmul_calls,
        "bf16_tensor_core_attention_dk_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_dk_matmul_calls,
        "bf16_tensor_core_attention_dv_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_dv_matmul_calls,
        "bf16_scalar_matmul_fallback_calls": input.tensor_core_counters.bf16_scalar_matmul_fallback_calls,
    });
    let cuda_runtime_json = cuda_runtime_counters_json(input.cuda_runtime_counters);
    let tensor_core_coverage_json =
        serde_json::to_value(&input.tensor_core_coverage).map_err(|err| {
            TensorError::Io(format!("failed to serialize Tensor Core coverage: {err}"))
        })?;
    let tensor_core_pad_crop = tensor_core_pad_crop_report_from_json(
        &cuda_runtime_json,
        &tensor_core_json,
        &tensor_core_coverage_json,
        "single_rank",
    );
    let json = serde_json::json!({
        "command": "train-lm",
        "model_family": "tiny_transformer",
        "start_step": input.start_step,
        "final_step": input.final_step,
        "initial_loss": input.initial_loss,
        "final_loss": input.final_loss,
        "loss_reduction": reduction,
        "checkpoint": input.checkpoint,
        "device": device_label(input.device),
        "precision": input.precision.as_str(),
        "learning_rate": input.learning_rate,
        "micro_batch_size": input.micro_batch_size,
        "grad_accumulation_steps": input.grad_accumulation_steps,
        "effective_batch_size": input.micro_batch_size * input.grad_accumulation_steps,
        "loader": input.loader_report,
        "tokenizer": input.tokenizer_report,
        "performance": input.performance_report,
        "tensor_core": tensor_core_json,
        "cuda_runtime": cuda_runtime_json,
        "tensor_core_pad_crop": tensor_core_pad_crop,
        "amp_bf16_policy": amp_bf16_policy(),
        "amp_bf16_op_decisions": amp_bf16_op_decisions(),
        "amp_bf16_finite_checks": amp::amp_bf16_finite_check_events(),
        "amp_bf16_validation": amp::amp_bf16_validation_report(&amp_bf16_policy()),
        "cuda_host_staging": amp_bf16_cuda_host_staging_events(),
        "tensor_core_coverage": tensor_core_coverage_json,
    });
    write_json_file(input.path, json)
}

struct MemoryTrainReportInput<'a> {
    path: &'a Path,
    start_step: usize,
    final_step: usize,
    initial_loss: f64,
    final_loss: f64,
    checkpoint: String,
    device: Device,
    precision: Precision,
    learning_rate: f64,
    micro_batch_size: usize,
    grad_accumulation_steps: usize,
    memory_config: &'a MemoryTransformerConfig,
    tensor_core_counters: cuda::TensorCoreCounters,
    cuda_runtime_counters: cuda::CudaRuntimeCounters,
    memory_kernel_counters: cuda::MemoryKernelCounters,
    tensor_core_coverage: AmpBf16TensorCoreCoverage,
    memory_selection_report: serde_json::Value,
    memory_optimizer_report: serde_json::Value,
    smft_access_report: serde_json::Value,
    smft_artifacts_report: serde_json::Value,
    memory_table_checksum_sum: f64,
    memory_table_checksum_sumsq: f64,
    loader_report: serde_json::Value,
    performance_report: serde_json::Value,
    tokenizer_report: serde_json::Value,
}

fn write_memory_train_report(input: MemoryTrainReportInput<'_>) -> Result<()> {
    let reduction = if input.initial_loss == 0.0 {
        0.0
    } else {
        (input.initial_loss - input.final_loss) / input.initial_loss
    };
    let tensor_core_json = serde_json::json!({
        "bf16_mma_probe_calls": input.tensor_core_counters.bf16_mma_probe_calls,
        "bf16_tensor_core_matmul_calls": input.tensor_core_counters.bf16_tensor_core_matmul_calls,
        "bf16_tensor_core_matmul_forward_calls": input.tensor_core_counters.bf16_tensor_core_matmul_forward_calls,
        "bf16_tensor_core_matmul_backward_calls": input.tensor_core_counters.bf16_tensor_core_matmul_backward_calls,
        "bf16_tensor_core_attention_forward_calls": input.tensor_core_counters.bf16_tensor_core_attention_forward_calls,
        "bf16_tensor_core_attention_qk_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_qk_matmul_calls,
        "bf16_tensor_core_attention_av_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_av_matmul_calls,
        "bf16_tensor_core_attention_backward_calls": input.tensor_core_counters.bf16_tensor_core_attention_backward_calls,
        "bf16_tensor_core_attention_score_grad_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_score_grad_matmul_calls,
        "bf16_tensor_core_attention_dq_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_dq_matmul_calls,
        "bf16_tensor_core_attention_dk_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_dk_matmul_calls,
        "bf16_tensor_core_attention_dv_matmul_calls": input.tensor_core_counters.bf16_tensor_core_attention_dv_matmul_calls,
        "bf16_scalar_matmul_fallback_calls": input.tensor_core_counters.bf16_scalar_matmul_fallback_calls,
    });
    let cuda_runtime_json = cuda_runtime_counters_json(input.cuda_runtime_counters);
    let memory_kernel_json = serde_json::json!({
        "lookup_rejected_calls": input.memory_kernel_counters.lookup_rejected_calls,
        "query_key_score_calls": input.memory_kernel_counters.query_key_score_calls,
        "topk_calls": input.memory_kernel_counters.topk_calls,
        "product_key_calls": input.memory_kernel_counters.product_key_calls,
        "softmax_topk_calls": input.memory_kernel_counters.softmax_topk_calls,
        "weighted_value_forward_calls": input.memory_kernel_counters.weighted_value_forward_calls,
        "weighted_value_backward_calls": input.memory_kernel_counters.weighted_value_backward_calls,
        "selected_key_backward_calls": input.memory_kernel_counters.selected_key_backward_calls,
        "scatter_add_rows_calls": input.memory_kernel_counters.scatter_add_rows_calls,
        "sparse_adamw_rows_calls": input.memory_kernel_counters.sparse_adamw_rows_calls,
        "sparse_adamw_compact_rows_calls": input
            .memory_kernel_counters
            .sparse_adamw_compact_rows_calls,
        "access_count_calls": input.memory_kernel_counters.access_count_calls,
        "gather_selected_rows_calls": input.memory_kernel_counters.gather_selected_rows_calls,
        "bool_mask_to_indices_calls": input.memory_kernel_counters.bool_mask_to_indices_calls,
        "selected_tokens": input.memory_kernel_counters.selected_tokens,
        "selected_rows": input.memory_kernel_counters.selected_rows,
    });
    let tensor_core_coverage_json =
        serde_json::to_value(&input.tensor_core_coverage).map_err(|err| {
            TensorError::Io(format!("failed to serialize Tensor Core coverage: {err}"))
        })?;
    let json = serde_json::json!({
        "command": "train-memory-lm",
        "model_family": "memory_transformer",
        "memory_config": input.memory_config,
        "start_step": input.start_step,
        "final_step": input.final_step,
        "initial_loss": input.initial_loss,
        "final_loss": input.final_loss,
        "loss_reduction": reduction,
        "checkpoint": input.checkpoint,
        "device": device_label(input.device),
        "precision": input.precision.as_str(),
        "learning_rate": input.learning_rate,
        "micro_batch_size": input.micro_batch_size,
        "grad_accumulation_steps": input.grad_accumulation_steps,
        "effective_batch_size": input.micro_batch_size * input.grad_accumulation_steps,
        "loader": input.loader_report,
        "tokenizer": input.tokenizer_report,
        "performance": input.performance_report,
        "memory_selection": input.memory_selection_report,
        "memory_optimizer": input.memory_optimizer_report,
        "smft_access": input.smft_access_report,
        "smft_artifacts": input.smft_artifacts_report,
        "memory_table_parameter_checksum_sum": input.memory_table_checksum_sum,
        "memory_table_parameter_checksum_sumsq": input.memory_table_checksum_sumsq,
        "tensor_core": tensor_core_json,
        "cuda_runtime": cuda_runtime_json,
        "amp_bf16_policy": amp_bf16_policy(),
        "amp_bf16_op_decisions": amp_bf16_op_decisions(),
        "amp_bf16_finite_checks": amp::amp_bf16_finite_check_events(),
        "amp_bf16_validation": amp::amp_bf16_validation_report(&amp_bf16_policy()),
        "cuda_host_staging": amp_bf16_cuda_host_staging_events(),
        "tensor_core_coverage": tensor_core_coverage_json,
        "cuda_memory_kernels": {
            "implemented": true,
            "forward_cpu_fallback_for_cuda": false,
            "kernel_surface": [
                "exact_topk_indices_f32",
                "selected_key_scores_forward_backward_f32_i64",
                "weighted_value_forward_backward_f32_i64",
                "selected_row_scatter_add_backward",
                "gather_selected_rows_f32_i64",
                "sparse_adamw_rows_f32_i64_optional_row_mask",
                "sparse_adamw_compact_rows_f32_i64_optional_row_mask",
                "access_count_rows_i64_u64",
                "selected_rows_to_f32_mask",
                "f32_mask_to_bool",
                "bool_mask_to_i64_indices",
                "i64_arange",
                "product_key_candidate_topk_f32",
                "product_key_half_row_split_i64",
                "product_key_selected_scores_forward_backward_f32_i64"
            ],
            "counters": memory_kernel_json,
        },
        "ddp_rank_report_contract": {
            "status": "single_rank_schema_ready",
            "required_rank_fields": [
                "model_family",
                "memory_config",
                "memory_selection",
                "memory_optimizer",
                "cuda_memory_kernels",
                "row_union_all_reduce_calls",
                "row_union_all_reduce_bytes",
                "row_union_candidate_rows",
                "memory_table_parameter_checksum_sum",
                "memory_table_parameter_checksum_sumsq"
            ],
            "note": "Memory-transformer DDP is implemented for Full/MemoryOnly updates and SparseRows row-union synchronization; SMFT DDP remains deferred until distributed mask intersection and refresh synchronization exist."
        },
    });
    write_json_file(input.path, json)
}

fn tensor_core_pad_crop_report_from_json(
    cuda_runtime: &serde_json::Value,
    tensor_core: &serde_json::Value,
    tensor_core_coverage: &serde_json::Value,
    aggregation: &str,
) -> serde_json::Value {
    let padded_tiles = json_u64(cuda_runtime, "tensor_core_padded_tiles");
    let remainder_tiles = json_u64(cuda_runtime, "tensor_core_remainder_tiles");
    let scalar_fallbacks = json_u64(tensor_core, "bf16_scalar_matmul_fallback_calls");
    let linear_totals = tensor_core_coverage
        .get("linear_totals")
        .unwrap_or(&serde_json::Value::Null);
    let linear_fallbacks = json_u64(linear_totals, "fallback_calls");
    let linear_tensor_core_calls = json_u64(linear_totals, "tensor_core_calls");
    let mut logical_shapes = Vec::new();
    let mut padded_shapes = Vec::new();
    let mut linear_modules = Vec::new();
    let mut shape_needs_padding = false;
    if let Some(modules) = tensor_core_coverage
        .get("linear_modules")
        .and_then(serde_json::Value::as_array)
    {
        for module in modules {
            let name = module
                .get("module")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            let m = json_u64(module, "last_m");
            let k = json_u64(module, "last_k");
            let n = json_u64(module, "last_n");
            if m == 0 || k == 0 || n == 0 {
                continue;
            }
            let padded_m = round_up_to_tile_u64(m, 16);
            let padded_k = round_up_to_tile_u64(k, 16);
            let padded_n = round_up_to_tile_u64(n, 8);
            let needs_padding = padded_m != m || padded_k != k || padded_n != n;
            shape_needs_padding |= needs_padding
                && module
                    .get("tensor_core_calls")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    > 0;
            logical_shapes.push(serde_json::json!({
                "module": name,
                "m": m,
                "k": k,
                "n": n,
            }));
            padded_shapes.push(serde_json::json!({
                "module": name,
                "m": padded_m,
                "k": padded_k,
                "n": padded_n,
            }));
            linear_modules.push(serde_json::json!({
                "module": name,
                "calls": json_u64(module, "calls"),
                "tensor_core_calls": json_u64(module, "tensor_core_calls"),
                "fallback_calls": json_u64(module, "fallback_calls"),
                "last_path": module.get("last_path").cloned().unwrap_or(serde_json::Value::Null),
                "logical_shape": {"m": m, "k": k, "n": n},
                "padded_shape": {"m": padded_m, "k": padded_k, "n": padded_n},
                "needs_padding": needs_padding,
            }));
        }
    }
    let used = padded_tiles > 0 || remainder_tiles > 0 || shape_needs_padding;
    let passed = used && scalar_fallbacks == 0 && linear_fallbacks == 0;
    let status = if passed {
        "passed"
    } else if used {
        "fallback_detected"
    } else {
        "not_used"
    };
    serde_json::json!({
        "schema_version": 1,
        "aggregation": aggregation,
        "used": used,
        "passed": passed,
        "status": status,
        "padded_tiles": padded_tiles,
        "remainder_tiles": remainder_tiles,
        "scalar_fallbacks": scalar_fallbacks,
        "linear_tensor_core_calls": linear_tensor_core_calls,
        "linear_fallbacks": linear_fallbacks,
        "tile_rules": {
            "m_multiple": 16,
            "k_multiple": 16,
            "n_multiple": 8,
        },
        "logical_shapes": logical_shapes,
        "padded_shapes": padded_shapes,
        "linear_modules": linear_modules,
    })
}

fn json_u64(value: &serde_json::Value, field: &str) -> u64 {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
}

fn round_up_to_tile_u64(value: u64, tile: u64) -> u64 {
    value.div_ceil(tile) * tile
}

fn cuda_runtime_counters_json(counters: cuda::CudaRuntimeCounters) -> serde_json::Value {
    let mut kernel_launch_families = serde_json::Map::new();
    for family in &counters.kernel_launch_families {
        kernel_launch_families.insert(
            family.label.clone(),
            serde_json::json!({
                "calls": family.calls,
                "elements": family.elements,
                "elapsed_us": family.elapsed_us,
            }),
        );
    }
    let mut json = serde_json::json!({
        "kernel_launch_calls": counters.kernel_launch_calls,
        "kernel_launch_elements": counters.kernel_launch_elements,
        "kernel_launch_families": kernel_launch_families,
        "host_sync_calls": counters.host_sync_calls,
        "stream_create_calls": counters.stream_create_calls,
        "stream_sync_calls": counters.stream_sync_calls,
        "event_create_calls": counters.event_create_calls,
        "event_record_calls": counters.event_record_calls,
        "event_query_calls": counters.event_query_calls,
        "event_elapsed_calls": counters.event_elapsed_calls,
        "h2d_bytes": counters.h2d_bytes,
        "d2h_bytes": counters.d2h_bytes,
        "d2d_bytes": counters.d2d_bytes,
        "allocation_active_bytes": counters.allocation_active_bytes,
        "allocation_reserved_bytes": counters.allocation_reserved_bytes,
        "allocation_high_water_bytes": counters.allocation_high_water_bytes,
        "allocation_calls": counters.allocation_calls,
        "allocation_cache_hits": counters.allocation_cache_hits,
        "allocation_frees": counters.allocation_frees,
        "allocation_deferred_frees": counters.allocation_deferred_frees,
        "allocation_pending_reclaims": counters.allocation_pending_reclaims,
        "module_load_calls": counters.module_load_calls,
        "module_cache_hits": counters.module_cache_hits,
        "tensor_core_padded_tiles": counters.tensor_core_padded_tiles,
        "tensor_core_remainder_tiles": counters.tensor_core_remainder_tiles,
        "tensor_core_cta_gemm_calls": counters.tensor_core_cta_gemm_calls,
        "tensor_core_cta_tiles": counters.tensor_core_cta_tiles,
        "tensor_core_cta_warps_launched": counters.tensor_core_cta_warps_launched,
        "tensor_core_mma_warp_tiles": counters.tensor_core_mma_warp_tiles,
        "tensor_core_staged_cta_gemm_calls": counters.tensor_core_staged_cta_gemm_calls,
        "tensor_core_staged_cta_gemm_elapsed_us": counters.tensor_core_staged_cta_gemm_elapsed_us,
        "tensor_core_shared_stage_tiles": counters.tensor_core_shared_stage_tiles,
        "tensor_core_shared_stage_bytes": counters.tensor_core_shared_stage_bytes,
        "tensor_core_wide_swizzled_cta_gemm_calls": counters.tensor_core_wide_swizzled_cta_gemm_calls,
        "tensor_core_wide_swizzled_cta_gemm_elapsed_us": counters.tensor_core_wide_swizzled_cta_gemm_elapsed_us,
        "tensor_core_swizzled_stage_tiles": counters.tensor_core_swizzled_stage_tiles,
        "tensor_core_swizzled_stage_bytes": counters.tensor_core_swizzled_stage_bytes,
        "tensor_core_ldmatrix_gemm_requested_calls": counters.tensor_core_ldmatrix_gemm_requested_calls,
        "tensor_core_ldmatrix_gemm_executed_calls": counters.tensor_core_ldmatrix_gemm_executed_calls,
        "tensor_core_ldmatrix_gemm_staged_fallback_calls": counters.tensor_core_ldmatrix_gemm_staged_fallback_calls,
        "tensor_core_ldmatrix_gemm_hard_require_failures": counters.tensor_core_ldmatrix_gemm_hard_require_failures,
        "tensor_core_ldmatrix_gemm_instructions": counters.tensor_core_ldmatrix_gemm_instructions,
        "tensor_core_ldmatrix_gemm_elapsed_us": counters.tensor_core_ldmatrix_gemm_elapsed_us,
        "tensor_core_cp_async_gemm_requested_calls": counters.tensor_core_cp_async_gemm_requested_calls,
        "tensor_core_cp_async_gemm_executed_calls": counters.tensor_core_cp_async_gemm_executed_calls,
        "tensor_core_cp_async_gemm_staged_fallback_calls": counters.tensor_core_cp_async_gemm_staged_fallback_calls,
        "tensor_core_cp_async_gemm_hard_require_failures": counters.tensor_core_cp_async_gemm_hard_require_failures,
        "tensor_core_cp_async_gemm_instructions": counters.tensor_core_cp_async_gemm_instructions,
        "tensor_core_cp_async_gemm_elapsed_us": counters.tensor_core_cp_async_gemm_elapsed_us,
        "tensor_core_global_cta_gemm_calls": counters.tensor_core_global_cta_gemm_calls,
        "tensor_core_global_cta_gemm_elapsed_us": counters.tensor_core_global_cta_gemm_elapsed_us,
        "tensor_core_legacy_warp_gemm_calls": counters.tensor_core_legacy_warp_gemm_calls,
        "tensor_core_legacy_warp_gemm_elapsed_us": counters.tensor_core_legacy_warp_gemm_elapsed_us,
        "tensor_core_attention_padded_tiles": counters.tensor_core_attention_padded_tiles,
        "tensor_core_attention_remainder_tiles": counters.tensor_core_attention_remainder_tiles,
        "tensor_core_attention_scalar_fallbacks": counters.tensor_core_attention_scalar_fallbacks,
        "tensor_core_attention_ldmatrix_requested_calls": counters.tensor_core_attention_ldmatrix_requested_calls,
        "tensor_core_attention_ldmatrix_global_fallback_calls": counters.tensor_core_attention_ldmatrix_global_fallback_calls,
        "tensor_core_attention_cp_async_requested_calls": counters.tensor_core_attention_cp_async_requested_calls,
        "tensor_core_attention_cp_async_global_fallback_calls": counters.tensor_core_attention_cp_async_global_fallback_calls,
        "bf16_attention_materialized_reference_calls": counters.bf16_attention_materialized_reference_calls,
        "flash_bf16_attention_requested_calls": counters.flash_bf16_attention_requested_calls,
        "flash_bf16_attention_executed_calls": counters.flash_bf16_attention_executed_calls,
        "flash_bf16_attention_fallback_calls": counters.flash_bf16_attention_fallback_calls,
        "flash_bf16_attention_scalar_fallback_calls": counters.flash_bf16_attention_scalar_fallback_calls,
        "flash_bf16_attention_qk_tile_calls": counters.flash_bf16_attention_qk_tile_calls,
        "flash_bf16_attention_av_tile_calls": counters.flash_bf16_attention_av_tile_calls,
        "flash_bf16_attention_ragged_tile_count": counters.flash_bf16_attention_ragged_tile_count,
        "flash_bf16_attention_causal_masked_tile_count": counters.flash_bf16_attention_causal_masked_tile_count,
        "flash_bf16_attention_elapsed_us": counters.flash_bf16_attention_elapsed_us,
        "flash_bf16_attention_hard_require_failures": counters.flash_bf16_attention_hard_require_failures,
        "flash_bf16_scalar_streaming_requested_calls": counters.flash_bf16_scalar_streaming_requested_calls,
        "flash_bf16_scalar_streaming_executed_calls": counters.flash_bf16_scalar_streaming_executed_calls,
        "flash_bf16_scalar_streaming_qk_tile_calls": counters.flash_bf16_scalar_streaming_qk_tile_calls,
        "flash_bf16_scalar_streaming_av_tile_calls": counters.flash_bf16_scalar_streaming_av_tile_calls,
        "flash_bf16_scalar_streaming_elapsed_us": counters.flash_bf16_scalar_streaming_elapsed_us,
        "flash_bf16_tensor_core_requested_calls": counters.flash_bf16_tensor_core_requested_calls,
        "flash_bf16_tensor_core_executed_calls": counters.flash_bf16_tensor_core_executed_calls,
        "flash_bf16_tensor_core_fallback_calls": counters.flash_bf16_tensor_core_fallback_calls,
        "flash_bf16_tensor_core_qk_mma_tile_calls": counters.flash_bf16_tensor_core_qk_mma_tile_calls,
        "flash_bf16_tensor_core_av_mma_tile_calls": counters.flash_bf16_tensor_core_av_mma_tile_calls,
        "flash_bf16_tensor_core_ragged_tile_count": counters.flash_bf16_tensor_core_ragged_tile_count,
        "flash_bf16_tensor_core_causal_masked_tile_count": counters.flash_bf16_tensor_core_causal_masked_tile_count,
        "flash_bf16_tensor_core_elapsed_us": counters.flash_bf16_tensor_core_elapsed_us,
    });
    json["kernel_launch_family_elapsed_us"] = counters.kernel_launch_family_elapsed_us.into();
    let object = json
        .as_object_mut()
        .expect("cuda runtime counters JSON is an object");
    object.insert(
        "flash_bf16_tensor_core_backward_requested_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_requested_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_executed_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_executed_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_fallback_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_fallback_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_row_dot_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_row_dot_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_dp_mma_tile_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_dp_mma_tile_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_dq_mma_tile_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_dq_mma_tile_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_dk_mma_tile_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_dk_mma_tile_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_dv_mma_tile_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_dv_mma_tile_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_scalar_tile_calls".to_string(),
        counters
            .flash_bf16_tensor_core_backward_scalar_tile_calls
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_ragged_tile_count".to_string(),
        counters
            .flash_bf16_tensor_core_backward_ragged_tile_count
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_causal_masked_tile_count".to_string(),
        counters
            .flash_bf16_tensor_core_backward_causal_masked_tile_count
            .into(),
    );
    object.insert(
        "flash_bf16_tensor_core_backward_elapsed_us".to_string(),
        counters.flash_bf16_tensor_core_backward_elapsed_us.into(),
    );
    json
}

fn eval_loss_scalar(loss: &Tensor, precision: Precision) -> Result<f64> {
    let value = if precision == Precision::AmpBf16 {
        amp::with_cuda_host_staging_allowed("amp-bf16 eval loss scalar logging", || {
            loss.data()[0] as f64
        })
    } else {
        loss.data()[0] as f64
    };
    amp::record_amp_bf16_finite_check("eval_loss", "loss", loss.dtype(), loss.device(), value)?;
    if !value.is_finite() {
        return Err(TensorError::Autograd(format!(
            "non-finite eval loss: {value}"
        )));
    }
    Ok(value)
}

fn device_label(device: Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(device_id) => format!("cuda:{device_id}"),
    }
}

struct GenerationReportInput<'a> {
    path: &'a Path,
    command: &'a str,
    model_family: &'a str,
    checkpoint: &'a Path,
    prompt: &'a str,
    decoded_text: &'a str,
    prompt_tokens: &'a [usize],
    output: &'a GenerationOutput,
    options: &'a GenerationOptions,
    seed: u64,
    device: Device,
    precision: Precision,
    tokenizer_report: serde_json::Value,
}

fn write_generation_report(input: GenerationReportInput<'_>) -> Result<()> {
    let generated_token_ids = input.output.tokens[input.prompt_tokens.len()..].to_vec();
    let json = serde_json::json!({
        "command": input.command,
        "model_family": input.model_family,
        "checkpoint": input.checkpoint.display().to_string(),
        "device": device_label(input.device),
        "precision": input.precision.as_str(),
        "prompt": input.prompt,
        "decoded_text": input.decoded_text,
        "prompt_tokens": input.prompt_tokens.len(),
        "generated_tokens": input.output.new_token_count,
        "total_tokens": input.output.tokens.len(),
        "tokenizer": input.tokenizer_report,
        "token_ids": &input.output.tokens,
        "generated_token_ids": generated_token_ids,
        "finish_reason": &input.output.finish_reason,
        "rng": {
            "seed": input.seed,
            "final_state": input.output.rng_state,
        },
        "options": input.options,
        "steps": &input.output.steps,
        "amp_bf16_policy": amp_bf16_policy(),
        "amp_bf16_op_decisions": amp_bf16_op_decisions(),
        "amp_bf16_finite_checks": amp::amp_bf16_finite_check_events(),
        "amp_bf16_validation": amp::amp_bf16_validation_report(&amp_bf16_policy()),
        "cuda_host_staging": amp_bf16_cuda_host_staging_events(),
    });
    write_json_file(input.path, json)
}

struct EvalReportInput<'a> {
    path: &'a Path,
    command: &'a str,
    model_family: &'a str,
    checkpoint: &'a Path,
    dataset_manifest: Option<&'a Path>,
    source: String,
    split: EvalSplit,
    batch_size: usize,
    max_batches: Option<usize>,
    metrics: &'a LmEvalMetrics,
    device: Device,
    precision: Precision,
    loader_report: serde_json::Value,
    tokenizer_report: serde_json::Value,
}

fn write_eval_report(input: EvalReportInput<'_>) -> Result<()> {
    let json = serde_json::json!({
        "command": input.command,
        "model_family": input.model_family,
        "checkpoint": input.checkpoint.display().to_string(),
        "device": device_label(input.device),
        "precision": input.precision.as_str(),
        "dataset_manifest": input.dataset_manifest.map(|path| path.display().to_string()),
        "source": input.source,
        "split": input.split.as_str(),
        "batch_size": input.batch_size,
        "max_batches": input.max_batches,
        "loader": input.loader_report,
        "tokenizer": input.tokenizer_report,
        "metrics": input.metrics,
        "amp_bf16_policy": amp_bf16_policy(),
        "amp_bf16_op_decisions": amp_bf16_op_decisions(),
        "amp_bf16_finite_checks": amp::amp_bf16_finite_check_events(),
        "amp_bf16_validation": amp::amp_bf16_validation_report(&amp_bf16_policy()),
        "cuda_host_staging": amp_bf16_cuda_host_staging_events(),
    });
    write_json_file(input.path, json)
}

fn write_cuda_smoke_report(path: &Path, report: &cuda::CudaSmokeReport) -> Result<()> {
    let json = serde_json::json!({
        "command": "gpu smoke",
        "status": "passed",
        "device": {
            "ordinal": report.device.ordinal,
            "name": report.device.name,
            "pci_bus_id": report.device.pci_bus_id,
            "compute_capability": format!(
                "{}.{}",
                report.device.compute_capability_major,
                report.device.compute_capability_minor
            ),
        },
        "len": report.len,
        "kernels": ["heirloom_add_f32", "heirloom_relu_f32"],
        "add_max_abs_error": report.add_max_abs_error,
        "relu_max_abs_error": report.relu_max_abs_error,
        "note": "This validates real CUDA Driver API allocation/copy/kernel/copy-back execution in heirloom-kernels. It is not yet a full Tensor CUDA backend or GPU training path.",
    });
    write_json_file(path, json)
}

fn write_cuda_topology_report(
    path: &Path,
    info: &cuda::CudaSystemInfo,
    peers: &[cuda::CudaPeerAccess],
) -> Result<()> {
    let devices = info
        .devices
        .iter()
        .map(|device| {
            serde_json::json!({
                "ordinal": device.ordinal,
                "name": device.name,
                "pci_bus_id": device.pci_bus_id,
                "compute_capability": format!(
                    "{}.{}",
                    device.compute_capability_major,
                    device.compute_capability_minor
                ),
            })
        })
        .collect::<Vec<_>>();
    let peer_access = peers
        .iter()
        .map(|peer| {
            serde_json::json!({
                "from": peer.from_ordinal,
                "to": peer.to_ordinal,
                "can_access": peer.can_access,
            })
        })
        .collect::<Vec<_>>();
    let json = serde_json::json!({
        "command": "gpu topology",
        "status": "passed",
        "driver_loaded": info.driver_loaded,
        "device_count": info.device_count,
        "devices": devices,
        "peer_access": peer_access,
        "note": "This reports CUDA-visible devices and cuDeviceCanAccessPeer results. Combine it with nvidia-smi topo/nvlink artifacts to distinguish NVLink, PCIe, and network topology.",
    });
    write_json_file(path, json)
}

fn write_tensor_core_probe_report(
    path: &Path,
    report: &cuda::TensorCoreProbeReport,
    counters: cuda::TensorCoreCounters,
) -> Result<()> {
    let json = serde_json::json!({
        "command": "gpu tensor-core-probe",
        "status": "passed",
        "device": {
            "ordinal": report.device_ordinal,
            "bf16_tensor_cores_supported": true,
        },
        "kernel": "heirloom_bf16_mma_probe",
        "expected_dot": report.expected_dot,
        "max_abs_error": report.max_abs_error,
        "sample_count": report.samples.len(),
        "samples": report.samples,
        "tensor_core": {
            "bf16_mma_probe_calls": counters.bf16_mma_probe_calls,
            "bf16_tensor_core_matmul_calls": counters.bf16_tensor_core_matmul_calls,
            "bf16_tensor_core_matmul_forward_calls": counters.bf16_tensor_core_matmul_forward_calls,
            "bf16_tensor_core_matmul_backward_calls": counters.bf16_tensor_core_matmul_backward_calls,
            "bf16_tensor_core_attention_forward_calls": counters.bf16_tensor_core_attention_forward_calls,
            "bf16_tensor_core_attention_qk_matmul_calls": counters.bf16_tensor_core_attention_qk_matmul_calls,
            "bf16_tensor_core_attention_av_matmul_calls": counters.bf16_tensor_core_attention_av_matmul_calls,
            "bf16_tensor_core_attention_backward_calls": counters.bf16_tensor_core_attention_backward_calls,
            "bf16_tensor_core_attention_score_grad_matmul_calls": counters.bf16_tensor_core_attention_score_grad_matmul_calls,
            "bf16_tensor_core_attention_dq_matmul_calls": counters.bf16_tensor_core_attention_dq_matmul_calls,
            "bf16_tensor_core_attention_dk_matmul_calls": counters.bf16_tensor_core_attention_dk_matmul_calls,
            "bf16_tensor_core_attention_dv_matmul_calls": counters.bf16_tensor_core_attention_dv_matmul_calls,
            "bf16_scalar_matmul_fallback_calls": counters.bf16_scalar_matmul_fallback_calls,
        },
        "note": "This validates CUDA Driver API JIT/execution of an sm80 BF16 mma.sync probe. It is not a general GEMM benchmark or proof that all transformer matmuls use Tensor Cores.",
    });
    write_json_file(path, json)
}

fn write_json_file(path: &Path, value: serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            TensorError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&value).unwrap() + "\n")
        .map_err(|err| TensorError::Io(format!("failed to write {}: {err}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_hex(byte: u8) -> String {
        (0..128).map(|_| format!("{byte:02x}")).collect()
    }

    fn env_value(env: &[LauncherEnvVar], name: &str) -> Option<String> {
        env.iter()
            .find(|var| var.name == name)
            .map(|var| var.value.clone())
    }

    fn padawan_tmp_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("heirloom-padawan-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_test_json(path: &Path, value: serde_json::Value) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, serde_json::to_string_pretty(&value).unwrap() + "\n").unwrap();
    }

    fn write_learning_sanity_sweep_fixture(root: &Path, world_size: usize, include_lr: bool) {
        let configs = [(0.001, 1usize), (0.001, 2), (0.0005, 1), (0.0005, 2)];
        let mut runs = Vec::new();
        for (index, (learning_rate, grad_accumulation)) in configs.iter().enumerate() {
            let learning_rate = *learning_rate;
            let grad_accumulation = *grad_accumulation;
            let report_name = format!("sweep-run-{index}.json");
            let global_effective_batch_size = 2 * grad_accumulation * world_size;
            let mut report = serde_json::json!({
                "command": "train-lm",
                "model_family": "tiny_transformer",
                "distributed": "nccl",
                "precision": "amp-bf16",
                "world_size": world_size,
                "devices": (0..world_size).map(|device| format!("cuda:{device}")).collect::<Vec<_>>(),
                "per_rank_batch_size": 2,
                "global_batch_size": 2 * world_size,
                "grad_accumulation_steps": grad_accumulation,
                "per_rank_effective_batch_size": 2 * grad_accumulation,
                "global_micro_batch_size": 2 * world_size,
                "global_effective_batch_size": global_effective_batch_size,
                "initial_loss": 4.0,
                "final_loss": 3.0,
                "loss_reduction": 0.25,
                "rank_loss_reductions": vec![0.25; world_size],
                "all_reduce_calls": 8,
                "all_reduce_bytes": 4096,
                "loader": {
                    "kind": "binary_shard_streaming"
                },
                "performance": {
                    "tokens_seen": 1024,
                    "micro_batch_size": 2,
                    "grad_accumulation_steps": grad_accumulation,
                    "data_parallel_world_size": world_size,
                    "global_effective_batch_size": global_effective_batch_size
                }
            });
            if include_lr {
                report["learning_rate"] = serde_json::json!(learning_rate);
            }
            write_test_json(&root.join(&report_name), report);
            runs.push(serde_json::json!({
                "label": format!("lr-{learning_rate}-ga-{grad_accumulation}"),
                "report": report_name,
                "expected_learning_rate": learning_rate,
                "expected_grad_accumulation_steps": grad_accumulation
            }));
        }
        write_test_json(
            &root.join("lr-grad-sweep.json"),
            serde_json::json!({
                "format": LEARNING_SANITY_LR_GRAD_SWEEP_FORMAT,
                "version": 0,
                "runs": runs
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": LEARNING_SANITY_MANIFEST_FORMAT,
                "version": 0,
                "stages": [
                    {
                        "stage_id": "lr_grad_accumulation_sweep",
                        "report": "lr-grad-sweep.json",
                        "expected_command": "train-lm",
                        "expected_model_family": "tiny_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "min_loss_reduction": 0.2
                    }
                ]
            }),
        );
    }

    #[test]
    fn learning_sanity_manifest_validates_loss_improvement() {
        let root = padawan_tmp_root("learning-sanity-pass");
        write_test_json(
            &root.join("dense-report.json"),
            serde_json::json!({
                "command": "train-lm",
                "model_family": "tiny_transformer",
                "initial_loss": 4.0,
                "final_loss": 3.0,
                "loss_reduction": 0.25,
                "loader": {
                    "kind": "binary_shard_streaming"
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "dense_fixed_shard",
                        "report": "dense-report.json",
                        "expected_command": "train-lm",
                        "expected_model_family": "tiny_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "min_loss_reduction": 0.2
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
        assert_eq!(report["min_loss_reduction_observed"], 0.25);
    }

    #[test]
    fn learning_sanity_manifest_rejects_flat_loss() {
        let root = padawan_tmp_root("learning-sanity-flat");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 5.0,
                "loss_reduction": 0.0,
                "memory_config": {
                    "memory_lookup": "exact",
                    "smft_mode": "disabled"
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "memory_exact_smft_disabled",
                        "report": "memory-report.json",
                        "expected_command": "train-memory-lm",
                        "expected_model_family": "memory_transformer",
                        "expected_memory_lookup": "exact",
                        "expected_smft_mode": "disabled",
                        "min_loss_reduction": 0.001
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("did not improve loss"));
    }

    #[test]
    fn learning_sanity_memory_layers_disabled_rejects_active_indices() {
        let root = padawan_tmp_root("learning-sanity-memory-disabled");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.0,
                "loss_reduction": 0.2,
                "loader": {
                    "kind": "binary_shard_streaming"
                },
                "memory_config": {
                    "n_layers": 2,
                    "memory_layer_indices": [1],
                    "memory_lookup": "exact",
                    "smft_mode": "disabled"
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "memory_layers_disabled",
                        "report": "memory-report.json",
                        "expected_command": "train-memory-lm",
                        "expected_model_family": "memory_transformer",
                        "expected_loader_kind": "binary_shard_streaming"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("expected memory layers disabled"));
    }

    #[test]
    fn learning_sanity_smft_enabled_validates_artifact_evidence() {
        let root = padawan_tmp_root("learning-sanity-smft-enabled");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.9,
                "loss_reduction": 0.02,
                "loader": {
                    "kind": "binary_shard_streaming"
                },
                "memory_config": {
                    "memory_lookup": "exact",
                    "memory_update_policy": "sparse_rows",
                    "smft_mode": "masked_memory_rows"
                },
                "memory_optimizer": {
                    "sparse_optimizer_updates_selected_rows": true,
                    "sparse_update_parameter_count": 2,
                    "row_mask_attached_sparse_update_count": 2,
                    "sparse_update_selected_row_events": 64,
                    "smft_row_mask": {
                        "applied": true,
                        "trainable_rows": 8
                    }
                },
                "smft_artifacts": {
                    "accumulated_counts": {
                        "total_events": 512,
                        "unique_rows": 16
                    },
                    "generated_mask": {
                        "trainable_rows": 8
                    },
                    "online_refresh": {
                        "enabled": true,
                        "refresh_count": 4,
                        "active_mask": {
                            "trainable_rows": 8
                        }
                    }
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "memory_exact_smft_enabled",
                        "report": "memory-report.json",
                        "expected_command": "train-memory-lm",
                        "expected_model_family": "memory_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "expected_memory_lookup": "exact",
                        "expected_memory_update_policy": "sparse_rows",
                        "expected_smft_mode": "masked_memory_rows"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
    }

    #[test]
    fn learning_sanity_smft_enabled_rejects_missing_artifacts() {
        let root = padawan_tmp_root("learning-sanity-smft-missing");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.9,
                "loss_reduction": 0.02,
                "memory_config": {
                    "memory_lookup": "exact",
                    "memory_update_policy": "sparse_rows",
                    "smft_mode": "masked_memory_rows"
                },
                "memory_optimizer": {
                    "sparse_optimizer_updates_selected_rows": true,
                    "sparse_update_parameter_count": 2,
                    "row_mask_attached_sparse_update_count": 2,
                    "sparse_update_selected_row_events": 64,
                    "smft_row_mask": {
                        "applied": true,
                        "trainable_rows": 8
                    }
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "memory_exact_smft_enabled",
                        "report": "memory-report.json"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("requires smft_artifacts evidence"));
    }

    #[test]
    fn learning_sanity_product_key_parity_validates_lookup_evidence() {
        let root = padawan_tmp_root("learning-sanity-product-key");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.0,
                "loss_reduction": 0.2,
                "loader": {
                    "kind": "binary_shard_streaming"
                },
                "memory_config": {
                    "memory_key_dim": 8,
                    "memory_lookup": "product_key",
                    "memory_slots": 16,
                    "memory_update_policy": "full",
                    "smft_mode": "disabled"
                },
                "memory_optimizer": {
                    "memory_table_parameter_count": 3
                },
                "memory_selection": {
                    "captured_memory_layers": 1,
                    "configured_memory_layers": 1,
                    "memory_lookup": "product_key",
                    "selected_row_events": 64,
                    "unique_selected_rows": 7
                },
                "amp_bf16_op_decisions": [
                    {
                        "kernel_path": "cpu_product_key_candidate_topk_memory_lookup"
                    }
                ]
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "product_key_parity",
                        "report": "memory-report.json",
                        "expected_command": "train-memory-lm",
                        "expected_model_family": "memory_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "expected_memory_lookup": "product_key",
                        "expected_memory_update_policy": "full",
                        "expected_smft_mode": "disabled"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
    }

    #[test]
    fn learning_sanity_product_key_parity_rejects_non_square_slots() {
        let root = padawan_tmp_root("learning-sanity-product-key-shape");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.0,
                "loss_reduction": 0.2,
                "memory_config": {
                    "memory_key_dim": 8,
                    "memory_lookup": "product_key",
                    "memory_slots": 18,
                    "smft_mode": "disabled"
                },
                "memory_optimizer": {
                    "memory_table_parameter_count": 3
                },
                "memory_selection": {
                    "captured_memory_layers": 1,
                    "configured_memory_layers": 1,
                    "memory_lookup": "product_key",
                    "selected_row_events": 64,
                    "unique_selected_rows": 7
                },
                "amp_bf16_op_decisions": [
                    {
                        "kernel_path": "cpu_product_key_candidate_topk_memory_lookup"
                    }
                ]
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "product_key_parity",
                        "report": "memory-report.json"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("memory_slots must be square"));
    }

    #[test]
    fn learning_sanity_lr_grad_sweep_validates_ddp_matrix() {
        let root = padawan_tmp_root("learning-sanity-lr-grad-sweep");
        write_learning_sanity_sweep_fixture(&root, 4, true);
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
        assert_eq!(report["stages"][0]["loss_reduction"], 0.25);
        assert_eq!(report["stages"][0]["best_loss_reduction"], 0.25);
        assert_eq!(report["stages"][0]["recommended_learning_rate"], 0.001);
        assert_eq!(
            report["stages"][0]["recommended_grad_accumulation_steps"],
            1
        );
        assert_eq!(report["stages"][0]["best_run"]["label"], "lr-0.001-ga-1");
        assert_eq!(report["stages"][0]["runs"].as_array().unwrap().len(), 4);
        assert_eq!(
            report["stages"][0]["min_observed_data_parallel_world_size"],
            4
        );
    }

    #[test]
    fn learning_sanity_lr_grad_sweep_rejects_weak_best_run() {
        let root = padawan_tmp_root("learning-sanity-lr-grad-weak-best");
        write_learning_sanity_sweep_fixture(&root, 4, true);
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": LEARNING_SANITY_MANIFEST_FORMAT,
                "version": 0,
                "stages": [
                    {
                        "stage_id": "lr_grad_accumulation_sweep",
                        "report": "lr-grad-sweep.json",
                        "expected_command": "train-lm",
                        "expected_model_family": "tiny_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "min_loss_reduction": 0.2,
                        "min_best_loss_reduction": 0.3
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("best loss_reduction 0.25 < required 0.3"));
    }

    #[test]
    fn learning_sanity_lr_grad_sweep_rejects_missing_learning_rate() {
        let root = padawan_tmp_root("learning-sanity-lr-grad-missing-lr");
        write_learning_sanity_sweep_fixture(&root, 4, false);
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("learning_rate must be numeric"));
    }

    #[test]
    fn learning_sanity_lr_grad_sweep_rejects_underpowered_world_size() {
        let root = padawan_tmp_root("learning-sanity-lr-grad-small-world");
        write_learning_sanity_sweep_fixture(&root, 2, true);
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("data_parallel_world_size 2 < required 4"));
    }

    #[test]
    fn learning_sanity_longer_32k_blend_validates_hardpath_bundle() {
        let report = validate_learning_sanity_manifest(Path::new(
            "tests/fixtures/learning_sanity/longer-32k-blend-valid.json",
        ))
        .unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
        assert_eq!(report["stages"][0]["tokenizer_vocab_size"], 32768);
        assert!((report["stages"][0]["loss_reduction"].as_f64().unwrap() - 0.16).abs() < 1.0e-12);
    }

    #[test]
    fn learning_sanity_longer_32k_blend_rejects_small_tokenizer() {
        let root = padawan_tmp_root("learning-sanity-32k-small-tokenizer");
        write_test_json(
            &root.join("summary.json"),
            serde_json::json!({
                "status": "passed",
                "tokenizer_version": 2,
                "tokenizer_vocab_size": 512,
                "reserved_tokens": 128,
                "manifest_version": 2,
                "manifest_storage": "binary_shards",
                "loader_kind": "binary_shard_streaming",
                "tokens_materialized": false,
                "target_tokens": 4096,
                "selected_tokens": 4096,
                "selected_docs": 8,
                "artifacts": {}
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "longer_32k_blend",
                        "report": "summary.json"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("tokenizer_vocab_size expected 32768"));
    }

    #[test]
    fn padawan_sha256_matches_known_vector() {
        assert_eq!(
            sha256_prefixed_hex(b"abc"),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn padawan_fixture_validates_and_verifies_in_rust() {
        let episodes = vec![PathBuf::from("padawan/fixtures/episode_valid.jsonl")];
        let root = Path::new("padawan/fixtures");
        let validation = padawan_validate_report(&episodes, Some(root)).unwrap();
        assert_eq!(validation["status"], "passed");
        assert_eq!(validation["episodes"], 1);
        assert_eq!(validation["sft_eligible"], 1);
        assert_eq!(validation["smft_eligible"], 1);

        let verification = padawan_verify_report(&episodes, Some(root), &[], &[]).unwrap();
        assert_eq!(verification["status"], "passed");
        assert_eq!(verification["family_counts"]["passed"], 4);
        assert_eq!(verification["family_counts"]["failed"], 0);
    }

    #[test]
    fn padawan_code_patch_verifier_rejects_non_sidecar_paths() {
        let root = padawan_tmp_root("bad-patch");
        std::fs::create_dir_all(root.join("artifacts")).unwrap();
        std::fs::write(
            root.join("artifacts/bad.diff"),
            "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@\n+bad\n",
        )
        .unwrap();
        let episode = serde_json::json!({
            "padawan": {
                "artifact_ref": "artifact://padawan/artifacts/bad.diff"
            },
            "verifier": {
                "unrelated_changes": 0
            }
        });
        let err = padawan_verify_code_patch(&episode, &root, &["padawan/".to_string()])
            .expect_err("patch verifier should reject non-sidecar paths");
        assert!(err
            .to_string()
            .contains("outside Padawan sidecar allowlist"));
    }

    #[test]
    fn padawan_json_tool_call_verifier_rejects_inline_results() {
        let root = padawan_tmp_root("bad-tool-call");
        write_test_json(
            &root.join("traces/trace.json"),
            serde_json::json!({
                "tool_calls": [
                    {
                        "tool": "shell",
                        "arguments": {},
                        "result": "inline result should live in observations"
                    }
                ]
            }),
        );
        write_test_json(
            &root.join("artifacts/bundle.json"),
            serde_json::json!({
                "artifacts": [
                    {
                        "name": "trace",
                        "ref": "artifact://padawan/traces/trace.json",
                        "media_type": "application/json"
                    }
                ]
            }),
        );
        let episode = serde_json::json!({
            "padawan": {
                "trace_ref": "artifact://padawan/traces/trace.json",
                "artifact_bundle_ref": "artifact://padawan/artifacts/bundle.json"
            }
        });
        let err = padawan_verify_json_tool_call(&episode, &root)
            .expect_err("tool-call verifier should reject inline results");
        assert!(err.to_string().contains("should not inline tool results"));
    }

    #[test]
    fn padawan_evidence_verifier_rejects_missing_observations() {
        let root = padawan_tmp_root("bad-evidence");
        write_test_json(
            &root.join("traces/trace.json"),
            serde_json::json!({
                "observations": [],
                "final_validation": {
                    "validator": "heirloom padawan verify",
                    "expected": "passes"
                }
            }),
        );
        std::fs::create_dir_all(root.join("finals")).unwrap();
        std::fs::write(root.join("finals/final.txt"), "done\n").unwrap();
        let episode = serde_json::json!({
            "padawan": {
                "trace_ref": "artifact://padawan/traces/trace.json",
                "final_response_ref": "artifact://padawan/finals/final.txt"
            },
            "verifier": {
                "failure_class": null
            }
        });
        let err = padawan_verify_evidence_citation(&episode, &root)
            .expect_err("evidence verifier should require observations");
        assert!(err.to_string().contains("at least one observation"));
    }

    #[test]
    fn padawan_memory_smft_verifier_rejects_row_budget_overflow() {
        let root = padawan_tmp_root("bad-smft");
        let rows = (0..33).collect::<Vec<usize>>();
        write_test_json(
            &root.join("memory/selection.json"),
            serde_json::json!({
                "format": "heirloom.padawan.memory_selection",
                "memory_slots": 1024,
                "layers": {
                    "layer_8": {
                        "selected_rows": rows,
                        "selected_row_events": 33
                    }
                }
            }),
        );
        write_test_json(
            &root.join("memory/counts.json"),
            serde_json::json!({
                "format": "heirloom.padawan.smft_access_counts",
                "memory_slots": 1024,
                "foreground_rows": rows,
                "background_rows": [99, 100],
                "foreground_background_lift": 4.5,
                "replay_jaccard_overlap": 0.34,
                "per_episode_row_cap": 32,
                "task_family_row_cap": 128
            }),
        );
        let episode = serde_json::json!({
            "selection": {
                "eligible_for_smft": true
            },
            "memory": {
                "selection_report_ref": "artifact://padawan/memory/selection.json",
                "smft_access_counts_ref": "artifact://padawan/memory/counts.json"
            }
        });
        let err = padawan_verify_memory_smft(&episode, &root)
            .expect_err("SMFT verifier should reject row budget overflow");
        assert!(err
            .to_string()
            .contains("selected rows exceed per-episode cap"));
    }

    #[test]
    fn tensor_core_flash_microbench_pass_requires_ok_and_passed_when_requested() {
        let error_report = serde_json::json!({
            "status": "error",
            "passed": false,
        });
        assert!(tensor_core_flash_microbench_passed(&error_report, false));
        assert!(!tensor_core_flash_microbench_passed(&error_report, true));

        let failed_report = serde_json::json!({
            "status": "ok",
            "passed": false,
        });
        assert!(!tensor_core_flash_microbench_passed(&failed_report, true));

        let passed_report = serde_json::json!({
            "status": "ok",
            "passed": true,
        });
        assert!(tensor_core_flash_microbench_passed(&passed_report, true));
    }

    #[test]
    fn flash_tensor_core_forward_mma_flops_count_head_dim_chunks() {
        let head_dim_16 = flash_tensor_core_forward_mma_flops(1, 1, 16, 16);
        let head_dim_64 = flash_tensor_core_forward_mma_flops(1, 1, 16, 64);

        assert_eq!(head_dim_16, 24_576.0);
        assert_eq!(head_dim_64, 294_912.0);
    }

    #[test]
    fn grad_accumulation_helpers_preserve_optimizer_step_semantics() {
        assert!(validate_grad_accumulation_steps(0).is_err());
        validate_grad_accumulation_steps(1).unwrap();
        assert_eq!(accumulated_training_tokens_seen(3, 2, 5, 4).unwrap(), 120);
        assert_eq!(ddp_sample_step(10, 2, 3, 4).unwrap(), 51);
    }

    #[test]
    fn aggregate_rank_performance_preserves_timing_buckets() {
        let rank_reports = vec![
            serde_json::json!({
                "performance": {
                    "tokens_seen": 100,
                    "train_elapsed_ms": 10,
                    "dataloader_elapsed_ms": 1,
                    "host_to_device_elapsed_ms": 2,
                    "forward_backward_elapsed_ms": 3,
                    "all_reduce_elapsed_ms": 4,
                    "optimizer_elapsed_ms": 5,
                    "host_to_device_cuda_elapsed_ms": 0.2,
                    "forward_backward_cuda_elapsed_ms": 1.5,
                    "all_reduce_cuda_elapsed_ms": 0.7,
                    "optimizer_cuda_elapsed_ms": 0.3,
                    "active_dense_flops_per_token_estimate": 1000.0
                }
            }),
            serde_json::json!({
                "performance": {
                    "tokens_seen": 200,
                    "train_elapsed_ms": 12,
                    "dataloader_elapsed_ms": 2,
                    "host_to_device_elapsed_ms": 1,
                    "forward_backward_elapsed_ms": 4,
                    "all_reduce_elapsed_ms": 6,
                    "optimizer_elapsed_ms": 3,
                    "host_to_device_cuda_elapsed_ms": 0.1,
                    "forward_backward_cuda_elapsed_ms": 2.5,
                    "all_reduce_cuda_elapsed_ms": 0.5,
                    "optimizer_cuda_elapsed_ms": 0.4,
                    "active_dense_flops_per_token_estimate": 1000.0
                }
            }),
        ];

        let report = aggregate_rank_performance(&rank_reports);

        assert_eq!(report["tokens_seen"], 300);
        assert_eq!(report["train_elapsed_ms"], 12);
        assert_eq!(report["dataloader_elapsed_ms"], 2);
        assert_eq!(report["host_to_device_elapsed_ms"], 2);
        assert_eq!(report["forward_backward_elapsed_ms"], 4);
        assert_eq!(report["all_reduce_elapsed_ms"], 6);
        assert_eq!(report["optimizer_elapsed_ms"], 5);
        assert_eq!(report["host_to_device_host_elapsed_ms"], 2);
        assert_eq!(report["forward_backward_host_elapsed_ms"], 4);
        assert_eq!(report["host_to_device_cuda_elapsed_ms"], 0.2);
        assert_eq!(report["forward_backward_cuda_elapsed_ms"], 2.5);
        assert_eq!(report["all_reduce_cuda_elapsed_ms"], 0.7);
        assert_eq!(report["optimizer_cuda_elapsed_ms"], 0.4);
        assert_eq!(report["cuda_event_timing_available"], true);
        assert_eq!(
            report["mfu_timing_source"]["dense_core_mfu_estimate"],
            "cuda_event_forward_backward_elapsed_ms"
        );
        assert!(report["tokens_per_second"].as_f64().unwrap() > 0.0);
        assert!(report["dense_core_mfu_estimate"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn aggregate_rank_cuda_runtime_sums_flash_backward_fields() {
        let rank_reports = vec![
            serde_json::json!({
                "cuda_runtime": {
                    "kernel_launch_family_elapsed_us": 800,
                    "kernel_launch_families": {
                        "rank2_elementwise": {"calls": 3, "elements": 30, "elapsed_us": 300},
                        "tensor_core_gemm_cp_async": {"calls": 5, "elements": 50, "elapsed_us": 500}
                    },
                    "flash_bf16_tensor_core_backward_requested_calls": 1,
                    "flash_bf16_tensor_core_backward_executed_calls": 2,
                    "flash_bf16_tensor_core_backward_fallback_calls": 3,
                    "flash_bf16_tensor_core_backward_row_dot_calls": 4,
                    "flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls": 5,
                    "flash_bf16_tensor_core_backward_dp_mma_tile_calls": 6,
                    "flash_bf16_tensor_core_backward_dq_mma_tile_calls": 7,
                    "flash_bf16_tensor_core_backward_dk_mma_tile_calls": 8,
                    "flash_bf16_tensor_core_backward_dv_mma_tile_calls": 9,
                    "flash_bf16_tensor_core_backward_scalar_tile_calls": 10,
                    "flash_bf16_tensor_core_backward_ragged_tile_count": 11,
                    "flash_bf16_tensor_core_backward_causal_masked_tile_count": 12,
                    "flash_bf16_tensor_core_backward_elapsed_us": 13
                }
            }),
            serde_json::json!({
                "cuda_runtime": {
                    "kernel_launch_family_elapsed_us": 1800,
                    "kernel_launch_families": {
                        "rank2_elementwise": {"calls": 7, "elements": 70, "elapsed_us": 700},
                        "flash_attention_bf16_tensor_core_backward": {"calls": 11, "elements": 110, "elapsed_us": 1100}
                    },
                    "flash_bf16_tensor_core_backward_requested_calls": 10,
                    "flash_bf16_tensor_core_backward_executed_calls": 20,
                    "flash_bf16_tensor_core_backward_fallback_calls": 30,
                    "flash_bf16_tensor_core_backward_row_dot_calls": 40,
                    "flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls": 50,
                    "flash_bf16_tensor_core_backward_dp_mma_tile_calls": 60,
                    "flash_bf16_tensor_core_backward_dq_mma_tile_calls": 70,
                    "flash_bf16_tensor_core_backward_dk_mma_tile_calls": 80,
                    "flash_bf16_tensor_core_backward_dv_mma_tile_calls": 90,
                    "flash_bf16_tensor_core_backward_scalar_tile_calls": 100,
                    "flash_bf16_tensor_core_backward_ragged_tile_count": 110,
                    "flash_bf16_tensor_core_backward_causal_masked_tile_count": 120,
                    "flash_bf16_tensor_core_backward_elapsed_us": 130
                }
            }),
        ];

        let report = aggregate_rank_cuda_runtime(&rank_reports);

        assert_eq!(
            report["kernel_launch_families"]["rank2_elementwise"]["calls"],
            10
        );
        assert_eq!(
            report["kernel_launch_families"]["rank2_elementwise"]["elements"],
            100
        );
        assert_eq!(
            report["kernel_launch_families"]["rank2_elementwise"]["elapsed_us"],
            1000
        );
        assert_eq!(
            report["kernel_launch_families"]["tensor_core_gemm_cp_async"]["calls"],
            5
        );
        assert_eq!(
            report["kernel_launch_families"]["flash_attention_bf16_tensor_core_backward"]["calls"],
            11
        );
        assert_eq!(report["kernel_launch_family_elapsed_us"], 2600);
        assert_eq!(
            report["flash_bf16_tensor_core_backward_requested_calls"],
            11
        );
        assert_eq!(report["flash_bf16_tensor_core_backward_executed_calls"], 22);
        assert_eq!(report["flash_bf16_tensor_core_backward_fallback_calls"], 33);
        assert_eq!(report["flash_bf16_tensor_core_backward_row_dot_calls"], 44);
        assert_eq!(
            report["flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls"],
            55
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_dp_mma_tile_calls"],
            66
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_dq_mma_tile_calls"],
            77
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_dk_mma_tile_calls"],
            88
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_dv_mma_tile_calls"],
            99
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_scalar_tile_calls"],
            110
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_ragged_tile_count"],
            121
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_causal_masked_tile_count"],
            132
        );
        assert_eq!(report["flash_bf16_tensor_core_backward_elapsed_us"], 143);
    }

    #[test]
    fn tensor_core_pad_crop_report_names_ragged_linear_evidence() {
        let cuda_runtime = serde_json::json!({
            "tensor_core_padded_tiles": 17,
            "tensor_core_remainder_tiles": 5,
        });
        let tensor_core = serde_json::json!({
            "bf16_scalar_matmul_fallback_calls": 0,
        });
        let coverage = serde_json::json!({
            "linear_totals": {
                "tensor_core_calls": 4,
                "fallback_calls": 0,
            },
            "linear_modules": [
                {
                    "module": "lm_head",
                    "calls": 4,
                    "tensor_core_calls": 4,
                    "fallback_calls": 0,
                    "last_path": "tensor_core",
                    "last_m": 15,
                    "last_k": 18,
                    "last_n": 281,
                }
            ],
        });

        let report =
            tensor_core_pad_crop_report_from_json(&cuda_runtime, &tensor_core, &coverage, "test");

        assert_eq!(report["used"], true);
        assert_eq!(report["passed"], true);
        assert_eq!(report["status"], "passed");
        assert_eq!(report["padded_tiles"], 17);
        assert_eq!(report["remainder_tiles"], 5);
        assert_eq!(report["scalar_fallbacks"], 0);
        assert_eq!(report["linear_fallbacks"], 0);
        assert_eq!(report["linear_modules"][0]["logical_shape"]["m"], 15);
        assert_eq!(report["linear_modules"][0]["padded_shape"]["m"], 16);
        assert_eq!(report["linear_modules"][0]["padded_shape"]["k"], 32);
        assert_eq!(report["linear_modules"][0]["padded_shape"]["n"], 288);
        assert_eq!(report["linear_modules"][0]["needs_padding"], true);
    }

    #[test]
    fn tensor_core_pad_crop_report_distinguishes_tile_aligned_shapes() {
        let cuda_runtime = serde_json::json!({
            "tensor_core_padded_tiles": 0,
            "tensor_core_remainder_tiles": 0,
        });
        let tensor_core = serde_json::json!({
            "bf16_scalar_matmul_fallback_calls": 0,
        });
        let coverage = serde_json::json!({
            "linear_totals": {
                "tensor_core_calls": 1,
                "fallback_calls": 0,
            },
            "linear_modules": [
                {
                    "module": "linear",
                    "calls": 1,
                    "tensor_core_calls": 1,
                    "fallback_calls": 0,
                    "last_path": "tensor_core",
                    "last_m": 16,
                    "last_k": 16,
                    "last_n": 8,
                }
            ],
        });

        let report =
            tensor_core_pad_crop_report_from_json(&cuda_runtime, &tensor_core, &coverage, "test");

        assert_eq!(report["used"], false);
        assert_eq!(report["passed"], false);
        assert_eq!(report["status"], "not_used");
        assert_eq!(report["linear_modules"][0]["needs_padding"], false);
    }

    #[test]
    fn cuda_runtime_counters_json_includes_flash_attention_fields() {
        let json = cuda_runtime_counters_json(cuda::CudaRuntimeCounters::default());

        assert_eq!(json["tensor_core_ldmatrix_gemm_executed_calls"], 0);
        assert_eq!(json["tensor_core_ldmatrix_gemm_hard_require_failures"], 0);
        assert_eq!(json["tensor_core_ldmatrix_gemm_instructions"], 0);
        assert_eq!(json["tensor_core_ldmatrix_gemm_elapsed_us"], 0);
        assert_eq!(json["tensor_core_cp_async_gemm_executed_calls"], 0);
        assert_eq!(json["tensor_core_cp_async_gemm_hard_require_failures"], 0);
        assert_eq!(json["tensor_core_cp_async_gemm_instructions"], 0);
        assert_eq!(json["tensor_core_cp_async_gemm_elapsed_us"], 0);
        assert_eq!(json["kernel_launch_family_elapsed_us"], 0);
        assert!(json["kernel_launch_families"]
            .as_object()
            .expect("kernel launch families is an object")
            .is_empty());
        assert_eq!(json["tensor_core_staged_cta_gemm_elapsed_us"], 0);
        assert_eq!(json["tensor_core_wide_swizzled_cta_gemm_elapsed_us"], 0);
        assert_eq!(json["tensor_core_global_cta_gemm_elapsed_us"], 0);
        assert_eq!(json["tensor_core_legacy_warp_gemm_elapsed_us"], 0);
        assert_eq!(json["bf16_attention_materialized_reference_calls"], 0);
        assert_eq!(json["flash_bf16_attention_requested_calls"], 0);
        assert_eq!(json["flash_bf16_attention_executed_calls"], 0);
        assert_eq!(json["flash_bf16_attention_fallback_calls"], 0);
        assert_eq!(json["flash_bf16_attention_scalar_fallback_calls"], 0);
        assert_eq!(json["flash_bf16_attention_qk_tile_calls"], 0);
        assert_eq!(json["flash_bf16_attention_av_tile_calls"], 0);
        assert_eq!(json["flash_bf16_attention_ragged_tile_count"], 0);
        assert_eq!(json["flash_bf16_attention_causal_masked_tile_count"], 0);
        assert_eq!(json["flash_bf16_attention_elapsed_us"], 0);
        assert_eq!(json["flash_bf16_attention_hard_require_failures"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_requested_calls"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_executed_calls"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_qk_tile_calls"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_av_tile_calls"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_elapsed_us"], 0);
        assert_eq!(json["flash_bf16_tensor_core_requested_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_executed_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_fallback_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_qk_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_av_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_ragged_tile_count"], 0);
        assert_eq!(json["flash_bf16_tensor_core_causal_masked_tile_count"], 0);
        assert_eq!(json["flash_bf16_tensor_core_elapsed_us"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_requested_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_executed_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_fallback_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_row_dot_calls"], 0);
        assert_eq!(
            json["flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls"],
            0
        );
        assert_eq!(json["flash_bf16_tensor_core_backward_dp_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_dq_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_dk_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_dv_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_scalar_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_ragged_tile_count"], 0);
        assert_eq!(
            json["flash_bf16_tensor_core_backward_causal_masked_tile_count"],
            0
        );
        assert_eq!(json["flash_bf16_tensor_core_backward_elapsed_us"], 0);
    }

    #[test]
    fn memory_optimizer_report_records_supplied_smft_row_mask() {
        let mut config = MemoryTransformerConfig::tiny(32);
        config.block_size = 4;
        config.n_layers = 2;
        config.d_model = 8;
        config.n_heads = 2;
        config.ff_hidden = 16;
        config.memory_layer_indices = vec![1];
        config.memory_slots = 8;
        config.memory_key_dim = 4;
        config.memory_value_dim = 8;
        config.memory_top_k = 2;
        config.memory_update_policy = MemoryUpdatePolicy::SparseRows;
        config.smft_mode = SmftMode::MaskedMemoryRows;
        let mut rng = HeirloomRng::new(515);
        let model = MemoryTransformerLm::new(config, &mut rng).unwrap();
        let input = Tensor::from_i64(vec![1, 2, 3, 4], &[1, 4], false).unwrap();
        let _ = model.forward(&input).unwrap();
        let mask = SmftRowMask {
            memory_slots: 8,
            trainable_rows: vec![1, 3],
            frozen_rows: 6,
            trainable_fraction: 0.25,
            scores: Vec::new(),
        };

        let report = memory_optimizer_report(&model, Some(&mask), Some("mask.json")).unwrap();

        assert_eq!(
            report["applied_path"],
            "cpu_sparse_rows_selected_memory_tables"
        );
        assert_eq!(report["sparse_update_parameter_count"], 2);
        assert_eq!(report["row_mask_attached_sparse_update_count"], 2);
        assert_eq!(
            report["sparse_updates_accumulate_dense_gradient_buffers"],
            true
        );
        assert_eq!(report["sparse_optimizer_updates_selected_rows"], true);
        assert_eq!(
            report["sparse_optimizer_gathers_compact_gradient_rows"],
            false
        );
        assert_eq!(report["compressed_sparse_gradient_transport"], false);
        assert_eq!(report["smft_row_mask"]["source"], "mask.json");
        assert_eq!(report["smft_row_mask"]["memory_slots"], 8);
        assert_eq!(report["smft_row_mask"]["trainable_rows"], 2);
        assert_eq!(report["smft_row_mask"]["frozen_rows"], 6);
    }

    #[test]
    fn memory_optimizer_report_records_product_key_smft_projection() {
        let mut config = MemoryTransformerConfig::tiny(32);
        config.block_size = 4;
        config.n_layers = 2;
        config.d_model = 8;
        config.n_heads = 2;
        config.ff_hidden = 16;
        config.memory_layer_indices = vec![1];
        config.memory_slots = 4;
        config.memory_key_dim = 4;
        config.memory_value_dim = 8;
        config.memory_top_k = 2;
        config.memory_lookup = MemoryLookupKind::ProductKey;
        config.memory_update_policy = MemoryUpdatePolicy::SparseRows;
        config.smft_mode = SmftMode::MaskedMemoryRows;
        let mut rng = HeirloomRng::new(616);
        let model = MemoryTransformerLm::new(config, &mut rng).unwrap();
        let input = Tensor::from_i64(vec![1, 2, 3, 4], &[1, 4], false).unwrap();
        let _ = model.forward(&input).unwrap();
        let mask = SmftRowMask {
            memory_slots: 4,
            trainable_rows: vec![0, 1],
            frozen_rows: 2,
            trainable_fraction: 0.5,
            scores: Vec::new(),
        };

        let report = memory_optimizer_report(&model, Some(&mask), Some("mask.json")).unwrap();

        assert_eq!(report["sparse_update_parameter_count"], 3);
        assert_eq!(report["row_mask_attached_sparse_update_count"], 3);
        let projection = &report["smft_row_mask"]["product_key_projection"];
        assert_eq!(projection["policy"], "conservative_all_slots");
        assert_eq!(projection["side"], 2);
        assert_eq!(projection["value_trainable_rows"], 2);
        assert_eq!(projection["left_trainable_rows"], 1);
        assert_eq!(projection["right_trainable_rows"], 0);
        assert_eq!(projection["half_key_rows_are_conservative"], true);
    }

    #[test]
    fn distributed_memory_evidence_requires_dense_counters_for_full_updates() {
        let err =
            validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
                memory_update_policy: &MemoryUpdatePolicy::Full,
                all_reduce_calls: 0,
                all_reduce_bytes: 0,
                row_union_all_reduce_calls: 0,
                row_union_all_reduce_bytes: 0,
                row_union_candidate_rows: 0,
                compact_gradient_all_reduce_calls: 0,
                compact_gradient_all_reduce_bytes: 0,
            })
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("without recorded gradient all-reduces"),
            "{err}"
        );

        validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
            memory_update_policy: &MemoryUpdatePolicy::Full,
            all_reduce_calls: 2,
            all_reduce_bytes: 128,
            row_union_all_reduce_calls: 0,
            row_union_all_reduce_bytes: 0,
            row_union_candidate_rows: 0,
            compact_gradient_all_reduce_calls: 0,
            compact_gradient_all_reduce_bytes: 0,
        })
        .unwrap();
    }

    #[test]
    fn distributed_memory_sparse_evidence_requires_compact_gradient_transport() {
        let err =
            validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
                memory_update_policy: &MemoryUpdatePolicy::SparseRows,
                all_reduce_calls: 4,
                all_reduce_bytes: 256,
                row_union_all_reduce_calls: 2,
                row_union_all_reduce_bytes: 64,
                row_union_candidate_rows: 8,
                compact_gradient_all_reduce_calls: 0,
                compact_gradient_all_reduce_bytes: 0,
            })
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("without recorded compact sparse all-reduces"),
            "{err}"
        );

        validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
            memory_update_policy: &MemoryUpdatePolicy::SparseRows,
            all_reduce_calls: 4,
            all_reduce_bytes: 256,
            row_union_all_reduce_calls: 2,
            row_union_all_reduce_bytes: 64,
            row_union_candidate_rows: 8,
            compact_gradient_all_reduce_calls: 2,
            compact_gradient_all_reduce_bytes: 512,
        })
        .unwrap();
    }

    #[test]
    fn parses_marked_nccl_unique_id_from_noisy_stdout() {
        let hex = valid_hex(0xab);
        let stdout =
            format!("NCCL INFO graph line\n{NCCL_UNIQUE_ID_HELPER_MARKER}{hex}\nNCCL INFO done\n");

        assert_eq!(parse_nccl_unique_id_helper_stdout(&stdout).unwrap(), hex);
    }

    #[test]
    fn parses_exact_hex_nccl_unique_id_for_backwards_compatibility() {
        let hex = valid_hex(0x17);

        assert_eq!(parse_nccl_unique_id_helper_stdout(&hex).unwrap(), hex);
    }

    #[test]
    fn rejects_missing_nccl_unique_id_marker_or_exact_hex_line() {
        let err = parse_nccl_unique_id_helper_stdout("NCCL INFO only\n").unwrap_err();

        assert!(format!("{err}").contains("did not contain"));
    }

    #[test]
    fn rejects_invalid_marked_nccl_unique_id_payload() {
        let stdout = format!("{NCCL_UNIQUE_ID_HELPER_MARKER}abc\n");
        let err = parse_nccl_unique_id_helper_stdout(&stdout).unwrap_err();

        assert!(format!("{err}").contains("invalid id"));
    }

    #[test]
    fn rejects_multiple_nccl_unique_id_candidates() {
        let first = valid_hex(0x01);
        let second = valid_hex(0x02);
        let stdout = format!(
            "{NCCL_UNIQUE_ID_HELPER_MARKER}{first}\n{NCCL_UNIQUE_ID_HELPER_MARKER}{second}"
        );
        let err = parse_nccl_unique_id_helper_stdout(&stdout).unwrap_err();

        assert!(format!("{err}").contains("multiple candidate ids"));
    }

    #[test]
    fn nccl_launcher_env_defaults_to_single_node_loopback_bootstrap() {
        let env = launcher_env_from_lookup(|_| None);

        assert_eq!(env_value(&env, "NCCL_DEBUG").as_deref(), Some("INFO"));
        assert_eq!(
            env_value(&env, "NCCL_DEBUG_SUBSYS").as_deref(),
            Some("INIT,COLL,GRAPH")
        );
        assert_eq!(env_value(&env, "NCCL_NET_PLUGIN").as_deref(), Some("none"));
        assert_eq!(env_value(&env, "NCCL_SOCKET_IFNAME").as_deref(), Some("lo"));
        assert_eq!(env_value(&env, "NCCL_IB_DISABLE").as_deref(), Some("1"));
        assert!(env_value(&env, "NCCL_CUMEM_ENABLE").is_none());
        assert!(env_value(&env, "NCCL_CUMEM_HOST_ENABLE").is_none());
        assert!(env_value(&env, "NCCL_P2P_DISABLE").is_none());
        assert!(env_value(&env, "NCCL_P2P_LEVEL").is_none());
        assert!(env_value(&env, "HEIRLOOM_NCCL_TRACE").is_none());
    }

    #[test]
    fn nccl_launcher_env_ignores_inherited_vertex_bootstrap_overrides() {
        let env = launcher_env_from_lookup(|name| match name {
            "NCCL_SOCKET_IFNAME" => Some("^cbr,veth,docker,lo,cali,gke,node,cilium".to_string()),
            "NCCL_IB_DISABLE" => Some("0".to_string()),
            "NCCL_NET_PLUGIN" => Some("FastSocket".to_string()),
            "HEIRLOOM_NCCL_TRACE" => Some("1".to_string()),
            _ => None,
        });

        assert_eq!(env_value(&env, "NCCL_SOCKET_IFNAME").as_deref(), Some("lo"));
        assert_eq!(env_value(&env, "NCCL_IB_DISABLE").as_deref(), Some("1"));
        assert_eq!(env_value(&env, "NCCL_NET_PLUGIN").as_deref(), Some("none"));
        assert_eq!(env_value(&env, "HEIRLOOM_NCCL_TRACE").as_deref(), Some("1"));
    }

    #[test]
    fn nccl_launcher_env_preserves_heirloom_specific_overrides() {
        let env = launcher_env_from_lookup(|name| match name {
            "HEIRLOOM_NCCL_SOCKET_IFNAME" => Some("eth0".to_string()),
            "HEIRLOOM_NCCL_IB_DISABLE" => Some("0".to_string()),
            "HEIRLOOM_NCCL_NET_PLUGIN" => Some("none".to_string()),
            "HEIRLOOM_NCCL_CUMEM_ENABLE" => Some("0".to_string()),
            "HEIRLOOM_NCCL_CUMEM_HOST_ENABLE" => Some("0".to_string()),
            "HEIRLOOM_NCCL_P2P_DISABLE" => Some("1".to_string()),
            "HEIRLOOM_NCCL_P2P_LEVEL" => Some("NVL".to_string()),
            "HEIRLOOM_NCCL_TRACE" => Some("1".to_string()),
            _ => None,
        });

        assert_eq!(
            env_value(&env, "NCCL_SOCKET_IFNAME").as_deref(),
            Some("eth0")
        );
        assert_eq!(env_value(&env, "NCCL_IB_DISABLE").as_deref(), Some("0"));
        assert_eq!(env_value(&env, "NCCL_NET_PLUGIN").as_deref(), Some("none"));
        assert_eq!(env_value(&env, "NCCL_CUMEM_ENABLE").as_deref(), Some("0"));
        assert_eq!(
            env_value(&env, "NCCL_CUMEM_HOST_ENABLE").as_deref(),
            Some("0")
        );
        assert_eq!(env_value(&env, "NCCL_P2P_DISABLE").as_deref(), Some("1"));
        assert_eq!(env_value(&env, "NCCL_P2P_LEVEL").as_deref(), Some("NVL"));
        assert_eq!(env_value(&env, "HEIRLOOM_NCCL_TRACE").as_deref(), Some("1"));
    }

    #[test]
    fn ddp_step_checksum_drifts_reports_synced_ranks() {
        let reports = vec![
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0},
                    {"step": 2, "parameter_checksum_sum": 11.0, "parameter_checksum_sumsq": 31.0}
                ]
            }),
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0},
                    {"step": 2, "parameter_checksum_sum": 11.0, "parameter_checksum_sumsq": 31.0}
                ]
            }),
        ];

        let drifts = ddp_step_checksum_drifts(&reports).unwrap();

        assert_eq!(drifts.len(), 2);
        assert_eq!(
            max_step_checksum_error(&drifts, "parameter_checksum_sum_max_error"),
            0.0
        );
        assert_eq!(
            max_step_checksum_error(&drifts, "parameter_checksum_sumsq_max_error"),
            0.0
        );
    }

    #[test]
    fn ddp_step_checksum_drifts_detects_rank_drift() {
        let reports = vec![
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0}
                ]
            }),
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.25, "parameter_checksum_sumsq": 30.5}
                ]
            }),
        ];

        let drifts = ddp_step_checksum_drifts(&reports).unwrap();

        assert_eq!(
            max_step_checksum_error(&drifts, "parameter_checksum_sum_max_error"),
            0.25
        );
        assert_eq!(
            max_step_checksum_error(&drifts, "parameter_checksum_sumsq_max_error"),
            0.5
        );
    }

    #[test]
    fn ddp_step_checksum_drifts_rejects_misaligned_steps() {
        let reports = vec![
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0}
                ]
            }),
            serde_json::json!({
                "step_checksums": [
                    {"step": 2, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0}
                ]
            }),
        ];

        let err = ddp_step_checksum_drifts(&reports).unwrap_err();

        assert!(format!("{err}").contains("does not match rank 0 step"));
    }

    #[test]
    fn ddp_memory_step_checksum_drifts_report_memory_table_sync() {
        let reports = vec![
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "memory_table_checksum_sum": 4.0, "memory_table_checksum_sumsq": 16.0},
                    {"step": 2, "memory_table_checksum_sum": 5.0, "memory_table_checksum_sumsq": 25.0}
                ]
            }),
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "memory_table_checksum_sum": 4.0, "memory_table_checksum_sumsq": 16.0},
                    {"step": 2, "memory_table_checksum_sum": 5.125, "memory_table_checksum_sumsq": 25.5}
                ]
            }),
        ];

        let drifts = ddp_memory_step_checksum_drifts(&reports).unwrap();

        assert_eq!(drifts.len(), 2);
        assert_eq!(
            max_step_checksum_error(&drifts, "memory_table_checksum_sum_max_error"),
            0.125
        );
        assert_eq!(
            max_step_checksum_error(&drifts, "memory_table_checksum_sumsq_max_error"),
            0.5
        );
        assert_eq!(drifts[0]["memory_table_checksum_sum_max_error"], 0.0);
    }
}
