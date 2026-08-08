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
