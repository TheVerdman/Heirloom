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
                    "product_key_selected_score_forward_calls": memory_kernel_counters.product_key_selected_score_forward_calls,
                    "product_key_backward_query_calls": memory_kernel_counters.product_key_backward_query_calls,
                    "product_key_backward_half_key_calls": memory_kernel_counters.product_key_backward_half_key_calls,
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
