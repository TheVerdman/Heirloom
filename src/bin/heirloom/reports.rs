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
