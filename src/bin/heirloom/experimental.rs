const PADAWAN_REF_PREFIX: &str = "artifact://padawan/";
const DEFAULT_PADAWAN_ALLOWED_PATH_PREFIXES: &[&str] = &[
    "padawan/",
    "src/bin/heirloom.rs",
    "src/bin/heirloom/",
    "docs/experimental/padawan/PADAWAN_LOOP_DESIGN.md",
    "docs/history/internal/CODEX_GOAL_LOOP.md",
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
