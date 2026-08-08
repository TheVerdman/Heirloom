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
