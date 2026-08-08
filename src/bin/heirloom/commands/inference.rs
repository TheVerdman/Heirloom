fn run_inference_command(command: Commands) -> Result<()> {
    match command {
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
        _ => unreachable!("inference dispatcher received a non-inference command"),
    }
    Ok(())
}
