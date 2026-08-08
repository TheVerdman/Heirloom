fn run_data_command(command: DataCommands) -> Result<()> {
    match command {
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
    }
    Ok(())
}
