fn run_tokenizer_command(command: TokenizerCommands) -> Result<()> {
    match command {
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
    }
    Ok(())
}
