fn run_training_command(command: Commands) -> Result<()> {
    match command {
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
        _ => unreachable!("training dispatcher received a non-training command"),
    }
    Ok(())
}
