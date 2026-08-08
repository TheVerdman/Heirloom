fn run_gpu_command(command: GpuCommands) -> Result<()> {
    match command {
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
    }
    Ok(())
}
