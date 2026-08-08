use heirloom::checkpoint::{load_lm_checkpoint_on_device, save_lm_checkpoint_with_dataset_state};
use heirloom::data::TokenDatasetState;
use heirloom::nn::{
    amp_bf16_tensor_core_coverage, reset_amp_bf16_tensor_core_coverage, AdamW, GenerationOptions,
    Linear, Module, Optimizer, TinyTransformerConfig, TinyTransformerLm,
};
use heirloom::rng::HeirloomRng;
use heirloom::tokenizer::BpeTokenizer;
use heirloom::{DType, Device, Tensor, TensorError};
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

static TENSOR_CORE_COUNTER_LOCK: Mutex<()> = Mutex::new(());

fn require_cuda_hardware() {
    assert_eq!(
        std::env::var("HEIRLOOM_CUDA_TESTS").ok().as_deref(),
        Some("1"),
        "ignored CUDA tests must be run through scripts/test_gpu.sh cuda"
    );
    assert!(
        heirloom_kernels::cuda::is_available(),
        "HEIRLOOM_CUDA_TESTS=1 but the CUDA Driver API is unavailable"
    );
}

fn require_bf16_tensor_cores() {
    assert!(
        heirloom_kernels::cuda::device_supports_bf16_tensor_cores(0)
            .expect("query BF16 Tensor Core support"),
        "this CUDA test requires a device with BF16 Tensor Core support"
    );
}

fn tensor_core_counter_guard() -> MutexGuard<'static, ()> {
    TENSOR_CORE_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct EnvVarGuard {
    key: &'static str,
    old: Option<String>,
}

impl EnvVarGuard {
    fn remove(key: &'static str) -> Self {
        let old = std::env::var(key).ok();
        std::env::remove_var(key);
        Self { key, old }
    }

    fn set(key: &'static str, value: &str) -> Self {
        let old = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, old }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = self.old.as_deref() {
            std::env::set_var(self.key, value);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

#[test]
fn bfloat16_cpu_round_trip_and_casts_are_explicit() {
    let tensor = Tensor::from_f32(vec![1.0, -2.5, 3.25, 0.0], &[2, 2], false).unwrap();
    let bf16 = tensor.to_dtype(DType::BFloat16).unwrap();

    assert_eq!(bf16.dtype(), DType::BFloat16);
    assert_eq!(bf16.device(), Device::Cpu);
    assert_eq!(bf16.data_bf16_bits().unwrap().len(), 4);

    let restored = bf16.to_dtype(DType::F32).unwrap();
    let data = restored.data_f32().unwrap();
    assert!((data[0] - 1.0).abs() <= 0.01);
    assert!((data[1] + 2.5).abs() <= 0.01);
    assert!((data[2] - 3.25).abs() <= 0.01);
    assert_eq!(data[3], 0.0);
}

#[test]
fn bfloat16_zeros_and_ones_use_the_new_dtype() {
    let zeros = Tensor::zeros_with_dtype(&[3], DType::BFloat16, false).unwrap();
    let ones = Tensor::ones_with_dtype(&[3], DType::BFloat16, false).unwrap();

    assert_eq!(zeros.dtype(), DType::BFloat16);
    assert_eq!(zeros.data(), vec![0.0, 0.0, 0.0]);
    assert_eq!(ones.data(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn bfloat16_casts_are_differentiable_while_gradients_stay_f32() {
    let tensor = Tensor::from_f32(vec![1.0, -2.0, 3.5, 0.25], &[4], true).unwrap();
    let restored = tensor
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    restored.sum().unwrap().backward().unwrap();

    assert_close(&tensor.grad().unwrap(), &[1.0, 1.0, 1.0, 1.0], 1e-6);
    let grad = tensor.grad_tensor().unwrap();
    assert_eq!(grad.dtype(), DType::F64);
    assert_eq!(grad.device(), Device::Cpu);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_storage_round_trip_is_gated() {
    require_cuda_hardware();

    let tensor = Tensor::from_f32(vec![0.5, -1.25, 3.0, 8.0], &[2, 2], false).unwrap();
    let cuda = tensor.cuda(0).unwrap();
    assert_eq!(cuda.device(), Device::Cuda(0));
    assert_eq!(cuda.dtype(), DType::F32);
    assert_eq!(cuda.data_f32().unwrap(), vec![0.5, -1.25, 3.0, 8.0]);

    let cpu = cuda.cpu().unwrap();
    assert_eq!(cpu.device(), Device::Cpu);
    assert_eq!(cpu.data_f32().unwrap(), vec![0.5, -1.25, 3.0, 8.0]);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_f32_bf16_roundtrip_matches_two_cast_path_and_counts_one_family() {
    require_cuda_hardware();

    let _counter_guard = tensor_core_counter_guard();
    let values = vec![
        0.0,
        1.0,
        -2.5,
        std::f32::consts::PI,
        1.0 / 3.0,
        65504.0,
        -0.03125,
        42.125,
    ];
    let input = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &values).unwrap();
    let bf16 = heirloom_kernels::cuda::f32_to_bf16_buffer(&input).unwrap();
    let expected = heirloom_kernels::cuda::bf16_to_f32_buffer(&bf16)
        .unwrap()
        .to_f32()
        .unwrap();

    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let fused = heirloom_kernels::cuda::f32_bf16_roundtrip_f32_buffer(&input)
        .unwrap()
        .to_f32()
        .unwrap();
    assert_close(&fused, &expected, 0.0);

    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert_eq!(runtime.kernel_launch_calls, 1);
    assert_eq!(runtime.kernel_launch_elements, values.len());
    let roundtrip = runtime
        .kernel_launch_families
        .iter()
        .find(|family| family.label == "bf16_roundtrip")
        .expect("bf16_roundtrip launch family should be recorded");
    assert_eq!(roundtrip.calls, 1);
    assert_eq!(roundtrip.elements, values.len());
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_bf16_tensor_core_probe_is_gated() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    heirloom_kernels::cuda::reset_tensor_core_counters();
    let report = heirloom_kernels::cuda::bf16_mma_probe(0).unwrap();
    assert_eq!(report.device_ordinal, 0);
    assert_eq!(report.samples.len(), 32);
    assert!(
        report.max_abs_error <= 1e-4,
        "BF16 MMA probe expected all-ones 16-wide dot products to equal {}, max_abs_error={}, samples={:?}",
        report.expected_dot,
        report.max_abs_error,
        report.samples
    );
    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_mma_probe_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_calls, 0);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 0);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 0);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);
}

#[test]
fn bf16_tensor_core_shape_support_distinguishes_exact_and_padded_paths() {
    assert!(heirloom_kernels::cuda::bf16_tensor_core_matmul_exact_tile_shape_supported(16, 16, 8));
    assert!(
        heirloom_kernels::cuda::bf16_tensor_core_matmul_exact_tile_shape_supported(32, 64, 264)
    );
    assert!(!heirloom_kernels::cuda::bf16_tensor_core_matmul_exact_tile_shape_supported(16, 8, 16));
    assert!(!heirloom_kernels::cuda::bf16_tensor_core_matmul_exact_tile_shape_supported(15, 16, 8));
    assert!(!heirloom_kernels::cuda::bf16_tensor_core_matmul_exact_tile_shape_supported(16, 16, 7));

    assert!(heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(16, 16, 8));
    assert!(heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(32, 64, 264));
    assert!(heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(16, 8, 16));
    assert!(heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(15, 16, 8));
    assert!(heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(16, 15, 8));
    assert!(heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(16, 16, 7));
    assert!(!heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(0, 16, 8));
}

#[test]
fn bf16_tensor_core_attention_shape_support_distinguishes_exact_and_ragged_edges() {
    let exact = heirloom_kernels::cuda::CausalAttentionDims {
        batch: 1,
        time: 16,
        channels: 16,
        n_heads: 1,
    };
    let ragged_time_and_head = heirloom_kernels::cuda::CausalAttentionDims {
        batch: 1,
        time: 15,
        channels: 17,
        n_heads: 1,
    };
    let invalid_heads = heirloom_kernels::cuda::CausalAttentionDims {
        batch: 1,
        time: 15,
        channels: 17,
        n_heads: 0,
    };
    let invalid_channels = heirloom_kernels::cuda::CausalAttentionDims {
        batch: 1,
        time: 15,
        channels: 17,
        n_heads: 2,
    };

    assert!(
        heirloom_kernels::cuda::causal_attention_bf16_tensor_core_exact_tile_shape_supported(exact)
    );
    assert!(heirloom_kernels::cuda::causal_attention_bf16_tensor_core_shape_supported(exact));
    assert!(
        !heirloom_kernels::cuda::causal_attention_bf16_tensor_core_exact_tile_shape_supported(
            ragged_time_and_head
        )
    );
    assert!(
        heirloom_kernels::cuda::causal_attention_bf16_tensor_core_shape_supported(
            ragged_time_and_head
        )
    );
    assert!(
        !heirloom_kernels::cuda::causal_attention_bf16_tensor_core_shape_supported(invalid_heads)
    );
    assert!(
        !heirloom_kernels::cuda::causal_attention_bf16_tensor_core_shape_supported(
            invalid_channels
        )
    );
}

#[test]
fn flash_bf16_attention_guards_default_off_without_cuda() {
    let _counter_guard = tensor_core_counter_guard();
    let _flash = EnvVarGuard::remove("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION");
    let _require = EnvVarGuard::remove("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION");
    let _timing = EnvVarGuard::remove("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING");
    let _backward = EnvVarGuard::remove("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD");
    let _ldmatrix_require = EnvVarGuard::remove("HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_LDMATRIX_GEMM");
    let _cp_async_require = EnvVarGuard::remove("HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM");

    assert!(!heirloom_kernels::cuda::flash_bf16_attention_enabled());
    assert!(!heirloom_kernels::cuda::require_flash_bf16_attention_enabled());
    assert!(!heirloom_kernels::cuda::flash_bf16_attention_timing_enabled());
    assert!(!heirloom_kernels::cuda::flash_bf16_tensor_core_attention_backward_enabled());
    assert!(!heirloom_kernels::cuda::require_tensor_core_ldmatrix_gemm_enabled());
    assert!(!heirloom_kernels::cuda::require_tensor_core_cp_async_gemm_enabled());
}

#[test]
fn flash_bf16_attention_hard_require_error_names_policy_without_cuda() {
    let _counter_guard = tensor_core_counter_guard();
    let _require = EnvVarGuard::set("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION", "1");

    assert!(heirloom_kernels::cuda::require_flash_bf16_attention_enabled());
    let err = heirloom_kernels::cuda::flash_bf16_attention_required_error(
        "autograd requires the materialized attention buffer",
    );
    assert!(err
        .to_string()
        .contains("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION=1"));
    assert!(err.to_string().contains("flash BF16 causal attention"));
}

#[test]
fn bf16_tensor_core_cta_plan_reports_tile_geometry_without_cuda() {
    let plan = heirloom_kernels::cuda::bf16_tensor_core_matmul_cta_plan(32, 64, 16).unwrap();

    assert_eq!(plan.cta_m, 32);
    assert_eq!(plan.cta_n, 16);
    assert_eq!(plan.warps_per_cta, 4);
    assert_eq!(plan.warp_m, 16);
    assert_eq!(plan.warp_n, 8);
    assert_eq!(plan.grid_x, 1);
    assert_eq!(plan.grid_y, 1);
    assert_eq!(plan.cta_tiles, 1);
    assert_eq!(plan.active_mma_warp_tiles, 4);
    assert_eq!(plan.launched_warps, 4);
    assert_eq!(plan.k_tiles, 4);
    assert_eq!(plan.shared_stage_tiles, 4);
    assert_eq!(plan.shared_stage_bytes, 6144);

    let ragged_after_padding =
        heirloom_kernels::cuda::bf16_tensor_core_matmul_cta_plan(16, 32, 8).unwrap();
    assert_eq!(ragged_after_padding.grid_x, 1);
    assert_eq!(ragged_after_padding.grid_y, 1);
    assert_eq!(ragged_after_padding.cta_tiles, 1);
    assert_eq!(ragged_after_padding.active_mma_warp_tiles, 1);
    assert_eq!(ragged_after_padding.launched_warps, 4);
    assert_eq!(ragged_after_padding.k_tiles, 2);
    assert_eq!(ragged_after_padding.shared_stage_tiles, 2);
    assert_eq!(ragged_after_padding.shared_stage_bytes, 3072);

    assert!(heirloom_kernels::cuda::bf16_tensor_core_matmul_cta_plan(15, 16, 8).is_err());

    let wide =
        heirloom_kernels::cuda::bf16_tensor_core_matmul_wide_swizzled_cta_plan(32, 64, 32).unwrap();
    assert_eq!(wide.cta_m, 32);
    assert_eq!(wide.cta_n, 32);
    assert_eq!(wide.warps_per_cta, 8);
    assert_eq!(wide.warp_m, 16);
    assert_eq!(wide.warp_n, 8);
    assert_eq!(wide.grid_x, 1);
    assert_eq!(wide.grid_y, 1);
    assert_eq!(wide.cta_tiles, 1);
    assert_eq!(wide.active_mma_warp_tiles, 8);
    assert_eq!(wide.launched_warps, 8);
    assert_eq!(wide.k_tiles, 4);
    assert_eq!(wide.shared_stage_tiles, 4);
    assert_eq!(wide.shared_stage_bytes, 8192);

    let wide_edge =
        heirloom_kernels::cuda::bf16_tensor_core_matmul_wide_swizzled_cta_plan(16, 32, 8).unwrap();
    assert_eq!(wide_edge.grid_x, 1);
    assert_eq!(wide_edge.grid_y, 1);
    assert_eq!(wide_edge.cta_tiles, 1);
    assert_eq!(wide_edge.active_mma_warp_tiles, 1);
    assert_eq!(wide_edge.launched_warps, 8);
    assert_eq!(wide_edge.k_tiles, 2);
    assert_eq!(wide_edge.shared_stage_tiles, 2);
    assert_eq!(wide_edge.shared_stage_bytes, 4096);

    assert!(
        heirloom_kernels::cuda::bf16_tensor_core_matmul_wide_swizzled_cta_plan(15, 16, 8).is_err()
    );
}

#[test]
fn cuda_runtime_counters_reset_to_zero_without_cuda() {
    if std::env::var("HEIRLOOM_CUDA_TESTS").as_deref() == Ok("1") {
        return;
    }

    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let counters = heirloom_kernels::cuda::cuda_runtime_counters();

    assert_eq!(counters.kernel_launch_calls, 0);
    assert_eq!(counters.host_sync_calls, 0);
    assert_eq!(counters.stream_create_calls, 0);
    assert_eq!(counters.event_create_calls, 0);
    assert_eq!(counters.h2d_bytes, 0);
    assert_eq!(counters.d2h_bytes, 0);
    assert_eq!(counters.allocation_active_bytes, 0);
    assert_eq!(counters.allocation_reserved_bytes, 0);
    assert_eq!(counters.module_load_calls, 0);
    assert_eq!(counters.module_cache_hits, 0);
    assert_eq!(counters.tensor_core_cta_gemm_calls, 0);
    assert_eq!(counters.tensor_core_cta_tiles, 0);
    assert_eq!(counters.tensor_core_cta_warps_launched, 0);
    assert_eq!(counters.tensor_core_mma_warp_tiles, 0);
    assert_eq!(counters.tensor_core_staged_cta_gemm_calls, 0);
    assert_eq!(counters.tensor_core_shared_stage_tiles, 0);
    assert_eq!(counters.tensor_core_shared_stage_bytes, 0);
    assert_eq!(counters.tensor_core_wide_swizzled_cta_gemm_calls, 0);
    assert_eq!(counters.tensor_core_swizzled_stage_tiles, 0);
    assert_eq!(counters.tensor_core_swizzled_stage_bytes, 0);
    assert_eq!(counters.tensor_core_ldmatrix_gemm_requested_calls, 0);
    assert_eq!(counters.tensor_core_ldmatrix_gemm_executed_calls, 0);
    assert_eq!(counters.tensor_core_ldmatrix_gemm_staged_fallback_calls, 0);
    assert_eq!(counters.tensor_core_ldmatrix_gemm_hard_require_failures, 0);
    assert_eq!(counters.tensor_core_ldmatrix_gemm_instructions, 0);
    assert_eq!(counters.tensor_core_cp_async_gemm_requested_calls, 0);
    assert_eq!(counters.tensor_core_cp_async_gemm_executed_calls, 0);
    assert_eq!(counters.tensor_core_cp_async_gemm_staged_fallback_calls, 0);
    assert_eq!(counters.tensor_core_cp_async_gemm_hard_require_failures, 0);
    assert_eq!(counters.tensor_core_cp_async_gemm_instructions, 0);
    assert_eq!(counters.tensor_core_global_cta_gemm_calls, 0);
    assert_eq!(counters.tensor_core_legacy_warp_gemm_calls, 0);
    assert_eq!(counters.tensor_core_attention_padded_tiles, 0);
    assert_eq!(counters.tensor_core_attention_remainder_tiles, 0);
    assert_eq!(counters.tensor_core_attention_scalar_fallbacks, 0);
    assert_eq!(counters.tensor_core_attention_ldmatrix_requested_calls, 0);
    assert_eq!(
        counters.tensor_core_attention_ldmatrix_global_fallback_calls,
        0
    );
    assert_eq!(counters.tensor_core_attention_cp_async_requested_calls, 0);
    assert_eq!(
        counters.tensor_core_attention_cp_async_global_fallback_calls,
        0
    );
    assert_eq!(counters.bf16_attention_materialized_reference_calls, 0);
    assert_eq!(counters.flash_bf16_attention_requested_calls, 0);
    assert_eq!(counters.flash_bf16_attention_executed_calls, 0);
    assert_eq!(counters.flash_bf16_attention_fallback_calls, 0);
    assert_eq!(counters.flash_bf16_attention_scalar_fallback_calls, 0);
    assert_eq!(counters.flash_bf16_attention_qk_tile_calls, 0);
    assert_eq!(counters.flash_bf16_attention_av_tile_calls, 0);
    assert_eq!(counters.flash_bf16_attention_ragged_tile_count, 0);
    assert_eq!(counters.flash_bf16_attention_causal_masked_tile_count, 0);
    assert_eq!(counters.flash_bf16_attention_elapsed_us, 0);
    assert_eq!(counters.flash_bf16_attention_hard_require_failures, 0);
    assert_eq!(counters.flash_bf16_scalar_streaming_requested_calls, 0);
    assert_eq!(counters.flash_bf16_scalar_streaming_executed_calls, 0);
    assert_eq!(counters.flash_bf16_scalar_streaming_qk_tile_calls, 0);
    assert_eq!(counters.flash_bf16_scalar_streaming_av_tile_calls, 0);
    assert_eq!(counters.flash_bf16_scalar_streaming_elapsed_us, 0);
    assert_eq!(counters.flash_bf16_tensor_core_requested_calls, 0);
    assert_eq!(counters.flash_bf16_tensor_core_executed_calls, 0);
    assert_eq!(counters.flash_bf16_tensor_core_fallback_calls, 0);
    assert_eq!(counters.flash_bf16_tensor_core_qk_mma_tile_calls, 0);
    assert_eq!(counters.flash_bf16_tensor_core_av_mma_tile_calls, 0);
    assert_eq!(counters.flash_bf16_tensor_core_ragged_tile_count, 0);
    assert_eq!(counters.flash_bf16_tensor_core_causal_masked_tile_count, 0);
    assert_eq!(counters.flash_bf16_tensor_core_elapsed_us, 0);
    assert_eq!(counters.flash_bf16_tensor_core_backward_requested_calls, 0);
    assert_eq!(counters.flash_bf16_tensor_core_backward_executed_calls, 0);
    assert_eq!(counters.flash_bf16_tensor_core_backward_fallback_calls, 0);
    assert_eq!(counters.flash_bf16_tensor_core_backward_row_dot_calls, 0);
    assert_eq!(
        counters.flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls,
        0
    );
    assert_eq!(
        counters.flash_bf16_tensor_core_backward_dp_mma_tile_calls,
        0
    );
    assert_eq!(
        counters.flash_bf16_tensor_core_backward_dq_mma_tile_calls,
        0
    );
    assert_eq!(
        counters.flash_bf16_tensor_core_backward_dk_mma_tile_calls,
        0
    );
    assert_eq!(
        counters.flash_bf16_tensor_core_backward_dv_mma_tile_calls,
        0
    );
    assert_eq!(
        counters.flash_bf16_tensor_core_backward_scalar_tile_calls,
        0
    );
    assert_eq!(
        counters.flash_bf16_tensor_core_backward_ragged_tile_count,
        0
    );
    assert_eq!(
        counters.flash_bf16_tensor_core_backward_causal_masked_tile_count,
        0
    );
    assert_eq!(counters.flash_bf16_tensor_core_backward_elapsed_us, 0);
}

#[test]
fn cuda_flash_bf16_tensor_core_env_guard_defaults_off() {
    let _guard = tensor_core_counter_guard();
    let _flash = EnvVarGuard::remove("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION");
    let _tensor_core = EnvVarGuard::remove("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE");
    let _backward = EnvVarGuard::remove("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD");
    assert!(!heirloom_kernels::cuda::flash_bf16_attention_enabled());
    assert!(!heirloom_kernels::cuda::flash_bf16_tensor_core_attention_enabled());
    assert!(!heirloom_kernels::cuda::flash_bf16_tensor_core_attention_backward_enabled());

    let _flash = EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION", "1");
    let _tensor_core = EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE", "1");
    let _backward = EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD", "1");
    assert!(heirloom_kernels::cuda::flash_bf16_attention_enabled());
    assert!(heirloom_kernels::cuda::flash_bf16_tensor_core_attention_enabled());
    assert!(heirloom_kernels::cuda::flash_bf16_tensor_core_attention_backward_enabled());
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_raw_bf16_tensor_core_matmul_matches_cpu_bf16_reference() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let input_values = (0..(16 * 16))
        .map(|index| ((index as f32 % 19.0) - 9.0) / 9.0)
        .collect::<Vec<_>>();
    let weight_values = (0..(8 * 16))
        .map(|index| ((index as f32 % 11.0) - 5.0) / 6.0)
        .collect::<Vec<_>>();
    let input_bf16 = Tensor::from_f32(input_values.clone(), &[16, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap();
    let weight_bf16 = Tensor::from_f32(weight_values.clone(), &[8, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap();
    let expected = input_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .matmul(
            &weight_bf16
                .to_dtype(DType::F32)
                .unwrap()
                .transpose()
                .unwrap(),
        )
        .unwrap()
        .data_f32()
        .unwrap();

    heirloom_kernels::cuda::reset_tensor_core_counters();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let left =
        heirloom_kernels::cuda::CudaBuffer::from_u16(0, &input_bf16.data_bf16_bits().unwrap())
            .unwrap();
    let right =
        heirloom_kernels::cuda::CudaBuffer::from_u16(0, &weight_bf16.data_bf16_bits().unwrap())
            .unwrap();
    let output =
        heirloom_kernels::cuda::matmul_bf16_tensor_core_rhs_t_f32_buffers(&left, &right, 16, 16, 8)
            .unwrap();

    assert_close(&output.to_f32().unwrap(), &expected, 5e-2);
    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 0);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    if heirloom_kernels::cuda::tensor_core_legacy_warp_gemm_enabled() {
        assert_eq!(runtime.tensor_core_legacy_warp_gemm_calls, 1);
    } else if heirloom_kernels::cuda::tensor_core_global_cta_gemm_enabled() {
        assert_eq!(runtime.tensor_core_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_global_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_staged_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_wide_swizzled_cta_gemm_calls, 0);
    } else if heirloom_kernels::cuda::tensor_core_wide_swizzled_gemm_enabled() {
        assert_eq!(runtime.tensor_core_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_cta_tiles, 1);
        assert_eq!(runtime.tensor_core_cta_warps_launched, 8);
        assert_eq!(runtime.tensor_core_mma_warp_tiles, 1);
        assert_eq!(runtime.tensor_core_wide_swizzled_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_swizzled_stage_tiles, 1);
        assert_eq!(runtime.tensor_core_swizzled_stage_bytes, 2048);
        assert_eq!(runtime.tensor_core_staged_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_global_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_legacy_warp_gemm_calls, 0);
    } else if heirloom_kernels::cuda::tensor_core_cp_async_gemm_enabled() {
        assert_eq!(runtime.tensor_core_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_cta_tiles, 1);
        assert_eq!(runtime.tensor_core_cta_warps_launched, 4);
        assert_eq!(runtime.tensor_core_mma_warp_tiles, 1);
        assert_eq!(runtime.tensor_core_staged_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_wide_swizzled_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_global_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_legacy_warp_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_cp_async_gemm_requested_calls, 1);
        assert_eq!(runtime.tensor_core_cp_async_gemm_executed_calls, 1);
        assert_eq!(runtime.tensor_core_cp_async_gemm_staged_fallback_calls, 0);
        assert!(runtime.tensor_core_cp_async_gemm_instructions > 0);
        assert_eq!(runtime.tensor_core_ldmatrix_gemm_requested_calls, 0);
        assert_eq!(runtime.tensor_core_ldmatrix_gemm_staged_fallback_calls, 0);
    } else if heirloom_kernels::cuda::tensor_core_ldmatrix_gemm_enabled()
        && !heirloom_kernels::cuda::tensor_core_cp_async_gemm_enabled()
    {
        assert_eq!(runtime.tensor_core_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_cta_tiles, 1);
        assert_eq!(runtime.tensor_core_cta_warps_launched, 4);
        assert_eq!(runtime.tensor_core_mma_warp_tiles, 1);
        assert_eq!(runtime.tensor_core_staged_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_wide_swizzled_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_global_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_legacy_warp_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_ldmatrix_gemm_requested_calls, 1);
        assert_eq!(runtime.tensor_core_ldmatrix_gemm_executed_calls, 1);
        assert_eq!(runtime.tensor_core_ldmatrix_gemm_staged_fallback_calls, 0);
        assert!(runtime.tensor_core_ldmatrix_gemm_instructions > 0);
        assert_eq!(runtime.tensor_core_cp_async_gemm_requested_calls, 0);
        assert_eq!(runtime.tensor_core_cp_async_gemm_staged_fallback_calls, 0);
    } else {
        assert_eq!(runtime.tensor_core_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_cta_tiles, 1);
        assert_eq!(runtime.tensor_core_cta_warps_launched, 4);
        assert_eq!(runtime.tensor_core_mma_warp_tiles, 1);
        assert_eq!(runtime.tensor_core_staged_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_shared_stage_tiles, 1);
        assert_eq!(runtime.tensor_core_shared_stage_bytes, 1536);
        assert_eq!(runtime.tensor_core_wide_swizzled_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_global_cta_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_legacy_warp_gemm_calls, 0);
        assert_eq!(runtime.tensor_core_cp_async_gemm_requested_calls, 0);
        assert_eq!(runtime.tensor_core_cp_async_gemm_staged_fallback_calls, 0);
        assert_eq!(runtime.tensor_core_ldmatrix_gemm_requested_calls, 0);
        assert_eq!(runtime.tensor_core_ldmatrix_gemm_staged_fallback_calls, 0);
    }
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_raw_bf16_tensor_core_normal_rhs_matmul_matches_cpu_bf16_reference() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let input_values = (0..(16 * 16))
        .map(|index| ((index as f32 % 23.0) - 11.0) / 11.0)
        .collect::<Vec<_>>();
    let right_values = (0..(16 * 8))
        .map(|index| ((index as f32 % 13.0) - 6.0) / 7.0)
        .collect::<Vec<_>>();
    let input_bf16 = Tensor::from_f32(input_values.clone(), &[16, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap();
    let right_bf16 = Tensor::from_f32(right_values.clone(), &[16, 8], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap();
    let expected = input_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .matmul(&right_bf16.to_dtype(DType::F32).unwrap())
        .unwrap()
        .data_f32()
        .unwrap();

    heirloom_kernels::cuda::reset_tensor_core_counters();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let left =
        heirloom_kernels::cuda::CudaBuffer::from_u16(0, &input_bf16.data_bf16_bits().unwrap())
            .unwrap();
    let right =
        heirloom_kernels::cuda::CudaBuffer::from_u16(0, &right_bf16.data_bf16_bits().unwrap())
            .unwrap();
    let output = heirloom_kernels::cuda::matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward(
        &left, &right, 16, 16, 8,
    )
    .unwrap();

    assert_close(&output.to_f32().unwrap(), &expected, 5e-2);
    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 0);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 1);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    let normal_rhs_calls = runtime
        .kernel_launch_families
        .iter()
        .find(|family| family.label == "tensor_core_gemm_normal_rhs_staged")
        .map(|family| family.calls)
        .unwrap_or(0);
    assert_eq!(normal_rhs_calls, 1);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_transpose2d_pair_bf16_matches_cpu_transposes_and_counts_one_family() {
    require_cuda_hardware();

    let first_rows = 3usize;
    let first_cols = 5usize;
    let second_rows = 4usize;
    let second_cols = 2usize;
    let first_bits = (0..(first_rows * first_cols))
        .map(|index| (index as u16).wrapping_mul(17).wrapping_add(3))
        .collect::<Vec<_>>();
    let second_bits = (0..(second_rows * second_cols))
        .map(|index| (index as u16).wrapping_mul(31).wrapping_add(7))
        .collect::<Vec<_>>();
    let transpose_bits = |values: &[u16], rows: usize, cols: usize| {
        let mut out = vec![0u16; values.len()];
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = values[row * cols + col];
            }
        }
        out
    };

    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let first = heirloom_kernels::cuda::CudaBuffer::from_u16(0, &first_bits).unwrap();
    let second = heirloom_kernels::cuda::CudaBuffer::from_u16(0, &second_bits).unwrap();
    let (first_t, second_t) = heirloom_kernels::cuda::transpose2d_pair_bf16_buffers(
        &first,
        first_rows,
        first_cols,
        &second,
        second_rows,
        second_cols,
    )
    .unwrap();

    assert_eq!(
        first_t.to_u16().unwrap(),
        transpose_bits(&first_bits, first_rows, first_cols)
    );
    assert_eq!(
        second_t.to_u16().unwrap(),
        transpose_bits(&second_bits, second_rows, second_cols)
    );
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    let pair_calls = runtime
        .kernel_launch_families
        .iter()
        .find(|family| family.label == "transpose2d_pair_bf16")
        .map(|family| family.calls)
        .unwrap_or(0);
    assert_eq!(pair_calls, 1);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_raw_bf16_tensor_core_matmul_pads_ragged_shapes() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let (m, k, n) = (15usize, 17usize, 7usize);
    let input_values = (0..(m * k))
        .map(|index| ((index as f32 % 23.0) - 11.0) / 11.0)
        .collect::<Vec<_>>();
    let weight_values = (0..(n * k))
        .map(|index| ((index as f32 % 17.0) - 8.0) / 9.0)
        .collect::<Vec<_>>();
    let input_bf16 = Tensor::from_f32(input_values.clone(), &[m, k], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap();
    let weight_bf16 = Tensor::from_f32(weight_values.clone(), &[n, k], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap();
    let expected = input_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .matmul(
            &weight_bf16
                .to_dtype(DType::F32)
                .unwrap()
                .transpose()
                .unwrap(),
        )
        .unwrap()
        .data_f32()
        .unwrap();

    heirloom_kernels::cuda::reset_tensor_core_counters();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let left =
        heirloom_kernels::cuda::CudaBuffer::from_u16(0, &input_bf16.data_bf16_bits().unwrap())
            .unwrap();
    let right =
        heirloom_kernels::cuda::CudaBuffer::from_u16(0, &weight_bf16.data_bf16_bits().unwrap())
            .unwrap();
    let output =
        heirloom_kernels::cuda::matmul_bf16_tensor_core_rhs_t_f32_buffers(&left, &right, m, k, n)
            .unwrap();

    assert_eq!(output.len(), m * n);
    assert_close(&output.to_f32().unwrap(), &expected, 8e-2);
    let tensor_core = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(tensor_core.bf16_tensor_core_matmul_calls, 1);
    assert_eq!(tensor_core.bf16_scalar_matmul_fallback_calls, 0);
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert!(runtime.tensor_core_padded_tiles > 0);
    assert!(runtime.tensor_core_remainder_tiles > 0);
    if !heirloom_kernels::cuda::tensor_core_legacy_warp_gemm_enabled() {
        assert_eq!(runtime.tensor_core_cta_gemm_calls, 1);
        assert_eq!(runtime.tensor_core_cta_tiles, 1);
        let expected_warps_launched =
            if heirloom_kernels::cuda::tensor_core_wide_swizzled_gemm_enabled() {
                8
            } else {
                4
            };
        assert_eq!(
            runtime.tensor_core_cta_warps_launched,
            expected_warps_launched
        );
        assert_eq!(runtime.tensor_core_mma_warp_tiles, 1);
        assert_eq!(runtime.tensor_core_legacy_warp_gemm_calls, 0);
        if heirloom_kernels::cuda::tensor_core_global_cta_gemm_enabled() {
            assert_eq!(runtime.tensor_core_global_cta_gemm_calls, 1);
            assert_eq!(runtime.tensor_core_staged_cta_gemm_calls, 0);
            assert_eq!(runtime.tensor_core_wide_swizzled_cta_gemm_calls, 0);
        } else if heirloom_kernels::cuda::tensor_core_wide_swizzled_gemm_enabled() {
            assert_eq!(runtime.tensor_core_wide_swizzled_cta_gemm_calls, 1);
            assert!(runtime.tensor_core_swizzled_stage_tiles > 0);
            assert!(runtime.tensor_core_swizzled_stage_bytes > 0);
            assert_eq!(runtime.tensor_core_staged_cta_gemm_calls, 0);
            assert_eq!(runtime.tensor_core_global_cta_gemm_calls, 0);
        } else {
            assert_eq!(runtime.tensor_core_staged_cta_gemm_calls, 1);
            assert!(runtime.tensor_core_shared_stage_tiles > 0);
            assert!(runtime.tensor_core_shared_stage_bytes > 0);
            assert_eq!(runtime.tensor_core_wide_swizzled_cta_gemm_calls, 0);
            assert_eq!(runtime.tensor_core_global_cta_gemm_calls, 0);
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_linear_amp_bf16_uses_tensor_core_matmul_when_tile_aligned() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let _normal_rhs_guard = EnvVarGuard::remove("HEIRLOOM_CUDA_TENSOR_CORE_NORMAL_RHS_GEMM");
    let input_values = (0..(16 * 16))
        .map(|index| ((index as f32 % 17.0) - 8.0) / 8.0)
        .collect::<Vec<_>>();
    let weight_values = (0..(16 * 16))
        .map(|index| ((index as f32 % 13.0) - 6.0) / 7.0)
        .collect::<Vec<_>>();
    let bias_values = (0..16)
        .map(|index| (index as f32 - 3.0) / 11.0)
        .collect::<Vec<_>>();

    let cpu_input = Tensor::from_f32(input_values.clone(), &[16, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let cpu_weight = Tensor::from_f32(weight_values.clone(), &[16, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let cpu_bias = Tensor::from_f32(bias_values.clone(), &[16], false).unwrap();
    let expected = cpu_input
        .matmul(&cpu_weight.transpose().unwrap())
        .unwrap()
        .add(&cpu_bias)
        .unwrap()
        .data_f32()
        .unwrap();

    let linear = Linear::new(16, 16).unwrap();
    linear.weight().copy_from_data(&weight_values).unwrap();
    linear.bias().copy_from_data(&bias_values).unwrap();
    let linear = linear.to_device(Device::Cuda(0)).unwrap();
    let input = Tensor::from_f32(input_values, &[16, 16], true)
        .unwrap()
        .cuda(0)
        .unwrap();

    heirloom_kernels::cuda::reset_tensor_core_counters();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    reset_amp_bf16_tensor_core_coverage();
    let output = linear.forward_amp_bf16(&input).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_close(&output.data_f32().unwrap(), &expected, 5e-2);
    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 0);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);
    let coverage = amp_bf16_tensor_core_coverage();
    assert_eq!(coverage.linear_totals.calls, 1);
    assert_eq!(coverage.linear_totals.tensor_core_calls, 1);
    assert_eq!(coverage.linear_totals.fallback_calls, 0);
    assert_eq!(coverage.linear_modules.len(), 1);
    assert_eq!(coverage.linear_modules[0].module, "linear");
    assert_eq!(coverage.linear_modules[0].last_m, 16);
    assert_eq!(coverage.linear_modules[0].last_k, 16);
    assert_eq!(coverage.linear_modules[0].last_n, 16);
    assert_eq!(coverage.linear_modules[0].last_path, "tensor_core");
    if heirloom_kernels::cuda::tensor_core_cp_async_gemm_enabled() {
        let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
        assert_eq!(runtime.tensor_core_cp_async_gemm_executed_calls, 1);
        let bias_forward_calls = runtime
            .kernel_launch_families
            .iter()
            .find(|family| family.label == "bias_2d")
            .map(|family| family.calls)
            .unwrap_or(0);
        assert_eq!(bias_forward_calls, 0);
    }

    output.sum().unwrap().backward().unwrap();
    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_matmul_calls, 3);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 2);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    let materialize_calls = runtime
        .kernel_launch_families
        .iter()
        .find(|family| family.label == "materialize_matrix_layout")
        .map(|family| family.calls)
        .unwrap_or(0);
    assert_eq!(materialize_calls, 1);
    let normal_rhs_calls = runtime
        .kernel_launch_families
        .iter()
        .find(|family| family.label == "tensor_core_gemm_normal_rhs_staged")
        .map(|family| family.calls)
        .unwrap_or(0);
    assert_eq!(normal_rhs_calls, 0);
    let pair_transpose_calls = runtime
        .kernel_launch_families
        .iter()
        .find(|family| family.label == "transpose2d_pair_bf16")
        .map(|family| family.calls)
        .unwrap_or(0);
    assert_eq!(pair_transpose_calls, 1);

    let cpu_input_data = cpu_input.data_f32().unwrap();
    let cpu_weight_data = cpu_weight.data_f32().unwrap();
    let expected_input_grad = (0..(16 * 16))
        .map(|index| {
            let k = index % 16;
            (0..16).map(|n| cpu_weight_data[n * 16 + k]).sum()
        })
        .collect::<Vec<f32>>();
    let expected_weight_grad = (0..(16 * 16))
        .map(|index| {
            let k = index % 16;
            (0..16).map(|m| cpu_input_data[m * 16 + k]).sum()
        })
        .collect::<Vec<f32>>();

    let input_grad = input.grad_tensor().unwrap();
    assert_eq!(input_grad.device(), Device::Cuda(0));
    assert_eq!(input_grad.dtype(), DType::F32);
    assert_close(&input_grad.data_f32().unwrap(), &expected_input_grad, 5e-2);

    let weight_grad = linear.weight().grad_tensor().unwrap();
    assert_eq!(weight_grad.device(), Device::Cuda(0));
    assert_eq!(weight_grad.dtype(), DType::F32);
    assert_close(
        &weight_grad.data_f32().unwrap(),
        &expected_weight_grad,
        5e-2,
    );
    let bias_grad = linear.bias().grad_tensor().unwrap();
    assert_eq!(bias_grad.device(), Device::Cuda(0));
    assert_eq!(bias_grad.dtype(), DType::F32);
    assert_close(&bias_grad.data_f32().unwrap(), &[16.0; 16], 1e-6);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_raw_bf16_tensor_core_matmul_bias_cp_async_matches_cpu_bf16_reference() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let _cp_async_guard = EnvVarGuard::set("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM", "1");
    let input_values = (0..(16 * 16))
        .map(|index| ((index as f32 % 17.0) - 8.0) / 8.0)
        .collect::<Vec<_>>();
    let weight_values = (0..(8 * 16))
        .map(|index| ((index as f32 % 13.0) - 6.0) / 7.0)
        .collect::<Vec<_>>();
    let bias_values = (0..8)
        .map(|index| (index as f32 - 4.0) / 9.0)
        .collect::<Vec<_>>();
    let input_bf16 = Tensor::from_f32(input_values.clone(), &[16, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap();
    let weight_bf16 = Tensor::from_f32(weight_values.clone(), &[8, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap();
    let expected = input_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .matmul(
            &weight_bf16
                .to_dtype(DType::F32)
                .unwrap()
                .transpose()
                .unwrap(),
        )
        .unwrap()
        .add(&Tensor::from_f32(bias_values.clone(), &[8], false).unwrap())
        .unwrap()
        .data_f32()
        .unwrap();

    heirloom_kernels::cuda::reset_tensor_core_counters();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let left =
        heirloom_kernels::cuda::CudaBuffer::from_u16(0, &input_bf16.data_bf16_bits().unwrap())
            .unwrap();
    let right =
        heirloom_kernels::cuda::CudaBuffer::from_u16(0, &weight_bf16.data_bf16_bits().unwrap())
            .unwrap();
    let bias = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &bias_values).unwrap();
    let output = heirloom_kernels::cuda::matmul_bf16_tensor_core_rhs_t_bias_f32_buffers(
        &left, &right, &bias, 16, 16, 8,
    )
    .unwrap();

    assert_close(&output.to_f32().unwrap(), &expected, 5e-2);
    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 0);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert_eq!(runtime.tensor_core_cp_async_gemm_requested_calls, 1);
    assert_eq!(runtime.tensor_core_cp_async_gemm_executed_calls, 1);
    assert_eq!(runtime.tensor_core_cp_async_gemm_staged_fallback_calls, 0);
    let bias_calls = runtime
        .kernel_launch_families
        .iter()
        .find(|family| family.label == "bias_2d")
        .map(|family| family.calls)
        .unwrap_or(0);
    assert_eq!(bias_calls, 0);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_causal_attention_amp_bf16_tensor_core_forward_is_gated() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let query_values = (0..(16 * 16))
        .map(|index| ((index as f32 % 17.0) - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let key_values = (0..(16 * 16))
        .map(|index| ((index as f32 % 19.0) - 9.0) / 13.0)
        .collect::<Vec<_>>();
    let value_values = (0..(16 * 16))
        .map(|index| ((index as f32 % 23.0) - 11.0) / 17.0)
        .collect::<Vec<_>>();

    let cpu_query_values = Tensor::from_f32(query_values.clone(), &[1, 16, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_key_values = Tensor::from_f32(key_values.clone(), &[1, 16, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_value_values = Tensor::from_f32(value_values.clone(), &[1, 16, 16], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_query = Tensor::from_f32(cpu_query_values, &[1, 16, 16], true).unwrap();
    let cpu_key = Tensor::from_f32(cpu_key_values, &[1, 16, 16], true).unwrap();
    let cpu_value = Tensor::from_f32(cpu_value_values, &[1, 16, 16], true).unwrap();
    let expected = cpu_query
        .causal_self_attention(&cpu_key, &cpu_value, 1)
        .unwrap()
        .data_f32()
        .unwrap();
    cpu_query
        .causal_self_attention(&cpu_key, &cpu_value, 1)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let query = Tensor::from_f32(query_values, &[1, 16, 16], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let key = Tensor::from_f32(key_values, &[1, 16, 16], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let value = Tensor::from_f32(value_values, &[1, 16, 16], true)
        .unwrap()
        .cuda(0)
        .unwrap();

    heirloom_kernels::cuda::reset_tensor_core_counters();
    let output = query
        .causal_self_attention_amp_bf16_tensor_core(&key, &value, 1)
        .unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_close(&output.data_f32().unwrap(), &expected, 8e-2);

    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_attention_forward_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_qk_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_av_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 2);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 0);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);

    output.sum().unwrap().backward().unwrap();
    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_attention_forward_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_qk_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_av_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_backward_calls, 1);
    assert_eq!(
        counters.bf16_tensor_core_attention_score_grad_matmul_calls,
        1
    );
    assert_eq!(counters.bf16_tensor_core_attention_dq_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_dk_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_dv_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_calls, 6);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 2);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 4);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);

    let query_grad = query.grad_tensor().unwrap();
    let key_grad = key.grad_tensor().unwrap();
    let value_grad = value.grad_tensor().unwrap();
    assert_eq!(query_grad.device(), Device::Cuda(0));
    assert_eq!(key_grad.device(), Device::Cuda(0));
    assert_eq!(value_grad.device(), Device::Cuda(0));
    assert_eq!(query_grad.dtype(), DType::F32);
    assert_eq!(key_grad.dtype(), DType::F32);
    assert_eq!(value_grad.dtype(), DType::F32);
    assert_close(
        &query_grad.data_f32().unwrap(),
        &cpu_query.grad().unwrap(),
        1.5e-1,
    );
    assert_close(
        &key_grad.data_f32().unwrap(),
        &cpu_key.grad().unwrap(),
        1.5e-1,
    );
    assert_close(
        &value_grad.data_f32().unwrap(),
        &cpu_value.grad().unwrap(),
        1.5e-1,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_causal_attention_amp_bf16_tensor_core_handles_ragged_edges() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let time = 15;
    let head_dim = 17;
    let len = time * head_dim;
    let query_values = (0..len)
        .map(|index| ((index as f32 % 29.0) - 14.0) / 17.0)
        .collect::<Vec<_>>();
    let key_values = (0..len)
        .map(|index| ((index as f32 % 31.0) - 15.0) / 19.0)
        .collect::<Vec<_>>();
    let value_values = (0..len)
        .map(|index| ((index as f32 % 37.0) - 18.0) / 23.0)
        .collect::<Vec<_>>();

    let cpu_query_values = Tensor::from_f32(query_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_key_values = Tensor::from_f32(key_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_value_values = Tensor::from_f32(value_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_query = Tensor::from_f32(cpu_query_values, &[1, time, head_dim], true).unwrap();
    let cpu_key = Tensor::from_f32(cpu_key_values, &[1, time, head_dim], true).unwrap();
    let cpu_value = Tensor::from_f32(cpu_value_values, &[1, time, head_dim], true).unwrap();
    let expected = cpu_query
        .causal_self_attention(&cpu_key, &cpu_value, 1)
        .unwrap()
        .data_f32()
        .unwrap();
    cpu_query
        .causal_self_attention(&cpu_key, &cpu_value, 1)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let query = Tensor::from_f32(query_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let key = Tensor::from_f32(key_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let value = Tensor::from_f32(value_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();

    heirloom_kernels::cuda::reset_tensor_core_counters();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let output = query
        .causal_self_attention_amp_bf16_tensor_core(&key, &value, 1)
        .unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_close(&output.data_f32().unwrap(), &expected, 1.2e-1);

    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_attention_forward_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_qk_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_av_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 2);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert!(runtime.tensor_core_attention_padded_tiles > 0);
    assert!(runtime.tensor_core_attention_remainder_tiles > 0);
    assert_eq!(runtime.tensor_core_attention_scalar_fallbacks, 0);

    output.sum().unwrap().backward().unwrap();
    let counters = heirloom_kernels::cuda::tensor_core_counters();
    assert_eq!(counters.bf16_tensor_core_attention_backward_calls, 1);
    assert_eq!(
        counters.bf16_tensor_core_attention_score_grad_matmul_calls,
        1
    );
    assert_eq!(counters.bf16_tensor_core_attention_dq_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_dk_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_attention_dv_matmul_calls, 1);
    assert_eq!(counters.bf16_tensor_core_matmul_calls, 6);
    assert_eq!(counters.bf16_tensor_core_matmul_forward_calls, 2);
    assert_eq!(counters.bf16_tensor_core_matmul_backward_calls, 4);
    assert_eq!(counters.bf16_scalar_matmul_fallback_calls, 0);
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert!(runtime.tensor_core_attention_padded_tiles > 0);
    assert!(runtime.tensor_core_attention_remainder_tiles > 0);
    assert_eq!(runtime.tensor_core_attention_scalar_fallbacks, 0);

    let query_grad = query.grad_tensor().unwrap();
    let key_grad = key.grad_tensor().unwrap();
    let value_grad = value.grad_tensor().unwrap();
    assert_eq!(query_grad.device(), Device::Cuda(0));
    assert_eq!(key_grad.device(), Device::Cuda(0));
    assert_eq!(value_grad.device(), Device::Cuda(0));
    assert_eq!(query_grad.dtype(), DType::F32);
    assert_eq!(key_grad.dtype(), DType::F32);
    assert_eq!(value_grad.dtype(), DType::F32);
    assert_close(
        &query_grad.data_f32().unwrap(),
        &cpu_query.grad().unwrap(),
        2.5e-1,
    );
    assert_close(
        &key_grad.data_f32().unwrap(),
        &cpu_key.grad().unwrap(),
        2.5e-1,
    );
    assert_close(
        &value_grad.data_f32().unwrap(),
        &cpu_value.grad().unwrap(),
        2.5e-1,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_flash_bf16_attention_forward_matches_cpu_for_aligned_and_ragged_shapes() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    for (time, head_dim, tolerance, expect_ragged) in
        [(16, 16, 1.2e-1, false), (15, 17, 1.8e-1, true)]
    {
        let len = time * head_dim;
        let query_values = (0..len)
            .map(|index| ((index as f32 % 29.0) - 14.0) / 17.0)
            .collect::<Vec<_>>();
        let key_values = (0..len)
            .map(|index| ((index as f32 % 31.0) - 15.0) / 19.0)
            .collect::<Vec<_>>();
        let value_values = (0..len)
            .map(|index| ((index as f32 % 37.0) - 18.0) / 23.0)
            .collect::<Vec<_>>();
        let cpu_query = Tensor::from_f32(query_values.clone(), &[1, time, head_dim], false)
            .unwrap()
            .to_dtype(DType::BFloat16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let cpu_key = Tensor::from_f32(key_values.clone(), &[1, time, head_dim], false)
            .unwrap()
            .to_dtype(DType::BFloat16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let cpu_value = Tensor::from_f32(value_values.clone(), &[1, time, head_dim], false)
            .unwrap()
            .to_dtype(DType::BFloat16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = cpu_query
            .causal_self_attention(&cpu_key, &cpu_value, 1)
            .unwrap()
            .data_f32()
            .unwrap();
        let query = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &query_values).unwrap();
        let key = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &key_values).unwrap();
        let value = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &value_values).unwrap();
        let dims = heirloom_kernels::cuda::CausalAttentionDims {
            batch: 1,
            time,
            channels: head_dim,
            n_heads: 1,
        };

        heirloom_kernels::cuda::reset_tensor_core_counters();
        heirloom_kernels::cuda::reset_cuda_runtime_counters();
        let output = heirloom_kernels::cuda::causal_attention_bf16_flash_forward_f32_buffers(
            &query, &key, &value, dims,
        )
        .unwrap();
        assert_close(&output.to_f32().unwrap(), &expected, tolerance);
        let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
        assert_eq!(runtime.flash_bf16_attention_requested_calls, 1);
        assert_eq!(runtime.flash_bf16_attention_executed_calls, 1);
        assert_eq!(runtime.flash_bf16_attention_fallback_calls, 0);
        assert_eq!(runtime.flash_bf16_scalar_streaming_requested_calls, 1);
        assert_eq!(runtime.flash_bf16_scalar_streaming_executed_calls, 1);
        assert_eq!(runtime.flash_bf16_tensor_core_executed_calls, 0);
        assert_eq!(runtime.flash_bf16_tensor_core_qk_mma_tile_calls, 0);
        assert_eq!(runtime.flash_bf16_tensor_core_av_mma_tile_calls, 0);
        assert!(runtime.flash_bf16_attention_qk_tile_calls > 0);
        assert!(runtime.flash_bf16_attention_av_tile_calls > 0);
        assert!(runtime.flash_bf16_attention_causal_masked_tile_count > 0);
        let tensor_core = heirloom_kernels::cuda::tensor_core_counters();
        assert_eq!(tensor_core.bf16_tensor_core_attention_forward_calls, 0);
        assert_eq!(tensor_core.bf16_tensor_core_attention_qk_matmul_calls, 0);
        assert_eq!(tensor_core.bf16_tensor_core_attention_av_matmul_calls, 0);
        assert_eq!(tensor_core.bf16_tensor_core_matmul_forward_calls, 0);
        if expect_ragged {
            assert!(runtime.flash_bf16_attention_ragged_tile_count > 0);
        } else {
            assert_eq!(runtime.flash_bf16_attention_ragged_tile_count, 0);
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_flash_bf16_attention_env_guard_and_grad_fallback_policy() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let _flash = EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION", "1");
    let _tensor_core_flash = EnvVarGuard::remove("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE");
    let _require_removed = EnvVarGuard::remove("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION");
    let time = 16;
    let head_dim = 16;
    let len = time * head_dim;
    let query_values = (0..len)
        .map(|index| ((index as f32 % 29.0) - 14.0) / 17.0)
        .collect::<Vec<_>>();
    let key_values = (0..len)
        .map(|index| ((index as f32 % 31.0) - 15.0) / 19.0)
        .collect::<Vec<_>>();
    let value_values = (0..len)
        .map(|index| ((index as f32 % 37.0) - 18.0) / 23.0)
        .collect::<Vec<_>>();

    let query = Tensor::from_f32(query_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let key = Tensor::from_f32(key_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let value = Tensor::from_f32(value_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let output = query
        .causal_self_attention_amp_bf16_tensor_core(&key, &value, 1)
        .unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert_eq!(runtime.flash_bf16_attention_requested_calls, 1);
    assert_eq!(runtime.flash_bf16_attention_executed_calls, 1);
    assert_eq!(runtime.flash_bf16_attention_fallback_calls, 0);
    assert_eq!(runtime.flash_bf16_scalar_streaming_executed_calls, 1);
    assert_eq!(runtime.flash_bf16_tensor_core_executed_calls, 0);

    let query_grad = Tensor::from_f32(query_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let key_grad = Tensor::from_f32(key_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let value_grad = Tensor::from_f32(value_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    let output = query_grad
        .causal_self_attention_amp_bf16_tensor_core(&key_grad, &value_grad, 1)
        .unwrap();
    output.sum().unwrap().backward().unwrap();
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert_eq!(runtime.flash_bf16_attention_requested_calls, 1);
    assert_eq!(runtime.flash_bf16_attention_executed_calls, 0);
    assert_eq!(runtime.flash_bf16_attention_fallback_calls, 1);
    assert_eq!(runtime.flash_bf16_attention_scalar_fallback_calls, 1);
    assert_eq!(runtime.flash_bf16_scalar_streaming_executed_calls, 0);

    let _require = EnvVarGuard::set("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION", "1");
    let query = Tensor::from_f32(vec![0.25; len], &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let key = Tensor::from_f32(vec![0.125; len], &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let value = Tensor::from_f32(vec![0.5; len], &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let err = query
        .causal_self_attention_amp_bf16_tensor_core(&key, &value, 1)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION=1"));
    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert_eq!(runtime.flash_bf16_attention_hard_require_failures, 1);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_flash_bf16_tensor_core_forward_matches_cpu_and_counts_mma_tiles() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let _flash = EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION", "1");
    let _tensor_core_flash =
        EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE", "1");
    let _require = EnvVarGuard::set("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION", "1");
    for (time, head_dim, tolerance, expect_ragged) in
        [(16, 16, 2.5e-1, false), (15, 17, 3.0e-1, true)]
    {
        let len = time * head_dim;
        let query_values = (0..len)
            .map(|index| ((index as f32 % 29.0) - 14.0) / 17.0)
            .collect::<Vec<_>>();
        let key_values = (0..len)
            .map(|index| ((index as f32 % 31.0) - 15.0) / 19.0)
            .collect::<Vec<_>>();
        let value_values = (0..len)
            .map(|index| ((index as f32 % 37.0) - 18.0) / 23.0)
            .collect::<Vec<_>>();
        let expected = Tensor::from_f32(query_values.clone(), &[1, time, head_dim], false)
            .unwrap()
            .to_dtype(DType::BFloat16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .causal_self_attention(
                &Tensor::from_f32(key_values.clone(), &[1, time, head_dim], false)
                    .unwrap()
                    .to_dtype(DType::BFloat16)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap(),
                &Tensor::from_f32(value_values.clone(), &[1, time, head_dim], false)
                    .unwrap()
                    .to_dtype(DType::BFloat16)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap(),
                1,
            )
            .unwrap()
            .data_f32()
            .unwrap();
        let query = Tensor::from_f32(query_values, &[1, time, head_dim], false)
            .unwrap()
            .cuda(0)
            .unwrap();
        let key = Tensor::from_f32(key_values, &[1, time, head_dim], false)
            .unwrap()
            .cuda(0)
            .unwrap();
        let value = Tensor::from_f32(value_values, &[1, time, head_dim], false)
            .unwrap()
            .cuda(0)
            .unwrap();

        heirloom_kernels::cuda::reset_tensor_core_counters();
        heirloom_kernels::cuda::reset_cuda_runtime_counters();
        let output = query
            .causal_self_attention_amp_bf16_tensor_core(&key, &value, 1)
            .unwrap();
        assert_close(
            &output.cpu().unwrap().data_f32().unwrap(),
            &expected,
            tolerance,
        );
        let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
        assert_eq!(runtime.flash_bf16_attention_requested_calls, 1);
        assert_eq!(runtime.flash_bf16_attention_executed_calls, 1);
        assert_eq!(runtime.flash_bf16_attention_fallback_calls, 0);
        assert_eq!(runtime.flash_bf16_tensor_core_requested_calls, 1);
        assert_eq!(runtime.flash_bf16_tensor_core_executed_calls, 1);
        assert_eq!(runtime.flash_bf16_tensor_core_fallback_calls, 0);
        assert!(runtime.flash_bf16_tensor_core_qk_mma_tile_calls > 0);
        assert!(runtime.flash_bf16_tensor_core_av_mma_tile_calls > 0);
        assert_eq!(runtime.flash_bf16_scalar_streaming_executed_calls, 0);
        let tensor_core = heirloom_kernels::cuda::tensor_core_counters();
        assert!(tensor_core.bf16_tensor_core_attention_forward_calls > 0);
        assert!(tensor_core.bf16_tensor_core_attention_qk_matmul_calls > 0);
        assert!(tensor_core.bf16_tensor_core_attention_av_matmul_calls > 0);
        if expect_ragged {
            assert!(runtime.flash_bf16_tensor_core_ragged_tile_count > 0);
        } else {
            assert_eq!(runtime.flash_bf16_tensor_core_ragged_tile_count, 0);
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_flash_bf16_tensor_core_backward_compact_state_matches_cpu_and_counts_honestly() {
    require_cuda_hardware();
    require_bf16_tensor_cores();

    let _counter_guard = tensor_core_counter_guard();
    let _flash = EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION", "1");
    let _tensor_core_flash =
        EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE", "1");
    let _backward = EnvVarGuard::set("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD", "1");
    let _require = EnvVarGuard::set("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION", "1");
    let time = 16;
    let head_dim = 16;
    let len = time * head_dim;
    let query_values = (0..len)
        .map(|index| ((index as f32 % 17.0) - 8.0) / 19.0)
        .collect::<Vec<_>>();
    let key_values = (0..len)
        .map(|index| ((index as f32 % 19.0) - 9.0) / 23.0)
        .collect::<Vec<_>>();
    let value_values = (0..len)
        .map(|index| ((index as f32 % 23.0) - 11.0) / 29.0)
        .collect::<Vec<_>>();

    let cpu_query_values = Tensor::from_f32(query_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_key_values = Tensor::from_f32(key_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_value_values = Tensor::from_f32(value_values.clone(), &[1, time, head_dim], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();
    let cpu_query = Tensor::from_f32(cpu_query_values, &[1, time, head_dim], true).unwrap();
    let cpu_key = Tensor::from_f32(cpu_key_values, &[1, time, head_dim], true).unwrap();
    let cpu_value = Tensor::from_f32(cpu_value_values, &[1, time, head_dim], true).unwrap();
    cpu_query
        .causal_self_attention(&cpu_key, &cpu_value, 1)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let query = Tensor::from_f32(query_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let key = Tensor::from_f32(key_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let value = Tensor::from_f32(value_values, &[1, time, head_dim], true)
        .unwrap()
        .cuda(0)
        .unwrap();

    heirloom_kernels::cuda::reset_tensor_core_counters();
    heirloom_kernels::cuda::reset_cuda_runtime_counters();
    query
        .causal_self_attention_amp_bf16_tensor_core(&key, &value, 1)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let query_grad = query.grad_tensor().unwrap();
    let key_grad = key.grad_tensor().unwrap();
    let value_grad = value.grad_tensor().unwrap();
    assert_eq!(query_grad.device(), Device::Cuda(0));
    assert_eq!(key_grad.device(), Device::Cuda(0));
    assert_eq!(value_grad.device(), Device::Cuda(0));
    assert_close(
        &query_grad.data_f32().unwrap(),
        &cpu_query.grad().unwrap(),
        3.5e-1,
    );
    assert_close(
        &key_grad.data_f32().unwrap(),
        &cpu_key.grad().unwrap(),
        3.5e-1,
    );
    assert_close(
        &value_grad.data_f32().unwrap(),
        &cpu_value.grad().unwrap(),
        3.5e-1,
    );

    let runtime = heirloom_kernels::cuda::cuda_runtime_counters();
    assert_eq!(runtime.flash_bf16_tensor_core_executed_calls, 1);
    assert_eq!(runtime.flash_bf16_tensor_core_fallback_calls, 0);
    assert_eq!(runtime.flash_bf16_tensor_core_backward_requested_calls, 1);
    assert_eq!(runtime.flash_bf16_tensor_core_backward_executed_calls, 1);
    assert_eq!(runtime.flash_bf16_tensor_core_backward_fallback_calls, 0);
    assert!(runtime.flash_bf16_tensor_core_backward_row_dot_calls > 0);
    assert_eq!(runtime.flash_bf16_tensor_core_backward_scalar_tile_calls, 0);
    assert!(runtime.flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls > 0);
    assert!(runtime.flash_bf16_tensor_core_backward_dp_mma_tile_calls > 0);
    assert!(runtime.flash_bf16_tensor_core_backward_dq_mma_tile_calls > 0);
    assert!(runtime.flash_bf16_tensor_core_backward_dk_mma_tile_calls > 0);
    assert!(runtime.flash_bf16_tensor_core_backward_dv_mma_tile_calls > 0);
    assert_eq!(runtime.bf16_attention_materialized_reference_calls, 0);
    let tensor_core = heirloom_kernels::cuda::tensor_core_counters();
    assert!(tensor_core.bf16_tensor_core_attention_forward_calls > 0);
    assert!(tensor_core.bf16_tensor_core_attention_backward_calls > 0);
    assert!(tensor_core.bf16_tensor_core_attention_score_grad_matmul_calls > 0);
    assert!(tensor_core.bf16_tensor_core_attention_dq_matmul_calls > 0);
    assert!(tensor_core.bf16_tensor_core_attention_dk_matmul_calls > 0);
    assert!(tensor_core.bf16_tensor_core_attention_dv_matmul_calls > 0);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_bfloat16_cast_round_trip_and_backward_stay_on_device() {
    require_cuda_hardware();

    let values = vec![1.0, -2.5, 3.25, 0.33325195];
    let expected = Tensor::from_f32(values.clone(), &[4], false)
        .unwrap()
        .to_dtype(DType::BFloat16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .data_f32()
        .unwrap();

    let tensor = Tensor::from_f32(values, &[4], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let bf16 = tensor.to_dtype(DType::BFloat16).unwrap();
    bf16.retain_grad().unwrap();
    assert_eq!(bf16.device(), Device::Cuda(0));
    assert_eq!(bf16.dtype(), DType::BFloat16);

    let restored = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(restored.device(), Device::Cuda(0));
    assert_eq!(restored.dtype(), DType::F32);
    assert_close(&restored.data_f32().unwrap(), &expected, 1e-6);

    restored.sum().unwrap().backward().unwrap();
    let source_grad = tensor.grad_tensor().unwrap();
    assert_eq!(source_grad.device(), Device::Cuda(0));
    assert_eq!(source_grad.dtype(), DType::F32);
    assert_close(
        &source_grad.data_f32().unwrap(),
        &[1.0, 1.0, 1.0, 1.0],
        1e-6,
    );
    let bf16_grad = bf16.grad_tensor().unwrap();
    assert_eq!(bf16_grad.device(), Device::Cuda(0));
    assert_eq!(bf16_grad.dtype(), DType::F32);
    assert_close(&bf16_grad.data_f32().unwrap(), &[1.0, 1.0, 1.0, 1.0], 1e-6);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_add_and_relu_are_tensor_kernels() {
    require_cuda_hardware();

    let tensor = Tensor::from_f32(vec![1.0, 2.0], &[2], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let added = tensor.add(&tensor).unwrap();
    assert_eq!(added.device(), Device::Cuda(0));
    assert_eq!(added.dtype(), DType::F32);
    assert_eq!(added.data_f32().unwrap(), vec![2.0, 4.0]);

    let relu = Tensor::from_f32(vec![-3.0, 0.5, 4.0], &[3], false)
        .unwrap()
        .cuda(0)
        .unwrap()
        .relu()
        .unwrap();
    assert_eq!(relu.device(), Device::Cuda(0));
    assert_eq!(relu.data_f32().unwrap(), vec![0.0, 0.5, 4.0]);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_gelu_matches_cpu() {
    require_cuda_hardware();

    let values = vec![-3.0, -1.0, -0.25, 0.0, 0.5, 2.0];
    let cpu = Tensor::from_f32(values.clone(), &[2, 3], false)
        .unwrap()
        .gelu()
        .unwrap();
    let cuda = Tensor::from_f32(values, &[2, 3], false)
        .unwrap()
        .cuda(0)
        .unwrap()
        .gelu()
        .unwrap();
    assert_eq!(cuda.device(), Device::Cuda(0));
    assert_eq!(cuda.dtype(), DType::F32);
    assert_close(&cuda.data_f32().unwrap(), &cpu.data_f32().unwrap(), 3e-4);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_sub_mul_div_are_tensor_kernels() {
    require_cuda_hardware();

    let left = Tensor::from_f32(vec![6.0, 8.0], &[2], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let right = Tensor::from_f32(vec![3.0, 2.0], &[2], false)
        .unwrap()
        .cuda(0)
        .unwrap();

    let sub = left.sub(&right).unwrap();
    assert_eq!(sub.device(), Device::Cuda(0));
    assert_eq!(sub.data_f32().unwrap(), vec![3.0, 6.0]);

    let mul = left.mul(&right).unwrap();
    assert_eq!(mul.device(), Device::Cuda(0));
    assert_eq!(mul.data_f32().unwrap(), vec![18.0, 16.0]);

    let div = left.div(&right).unwrap();
    assert_eq!(div.device(), Device::Cuda(0));
    assert_eq!(div.data_f32().unwrap(), vec![2.0, 4.0]);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_sum_and_mean_are_tensor_kernels() {
    require_cuda_hardware();

    let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0], &[3], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let sum = tensor.sum().unwrap();
    assert_eq!(sum.device(), Device::Cuda(0));
    assert_eq!(sum.shape(), Vec::<usize>::new());
    assert_eq!(sum.data_f32().unwrap(), vec![6.0]);

    let mean = tensor.mean().unwrap();
    assert_eq!(mean.device(), Device::Cuda(0));
    assert_eq!(mean.data_f32().unwrap(), vec![2.0]);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_rank2_matmul_is_a_tensor_kernel() {
    require_cuda_hardware();

    let left = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let right = Tensor::from_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], false)
        .unwrap()
        .cuda(0)
        .unwrap();

    let output = left.matmul(&right).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), vec![2, 2]);
    assert_close(
        &output.data_f32().unwrap(),
        &[58.0, 64.0, 139.0, 154.0],
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn unsupported_cuda_batched_matmul_errors_without_cpu_fallback() {
    require_cuda_hardware();

    let left = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let right = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let err = left.matmul(&right).unwrap_err();
    match err {
        TensorError::Shape(message) => assert!(message.contains("only rank-2")),
        other => panic!("expected CUDA rank-2 shape error, got {other:?}"),
    }
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_add_backward_accumulates_on_device() {
    require_cuda_hardware();

    let tensor = Tensor::from_f32(vec![1.0, 2.0], &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let output = tensor.add(&tensor).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    output.backward_with_grad(vec![1.0, 2.0]).unwrap();

    let grad = tensor.grad_tensor().unwrap();
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.dtype(), DType::F32);
    assert_eq!(grad.data_f32().unwrap(), vec![2.0, 4.0]);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_relu_backward_masks_on_device() {
    require_cuda_hardware();

    let tensor = Tensor::from_f32(vec![-2.0, 0.0, 3.0], &[3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let output = tensor.relu().unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    output.backward_with_grad(vec![1.0, 2.0, 3.0]).unwrap();

    let grad = tensor.grad_tensor().unwrap();
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.dtype(), DType::F32);
    assert_eq!(grad.data_f32().unwrap(), vec![0.0, 0.0, 3.0]);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_gelu_backward_matches_cpu_and_stays_on_device() {
    require_cuda_hardware();

    let values = vec![-2.0, -0.75, 0.0, 0.5, 1.5, 3.0];
    let seed = vec![1.0, -0.5, 2.0, 0.25, -1.0, 1.75];

    let cpu = Tensor::from_f32(values.clone(), &[2, 3], true).unwrap();
    cpu.gelu()
        .unwrap()
        .backward_with_grad(seed.clone())
        .unwrap();
    let expected = cpu.grad().unwrap();

    let cuda = Tensor::from_f32(values, &[2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    cuda.gelu().unwrap().backward_with_grad(seed).unwrap();
    let grad = cuda.grad_tensor().unwrap();
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.dtype(), DType::F32);
    assert_close(&grad.data_f32().unwrap(), &expected, 3e-4);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_binary_backward_formulas_stay_on_device() {
    require_cuda_hardware();

    let x = Tensor::from_f32(vec![2.0, 4.0], &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let y = Tensor::from_f32(vec![3.0, 4.0], &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    x.mul(&y)
        .unwrap()
        .backward_with_grad(vec![1.0, 2.0])
        .unwrap();
    assert_eq!(x.grad_tensor().unwrap().device(), Device::Cuda(0));
    assert_eq!(x.grad_tensor().unwrap().data_f32().unwrap(), vec![3.0, 8.0]);
    assert_eq!(y.grad_tensor().unwrap().data_f32().unwrap(), vec![2.0, 8.0]);

    let x = Tensor::from_f32(vec![6.0, 8.0], &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let y = Tensor::from_f32(vec![3.0, 2.0], &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    x.div(&y)
        .unwrap()
        .backward_with_grad(vec![1.0, 2.0])
        .unwrap();
    assert_close(
        &x.grad_tensor().unwrap().data_f32().unwrap(),
        &[1.0 / 3.0, 1.0],
        1e-5,
    );
    assert_close(
        &y.grad_tensor().unwrap().data_f32().unwrap(),
        &[-2.0 / 3.0, -4.0],
        1e-5,
    );

    let x = Tensor::from_f32(vec![1.0, 2.0], &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let y = Tensor::from_f32(vec![3.0, 4.0], &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    x.sub(&y)
        .unwrap()
        .backward_with_grad(vec![1.0, 2.0])
        .unwrap();
    assert_eq!(x.grad_tensor().unwrap().data_f32().unwrap(), vec![1.0, 2.0]);
    assert_eq!(
        y.grad_tensor().unwrap().data_f32().unwrap(),
        vec![-1.0, -2.0]
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_reduction_backward_fills_device_gradients() {
    require_cuda_hardware();

    let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0], &[3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    tensor.sum().unwrap().backward_with_grad(vec![2.0]).unwrap();
    let grad = tensor.grad_tensor().unwrap();
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.data_f32().unwrap(), vec![2.0, 2.0, 2.0]);

    let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0], &[3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    tensor
        .mean()
        .unwrap()
        .backward_with_grad(vec![3.0])
        .unwrap();
    let grad = tensor.grad_tensor().unwrap();
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.data_f32().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_matmul_backward_formulas_stay_on_device() {
    require_cuda_hardware();

    let left = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let right = Tensor::from_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], true)
        .unwrap()
        .cuda(0)
        .unwrap();

    let output = left.matmul(&right).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    output.backward_with_grad(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let left_grad = left.grad_tensor().unwrap();
    assert_eq!(left_grad.device(), Device::Cuda(0));
    assert_eq!(left_grad.dtype(), DType::F32);
    assert_close(
        &left_grad.data_f32().unwrap(),
        &[23.0, 29.0, 35.0, 53.0, 67.0, 81.0],
        1e-5,
    );

    let right_grad = right.grad_tensor().unwrap();
    assert_eq!(right_grad.device(), Device::Cuda(0));
    assert_eq!(right_grad.dtype(), DType::F32);
    assert_close(
        &right_grad.data_f32().unwrap(),
        &[13.0, 18.0, 17.0, 24.0, 21.0, 30.0],
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_rank2_matmul_accepts_transposed_rhs_view() {
    require_cuda_hardware();

    let left_data = vec![1.0, -2.0, 3.0, 4.0, 0.5, -1.5];
    let weight_data = vec![2.0, -1.0, 0.25, -0.5, 3.0, 1.5];
    let cpu_left = Tensor::from_f32(left_data.clone(), &[2, 3], false).unwrap();
    let cpu_weight = Tensor::from_f32(weight_data.clone(), &[2, 3], false).unwrap();
    let expected = cpu_left
        .matmul(&cpu_weight.transpose().unwrap())
        .unwrap()
        .data_f32()
        .unwrap();

    let left = Tensor::from_f32(left_data, &[2, 3], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(weight_data, &[2, 3], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let output = left.matmul(&weight.transpose().unwrap()).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), vec![2, 2]);
    assert_close(&output.data_f32().unwrap(), &expected, 1e-5);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_transposed_rhs_matmul_backward_reaches_original_weight() {
    require_cuda_hardware();

    let left_data = vec![1.0, -2.0, 3.0, 4.0, 0.5, -1.5];
    let weight_data = vec![2.0, -1.0, 0.25, -0.5, 3.0, 1.5];
    let seed = vec![1.0, -0.5, 2.0, 0.25];

    let cpu_left = Tensor::from_f32(left_data.clone(), &[2, 3], true).unwrap();
    let cpu_weight = Tensor::from_f32(weight_data.clone(), &[2, 3], true).unwrap();
    cpu_left
        .matmul(&cpu_weight.transpose().unwrap())
        .unwrap()
        .backward_with_grad(seed.clone())
        .unwrap();

    let left = Tensor::from_f32(left_data, &[2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(weight_data, &[2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    left.matmul(&weight.transpose().unwrap())
        .unwrap()
        .backward_with_grad(seed)
        .unwrap();

    let left_grad = left.grad_tensor().unwrap();
    assert_eq!(left_grad.device(), Device::Cuda(0));
    assert_eq!(left_grad.dtype(), DType::F32);
    assert_close(
        &left_grad.data_f32().unwrap(),
        &cpu_left.grad().unwrap(),
        1e-5,
    );
    let weight_grad = weight.grad_tensor().unwrap();
    assert_eq!(weight_grad.device(), Device::Cuda(0));
    assert_eq!(weight_grad.dtype(), DType::F32);
    assert_close(
        &weight_grad.data_f32().unwrap(),
        &cpu_weight.grad().unwrap(),
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_bias_add_broadcast_forward_backward_stays_on_device() {
    require_cuda_hardware();

    let matrix_data = vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
    let bias_data = vec![0.25, -1.0, 2.0];
    let seed = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let cpu_matrix = Tensor::from_f32(matrix_data.clone(), &[2, 3], true).unwrap();
    let cpu_bias = Tensor::from_f32(bias_data.clone(), &[3], true).unwrap();
    let expected = cpu_matrix.add(&cpu_bias).unwrap().data_f32().unwrap();
    cpu_matrix
        .add(&cpu_bias)
        .unwrap()
        .backward_with_grad(seed.clone())
        .unwrap();

    let matrix = Tensor::from_f32(matrix_data, &[2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let bias = Tensor::from_f32(bias_data, &[3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let output = matrix.add(&bias).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), vec![2, 3]);
    assert_close(&output.data_f32().unwrap(), &expected, 1e-6);
    output.backward_with_grad(seed).unwrap();

    let matrix_grad = matrix.grad_tensor().unwrap();
    assert_eq!(matrix_grad.device(), Device::Cuda(0));
    assert_close(
        &matrix_grad.data_f32().unwrap(),
        &cpu_matrix.grad().unwrap(),
        1e-6,
    );
    let bias_grad = bias.grad_tensor().unwrap();
    assert_eq!(bias_grad.device(), Device::Cuda(0));
    assert_close(
        &bias_grad.data_f32().unwrap(),
        &cpu_bias.grad().unwrap(),
        1e-6,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_linear_projection_shape_path_matches_cpu_backward() {
    require_cuda_hardware();

    let input_data = vec![
        0.5, -1.0, 2.0, 1.5, 0.25, -0.75, -2.0, 1.0, 0.5, 3.0, -1.5, 0.0,
    ];
    let weight_data = vec![0.25, -0.5, 1.0, -1.5, 0.75, 0.5];
    let bias_data = vec![0.1, -0.2];
    let seed = vec![1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -0.25];

    let cpu_input = Tensor::from_f32(input_data.clone(), &[2, 2, 3], true).unwrap();
    let cpu_weight = Tensor::from_f32(weight_data.clone(), &[2, 3], true).unwrap();
    let cpu_bias = Tensor::from_f32(bias_data.clone(), &[2], true).unwrap();
    let cpu_flat = cpu_input.reshape(&[4, 3]).unwrap();
    let cpu_output = cpu_flat
        .matmul(&cpu_weight.transpose().unwrap())
        .unwrap()
        .add(&cpu_bias)
        .unwrap()
        .reshape(&[2, 2, 2])
        .unwrap();
    let expected_output = cpu_output.data_f32().unwrap();
    cpu_output.backward_with_grad(seed.clone()).unwrap();

    let input = Tensor::from_f32(input_data, &[2, 2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(weight_data, &[2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let bias = Tensor::from_f32(bias_data, &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let flat = input.reshape(&[4, 3]).unwrap();
    let output = flat
        .matmul(&weight.transpose().unwrap())
        .unwrap()
        .add(&bias)
        .unwrap()
        .reshape(&[2, 2, 2])
        .unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), vec![2, 2, 2]);
    assert_close(&output.data_f32().unwrap(), &expected_output, 1e-5);
    output.backward_with_grad(seed).unwrap();

    let input_grad = input.grad_tensor().unwrap();
    assert_eq!(input_grad.device(), Device::Cuda(0));
    assert_close(
        &input_grad.data_f32().unwrap(),
        &cpu_input.grad().unwrap(),
        1e-5,
    );
    let weight_grad = weight.grad_tensor().unwrap();
    assert_eq!(weight_grad.device(), Device::Cuda(0));
    assert_close(
        &weight_grad.data_f32().unwrap(),
        &cpu_weight.grad().unwrap(),
        1e-5,
    );
    let bias_grad = bias.grad_tensor().unwrap();
    assert_eq!(bias_grad.device(), Device::Cuda(0));
    assert_close(
        &bias_grad.data_f32().unwrap(),
        &cpu_bias.grad().unwrap(),
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_embedding_forward_gathers_rows_on_device() {
    require_cuda_hardware();

    let indices = Tensor::from_i64(vec![0, 1, 0], &[3], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)
        .unwrap()
        .cuda(0)
        .unwrap();

    let output = indices.embedding(&weight).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), vec![3, 2]);
    assert_close(
        &output.data_f32().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
        1e-6,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_embedding_backward_scatter_adds_repeated_rows_on_device() {
    require_cuda_hardware();

    let indices = Tensor::from_i64(vec![0, 1, 0], &[3], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)
        .unwrap()
        .cuda(0)
        .unwrap();

    indices
        .embedding(&weight)
        .unwrap()
        .backward_with_grad(vec![2.0, 2.0, 1.0, 1.0, 3.0, 3.0])
        .unwrap();

    let grad = weight.grad_tensor().unwrap();
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.dtype(), DType::F32);
    assert_close(&grad.data_f32().unwrap(), &[5.0, 5.0, 1.0, 1.0], 1e-6);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_embedding_out_of_range_errors_without_cpu_fallback() {
    require_cuda_hardware();

    let indices = Tensor::from_i64(vec![2], &[1], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)
        .unwrap()
        .cuda(0)
        .unwrap();

    let err = indices.embedding(&weight).unwrap_err();
    match err {
        TensorError::Device(message) => assert!(message.contains("out-of-range")),
        other => panic!("expected CUDA device index error, got {other:?}"),
    }
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_layer_norm_forward_matches_cpu() {
    require_cuda_hardware();

    let input_data = vec![1.0, 2.0, 4.0, -1.0, 0.5, 3.0];
    let weight_data = vec![1.0, 1.5, 0.5];
    let bias_data = vec![0.1, -0.2, 0.3];
    let cpu = Tensor::from_f32(input_data.clone(), &[2, 3], false)
        .unwrap()
        .layer_norm_last_dim(
            &Tensor::from_f32(weight_data.clone(), &[3], false).unwrap(),
            &Tensor::from_f32(bias_data.clone(), &[3], false).unwrap(),
            1e-5,
        )
        .unwrap()
        .data_f32()
        .unwrap();

    let input = Tensor::from_f32(input_data, &[2, 3], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(weight_data, &[3], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let bias = Tensor::from_f32(bias_data, &[3], false)
        .unwrap()
        .cuda(0)
        .unwrap();

    let output = input.layer_norm_last_dim(&weight, &bias, 1e-5).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), vec![2, 3]);
    assert_close(&output.data_f32().unwrap(), &cpu, 1e-4);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_layer_norm_backward_matches_cpu_and_stays_on_device() {
    require_cuda_hardware();

    let input_data = vec![1.0, 2.0, 4.0, -1.0, 0.5, 3.0];
    let weight_data = vec![1.0, 1.5, 0.5];
    let bias_data = vec![0.1, -0.2, 0.3];
    let seed = vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75];

    let cpu_input = Tensor::from_f32(input_data.clone(), &[2, 3], true).unwrap();
    let cpu_weight = Tensor::from_f32(weight_data.clone(), &[3], true).unwrap();
    let cpu_bias = Tensor::from_f32(bias_data.clone(), &[3], true).unwrap();
    cpu_input
        .layer_norm_last_dim(&cpu_weight, &cpu_bias, 1e-5)
        .unwrap()
        .backward_with_grad(seed.clone())
        .unwrap();

    let input = Tensor::from_f32(input_data, &[2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(weight_data, &[3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let bias = Tensor::from_f32(bias_data, &[3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    input
        .layer_norm_last_dim(&weight, &bias, 1e-5)
        .unwrap()
        .backward_with_grad(seed)
        .unwrap();

    let input_grad = input.grad_tensor().unwrap();
    assert_eq!(input_grad.device(), Device::Cuda(0));
    assert_close(
        &input_grad.data_f32().unwrap(),
        &cpu_input.grad().unwrap(),
        2e-4,
    );
    let weight_grad = weight.grad_tensor().unwrap();
    assert_eq!(weight_grad.device(), Device::Cuda(0));
    assert_close(
        &weight_grad.data_f32().unwrap(),
        &cpu_weight.grad().unwrap(),
        2e-4,
    );
    let bias_grad = bias.grad_tensor().unwrap();
    assert_eq!(bias_grad.device(), Device::Cuda(0));
    assert_close(
        &bias_grad.data_f32().unwrap(),
        &cpu_bias.grad().unwrap(),
        2e-4,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_cross_entropy_forward_matches_cpu() {
    require_cuda_hardware();

    let logits_data = vec![1.0, 2.0, 0.0, -1.0, 0.5, 3.0];
    let targets = [1, 2];
    let cpu = Tensor::from_f32(logits_data.clone(), &[2, 3], false)
        .unwrap()
        .cross_entropy_for_logits(&targets)
        .unwrap()
        .data_f32()
        .unwrap();
    let logits = Tensor::from_f32(logits_data, &[2, 3], false)
        .unwrap()
        .cuda(0)
        .unwrap();

    let loss = logits.cross_entropy_for_logits(&targets).unwrap();
    assert_eq!(loss.device(), Device::Cuda(0));
    assert_eq!(loss.shape(), Vec::<usize>::new());
    assert_close(&loss.data_f32().unwrap(), &cpu, 2e-3);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_cross_entropy_backward_matches_cpu_and_stays_on_device() {
    require_cuda_hardware();

    let logits_data = vec![1.0, 2.0, 0.0, -1.0, 0.5, 3.0];
    let targets = [1, 2];
    let cpu_logits = Tensor::from_f32(logits_data.clone(), &[2, 3], true).unwrap();
    cpu_logits
        .cross_entropy_for_logits(&targets)
        .unwrap()
        .backward()
        .unwrap();

    let logits = Tensor::from_f32(logits_data, &[2, 3], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    logits
        .cross_entropy_for_logits(&targets)
        .unwrap()
        .backward()
        .unwrap();

    let grad = logits.grad_tensor().unwrap();
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.dtype(), DType::F32);
    assert_close(&grad.data_f32().unwrap(), &cpu_logits.grad().unwrap(), 2e-3);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_causal_attention_forward_matches_cpu() {
    require_cuda_hardware();

    let query_data = vec![
        0.2, -0.1, 0.4, 0.7, 0.0, 0.3, -0.5, 0.6, 0.8, -0.4, 0.1, -0.2,
    ];
    let key_data = vec![
        -0.3, 0.5, 0.2, -0.6, 0.9, -0.2, 0.0, 0.4, -0.1, 0.3, 0.7, -0.8,
    ];
    let value_data = vec![
        0.6, -0.7, 0.1, 0.2, -0.4, 0.5, 0.8, -0.1, 0.3, 0.0, -0.2, 0.9,
    ];
    let cpu = Tensor::from_f32(query_data.clone(), &[1, 3, 4], false)
        .unwrap()
        .causal_self_attention(
            &Tensor::from_f32(key_data.clone(), &[1, 3, 4], false).unwrap(),
            &Tensor::from_f32(value_data.clone(), &[1, 3, 4], false).unwrap(),
            2,
        )
        .unwrap()
        .data_f32()
        .unwrap();

    let query = Tensor::from_f32(query_data, &[1, 3, 4], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let key = Tensor::from_f32(key_data, &[1, 3, 4], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let value = Tensor::from_f32(value_data, &[1, 3, 4], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let output = query.causal_self_attention(&key, &value, 2).unwrap();
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), vec![1, 3, 4]);
    assert_close(&output.data_f32().unwrap(), &cpu, 3e-3);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_causal_attention_backward_matches_cpu_and_stays_on_device() {
    require_cuda_hardware();

    let query_data = vec![
        0.2, -0.1, 0.4, 0.7, 0.0, 0.3, -0.5, 0.6, 0.8, -0.4, 0.1, -0.2,
    ];
    let key_data = vec![
        -0.3, 0.5, 0.2, -0.6, 0.9, -0.2, 0.0, 0.4, -0.1, 0.3, 0.7, -0.8,
    ];
    let value_data = vec![
        0.6, -0.7, 0.1, 0.2, -0.4, 0.5, 0.8, -0.1, 0.3, 0.0, -0.2, 0.9,
    ];
    let grad_output = vec![
        1.0, -0.5, 0.25, 0.75, -1.0, 0.4, 0.6, -0.3, 0.2, 0.9, -0.8, 0.1,
    ];

    let cpu_query = Tensor::from_f32(query_data.clone(), &[1, 3, 4], true).unwrap();
    let cpu_key = Tensor::from_f32(key_data.clone(), &[1, 3, 4], true).unwrap();
    let cpu_value = Tensor::from_f32(value_data.clone(), &[1, 3, 4], true).unwrap();
    cpu_query
        .causal_self_attention(&cpu_key, &cpu_value, 2)
        .unwrap()
        .backward_with_grad(grad_output.clone())
        .unwrap();

    let query = Tensor::from_f32(query_data, &[1, 3, 4], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let key = Tensor::from_f32(key_data, &[1, 3, 4], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let value = Tensor::from_f32(value_data, &[1, 3, 4], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    query
        .causal_self_attention(&key, &value, 2)
        .unwrap()
        .backward_with_grad(grad_output)
        .unwrap();

    let query_grad = query.grad_tensor().unwrap();
    let key_grad = key.grad_tensor().unwrap();
    let value_grad = value.grad_tensor().unwrap();
    assert_eq!(query_grad.device(), Device::Cuda(0));
    assert_eq!(key_grad.device(), Device::Cuda(0));
    assert_eq!(value_grad.device(), Device::Cuda(0));
    assert_eq!(query_grad.dtype(), DType::F32);
    assert_eq!(key_grad.dtype(), DType::F32);
    assert_eq!(value_grad.dtype(), DType::F32);
    assert_close(
        &query_grad.data_f32().unwrap(),
        &cpu_query.grad().unwrap(),
        4e-3,
    );
    assert_close(
        &key_grad.data_f32().unwrap(),
        &cpu_key.grad().unwrap(),
        4e-3,
    );
    assert_close(
        &value_grad.data_f32().unwrap(),
        &cpu_value.grad().unwrap(),
        4e-3,
    );
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_sgd_update_runs_on_device_after_backward() {
    require_cuda_hardware();

    let parameter = Tensor::from_f32(vec![1.0, -2.0], &[2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let loss = parameter.mul(&parameter).unwrap().mean().unwrap();
    assert_eq!(loss.device(), Device::Cuda(0));
    loss.backward().unwrap();

    let grad = parameter.grad_tensor().unwrap();
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.dtype(), DType::F32);
    assert_close(&grad.data_f32().unwrap(), &[1.0, -2.0], 1e-6);

    parameter.apply_sgd(0.1).unwrap();
    assert_close(&parameter.data_f32().unwrap(), &[0.9, -1.8], 1e-6);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_adamw_update_matches_cpu_and_exports_state() {
    require_cuda_hardware();

    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let cpu_param = Tensor::from_f32(initial.clone(), &[2, 2], true).unwrap();
    let mut cpu_opt = AdamW::new(vec![cpu_param.clone()], 0.05)
        .unwrap()
        .with_weight_decay(0.01)
        .unwrap()
        .with_clip_norm(Some(1.0))
        .unwrap();
    cpu_param
        .mul(&cpu_param)
        .unwrap()
        .mean()
        .unwrap()
        .backward()
        .unwrap();
    cpu_opt.step_mut().unwrap();

    let cuda_param = Tensor::from_f32(initial, &[2, 2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let mut cuda_opt = AdamW::new(vec![cuda_param.clone()], 0.05)
        .unwrap()
        .with_weight_decay(0.01)
        .unwrap()
        .with_clip_norm(Some(1.0))
        .unwrap();
    cuda_param
        .mul(&cuda_param)
        .unwrap()
        .mean()
        .unwrap()
        .backward()
        .unwrap();
    cuda_opt.step_mut().unwrap();

    assert_eq!(cuda_param.device(), Device::Cuda(0));
    assert_close(
        &cuda_param.data_f32().unwrap(),
        &cpu_param.data_f32().unwrap(),
        2e-4,
    );

    let cpu_state = cpu_opt.state();
    let cuda_state = cuda_opt.try_state().unwrap();
    assert_eq!(cuda_state.step, 1);
    assert_eq!(cuda_state.m.len(), 1);
    assert_eq!(cuda_state.v.len(), 1);
    assert_close_f64(&cuda_state.m[0], &cpu_state.m[0], 2e-4);
    assert_close_f64(&cuda_state.v[0], &cpu_state.v[0], 2e-4);
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_adamw_trains_linear_projection_loss_down() {
    require_cuda_hardware();

    let input = Tensor::from_f32(
        vec![
            -1.0, 0.5, 0.0, 1.0, 1.5, -0.5, 2.0, 1.0, -2.0, -1.0, 0.75, 1.25,
        ],
        &[6, 2],
        false,
    )
    .unwrap()
    .cuda(0)
    .unwrap();
    let target_values = vec![-2.0, -0.5, 4.0, 3.5, -3.5, -0.75];
    let target = Tensor::from_f32(target_values, &[6, 1], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let weight = Tensor::from_f32(vec![0.0, 0.0], &[1, 2], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let bias = Tensor::from_f32(vec![0.0], &[1], true)
        .unwrap()
        .cuda(0)
        .unwrap();
    let mut optimizer = AdamW::new(vec![weight.clone(), bias.clone()], 0.08)
        .unwrap()
        .with_clip_norm(Some(10.0))
        .unwrap();

    let mut initial_loss = None;
    for step in 0..80 {
        optimizer.zero_grad();
        let prediction = input
            .matmul(&weight.transpose().unwrap())
            .unwrap()
            .add(&bias)
            .unwrap();
        let loss = prediction.mse_loss(&target).unwrap();
        let value = loss.data_f32().unwrap()[0];
        if step == 0 {
            initial_loss = Some(value);
        }
        loss.backward().unwrap();
        optimizer.step_mut().unwrap();
    }

    optimizer.zero_grad();
    let prediction = input
        .matmul(&weight.transpose().unwrap())
        .unwrap()
        .add(&bias)
        .unwrap();
    let final_loss = prediction.mse_loss(&target).unwrap().data_f32().unwrap()[0];
    let initial_loss = initial_loss.unwrap();
    assert!(
        final_loss < initial_loss * 0.25,
        "expected CUDA AdamW training to reduce loss substantially, initial={initial_loss} final={final_loss}"
    );
    assert!(weight.grad_tensor().is_none());
    assert!(bias.grad_tensor().is_none());
    assert_eq!(weight.device(), Device::Cuda(0));
    assert_eq!(bias.device(), Device::Cuda(0));
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_tiny_transformer_bf16_activation_step_stays_on_device() {
    require_cuda_hardware();

    let config = TinyTransformerConfig {
        vocab_size: 16,
        block_size: 4,
        d_model: 4,
        n_heads: 2,
        ff_hidden: 8,
    };
    let mut rng = HeirloomRng::new(29);
    let model = TinyTransformerLm::new(config, &mut rng)
        .unwrap()
        .to_device(Device::Cuda(0))
        .unwrap();
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 4, 3, 2, 1], &[2, 4], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let target = Tensor::from_i64(vec![2, 3, 4, 5, 3, 2, 1, 0], &[2, 4], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let mut optimizer = AdamW::new(model.parameters(), 0.01)
        .unwrap()
        .with_clip_norm(Some(1.0))
        .unwrap();

    optimizer.zero_grad();
    let loss = model.loss_bf16_activations(&input, &target).unwrap();
    assert_eq!(loss.device(), Device::Cuda(0));
    assert_eq!(loss.dtype(), DType::F32);
    assert!(loss.data_f32().unwrap()[0].is_finite());
    loss.backward().unwrap();
    assert!(model.parameters().iter().any(|parameter| {
        parameter
            .grad_tensor()
            .is_some_and(|grad| grad.device() == Device::Cuda(0) && grad.dtype() == DType::F32)
    }));
    optimizer.step_mut().unwrap();
    optimizer.zero_grad();

    let next_loss = model
        .loss_bf16_activations(&input, &target)
        .unwrap()
        .data_f32()
        .unwrap()[0];
    assert!(next_loss.is_finite());
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_tiny_transformer_lm_step_and_checkpoint_stay_on_device() {
    require_cuda_hardware();

    let config = TinyTransformerConfig {
        vocab_size: 16,
        block_size: 4,
        d_model: 4,
        n_heads: 2,
        ff_hidden: 8,
    };
    let mut rng = HeirloomRng::new(17);
    let model = TinyTransformerLm::new(config, &mut rng)
        .unwrap()
        .to_device(Device::Cuda(0))
        .unwrap();
    assert_eq!(model.device(), Device::Cuda(0));
    assert!(model
        .parameters()
        .iter()
        .all(|parameter| parameter.device() == Device::Cuda(0)));

    let input = Tensor::from_i64(vec![1, 2, 3, 4, 4, 3, 2, 1], &[2, 4], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let target = Tensor::from_i64(vec![2, 3, 4, 5, 3, 2, 1, 0], &[2, 4], false)
        .unwrap()
        .cuda(0)
        .unwrap();
    let mut optimizer = AdamW::new(model.parameters(), 0.01)
        .unwrap()
        .with_clip_norm(Some(1.0))
        .unwrap();

    optimizer.zero_grad();
    let loss = model.loss(&input, &target).unwrap();
    assert_eq!(loss.device(), Device::Cuda(0));
    assert_eq!(loss.dtype(), DType::F32);
    let initial_loss = loss.data_f32().unwrap()[0];
    assert!(initial_loss.is_finite());
    loss.backward().unwrap();
    assert!(model.parameters().iter().any(|parameter| {
        parameter
            .grad_tensor()
            .is_some_and(|grad| grad.device() == Device::Cuda(0))
    }));
    optimizer.step_mut().unwrap();
    optimizer.zero_grad();

    let next_loss = model.loss(&input, &target).unwrap().data_f32().unwrap()[0];
    assert!(next_loss.is_finite());

    let dir = temp_dir("cuda_lm_checkpoint");
    let tokenizer = BpeTokenizer::train("one two three four five", 280).unwrap();
    save_lm_checkpoint_with_dataset_state(
        &dir,
        &model,
        &optimizer,
        &tokenizer,
        TokenDatasetState::from_seed(123),
        Some("manifest.json".to_string()),
    )
    .unwrap();
    let loaded = load_lm_checkpoint_on_device(&dir, Device::Cuda(0)).unwrap();
    assert_eq!(loaded.model.device(), Device::Cuda(0));
    assert_eq!(loaded.metadata.step, optimizer.step_index());
    let loaded_loss = loaded.model.loss(&input, &target).unwrap();
    assert_eq!(loaded_loss.device(), Device::Cuda(0));
    assert!(loaded_loss.data_f32().unwrap()[0].is_finite());

    let metrics = loaded
        .model
        .evaluate_token_loss(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 2, Some(1))
        .unwrap();
    assert_eq!(metrics.batches, 1);
    assert!(metrics.loss.is_finite());
    assert!(metrics.perplexity.is_finite());

    let mut generation_rng = HeirloomRng::new(99);
    let generated = loaded
        .model
        .generate(
            &[1, 2],
            &GenerationOptions {
                max_new_tokens: 2,
                eos_id: 0,
                temperature: 0.8,
                top_k: Some(4),
                top_p: Some(0.95),
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
            },
            &mut generation_rng,
        )
        .unwrap();
    assert!(generated.tokens.len() >= 2);
    assert!(generated.new_token_count <= 2);
    fs::remove_dir_all(dir).unwrap();
}

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "heirloom_{name}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
