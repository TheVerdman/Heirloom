fn require_nccl_hardware() {
    assert_eq!(
        std::env::var("HEIRLOOM_NCCL_TESTS").ok().as_deref(),
        Some("1"),
        "ignored NCCL tests must be run through scripts/test_gpu.sh nccl"
    );
    assert!(
        heirloom_kernels::cuda::is_available(),
        "HEIRLOOM_NCCL_TESTS=1 but the CUDA Driver API is unavailable"
    );
}

#[test]
#[ignore = "requires CUDA hardware and NCCL; run scripts/test_gpu.sh nccl"]
fn nccl_single_rank_all_reduce_f32() {
    require_nccl_hardware();

    let unique_id = heirloom_kernels::cuda::nccl_unique_id().unwrap();
    let mut communicator =
        heirloom_kernels::cuda::NcclCommunicator::init_rank(0, 0, 1, unique_id).unwrap();
    let buffer = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &[1.0, 2.0, -3.0, 4.5]).unwrap();
    let stats = communicator.all_reduce_sum_in_place_f32(&buffer).unwrap();

    assert_eq!(stats.calls, 1);
    assert_eq!(stats.bytes, 4 * std::mem::size_of::<f32>());
    assert_close(&buffer.to_f32().unwrap(), &[1.0, 2.0, -3.0, 4.5], 1e-6);
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}
