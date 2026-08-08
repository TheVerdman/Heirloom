const CUDA_STORAGE_TESTS: &str = include_str!("cuda_storage.rs");
const MEMORY_TRANSFORMER_TESTS: &str = include_str!("memory_transformer.rs");
const NCCL_TESTS: &str = include_str!("nccl.rs");
const KERNEL_CUDA_TESTS: &str = include_str!("../heirloom-kernels/src/cuda.rs");

#[test]
fn hardware_tests_are_explicitly_ignored() {
    assert_hardware_tests_are_ignored(CUDA_STORAGE_TESTS, "require_cuda_hardware();", "CUDA");
    assert_hardware_tests_are_ignored(MEMORY_TRANSFORMER_TESTS, "require_cuda_hardware();", "CUDA");
    assert_hardware_tests_are_ignored(NCCL_TESTS, "require_nccl_hardware();", "NCCL");
    assert_hardware_tests_are_ignored(KERNEL_CUDA_TESTS, "smoke_f32(0, 64)", "kernel CUDA");
}

#[test]
fn hardware_tests_do_not_use_successful_early_returns_as_skips() {
    for source in [
        CUDA_STORAGE_TESTS,
        MEMORY_TRANSFORMER_TESTS,
        NCCL_TESTS,
        KERNEL_CUDA_TESTS,
    ] {
        assert!(
            !source.contains("!= Some(\"1\")"),
            "hardware tests must be ignored, not return success when their environment variable is absent"
        );
    }
    assert!(
        !CUDA_STORAGE_TESTS
            .contains("if !heirloom_kernels::cuda::device_supports_bf16_tensor_cores(0).unwrap()"),
        "BF16 hardware tests must fail when the required device capability is absent"
    );
}

fn assert_hardware_tests_are_ignored(source: &str, marker: &str, kind: &str) {
    let mut classified = 0;
    for test in source.split("#[test]").skip(1) {
        if !test.contains(marker) {
            continue;
        }
        classified += 1;
        assert!(
            test.contains("#[ignore ="),
            "{kind} test is missing an explicit ignore classification:\n{test}"
        );
    }
    assert!(
        classified > 0,
        "expected at least one classified {kind} test"
    );
}
