#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-all}"

run_cuda() {
  HEIRLOOM_CUDA_TESTS=1 cargo test -p heirloom-kernels \
    cuda::tests::cuda_smoke_f32_executes_on_device -- --ignored --exact --test-threads=1
  HEIRLOOM_CUDA_TESTS=1 cargo test -p heirloom --test cuda_storage -- --ignored --test-threads=1
  HEIRLOOM_CUDA_TESTS=1 cargo test -p heirloom --test memory_transformer -- --ignored --test-threads=1
}

run_nccl() {
  HEIRLOOM_NCCL_TESTS=1 cargo test -p heirloom --test nccl -- --ignored --test-threads=1
}

case "$mode" in
  cuda)
    run_cuda
    ;;
  nccl)
    run_nccl
    ;;
  all)
    run_cuda
    run_nccl
    ;;
  *)
    echo "usage: $0 [cuda|nccl|all]" >&2
    exit 2
    ;;
esac
