# GPU validation evidence

This is a concise record of one retained hardware run. It is historical evidence, not a claim that GPU tests ran on the current commit or during the August 2026 committee-readiness pass.

## Environment

- Date: 2026-06-06
- Hardware: 4 × NVIDIA A100-SXM4-80GB, compute capability 8.0, same-node NV12 links
- NVIDIA driver: 535.288.01
- Reported CUDA compatibility: 12.2
- NCCL: 2.21.5 (`nccl_version=22105`)
- Rust: not captured in the retained summary. The run predates the repository's Rust 1.95.0 pin, so it must not be presented as validation of that pinned toolchain.
- Cloud provenance: a successful private Vertex custom job was retained; project, bucket, and account identifiers are intentionally omitted because they are not part of public reproducibility.

## Commands that executed

The CUDA integration suite used the then-current opt-in command:

```bash
HEIRLOOM_CUDA_TESTS=1 \
  cargo test --workspace --test cuda_storage -- --nocapture
```

The four-rank NCCL preflight used:

```bash
cargo run --bin heirloom -- gpu nccl-probe \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --len 1024 \
  --probe-kind all-reduce \
  --timeout-secs 120 \
  --rank-start-timeout-secs 10 \
  --kill-grace-secs 5 \
  --report nccl-probe.json
```

The bounded public-data reference tier was launched with the repository wrapper and its `full` profile:

```bash
HEIRLOOM_TINYSTORIES_CUDA_MODE=full \
HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl \
HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16 \
  ./scripts/run_tinystories_cuda_reference.sh runs/tinystories-cuda-reference
```

For the current repository, the classified hardware-test entry points are:

```bash
./scripts/test_gpu.sh cuda
./scripts/test_gpu.sh nccl
```

## What executed and passed

- CUDA device discovery, peer-access topology, per-device smoke checks, and BF16 Tensor Core probes ran on all four GPUs.
- The CUDA storage/autograd integration suite passed on the A100 worker. This historical suite predates the current explicit `#[ignore]` classification and therefore has a different test count.
- Four-rank NCCL all-reduce passed: expected sum `10.0`, maximum absolute error `0.0`, four calls, and 16,384 aggregate bytes.
- A 500-step TinyStories-valid DDP training run used per-rank batch 4, global batch 16, block 64, `d_model=64`, four heads, and `ff_hidden=256`.
- Training loss fell from `7.063565` to `3.666313`; held-out loss was `3.581789` over 51,200 tokens.
- Training recorded 44,000 gradient all-reduces and 1,490,432,000 all-reduce bytes; per-step and final parameter-checksum drift were `0.0`.
- BF16 Tensor Core counters were positive and scalar matmul fallback count was `0` for this run.

## Limits of this evidence

This run demonstrates a bounded, same-node 4 × A100 CUDA/NCCL path on an older source snapshot. It does not establish current-commit GPU correctness, the new Rust 1.95.0 gate on GPU, multi-node scaling, production reliability, or broad model-quality claims. No GPU was available and no billable cloud resource was created for the committee-readiness pass; fresh hardware validation remains a follow-up item.
