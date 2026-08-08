# GPU validation evidence

This page keeps current-code single-GPU evidence separate from an older multi-GPU run. Neither result is generalized beyond the exact command, source revision, and hardware recorded here.

## Current-code validation — August 8, 2026

Vertex custom job `4261257102716043264` reached `JOB_STATE_SUCCEEDED` with no retry or error. It started at `2026-08-08T22:12:24Z`, ended at `2026-08-08T22:17:26Z`, and validated source revision `74307a195a9bba0ad53117efd160df77801445da`. The later evidence-publication commit changes Markdown only.

### Environment

- Hardware: 1 × NVIDIA A100-SXM4-80GB, compute capability 8.0
- Worker: one `a2-ultragpu-1g` replica with a 300 GB SSD
- NVIDIA driver: 535.309.01
- Reported CUDA compatibility / CUDA driver API: 12.2 / 12020
- NCCL: 2.21.5+cuda12.4
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Cargo: `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`
- Executor image tag: `pytorch-gpu.2-4.py310:latest`; the mutable tag's digest was not captured

### Exact launch

Project and bucket identifiers are replaced with caller-owned placeholders; every validation control below is the value used for the successful run.

```bash
PROJECT_ID=<existing-project> \
BUCKET=gs://<existing-artifact-bucket> \
REGION=us-central1 \
HEIRLOOM_VERTEX_VALIDATE_MODE=quick \
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=1 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_TESTS=1 \
HEIRLOOM_RUN_TENSOR_CORE_MICROBENCH=1 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ITERATIONS=16 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_WARMUP=4 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_SECTIONS=gemm,attention \
STREAM_LOGS=true \
  zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The launcher refused dirty input, packaged exact `HEAD` with `git archive`, and wrote the source revision and repository-resolved toolchain into `summary.json`.

### What executed

- CUDA device discovery and topology passed for the one visible A100.
- CUDA Driver API add/ReLU smoke passed over 4,096 elements; both maximum absolute errors were `0.0`.
- The SM80 BF16 `mma.sync` probe produced the expected dot product `16.0` across 32 samples with maximum absolute error `0.0`.
- `./scripts/test_gpu.sh nccl` passed 1/1 single-rank f32 all-reduce test. This proves the wrapper can load NCCL, create rank 0 of 1, and execute a collective; it is not multi-rank evidence.
- `./scripts/test_gpu.sh cuda` passed 61/61 hardware tests: 1 kernel smoke unit test, 46 tensor/storage/autograd tests, and 14 memory-transformer tests. The memory suite included exact product-key forward and backward parity plus the end-to-end CUDA memory-transformer path.
- The worker quick gate passed `cargo fmt --all --check`, `cargo test --workspace --exclude heirloom-python`, and Clippy over all non-Python workspace targets with `-D warnings`.

### Existing microbenchmarks

Each measurement used 4 warmups and 16 measured iterations. These are kernel/runtime microbenchmarks, not end-to-end model training throughput.

| Section | Shape and path | Measured elapsed | Throughput / comparison | Maximum absolute error |
| --- | --- | ---: | ---: | ---: |
| GEMM | `m=4096, k=1024, n=4096`; `cp_async_double_buffered_ldmatrix_a_mma` | 38.6287 ms | 14.2318 TFLOP/s | 0.00001335 |
| Attention current materialized | batch 4, heads 16, time 512, head dim 64 | 97.3351 ms | baseline | 0.00131983 |
| Attention Tensor Core flash | same shape; `tensor_core_tiled_flash_forward_mma` | 51.5926 ms | 1.8866× versus current materialized | 0.00145862 |

The attention run also measured the scalar streaming reference at 609.7031 ms. No claim is made that these guarded paths are generally optimal or enabled for every model shape.

### Artifact integrity

Raw artifacts remain in the private caller-owned validation bucket. These SHA-256 values let an exported artifact be checked against the files used for this report:

```text
cbd8440024a200b979829116124bd6641b1b09495981c141c65a74ef0714c682  summary.json
8380e28dfb13417acace97d79857302eadcf20aba93766bd15bc300d886a78bf  rust-toolchain.json
28e401d8dd44d3539854f43b9624f5d70b0b2e9d489803d61eb7e307eef4ea95  gpu-smoke-device-0.json
8678cd9b4ed09252ea3760cfc90a6e222a6dc03c1d5e5cea7cfb7a476d53088b  tensor-core-probe-device-0.json
cb61e1707c7d27aed54be5eb4bcc6ecd1565d979466cdff65fc3cf7ade79ef17  cuda-storage-tests.txt
bf58bba31d17a8af3f8be17df68d58becb0c9b200d0576fda32e98cb08e892cc  nccl-tests.txt
167ab6c22a5dcd53912e16bb91509b1041f43ae2fabb527823d3fe564956dd0d  tensor-core-microbench-gemm.json
69ed7297d4420fead6af554b5ef523206d8d3f82d67e11df912a80e5b96e1a90  tensor-core-microbench-attention.json
```

### Limits of the current-code evidence

This run is strong evidence for the advertised single-A100 CUDA boundary and memory-transformer integration tests on the validated source revision. It does not establish current-revision multi-GPU behavior, multi-node communication, broad GPU portability, sustained training throughput, production reliability, or model quality. The executor image used a mutable `latest` tag, so the recorded driver, CUDA, NCCL, Rust, Cargo, command, and artifact hashes—not that tag alone—define the environment evidence.

## Historical 4 × A100 validation — June 6, 2026

This section preserves a broader same-node run from an older source snapshot. It is useful provenance, but it does not substitute for rerunning current code on four devices.

### Environment

- Date: 2026-06-06
- Hardware: 4 × NVIDIA A100-SXM4-80GB, compute capability 8.0, same-node NV12 links
- NVIDIA driver: 535.288.01
- Reported CUDA compatibility: 12.2
- NCCL: 2.21.5 (`nccl_version=22105`)
- Rust: not captured in the retained summary. The run predates the repository's Rust 1.95.0 pin, so it must not be presented as validation of that pinned toolchain.
- Cloud provenance: a successful private Vertex custom job was retained; project, bucket, and account identifiers are intentionally omitted because they are not part of public reproducibility.

### Commands that executed

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

### What executed and passed

- CUDA device discovery, peer-access topology, per-device smoke checks, and BF16 Tensor Core probes ran on all four GPUs.
- The CUDA storage/autograd integration suite passed on the A100 worker. This historical suite predates the current explicit `#[ignore]` classification and therefore has a different test count.
- Four-rank NCCL all-reduce passed: expected sum `10.0`, maximum absolute error `0.0`, four calls, and 16,384 aggregate bytes.
- A 500-step TinyStories-valid DDP training run used per-rank batch 4, global batch 16, block 64, `d_model=64`, four heads, and `ff_hidden=256`.
- Training loss fell from `7.063565` to `3.666313`; held-out loss was `3.581789` over 51,200 tokens.
- Training recorded 44,000 gradient all-reduces and 1,490,432,000 all-reduce bytes; per-step and final parameter-checksum drift were `0.0`.
- BF16 Tensor Core counters were positive and scalar matmul fallback count was `0` for this run.

### Limits of the historical evidence

This run demonstrates a bounded, same-node 4 × A100 CUDA/NCCL path on an older source snapshot. It does not establish current-revision multi-GPU correctness, the Rust 1.95.0 gate on four GPUs, multi-node scaling, production reliability, or broad model-quality claims.
