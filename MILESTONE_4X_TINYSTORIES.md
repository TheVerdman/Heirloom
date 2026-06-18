# Milestone: 4x A100 TinyStories-Valid DDP Gate

Date: 2026-06-06

This is the reviewer anchor for Heirloom's first full single-node, four-GPU public-data language-model training gate. It proves a narrow Rust-native CUDA/NCCL/AMP-BF16 path can train, resume, evaluate, and generate on the full TinyStories validation text split. It does not prove PyTorch parity, production distributed training, tuned kernels, multi-node scaling, or production tokenizer/data maturity.

## Vertex Job

- Project: `project-49b1b523-d248-434f-bd4`
- Region: `us-central1`
- Worker shape: `a2-ultragpu-4g`
- Accelerator: `4x NVIDIA_A100_80GB`
- Observed GPUs: `4x NVIDIA A100-SXM4-80GB`
- Vertex custom job id: `9051757496032559104`
- Display name: `heirloom-validate-quick-20260606-004954`
- State: `JOB_STATE_SUCCEEDED`
- Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260606-004954.tar.gz`
- Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954`

## Command Shape

The job used the existing Vertex validation wrapper and the full TinyStories CUDA reference tier:

```sh
HEIRLOOM_TINYSTORIES_CUDA_MODE=full \
HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl \
HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16 \
HEIRLOOM_TINYSTORIES_CUDA_RESUME_STEPS=5 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES=1 \
HEIRLOOM_ENABLE_NCCL_DEBUG=1 \
scripts/gcp/submit_vertex_heirloom_validate.sh
```

The effective training command was `train-lm --devices cuda:0,cuda:1,cuda:2,cuda:3 --distributed nccl --precision amp-bf16` with one hidden Rust rank per GPU.

## Artifacts

- Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/summary.json`
- NCCL probe: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/nccl-probe.json`
- TinyStories summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/summary.json`
- Train report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/report.json`
- Resume report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/resume-report.json`
- Eval report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/eval.json`
- Generation report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/generation.json`
- Train rank artifacts: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/train-ddp-ranks/`
- Resume rank artifacts: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/resume-ddp-ranks/`
- Checkpoint: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/checkpoint/`
- Resume checkpoint: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954/tinystories-cuda-reference/checkpoint-resume/`

## Data And Model

- Dataset source: full public `TinyStories-valid.txt`
- Tokenizer: Heirloom byte-level BPE-like tokenizer
- Vocab size: `1024`
- Train tokens: `8,962,067`
- Validation tokens: `471,688`
- Steps: `500`
- Resume steps: `5`
- Per-rank batch size: `4`
- Global batch size: `16`
- Block size: `64`
- `d_model`: `64`
- `n_heads`: `4`
- `ff_hidden`: `256`
- Precision: `amp-bf16`
- Distributed mode: single-node NCCL DDP, one Rust process per GPU

## Metrics

- NCCL preflight expected sum: `10.0`
- NCCL preflight max absolute error: `0.0`
- NCCL preflight all-reduce calls: `4`
- NCCL preflight all-reduce bytes: `16,384`
- NCCL version: `22105`
- Train loss: `7.063565 -> 3.666313`
- Train loss reduction: `48.1%`
- Train gradient all-reduce calls: `44,000`
- Train gradient all-reduce bytes: `1,490,432,000`
- Final parameter checksum drift: `0.0`
- Per-step parameter checksum drift: `0.0`
- Resume loss: `3.600464 -> 3.545029`
- Resume gradient all-reduce calls: `440`
- Resume final/per-step checksum drift: `0.0`
- Heldout validation loss: `3.581789`
- Heldout validation perplexity: `35.937761`
- Heldout eval batches: `200`
- Heldout eval tokens: `51,200`
- Generation prompt: `Once upon a time`
- Generation length: `120` new tokens
- Generation finish reason: `max_new_tokens`

## Tensor Core And AMP Evidence

- BF16 Tensor Core Linear forward calls: `78,000`
- BF16 Tensor Core Linear backward calls: `156,000`
- BF16 Tensor Core Linear total calls: `234,000`
- BF16 scalar matmul fallback calls: `0`
- Module-level Linear AMP calls: `3,500`
- Module-level Linear Tensor Core calls: `3,500`
- Module-level Linear fallbacks: `0`
- Tensor Core attention forward calls: `2,000`
- Attention QK matmuls: `32,000`
- Attention AV matmuls: `32,000`
- Attention backward calls: `2,000`
- Attention score-grad matmuls: `32,000`
- Attention dQ/dK/dV matmuls: `32,000` each

## Reproduction Notes

1. Confirm GCP access without printing secrets:

```sh
gcloud config get-value project
gcloud ai custom-jobs list --region=us-central1 --project=project-49b1b523-d248-434f-bd4 --limit=5
gcloud storage ls gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/
```

2. Submit through `scripts/gcp/submit_vertex_heirloom_validate.sh` with the command-shape environment above.
3. Expect the wrapper to upload `summary.json`, `nccl-probe.json`, train/eval/generation reports, rank artifacts, and checkpoint directories.
4. Do not dump full Vertex job JSON in chat or logs because job specs can contain environment values.
5. If reproducing after CUDA runtime hardening, require the same loss/eval/resume gates plus zero unexpected AMP CPU staging and complete CUDA runtime counters.

## Limitations

- This is single-node DDP only.
- It used same-host A100s; it is not multi-node rendezvous or elastic distributed training.
- CUDA kernels were correctness-first at the time of the run, not tuned production kernels.
- AMP BF16 existed as a narrow transformer path, not a complete PyTorch-style autocast system.
- Tensor Core GEMM was limited to compatible shapes and did not yet include a production allocator, stream/event lifecycle, or profiler-grade counters.
- Checkpoint serialization intentionally copied CUDA state to CPU.
- Generation intentionally materialized logits to host for sampling.
- Tokenizer/data code was sufficient for TinyStories-valid, not a mature GPT tokenizer/data pipeline.
- This milestone does not claim PyTorch parity.
