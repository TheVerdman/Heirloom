# Heirloom Vertex Validation

This directory contains opt-in GCP/Vertex validation scaffolding for Heirloom. Local validation remains CPU-only, no-download, and no-cloud by default.

## Existing Project

Use the existing local GCP setup from `/Users/andrewverdiramo/Desktop/VECL-QB`.

```text
project: project-49b1b523-d248-434f-bd4
region: us-central1
bucket: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts
```

Do not print `.env` contents or token values. The Heirloom validation launcher does not require `HF_TOKEN`, OpenAI, Anthropic, or other API tokens.

## Non-Secret Access Checks

Before launching from a local shell, verify access with non-secret commands:

```bash
gcloud config get-value project
gcloud ai custom-jobs list \
  --region=us-central1 \
  --project=project-49b1b523-d248-434f-bd4 \
  --limit=5 \
  --format='table(name,displayName,state,createTime)'
gcloud storage ls gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/
```

If running these from Codex under sandboxing, request command escalation. Avoid dumping full Vertex job JSON or unformatted custom-job listings because job specs can contain env values in other projects/jobs.

## Vertex Rust Validation

Launch an opt-in Vertex job that runs Heirloom validation on a GPU worker:

```bash
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

Defaults:

```text
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=1
HEIRLOOM_VERTEX_VALIDATE_MODE=quick
HEIRLOOM_GPU_SMOKE_DEVICES=0
HEIRLOOM_GPU_SMOKE_LEN=4096
HEIRLOOM_COLLECT_GPU_TOPOLOGY=1
HEIRLOOM_RUN_TENSOR_CORE_PROBE=1
HEIRLOOM_ENABLE_NCCL_DEBUG=1
STREAM_LOGS=true
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=0
```

Every GPU-required Vertex validation run now performs a real CUDA smoke before the Rust validation gate:

```text
cargo run --bin heirloom -- gpu info
cargo run --bin heirloom -- gpu topology --report <json>
cargo run --bin heirloom -- gpu smoke --device <n> --len <len> --report <json>
cargo run --bin heirloom -- gpu tensor-core-probe --device <n> --report <json>
HEIRLOOM_CUDA_TESTS=1 cargo test --workspace --test cuda_storage -- --nocapture
```

The smoke path dynamically loads the CUDA Driver API, allocates GPU memory, copies f32 inputs to device, launches embedded PTX kernels for `heirloom_add_f32` and `heirloom_relu_f32`, copies results back, checks max absolute error against CPU references, and uploads JSON reports under:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/
```

The gated Rust CUDA test then runs the Tensor CUDA storage, arithmetic, rank-2 dense/strided matmul, Linear-shaped bias-add/view-backward, autograd, reduction, SGD, AdamW, tiny CUDA Linear-shaped training, and tiny CUDA transformer forward/backward/AdamW/checkpoint smoke tests on the GPU. This is actual GPU kernel validation for the current CUDA Tensor boundary. It is still not a public-data GPU training run or production GPU training backend.

The launcher streams and uploads diagnostic logs for GPU info, each smoke run, each Tensor Core probe, CUDA storage tests, and any optional training/reference wrapper. Failed probe/smoke runs still upload their `.txt` transcripts when the command exits before JSON can be written. Successful Tensor Core probes also upload:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tensor-core-probe-device-<n>.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tensor-core-probe-device-<n>.txt
```

If the worker fails before the normal `summary.json` can be written, it attempts to upload:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/failure-summary.json
```

That file records the non-secret error type/message and points back to the artifact prefix for the uploaded transcripts.

When `HEIRLOOM_COLLECT_GPU_TOPOLOGY=1`, the launcher also uploads:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/nvidia-smi.txt
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/nvidia-smi-topo.txt
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/nvidia-smi-nvlink.txt
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/nvidia-smi-gpu-bus-ids.csv
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/heirloom-gpu-topology.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/heirloom-gpu-topology.txt
```

Use `nvidia-smi-topo.txt` and `nvidia-smi-nvlink.txt` to confirm whether the GPUs are NVLink-connected. Use `heirloom-gpu-topology.json` to confirm CUDA device ordinals, PCI bus IDs, and `cuDeviceCanAccessPeer` visibility from Heirloom's runtime.

For an opt-in CLI training gate on the GPU, enable the CUDA LM fixture:

```bash
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1 \
HEIRLOOM_CUDA_FIXTURE_DEVICE=cuda:0 \
HEIRLOOM_CUDA_FIXTURE_PRECISION=bf16 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

That fixture runs `scripts/cuda_train_lm_fixture.sh` inside the worker. It trains through the real `heirloom train-lm --device cuda:0` CLI on a tiny prepared corpus, requires loss decrease, resumes from the saved checkpoint, checks the resumed step count, and uploads:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/cuda-train-lm-fixture/train-report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/cuda-train-lm-fixture/resume-report.json
```

A successful reference run is documented in `runs/vertex-cuda-a100-20260603.md` under `heirloom-validate-quick-20260604-105451`.

For the first Tensor Core engine validation, run the CUDA fixture with AMP and the hard gate before attempting a TinyStories job:

```bash
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1 \
HEIRLOOM_CUDA_FIXTURE_DEVICE=cuda:0 \
HEIRLOOM_CUDA_FIXTURE_PRECISION=amp-bf16 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_TENSOR_CORES=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The fixture auto-selects tile-compatible defaults for AMP (`batch=4`, `block=4`, `d_model=16`, `n_heads=4`, `ff_hidden=64`, `vocab=272`) unless `HEIRLOOM_CUDA_FIXTURE_AUTO_TILE=0` is set. A good run must upload `train-report.json` and `resume-report.json` with `tensor_core.bf16_tensor_core_matmul_calls > 0`, `tensor_core.bf16_tensor_core_matmul_forward_calls > 0`, `tensor_core.bf16_tensor_core_matmul_backward_calls > 0`, `tensor_core.bf16_scalar_matmul_fallback_calls == 0`, `tensor_core_coverage.linear_totals.tensor_core_calls > 0`, `tensor_core_coverage.linear_totals.fallback_calls == 0`, `cuda_runtime.tensor_core_cta_gemm_calls > 0`, `cuda_runtime.tensor_core_mma_warp_tiles > 0`, `cuda_runtime.tensor_core_global_cta_gemm_calls == 0`, and `cuda_runtime.tensor_core_legacy_warp_gemm_calls == 0`. Default staged runs should also show positive `cuda_runtime.tensor_core_staged_cta_gemm_calls`, `cuda_runtime.tensor_core_shared_stage_tiles`, and `cuda_runtime.tensor_core_shared_stage_bytes`, with `cuda_runtime.tensor_core_wide_swizzled_cta_gemm_calls == 0`. Runs with `HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM=1` should instead show positive `cuda_runtime.tensor_core_wide_swizzled_cta_gemm_calls`, `cuda_runtime.tensor_core_swizzled_stage_tiles`, and `cuda_runtime.tensor_core_swizzled_stage_bytes`, with staged/shared counters at `0`. If this fails before training, inspect `tensor-core-probe-device-0.txt` and `tensor-core-probe-device-0.json`; if it fails during train/resume, inspect `cuda-train-lm-fixture/run.log` plus the fixture report counters, `tensor_core_coverage.linear_modules`, and shape/device error.

A successful Tensor Core Linear forward/backward fixture gate is documented in `runs/vertex-cuda-a100-20260603.md` under `heirloom-validate-quick-20260605-131318`.

For the ragged Tensor Core Linear pad/crop gate, keep it single-GPU and disable public-data work. This mode deliberately uses odd projection dimensions so the Tensor Core GEMM wrapper must pad BF16 inputs on device, crop the f32 output, and report positive padded/remainder tile counters:

```bash
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=1 \
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0 \
HEIRLOOM_CUDA_FIXTURE_DEVICE=cuda:0 \
HEIRLOOM_CUDA_FIXTURE_PRECISION=amp-bf16 \
HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES=1 \
HEIRLOOM_CUDA_FIXTURE_STEPS=40 \
HEIRLOOM_CUDA_FIXTURE_RESUME_STEPS=3 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_TENSOR_CORE_PADDING=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The CUDA storage test suite also includes a raw ragged `M=15, K=17, N=7` BF16 Tensor Core fixture. The training fixture uses `batch=3`, `block=5`, `d_model=18`, `n_heads=3`, and `ff_hidden=37` by default when `HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES=1`. Do not combine this ragged Linear gate with `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1`; the point is to isolate Linear GEMM pad/crop while attention takes the existing f32 CUDA path. A good run must show `tensor_core_pad_crop.status == "passed"` in both train and resume reports, positive `tensor_core_pad_crop.padded_tiles` and `tensor_core_pad_crop.remainder_tiles`, zero `tensor_core_pad_crop.scalar_fallbacks`, zero `tensor_core_pad_crop.linear_fallbacks`, positive `cuda_runtime.tensor_core_cta_gemm_calls`, positive `cuda_runtime.tensor_core_mma_warp_tiles`, zero `cuda_runtime.tensor_core_global_cta_gemm_calls`, and zero `cuda_runtime.tensor_core_legacy_warp_gemm_calls`. Default staged runs require positive staged/shared counters; wide-swizzled runs require positive wide/swizzled counters and zero staged/shared counters. The fixture also writes and uploads compact reviewer artifacts:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/cuda-train-lm-fixture/tensor-core-pad-crop-summary.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/cuda-train-lm-fixture/tensor-core-pad-crop-summary.txt
```

A successful ragged Tensor Core Linear pad/crop gate is documented in `runs/vertex-cuda-a100-20260603.md` under `heirloom-validate-quick-20260607-161144`; the promoted pad/crop report artifact gate is documented under `heirloom-validate-quick-20260607-171521`; the shared-memory-staged CTA Linear Tensor Core gate is documented under `heirloom-validate-quick-20260607-185948`; and the opt-in wide-swizzled CTA gate is documented under `heirloom-validate-quick-20260607-213948`.

For the first Tensor Core attention validation, add the attention hard gate. The fixture switches to a known-compatible small shape (`block=16`, `d_model=16`, `n_heads=1`) unless explicit fixture dimensions are provided:

```bash
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1 \
HEIRLOOM_CUDA_FIXTURE_DEVICE=cuda:0 \
HEIRLOOM_CUDA_FIXTURE_PRECISION=amp-bf16 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

A good run must show positive `tensor_core.bf16_tensor_core_attention_forward_calls`, `tensor_core.bf16_tensor_core_attention_qk_matmul_calls`, `tensor_core.bf16_tensor_core_attention_av_matmul_calls`, `tensor_core.bf16_tensor_core_attention_backward_calls`, and the score-grad/dQ/dK/dV attention backward matmul counters in both train and resume reports. This validates the BF16 Tensor Core QK/AV forward path plus the first Tensor Core attention-backward matmul decomposition. The softmax backward and transpose glue remain f32 CUDA kernels, and this is not a fused FlashAttention-style implementation.

A successful Tensor Core attention backward fixture gate is documented in `runs/vertex-cuda-a100-20260603.md` under `heirloom-validate-quick-20260605-160714`; the earlier forward-only gate is documented under `heirloom-validate-quick-20260605-151025`.

For guarded A100 instruction-path GEMM throughput validation, enable the Tensor Core microbench block and request only the GEMM section. This runs the CUDA storage test gate first, then runs `heirloom gpu tensor-core-microbench` with hard-required double-buffered `cp.async` GEMM:

```bash
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=1 \
HEIRLOOM_RUN_TENSOR_CORE_MICROBENCH=1 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_SECTIONS=gemm \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ITERATIONS=32 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_WARMUP=4 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_M=4096 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_K=1024 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_N=4096 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_BATCH=4 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEADS=16 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_TIME=512 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEAD_DIM=64 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The worker sets `HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1` and `HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1` only for the microbench command. GEMM-only Vertex job `9203551123560988672` passed this gate on A100 and microbench-validated `ldmatrix_a_shared_mma` plus `cp_async_double_buffered_ldmatrix_a_mma` with zero fallback counters. That job did not validate Tensor Core flash attention, and these GEMM tiers are still guarded throughput gates rather than default train-path production kernels.

A good GEMM run uploads:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tensor-core-microbench-gemm.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tensor-core-microbench-gemm.txt
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tensor-core-microbench-summary.json
```

The summary JSON must have `status == "passed"`, `quick_clippy == "run"` or `"skipped"`, `quick_clippy_reason` when skipped, and the requested/completed/failed microbench section audit fields. The GEMM JSON must have top-level `status == "passed"`, `gemm.ldmatrix_requested.passed == true`, `gemm.cp_async_requested.passed == true`, positive ldmatrix/cp.async runtime counters, and zero fallback counters.

For the separate Tensor Core flash attention forward gate, run only the attention section and keep it isolated from the GEMM evidence:

```bash
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=1 \
HEIRLOOM_RUN_TENSOR_CORE_MICROBENCH=1 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_SECTIONS=attention \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ITERATIONS=32 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_WARMUP=4 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_BATCH=4 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEADS=16 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_TIME=512 \
HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEAD_DIM=64 \
HEIRLOOM_CUDA_FLASH_BF16_ATTENTION=1 \
HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE=1 \
HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The attention JSON must have top-level `status == "passed"`, `attention.tensor_core_flash_forward.status == "ok"`, `attention.tensor_core_flash_forward.passed == true`, positive Tensor Core flash QK/AV MMA counters, and zero flash fallback counters before the forward path can be called A100-validated. Vertex job `6050468434448220160` passed this attention-only forward gate for `batch=4, heads=16, time=512, head_dim=64`. The later 4x A100 training-path gate `894480179806601216` validated exact-tile flash backward for the same target shape behind `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD=1`; keep this route guarded until ragged backward and broader shape coverage pass.

For the ragged Tensor Core attention edge gate, keep it single-GPU and enable the ragged attention fixture mode. This deliberately uses non-tile-aligned attention dimensions so the QK/AV and score-grad/dQ/dK/dV Tensor Core kernels must handle boundary-predicated edge loads/stores:

```bash
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=1 \
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0 \
HEIRLOOM_CUDA_FIXTURE_DEVICE=cuda:0 \
HEIRLOOM_CUDA_FIXTURE_PRECISION=amp-bf16 \
HEIRLOOM_CUDA_FIXTURE_RAGGED_ATTENTION_TENSOR_CORES=1 \
HEIRLOOM_CUDA_FIXTURE_STEPS=40 \
HEIRLOOM_CUDA_FIXTURE_RESUME_STEPS=3 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES=1 \
HEIRLOOM_EXPECT_TENSOR_CORE_PADDING=1 \
HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORE_PADDING=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The ragged attention fixture defaults to `batch=3`, `block=15`, `d_model=17`, `n_heads=1`, `ff_hidden=37`, and vocab `281`. A good run must show positive attention QK/AV/score-grad/dQ/dK/dV counters, positive `cuda_runtime.tensor_core_attention_padded_tiles`, positive `cuda_runtime.tensor_core_attention_remainder_tiles`, and `cuda_runtime.tensor_core_attention_scalar_fallbacks == 0` in both train and resume reports. Because the default ragged attention fixture also uses ragged Linear shapes, it should also pass the named `tensor_core_pad_crop` checks with zero Linear fallbacks. A successful ragged Tensor Core attention edge gate is documented in `runs/vertex-cuda-a100-20260603.md` under `heirloom-validate-quick-20260607-204532`.

For an opt-in small public-data CUDA reference run, enable:

```bash
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=1 \
HEIRLOOM_TINYSTORIES_CUDA_DEVICE=cuda:0 \
HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The first bounded 4x DDP training fixture has now passed on Vertex. It runs the tiny local LM corpus, one hidden Rust rank per GPU, NCCL f32 gradient all-reduce, local CUDA AdamW, checkpoint save/resume, and per-step parameter-checksum synchronization before any public-data DDP run. The recorded pass is `heirloom-validate-quick-20260605-234242` / job `6473235603130417152`.

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_NCCL_PROBE_KIND=all-reduce \
HEIRLOOM_NCCL_TRACE=1 \
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0 \
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1 \
HEIRLOOM_CUDA_FIXTURE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_CUDA_FIXTURE_DISTRIBUTED=nccl \
HEIRLOOM_CUDA_FIXTURE_PRECISION=amp-bf16 \
HEIRLOOM_CUDA_FIXTURE_STEPS=40 \
HEIRLOOM_CUDA_FIXTURE_RESUME_STEPS=3 \
HEIRLOOM_CUDA_FIXTURE_DDP_CHECKSUM_EVERY=1 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

This fixture uploads `cuda-train-lm-fixture/train-report.json`, `resume-report.json`, and per-rank launcher artifacts under `train-ddp-ranks/` and `resume-ddp-ranks/`. A good run must show positive aggregate `all_reduce_calls`/`all_reduce_bytes`, zero parameter-checksum drift beyond tolerance, rank-0-only checkpoint writes, and successful resume metadata. The recorded pass had `5.903006 -> 0.918672` train loss, `3520` gradient all-reduce calls, `7905280` all-reduce bytes, zero final/per-step checksum drift, and positive Tensor Core Linear counters with zero scalar fallbacks. Add `HEIRLOOM_REQUIRE_TENSOR_CORES=1` when validating default tile-compatible training shapes; it rejects unsupported projection or backward shapes/devices instead of falling back from the Tensor Core Linear path. Add `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1` only when the model has attention-compatible tiles such as `block_size%16=0` and `(d_model/n_heads)%16=0`; otherwise the run should fail early with a shape/device diagnostic.

For the memory-transformer sparse-row DDP row-union gate, use the separate memory fixture. It runs `train-memory-lm`, defaults to `--memory-update-policy sparse-rows`, resumes from checkpoint, and validates row-union report fields before upload:

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_NCCL_PROBE_KIND=all-reduce \
HEIRLOOM_NCCL_TRACE=1 \
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0 \
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=0 \
HEIRLOOM_RUN_CUDA_TRAIN_MEMORY_LM_FIXTURE=1 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED=nccl \
HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION=amp-bf16 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY=sparse-rows \
HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE=masked-memory-rows \
HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK=auto \
HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS=40 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS=3 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_CHECKSUM_EVERY=1 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

This fixture uploads `cuda-train-memory-lm-fixture/train-report.json`, `resume-report.json`, `summary.json`, `artifact-validator.txt`, per-rank launcher artifacts under `train-ddp-memory-ranks/` and `resume-ddp-memory-ranks/`, and a checkpoint prefix. With `HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK=auto`, the fixture writes a deterministic offline row mask and requires aggregate reports to record the mask. A good sparse-row DDP run must show positive aggregate `all_reduce_calls`/`all_reduce_bytes`, positive `row_union_all_reduce_calls`/`row_union_all_reduce_bytes`/`row_union_candidate_rows`, positive `compact_gradient_all_reduce_calls`/`compact_gradient_all_reduce_bytes`, `compressed_sparse_gradient_transport=true`, nonzero per-rank `memory_gradient_parameter_count`, positive per-rank `bool_mask_to_indices_calls`, `gather_selected_rows_calls`, and `sparse_adamw_compact_rows_calls`, zero memory-table checksum drift beyond tolerance, successful resume metadata, and SMFT row-mask evidence when mask mode is enabled. The fixture `summary.json` now mirrors those train/resume transport counters, rank compact-gradient counters, checksum drift fields, and rank-0 memory kernel counters so reviewers can inspect one non-secret summary before drilling into per-rank artifacts. This validates synchronized sparse-row memory DDP plus offline SMFT mask intersection, row-union mask compaction, compact sparse-gradient all-reduce, and compact selected-row sparse optimizer consumption; it does not prove distributed background-count mask generation or online SMFT refresh synchronization.

After downloading or copying a completed fixture directory, validate the artifacts without rerunning training:

```bash
python3 scripts/validate_memory_fixture_artifacts.py \
  /path/to/cuda-train-memory-lm-fixture \
  --expect-distributed \
  --expect-sparse-rows \
  --require-resume
```

For the paid 4x A100 32-block memory-transformer large-data reference, use the same fixture with explicit shape and data gates. This uses TinyStories-valid rather than the synthetic memory text, and the artifact validator requires the full model profile and a large prepared-token count:

```bash
STREAM_LOGS=false \
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_VERTEX_VALIDATE_MODE=quick \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_NCCL_PROBE_KIND=all-reduce \
HEIRLOOM_NCCL_TRACE=1 \
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0 \
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=0 \
HEIRLOOM_RUN_CUDA_TRAIN_MEMORY_LM_FIXTURE=1 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_SOURCE=tinystories-valid \
HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_SOURCE_BYTES=10000000 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_TRAIN_TOKENS=7000000 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_VOCAB_SIZE=1024 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED=nccl \
HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION=amp-bf16 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_N_LAYERS=32 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_BLOCK_SIZE=64 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_D_MODEL=64 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_N_HEADS=4 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_FF_HIDDEN=256 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LAYER_INDICES=8,16,24 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_SLOTS=1024 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_KEY_DIM=32 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_VALUE_DIM=64 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_TOP_K=4 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_HEADS=1 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LOOKUP=exact \
HEIRLOOM_CUDA_MEMORY_FIXTURE_SHARED_MEMORY=true \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_PLUS=true \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY=sparse-rows \
HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE=masked-memory-rows \
HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK=auto \
HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS=100 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS=5 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION=-1.0 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_BATCH_SIZE=1 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_LR=0.0003 \
HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_CHECKSUM_EVERY=5 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

For this tier, the wrapper runs `scripts/validate_memory_fixture_artifacts.py` with `--expect-n-layers 32`, `--expect-memory-layer-indices 8,16,24`, `--expect-data-source tinystories-valid`, minimum source/token thresholds, `--require-memory-kernel-counters`, and `--require-tensor-core-counters`. The `MIN_REDUCTION=-1.0` setting makes this a runtime/shape/resume/kernel proof rather than an early-training-quality gate; loss movement is still recorded in the reports. A passed artifact therefore proves the 32-block memory model, not just the small 4-layer fixture.

If the fixture failed before writing `summary.json`, run the same validator against the partial fixture directory. When `launcher-report.json` is present, it reports each rank's final stage, exit status, missing rank report status, and common unsupported-CUDA-operator messages instead of reducing the failure to a missing summary artifact.

A first bounded 4x TinyStories DDP public-data reference tier has now passed on Vertex. It uses the TinyStories validation text, a 1 MiB tokenizer/data slice, one hidden Rust rank per GPU, NCCL f32 gradient all-reduce, local CUDA AdamW, heldout eval/perplexity, generation, and DDP checksum reporting. The recorded pass is `heirloom-validate-quick-20260606-002304` / job `2438784393192407040`.

To reproduce that bounded reference tier:

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_NCCL_PROBE_KIND=all-reduce \
HEIRLOOM_NCCL_TRACE=1 \
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0 \
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=0 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=1 \
HEIRLOOM_TINYSTORIES_CUDA_MODE=reference \
HEIRLOOM_TINYSTORIES_CUDA_DEVICE=cuda:0 \
HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl \
HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The recorded bounded reference pass had train loss `6.579771 -> 4.123779`, loss reduction `0.373264`, validation loss `3.951583`, perplexity `52.017655`, `7040` gradient all-reduce calls, `26193920` all-reduce bytes, zero final/per-step parameter-checksum drift, `6720` BF16 Tensor Core Linear matmul calls, and zero scalar matmul fallbacks.

The full 4x TinyStories DDP gate has also passed. The recorded pass is `heirloom-validate-quick-20260606-004954` / job `9051757496032559104`. It uses the full TinyStories validation text, `500` steps, `d_model=64`, `block_size=64`, a 20% loss-reduction gate, a 5-step checkpoint-resume continuation, heldout eval, generation, NCCL all-reduce reporting, Linear Tensor Core hard gating, and attention Tensor Core hard gating.

To reproduce the full gate:

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_NCCL_PROBE_KIND=all-reduce \
HEIRLOOM_NCCL_TRACE=1 \
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0 \
HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=0 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=1 \
HEIRLOOM_TINYSTORIES_CUDA_MODE=full \
HEIRLOOM_TINYSTORIES_CUDA_DEVICE=cuda:0 \
HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl \
HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16 \
HEIRLOOM_TINYSTORIES_CUDA_BATCH=4 \
HEIRLOOM_TINYSTORIES_CUDA_BLOCK=64 \
HEIRLOOM_TINYSTORIES_CUDA_D_MODEL=64 \
HEIRLOOM_TINYSTORIES_CUDA_HEADS=4 \
HEIRLOOM_TINYSTORIES_CUDA_FF=256 \
HEIRLOOM_TINYSTORIES_CUDA_VOCAB=1024 \
HEIRLOOM_TINYSTORIES_CUDA_STEPS=500 \
HEIRLOOM_TINYSTORIES_CUDA_RESUME_STEPS=5 \
HEIRLOOM_TINYSTORIES_CUDA_MIN_REDUCTION=0.20 \
HEIRLOOM_TINYSTORIES_CUDA_EVAL_BATCHES=200 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The recorded full pass had train loss `7.063565 -> 3.666313`, loss reduction `0.480954`, validation loss `3.581789`, perplexity `35.937761`, `44000` gradient all-reduce calls, `1490432000` all-reduce bytes, zero final/per-step parameter-checksum drift, resume `500 -> 505`, `234000` BF16 Tensor Core matmul calls, positive attention Tensor Core QK/AV and backward matmul counters, and zero scalar matmul fallbacks.

The QB-native data hard path runs `scripts/run_qb_data_hardpath.sh`. It prepares manifest v2 binary token shards from repeated TinyStories plus deterministic QB trace inputs and writes host timing, CUDA-event timing, and MFU estimate fields. Dense mode trains/resumes through the streaming loader, then runs eval and generation. Memory mode is selected with `HEIRLOOM_QB_DATA_HARDPATH_MODEL=memory`; it trains/resumes with `train-memory-lm`, evaluates with `eval-memory-lm`, generates with `generate-memory-lm`, uses the same manifest v2 streaming eval loader, and validates memory selection, memory optimizer, SMFT access, sparse-row transport when enabled, and memory-table checksum evidence. Vertex uploads its artifacts under `qb-data-hardpath/`, including `summary.json`, `timings.json`, train/resume/eval/generation reports, the tokenizer, QB trace fixture, optional SMFT row mask, manifest v2, all shard metadata/payload files, DDP launcher reports, rank directories, and checkpoints.

The QB-native tokenizer hard path runs `scripts/run_qb_tokenizer_hardpath.sh` and is enabled with `HEIRLOOM_RUN_QB_TOKENIZER_HARDPATH=1`. It trains a native tokenizer artifact v2 from a corpus-blend manifest, validates reserved tokens, writes fertility reports, materializes governed source samples with `heirloom data materialize-blend`, prepares manifest v2 binary shards, and runs memory train/resume/eval/generation. `HEIRLOOM_QB_TOKENIZER_HARDPATH_MODE=smoke` uses non-TinyStories local fixtures plus any available VECL-QB v1-hard data; `full` requires materialized approved source paths via `HEIRLOOM_QB_TOKENIZER_DOLMA_PATH`, `HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH`, `HEIRLOOM_QB_TOKENIZER_OLMO3_PATH`, `HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH`, and `HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH`. Those variables may be local file paths, flat local source directories, single-object `gs://` URIs, or flat `gs://` prefixes ending in `/`; `gs://` inputs are staged onto the Vertex worker under `HEIRLOOM_QB_TOKENIZER_SOURCE_STAGE_DIR` or `qb-tokenizer-hardpath/source-slices` by default, while the generated corpus-blend manifest records the original GCS URI. Full mode defaults to `HEIRLOOM_QB_TOKENIZER_HARDPATH_TARGET_TOKENS=20000000000` and `HEIRLOOM_QB_TOKENIZER_HARDPATH_MATERIALIZE_MODE=full`; it also defaults `HEIRLOOM_QB_TOKENIZER_HARDPATH_CANDIDATE_TEXT_MODE=rescan`, exact bounded candidate retention to `HEIRLOOM_QB_TOKENIZER_HARDPATH_CANDIDATE_RETENTION_TOKEN_MULTIPLIER=1.0`, `HEIRLOOM_QB_TOKENIZER_HARDPATH_CANDIDATE_RETENTION_MIN_DOCS=100000`, and `HEIRLOOM_QB_TOKENIZER_HARDPATH_CANDIDATE_PRUNE_EVERY=50000`, plus materializer heartbeat logs to `HEIRLOOM_QB_TOKENIZER_HARDPATH_PROGRESS_EVERY_RECORDS=100000` and `HEIRLOOM_QB_TOKENIZER_HARDPATH_PROGRESS_EVERY_BYTES=1073741824`. Full mode also enables materializer scan checkpoints by default under `HEIRLOOM_QB_TOKENIZER_HARDPATH_CHECKPOINT_DIR` or `$out_dir/materializer-checkpoints`, every `HEIRLOOM_QB_TOKENIZER_HARDPATH_CHECKPOINT_EVERY_RECORDS=100000` records or `HEIRLOOM_QB_TOKENIZER_HARDPATH_CHECKPOINT_EVERY_BYTES=1073741824` bytes; set `HEIRLOOM_QB_TOKENIZER_HARDPATH_RESUME_CHECKPOINT=1` to resume compatible source scan checkpoints. Smoke mode defaults to a small `sample` materialization. Vertex uploads artifacts under `qb-tokenizer-hardpath/`, including `materialized/curation-report.json`, `materialized/source-index.json`, `materialized/selected-docs.jsonl`, `materialized/tokenizer-sample-manifest.json`, `materialized/prepared/manifest.json`, `materialized/prepared/shards/`, and full-mode materializer scan checkpoints.

Set `HEIRLOOM_QB_TOKENIZER_HARDPATH_PREPARED_MANIFEST` to reuse an existing
prepared manifest instead of running `data materialize-blend` again. The value
may be a local `manifest.json`, a local prepared directory containing
`manifest.json` plus `shards/`, a `gs://.../prepared/manifest.json` object, or a
`gs://.../prepared/` prefix. The hard path stages the prepared directory under
`materialized/prepared/`, preserves/copies materialization sidecars such as
`curation-report.json` when available, synthesizes minimal reuse metadata when
needed, and records `prepared_manifest_reused` plus
`prepared_manifest_source` in `summary.json`. If the prepared sidecars are not
available and the learning-sanity gate needs a selected-doc count, provide
`HEIRLOOM_QB_TOKENIZER_HARDPATH_PREPARED_SELECTED_DOCS`.

QB-1 target shape, locked on 2026-06-20, is a slightly-greater-than-1B
parameter memory transformer for the governed 20B-token corpus. The working
target is `vocab=32768`, `block=1024`, `n_layers=36`, `d_model=1536`,
`heads=24` (`head_dim=64`), `ff_hidden=6144`, memory layers `8,16,24,32`,
`memory_slots=1024`, `memory_key_dim=64`, `memory_value_dim=64`,
`memory_top_k=4`, `memory_heads=1`, shared memory tables, memory-plus gating,
sparse-row memory updates, and SMFT row masking. This shape is approximately
1.05B trainable parameters with the current shared-memory-table implementation,
not 1.05B dense-only parameters. The equivalent 36-layer all-dense transformer
would be approximately 1.12B parameters; replacing four FFN blocks with memory
blocks removes approximately 74M dense FFN parameters, while the shared
trainable key/value memory tables add only approximately 0.13M parameters
(`1024 x 64` key plus `1024 x 64` value). Memory projections plus those shared
tables account for approximately 1.3M parameters. The smaller `d_model=1024`
4-layer and 32-layer runs remain readiness, throughput, and evidence gates, not
the final QB-1 target.

Recommended production slice layout:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/
  source-slices/
    dolma-v1_7/dolma-v1_7-20b/part files...
    nemotron-cc-high-actual/nemotron-cc-high-actual-20b/part files...
    dolma3-dolmino-mix-100b-1125/dolmino-20b/part files...
    nemotron-cc-math/nemotron-cc-math-20b/part files...
    vecl-qb-v1-hard/corpus.jsonl
  tokenizer/
  materialized/
  prepared/
  reports/
```

The repeatable staging helper is:

```bash
python3 - <<'PY'
import urllib.request
url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_7.txt"
out = "/private/tmp/dolma-v1_7-urls.txt"
with urllib.request.urlopen(url, timeout=60) as src, open(out, "wb") as dst:
    dst.write(src.read())
print(out)
PY

python3 scripts/slice_hf_dataset.py \
  --repo-id allenai/dolma \
  --source-id allenai.dolma.v1_7 \
  --slug dolma-v1_7 \
  --preset dolma-qb \
  --url-list /private/tmp/dolma-v1_7-urls.txt \
  --target-tokens-estimate 100000000 \
  --shard-output-bytes 1073741824 \
  --out runs/qb-native-pretraining-v1/source-slices/dolma-v1_7/dolma-v1_7-100m \
  --report runs/qb-native-pretraining-v1/source-slices/dolma-v1_7/dolma-v1_7-100m.report.json \
  --upload \
  --project project-49b1b523-d248-434f-bd4

python3 scripts/slice_hf_dataset.py \
  --repo-id allenai/dolma3_dolmino_mix-100B-1125 \
  --source-id allenai.dolma3_dolmino_mix-100B-1125 \
  --slug dolma3-dolmino-mix-100b-1125 \
  --preset dolmino-qb \
  --target-tokens-estimate 100000000 \
  --shard-output-bytes 1073741824 \
  --out runs/qb-native-pretraining-v1/source-slices/dolma3-dolmino-mix-100b-1125/dolmino-100m \
  --report runs/qb-native-pretraining-v1/source-slices/dolma3-dolmino-mix-100b-1125/dolmino-100m.report.json \
  --upload \
  --project project-49b1b523-d248-434f-bd4

python3 scripts/stage_qb_source_slices.py --upload \
  --project project-49b1b523-d248-434f-bd4
```

`scripts/slice_hf_dataset.py` reads `HF_TOKEN` from the environment or
`/Users/andrewverdiramo/Desktop/VECL-QB/.env` only when it needs Hugging Face
network access. Reports record whether auth was used, but never record token
values. URL-list inputs such as Dolma's `urls/v1_7.txt` do not use or transmit
the HF token. With `--shard-output-bytes`, the slicer treats `--out` as a
directory, writes deterministic `*-part-NNNNN.jsonl` files, and reports a
file-set hash. `scripts/stage_qb_source_slices.py` records and uploads each
part while keeping the source-level entry addressable as a directory or GCS
prefix.

The first staged source is VECL-QB `v1-hard`:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/vecl-qb-v1-hard/corpus.jsonl
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/vecl-qb-v1-hard/metadata.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/source-slice-inventory.json
```

As of the first staging pass, the inventory has `1` available source, `4`
pending external sources, `25,000` records, and `43,971,635` corpus bytes.
See `docs/experimental/qb/QB_SOURCE_GOVERNANCE.md` for source approval ownership and the current
blocked/pending external-source verdicts.

The first full 4x A100 hard-path pass completed at `2026-06-11T01:36:13Z` (`2026-06-10` US/Eastern):

```text
display: heirloom-validate-quick-20260610-212624
job: projects/232930557062/locations/us-central1/customJobs/9079935230273388544
summary: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260610-212624/qb-data-hardpath/summary.json
```

The first full 4x A100 memory-transformer hard-path pass completed at `2026-06-11T03:29:29Z` (`2026-06-10` US/Eastern):

```text
display: heirloom-validate-quick-20260610-231715
job: projects/232930557062/locations/us-central1/customJobs/683395937506164736
summary: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260610-231715/qb-data-hardpath/summary.json
```

The memory run passed `scripts/validate_qb_data_hardpath_artifacts.py --profile full --expected-world-size 4` with `model_kind=memory`, `train-memory-lm`, `eval-memory-lm`, `generate-memory-lm`, manifest v2 `storage="binary_shards"`, `8962719` train tokens, `471722` valid tokens, `tokens_seen=25600`, `tokens_per_second=1982.9589465530596`, positive CUDA-event timing, positive DDP all-reduce/row-union/compact-gradient counters, positive Linear and attention Tensor Core counters, and zero scalar matmul fallbacks.

Validate a local or GCS hard-path summary with:

```bash
python3 scripts/validate_qb_data_hardpath_artifacts.py \
  gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260610-212624/qb-data-hardpath/summary.json
```

To run the full 4x A100 hard-path gate:

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_NCCL_PROBE_KIND=all-reduce \
HEIRLOOM_NCCL_TRACE=1 \
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0 \
HEIRLOOM_RUN_QB_DATA_HARDPATH=1 \
HEIRLOOM_QB_DATA_HARDPATH_MODE=full \
HEIRLOOM_QB_DATA_HARDPATH_SHARD_TOKENS=1000000 \
HEIRLOOM_QB_DATA_HARDPATH_GRAD_ACCUMULATION_STEPS=1 \
HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl \
HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16 \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The dense gate fails unless the manifest is version `2` with `storage="binary_shards"`, sources include TinyStories and the QB trace fixture, train/resume/eval/generation complete, DDP all-reduce calls and bytes are positive, checksum drift stays within tolerance, Linear and attention Tensor Core hard gates have zero scalar fallbacks, reports use `loader.kind="binary_shard_streaming"` with `tokens_materialized=false`, and performance reports include host timing buckets, CUDA-event timing buckets, tokens/sec, dense-core MFU estimate, end-to-end MFU estimate, micro-batch size, grad accumulation steps, and effective batch size. Full distributed CUDA gates must also show `performance.cuda_event_timing_available=true`, positive `forward_backward_cuda_elapsed_ms`, and positive `cuda_runtime.event_elapsed_calls`. The dense-core MFU estimate uses CUDA-event forward/backward elapsed time when available; end-to-end MFU remains based on host train elapsed time. The MFU denominator is the documented A100 SXM BF16 dense peak, and sparse/memory work is reported separately so sparse parameters do not inflate the estimate.

To run the memory-transformer hard path on the same manifest v2 substrate:

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_NCCL_PROBE_KIND=all-reduce \
HEIRLOOM_NCCL_TRACE=1 \
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0 \
HEIRLOOM_RUN_QB_DATA_HARDPATH=1 \
HEIRLOOM_QB_DATA_HARDPATH_MODEL=memory \
HEIRLOOM_QB_DATA_HARDPATH_MODE=full \
HEIRLOOM_QB_DATA_HARDPATH_STEPS=100 \
HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS=5 \
HEIRLOOM_QB_DATA_HARDPATH_BATCH=1 \
HEIRLOOM_QB_DATA_HARDPATH_LR=0.0003 \
HEIRLOOM_QB_DATA_HARDPATH_MIN_REDUCTION=-1.0 \
HEIRLOOM_QB_DATA_HARDPATH_SHARD_TOKENS=1000000 \
HEIRLOOM_QB_DATA_HARDPATH_GRAD_ACCUMULATION_STEPS=1 \
HEIRLOOM_QB_DATA_HARDPATH_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_QB_DATA_HARDPATH_DISTRIBUTED=nccl \
HEIRLOOM_QB_DATA_HARDPATH_PRECISION=amp-bf16 \
HEIRLOOM_QB_DATA_HARDPATH_MEMORY_UPDATE_POLICY=sparse-rows \
HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_MODE=masked-memory-rows \
HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_ROW_MASK=auto \
HEIRLOOM_REQUIRE_TENSOR_CORES=1 \
HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The memory gate fails unless `model_kind="memory"`, `train-memory-lm`, `eval-memory-lm`, and `generate-memory-lm` report `model_family="memory_transformer"`, train/eval loaders are `binary_shard_streaming` with `tokens_materialized=false`, memory layers are configured, memory-table gradient/checksum evidence is present, and sparse-row runs report positive row-union plus compact-gradient all-reduce counters.

To run the same memory-transformer manifest v2 substrate while hard-requiring the
guarded Tensor Core flash forward/backward route and double-buffered `cp.async`
GEMM, use the scoped wrapper:

```bash
scripts/gcp/submit_vertex_qb_memory_flash_gate.sh
```

This wrapper keeps the QB data hard path in memory mode, sets per-rank
`batch=1`, `time=512`, `heads=16`, `head_dim=64`, `d_model=1024`, `ff_hidden=4096`,
runs 10 train steps plus one resume step, and requires
`HEIRLOOM_CUDA_FLASH_BF16_ATTENTION=1`,
`HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE=1`,
`HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD=1`,
`HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION=1`, and
`HEIRLOOM_EXPECT_FLASH_BF16_ATTENTION=1`. The QB hard-path wrapper and validator
fail this gate unless train and resume reports show positive flash
forward/backward execution, positive QK/dP/dQ/dK/dV MMA counters, positive flash
CUDA-event timing, zero flash fallback, zero scalar backward tiles, zero hard
failures, zero materialized-reference BF16 attention, and positive
`cp.async` GEMM execution with zero fallback/hard failures.

Recorded gate: Vertex job `6192402191454568448` succeeded and uploaded artifacts
under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-010439/`.
The post-run validator passed with:

```bash
python3 scripts/validate_qb_data_hardpath_artifacts.py \
  --profile full \
  --expected-world-size 4 \
  --require-flash-attention \
  --require-flash-timing \
  gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-010439/qb-data-hardpath/summary.json
```

The train report recorded `loader.kind="binary_shard_streaming"`,
`tokens_materialized=false`, `tokens_seen=20480`,
`tokens_per_second=5441.0201912858665`,
`dense_core_mfu_estimate=0.0015274721106407851`, flash forward/backward executed
`40/40`, QK/dP/dQ/dK/dV backward MMA tiles
`41943040/41943040/5242880/5242880/5242880`, flash fallback/scalar backward
tiles/materialized-reference attention all `0`, `cp.async` GEMM executed `630`
with zero fallback/hard failures, NCCL all-reduce calls/bytes `80/1150976`,
row-union all-reduce calls/bytes `80/327680`, checksum drift `0.0`, and loss
from `7.31654691696167` to `7.253988265991211`. The resume report recorded
flash forward/backward executed `4/4` with the same zero-fallback contract.

For the scale/tuning pass, run the larger exact-tile QB memory wrapper:

```bash
scripts/gcp/submit_vertex_qb_memory_flash_scale_gate.sh
```

This keeps the same 4x A100 NCCL, manifest v2 streaming, sparse-row memory,
SMFT row-mask, Tensor Core flash forward/backward, and `cp.async` hard gates,
but changes the default training shape to per-rank `batch=1`, `block=1024`,
`d_model=1024`, `heads=16`, `head_dim=64`, `ff_hidden=4096`, and grad
accumulation `2`. Validate the uploaded summary with:

```bash
python3 scripts/validate_qb_data_hardpath_artifacts.py \
  --profile full \
  --expected-world-size 4 \
  --require-flash-attention \
  --require-flash-timing \
  --require-exact-tile-shape \
  --min-block-size 1024 \
  --min-d-model 1024 \
  --min-head-dim 64 \
  --min-grad-accumulation-steps 2 \
  --min-tokens-seen 65536 \
  gs://.../qb-data-hardpath/summary.json
```

Recorded scale gate: Vertex job `660398552299601920` succeeded and uploaded
artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-013844/`.
The validator command above returned `status="passed"`.

The train report recorded exact-tile shape `block=1024`, `d_model=1024`,
`heads=16`, `head_dim=64`, grad accumulation `2`, `tokens_seen=65536`,
`tokens_per_second=5892.465383923754`,
`dense_core_mfu_estimate=0.0015504495422534528`, and
`end_to_end_mfu_estimate=0.0015165646661225846`. Flash forward/backward executed
`64/64`, total flash was `600836` us or `9.16802978515625` us/token,
QK/dP/dQ/dK/dV backward MMA tiles were
`268435456/268435456/33554432/33554432/33554432`, flash fallback/scalar backward
tiles/materialized-reference attention were all `0`, and `cp.async` GEMM
executed `1008` with zero fallback/hard failures. NCCL all-reduce calls/bytes
were `64/970752`, row-union calls/bytes were `64/262144`, and checksum drift was
`0.0`. Loss moved `7.280417442321777 -> 7.3151936531066895`; this is expectedly
not a quality gate because the scale wrapper is a throughput/evidence baseline.

This is a scale/tuning evidence gate, not a 40% MFU pass/fail gate. Future runs
can add optional `--min-tokens-per-second` and `--min-dense-core-mfu` thresholds
against this baseline.

For the head-dim shape-coverage pass, isolate wider attention heads without
widening the dense backbone:

```bash
scripts/gcp/submit_vertex_qb_memory_flash_head_dim128_gate.sh
```

This keeps the same `block=1024`, `d_model=1024`, `ff_hidden=4096`, and grad
accumulation `2` scale shape, but changes `heads=8` so the flash attention
kernel runs `head_dim=128`. Validate the uploaded summary with:

```bash
python3 scripts/validate_qb_data_hardpath_artifacts.py \
  --profile full \
  --expected-world-size 4 \
  --require-flash-attention \
  --require-flash-timing \
  --require-exact-tile-shape \
  --min-block-size 1024 \
  --min-d-model 1024 \
  --min-head-dim 128 \
  --min-grad-accumulation-steps 2 \
  --min-tokens-seen 65536 \
  gs://.../qb-data-hardpath/summary.json
```

Recorded head-dim gate: Vertex job `7673373985324662784` succeeded and uploaded
artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-095048/`.
The validator command above returned `status="passed"`.

The train report recorded exact-tile shape `block=1024`, `d_model=1024`,
`heads=8`, `head_dim=128`, grad accumulation `2`, `tokens_seen=65536`,
`tokens_per_second=5715.182698177378`,
`dense_core_mfu_estimate=0.0014982224574761746`, and
`end_to_end_mfu_estimate=0.0014709367939840746`. Flash forward/backward executed
`64/64`, total flash was `968215` us or `14.773788452148438` us/token,
QK/dP/dQ/dK/dV backward MMA tiles were
`536870912/536870912/33554432/33554432/33554432`, flash fallback/scalar backward
tiles/materialized-reference attention were all `0`, and `cp.async` GEMM
executed `1008` with zero fallback/hard failures. NCCL all-reduce calls/bytes
were `64/940032`, row-union calls/bytes were `64/262144`, and checksum drift was
`0.0`. Loss moved `7.2832841873168945 -> 7.3172783851623535`; this remains a
throughput/shape-coverage gate, not a quality gate.

For measured throughput work, use the separate memory throughput wrapper instead
of the correctness gates:

```bash
scripts/gcp/submit_vertex_qb_memory_throughput.sh
```

This keeps the known-good QB memory hard path (`block=1024`, `d_model=1024`,
`heads=16`, `head_dim=64`, `ff_hidden=4096`) but raises grad accumulation to
`4`, runs a 16-step warmup train, then treats the 128-step resumed train as the
measured window. It sets `HEIRLOOM_QB_DATA_HARDPATH_PROFILE=throughput`,
`HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE=release`,
`HEIRLOOM_QB_DATA_HARDPATH_THROUGHPUT_REPORT=resume`, and
`HEIRLOOM_QB_DATA_HARDPATH_SKIP_EVAL_GENERATION=1`, so eval/generation does not
pollute the throughput run and the measured training binary is release-built.

Validate the uploaded summary with:

```bash
python3 scripts/validate_qb_memory_throughput_artifacts.py \
  --expected-world-size 4 \
  --require-flash-attention \
  --require-flash-timing \
  --require-cp-async-gemm \
  --require-exact-tile-shape \
  --require-release \
  --min-block-size 1024 \
  --min-d-model 1024 \
  --min-head-dim 64 \
  --min-grad-accumulation-steps 4 \
  --min-warmup-steps 16 \
  --min-measured-steps 128 \
  --min-tokens-seen 2097152 \
  gs://.../qb-data-hardpath/summary.json
```

This validator reads the selected throughput report from `summary.throughput`,
checks manifest v2 streaming, memory DDP/NCCL evidence, Tensor Core flash
forward/backward, double-buffered `cp.async` GEMM, timing buckets, and emits
bucket shares so MFU work starts from the measured bottleneck rather than the
end-to-end correctness gate.

For a short diagnostic run that times the selected dense Tensor Core GEMM tier
directly, set `HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING=1` and reduce the measured
step count before launch. This inserts per-GEMM CUDA event synchronizations, so
it is for attribution rather than comparable MFU measurement. Validate those
diagnostic artifacts with `--require-cp-async-gemm-timing`.

Recorded GEMM-timing diagnostic: Vertex job `7447965305537560576` succeeded and
uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-215507/`.
It used `HEIRLOOM_QB_DATA_HARDPATH_STEPS=2`,
`HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS=8`, and
`HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING=1`. The timing validator returned
`status="passed"` with `tokens_seen=131072`, `tokens_per_second=6002.289691807483`,
`dense_core_mfu_estimate=0.0015563596443726427`, flash at
`0.05486458994569623` of forward/backward CUDA, and timed `cp.async` GEMM at
`416223` us across `2016` calls, or `0.019202696445407154` of forward/backward
CUDA. Rank 0 recorded `19942` kernel launches and `3232` syncs in the measured
window, so the next MFU diagnostic should attribute launch families and then
fuse/reduce the dominant small kernels.

For the historical count-only launch-family diagnostic, keep GEMM timing
disabled and reduce the warmup/measured steps:

```bash
HEIRLOOM_QB_DATA_HARDPATH_STEPS=2 \
HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS=8 \
HEIRLOOM_QB_DATA_HARDPATH_LOG_EVERY=1 \
HEIRLOOM_QB_DATA_HARDPATH_RESUME_LOG_EVERY=4 \
scripts/gcp/submit_vertex_qb_memory_throughput.sh
```

Validate those diagnostic artifacts by adding `--require-kernel-launch-families`
and lowering the warmup/measured/token thresholds, for example:

```bash
python3 scripts/validate_qb_memory_throughput_artifacts.py \
  --expected-world-size 4 \
  --require-flash-attention \
  --require-flash-timing \
  --require-cp-async-gemm \
  --require-kernel-launch-families \
  --require-exact-tile-shape \
  --require-release \
  --min-block-size 1024 \
  --min-d-model 1024 \
  --min-head-dim 64 \
  --min-grad-accumulation-steps 4 \
  --min-warmup-steps 2 \
  --min-measured-steps 8 \
  --min-tokens-seen 131072 \
  gs://.../qb-data-hardpath/summary.json
```

For elapsed-time launch-family attribution, additionally enable per-launch CUDA
event timing. This synchronizes the compute stream after every kernel launch, so
use it only on the short diagnostic lane and do not compare its tokens/sec to
normal throughput runs:

```bash
HEIRLOOM_QB_DATA_HARDPATH_STEPS=2 \
HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS=8 \
HEIRLOOM_QB_DATA_HARDPATH_LOG_EVERY=1 \
HEIRLOOM_QB_DATA_HARDPATH_RESUME_LOG_EVERY=4 \
HEIRLOOM_CUDA_KERNEL_LAUNCH_FAMILY_TIMING=1 \
scripts/gcp/submit_vertex_qb_memory_throughput.sh
```

Validate elapsed attribution with `--require-kernel-launch-family-timing`:

```bash
python3 scripts/validate_qb_memory_throughput_artifacts.py \
  --expected-world-size 4 \
  --require-flash-attention \
  --require-flash-timing \
  --require-cp-async-gemm \
  --require-kernel-launch-family-timing \
  --require-exact-tile-shape \
  --require-release \
  --min-block-size 1024 \
  --min-d-model 1024 \
  --min-head-dim 64 \
  --min-grad-accumulation-steps 4 \
  --min-warmup-steps 2 \
  --min-measured-steps 8 \
  --min-tokens-seen 131072 \
  gs://.../qb-data-hardpath/summary.json
```

Recorded launch-family diagnostic: Vertex job `2782376829070082048` succeeded
and uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-223949/`.
The validator returned `status="passed"` with `tokens_seen=131072`,
`tokens_per_second=5951.59605866594`,
`dense_core_mfu_estimate=0.001554337522768353`, flash at
`0.05480090885803048` of forward/backward CUDA, and `2016` executed
`cp.async` GEMM calls. Rank 0 recorded `19942` kernel launches. The top launch
families were `vector_elementwise` (`9270` calls, `46.48%`), `bias_2d`
(`3936` calls, `19.74%`), `tensor_core_gemm_cp_async` (`2016` calls, `10.11%`),
`materialize_matrix_layout` (`1344` calls, `6.74%`), `layer_norm` (`864` calls,
`4.33%`), `matmul_strided_f32_reference` (`640` calls, `3.21%`), and
`scaled_vector_elementwise` (`592` calls, `2.97%`). This points the next MFU
implementation pass at fusion/reduction of elementwise, bias, layout,
layernorm, and strided reference-matmul launches.

Recorded fused BF16 roundtrip verification: Vertex job `7398284972148129792`
succeeded and uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-230716/`.
This run used the same short `2 + 8` release-profile lane after replacing AMP
activation `f32 -> bf16 -> f32` with a single CUDA roundtrip kernel. The
validator returned `status="passed"` with `tokens_seen=131072`,
`tokens_per_second=5842.299977713395`,
`dense_core_mfu_estimate=0.0015595487947830517`, flash at
`0.05495560891383791` of forward/backward CUDA, and `2016` executed
`cp.async` GEMM calls. Compared with job `2782376829070082048`, rank-0 total
launches moved `19942 -> 19302`, `vector_elementwise` moved `9270 -> 7990`,
and a new `bf16_roundtrip` family recorded `640` calls. Forward/backward CUDA
elapsed time moved `21703.435668945312 -> 21630.9130859375` ms, but end-to-end
tokens/sec was lower in the short lane. Treat this as a launch-count cleanup,
not a proven throughput improvement. The next high-count candidate is Linear
bias fusion into the Tensor Core GEMM store path, followed by layout
materialization reduction.

Recorded fused Linear bias verification: Vertex job `5917310979254779904`
succeeded and uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260613-005643/`.
This run used the same short `2 + 8` release-profile lane after fusing Linear
bias into the exact-tile double-buffered `cp.async` Tensor Core GEMM store path.
The validator returned `status="passed"` with `tokens_seen=131072`,
`tokens_per_second=5887.701015182823`,
`dense_core_mfu_estimate=0.0015533740387336967`, flash at
`0.05469340232190727` of forward/backward CUDA, and `2016` executed
`cp.async` GEMM calls with zero staged fallback/hard failures. Compared with
job `7398284972148129792`, total measured rank-0 launches moved `19302 ->
18630`, `bias_2d` moved `3936 -> 3264`, and launched elements moved
`19912896596 -> 19006926932`. Dense-core MFU remained effectively flat in the
short lane, so treat this as a correct launch-count cleanup rather than a
proven throughput improvement. The next high-count candidates are
`materialize_matrix_layout`, `matmul_strided_f32_reference`, `layer_norm`, and
remaining elementwise chains.

Recorded contiguous BF16 layout-skip verification: Vertex job
`863075928694063104` succeeded and uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260613-021419/`.
This run used the same short `2 + 8` release-profile lane after changing CUDA
matmul backward to reuse BF16 buffers whose logical layout is already
contiguous row-major. The validator returned `status="passed"` with
`tokens_seen=131072`, `tokens_per_second=5977.653121722078`,
`dense_core_mfu_estimate=0.001552803693424789`, flash at
`0.05473895983926339` of forward/backward CUDA, and `2016` executed
`cp.async` GEMM calls with zero staged fallback/hard failures. Compared with
job `5917310979254779904`, total measured rank-0 launches moved `18630 ->
17958`, `materialize_matrix_layout` moved `1344 -> 672`, and launched elements
moved `19006926932 -> 18100957268`. Dense-core MFU remained effectively flat in
the short lane, so treat this as a launch/layout cleanup rather than a proven
throughput improvement. The next high-count candidates are the remaining
transposed layout materialization, `matmul_strided_f32_reference`,
`layer_norm`, and remaining elementwise chains.

Recorded paired BF16 transpose verification: Vertex job `5016520685036503040`
succeeded and uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260613-023553/`.
This run used the same short `2 + 8` release-profile lane after pairing the two
BF16 transposes that prepare Linear weight-gradient Tensor Core GEMM. The
validator returned `status="passed"` with `tokens_seen=131072`,
`tokens_per_second=5932.470353942246`,
`dense_core_mfu_estimate=0.0015577634370467514`, flash at
`0.05490387110109928` of forward/backward CUDA, and `2016` executed
`cp.async` GEMM calls with zero staged fallback/hard failures. Compared with
job `863075928694063104`, total measured rank-0 launches moved `17958 ->
17286`, `bias_2d` moved `3264 -> 1920`, and the new
`transpose2d_pair_bf16` family recorded `672` calls. Forward/backward CUDA
elapsed improved slightly in the short lane, while end-to-end tokens/sec moved
against the run, so treat this as a launch-count cleanup rather than a proven
throughput improvement. The next high-count candidates are the remaining
transposed layout materialization, `matmul_strided_f32_reference`,
`layer_norm`, and remaining elementwise chains.

Recorded normal-RHS Tensor Core GEMM experiment: Vertex job
`4641877491034619904` succeeded and uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260613-030621/`.
This run used the same short `2 + 8` release-profile lane after routing Linear
input-gradient backward through an exact-tile normal-RHS staged Tensor Core
GEMM. The validator returned `status="passed"` with `tokens_seen=131072`,
`tokens_per_second=5418.437370814386`,
`dense_core_mfu_estimate=0.0015460787869372994`, flash at
`0.0546513040301534` of forward/backward CUDA, `1344` executed `cp.async`
GEMM calls, and `672` `tensor_core_gemm_normal_rhs_staged` calls. Compared
with job `5016520685036503040`, total measured rank-0 launches moved `17286 ->
16614` and `materialize_matrix_layout` moved `672 -> 0`, but forward/backward
CUDA elapsed worsened `21538.968505859375 -> 21819.369567871094` ms on rank 0
and end-to-end tokens/sec regressed. The route is therefore kept opt-in behind
`HEIRLOOM_CUDA_TENSOR_CORE_NORMAL_RHS_GEMM=1` and is not the default measured
throughput path until the normal/strided RHS GEMM has an `ldmatrix`/`cp.async`
instruction path.

Recorded release-profile throughput baseline: Vertex job `1689902075711848448`
succeeded and uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-212601/`.
The validator command above returned `status="passed"` with `--require-release`.

The measured resumed train report recorded `warmup_steps=16`,
`measured_steps=128`, `tokens_seen=2097152`,
`tokens_per_second=6078.807167660886`,
`dense_core_mfu_estimate=0.0015686277322282277`, and
`end_to_end_mfu_estimate=0.0015645241103662447`. CUDA timing buckets were:
`train_elapsed_ms=344994`, `forward_backward_cuda_elapsed_ms=344091.47552490234`,
`all_reduce_cuda_elapsed_ms=1659.6593832969666`,
`host_to_device_cuda_elapsed_ms=199.64879997819665`, and
`optimizer_cuda_elapsed_ms=53.98527976870537`. Flash forward/backward executed
`2048/2048`, total flash was `18984547` us or `9.052537441253662` us/token, and
flash was `0.05517296518618946` of forward/backward CUDA time. `cp.async` GEMM
executed `32256` calls with zero fallback/hard failures. NCCL all-reduce
calls/bytes were `1024/13385728`, row-union calls/bytes were `1024/4194304`,
and checksum drift was `0.0`. Loss moved
`7.274963617324829 -> 7.251679062843323`.

Historical dev-profile baseline: Vertex job `2438906988738904064` succeeded and
uploaded artifacts under
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-203926/`.
The validator command above returned `status="passed"`.

The measured resumed train report recorded `warmup_steps=16`,
`measured_steps=128`, `tokens_seen=2097152`,
`tokens_per_second=6035.629795488428`,
`dense_core_mfu_estimate=0.0015616325048544738`, and
`end_to_end_mfu_estimate=0.0015534113973087484`. CUDA timing buckets were:
`train_elapsed_ms=347462`, `forward_backward_cuda_elapsed_ms=345632.81005859375`,
`all_reduce_cuda_elapsed_ms=1801.2379007339478`,
`host_to_device_cuda_elapsed_ms=460.14483174681664`, and
`optimizer_cuda_elapsed_ms=66.70793595910072`. Flash forward/backward executed
`2048/2048`, total flash was `18982070` us or `9.051356315612793` us/token, and
flash was `0.05491975717462137` of forward/backward CUDA time. `cp.async` GEMM
executed `32256` calls with zero fallback/hard failures. NCCL all-reduce
calls/bytes were `1024/13383680`, row-union calls/bytes were `1024/4194304`,
and checksum drift was `0.0`. Loss moved
`7.27497410774231 -> 7.251707077026367`.

Before another full 4x TinyStories run, prefer an isolated NCCL fabric probe:

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
HEIRLOOM_NCCL_PROBE_KIND=spawn \
HEIRLOOM_NCCL_TRACE=1 \
HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0 \
HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0 \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

Run the ladder in order by setting `HEIRLOOM_NCCL_PROBE_KIND` to `spawn`, then `cuda-context`, then `nccl-init`, then `all-reduce`. The default is `all-reduce`, but the staged ladder makes the first missing boundary obvious. Each probe runs `cargo run --bin heirloom -- gpu nccl-probe`, spawns one hidden rank per CUDA device, writes per-rank config/stage/stdout/stderr/report files, and uploads `nccl-probe.txt`, `nccl-probe.json`, `launcher-report.json`, and the `nccl-probe-ranks/` directory. It intentionally avoids tokenizer/model/checkpoint work so communicator-init failures are not buried under training logs.

For 4x runs, verify topology before interpreting performance: `a2-ultragpu-4g` should expose same-host A100s, but the report must prove what NCCL actually saw. `HEIRLOOM_ENABLE_NCCL_DEBUG=1` defaults NCCL logging to `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH` for distributed TinyStories reference runs and the NCCL probe; `HEIRLOOM_NCCL_TRACE=1` adds Rust-side stage logs around NCCL library loading, primary-context retention, stream creation, and `ncclCommInitRank`. The single-node launcher pins NCCL bootstrap to loopback with `NCCL_SOCKET_IFNAME=lo`, disables IB probing with `NCCL_IB_DISABLE=1`, and forces internal NCCL networking with `NCCL_NET_PLUGIN=none` unless the Heirloom-specific override variables below are set; this avoids inheriting Vertex base-image exclusions such as `NCCL_SOCKET_IFNAME=^...,lo,...` and avoids auto-loading provider plugins such as FastSocket during the single-node bootstrap probe.

Current status: 4x preflight has succeeded on Vertex with four `NVIDIA A100-SXM4-80GB` devices, peer access across every pair, `NV12` links reported by `nvidia-smi topo`, a clean four-rank NCCL init, a clean four-rank f32 all-reduce, a bounded 4x DDP tiny train/resume fixture, a bounded 4x TinyStories DDP reference tier, the full 500-step 4x TinyStories-valid DDP gate, and the full QB data hard-path gate. The failed and successful ladder runs are documented in `runs/vertex-cuda-a100-20260603.md`. `HEIRLOOM_NCCL_NET_PLUGIN=<value>` is forwarded to `NCCL_NET_PLUGIN` when a run intentionally wants a provider plugin instead of the launcher default `none`.

That runs `scripts/run_tinystories_cuda_reference.sh`. By default it downloads `TinyStories-valid.txt`, trains a tokenizer on the first `1048576` bytes, prepares train/valid token files, trains a small CUDA LM for `80` steps, runs CUDA `eval-lm` and CUDA `generate` with the selected precision, and uploads:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/summary.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/resume-report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/eval.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/generation.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/generation.txt
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/timings.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/launcher-report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/train-launcher-report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/resume-launcher-report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/ddp-ranks/
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/train-ddp-ranks/
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/resume-ddp-ranks/
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/checkpoint/
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/<display-name>/tinystories-cuda-reference/checkpoint-resume/
```

A successful public-data CUDA reference run is documented in `runs/vertex-cuda-a100-20260603.md` under `heirloom-validate-quick-20260604-185012`. A successful full uncapped TinyStories-valid CUDA gate is documented there under `heirloom-validate-quick-20260604-193310`. A successful `amp-bf16` public-data reference run with module-level Tensor Core coverage is documented there under `heirloom-validate-quick-20260605-122143`. A successful bounded 4x TinyStories DDP reference tier is documented there under `heirloom-validate-quick-20260606-002304`. The successful full 4x TinyStories DDP gate is documented there under `heirloom-validate-quick-20260606-004954`.

The CUDA reference wrapper supports tiers:

- `HEIRLOOM_TINYSTORIES_CUDA_MODE=smoke`: 1 MiB data slice, 40 small-model steps, relaxed convergence threshold.
- `HEIRLOOM_TINYSTORIES_CUDA_MODE=reference`: 1 MiB data slice, 80 small-model steps, default for quick public-data CUDA evidence.
- `HEIRLOOM_TINYSTORIES_CUDA_MODE=full`: full TinyStories validation file, 500 steps, `d_model=64`, `block_size=64`, a 20% loss-reduction gate, and by default a 5-step checkpoint-resume continuation.
- `HEIRLOOM_TINYSTORIES_CUDA_PRECISION=f32|bf16|amp-bf16`: selects the f32 path, BF16 activation-rounding path, or the current AMP policy surface.
- `HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`: passes `--devices` to `train-lm` for DDP.
- `HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl`: enables the NCCL rank-per-GPU training launcher.
- `HEIRLOOM_TINYSTORIES_CUDA_RESUME_STEPS=<n>`: runs an optional copied-checkpoint resume continuation and emits `resume-report.json`; `full` defaults to `5`, while `smoke` and `reference` default to `0`.
- `HEIRLOOM_COLLECT_GPU_TOPOLOGY=1`: uploads NVIDIA topology/NVLink logs plus Heirloom's CUDA peer-access matrix.
- `HEIRLOOM_ENABLE_NCCL_DEBUG=1`: enables NCCL init/collective/graph debug logging for distributed reference wrappers.
- `HEIRLOOM_RUN_NCCL_PROBE=1`: runs the isolated multi-rank NCCL init/all-reduce probe before CUDA storage tests and any LM reference run.
- `HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`: device list for the isolated NCCL probe; defaults to the GPU smoke device list when omitted.
- `HEIRLOOM_NCCL_PROBE_KIND=spawn|cuda-context|nccl-init|all-reduce`: selects the staged probe boundary; defaults to `all-reduce`.
- `HEIRLOOM_NCCL_PROBE_TIMEOUT_SECS=120`: Rust-side probe timeout, with a Python-side process-group timeout set slightly higher by the wrapper.
- `HEIRLOOM_NCCL_PROBE_RANK_START_TIMEOUT_SECS=10`: fails quickly when a rank never gets beyond the parent-written `spawned` stage.
- `HEIRLOOM_NCCL_PROBE_KILL_GRACE_SECS=5`: grace period after sibling-rank termination before the parent writes the aggregate launcher report.
- `HEIRLOOM_NCCL_TRACE=1`: adds Rust-side NCCL wrapper stage logs.
- `HEIRLOOM_NCCL_SOCKET_IFNAME=lo`: explicit single-node NCCL bootstrap interface override; defaults to `lo` inside the Rust launcher even if the Vertex image exports a different `NCCL_SOCKET_IFNAME`.
- `HEIRLOOM_NCCL_IB_DISABLE=1`: explicit NCCL IB-probing override; defaults to `1` for single-node Heirloom launcher runs.
- `HEIRLOOM_NCCL_NET_PLUGIN=none`: explicit NCCL net plugin override; defaults to `none` for single-node Heirloom launcher runs so provider plugins are not auto-loaded during bootstrap.
- `HEIRLOOM_NCCL_DEBUG` and `HEIRLOOM_NCCL_DEBUG_SUBSYS`: Heirloom-specific debug overrides that map to `NCCL_DEBUG` and `NCCL_DEBUG_SUBSYS` for rank workers.
- `HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0`: skips the broader gated CUDA storage/autograd tests when the goal is a narrow NCCL fabric probe; defaults to `1`.
- `HEIRLOOM_REQUIRE_TENSOR_CORES=1`: hard-fails `amp-bf16` when a projection shape/device cannot use the BF16 Tensor Core Linear forward path or when the compatible Linear backward matmuls cannot use Tensor Cores. The wrappers also fail if the train report has zero Tensor Core forward/backward matmul calls or if `tensor_core_coverage.linear_totals` records any Linear fallback. The default Linear path is the shared-memory staged CTA Tensor Core GEMM; `cuda_runtime.tensor_core_cta_gemm_calls`, `cuda_runtime.tensor_core_cta_tiles`, `cuda_runtime.tensor_core_cta_warps_launched`, `cuda_runtime.tensor_core_mma_warp_tiles`, `cuda_runtime.tensor_core_staged_cta_gemm_calls`, `cuda_runtime.tensor_core_shared_stage_tiles`, and `cuda_runtime.tensor_core_shared_stage_bytes` should be positive in default AMP Tensor Core fixture reports, while `cuda_runtime.tensor_core_wide_swizzled_cta_gemm_calls`, `cuda_runtime.tensor_core_global_cta_gemm_calls`, and `cuda_runtime.tensor_core_legacy_warp_gemm_calls` should stay `0`. It is still not proof of general batched Tensor Core attention GEMM or production AMP.
- `HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM=1`: selects the opt-in eight-warp, 32x32 output-region, XOR-swizzled shared-memory CTA Linear Tensor Core kernel. In this mode, `cuda_runtime.tensor_core_wide_swizzled_cta_gemm_calls`, `cuda_runtime.tensor_core_swizzled_stage_tiles`, and `cuda_runtime.tensor_core_swizzled_stage_bytes` should be positive, while staged/shared, global-fed, and legacy counters stay `0`. This is a validated kernel-family bite, not an `ldmatrix` or fully tuned GEMM.
- `HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM=1`: selects the previous global-memory-fed four-warp CTA Linear Tensor Core kernel for debugging the staged CTA engine. Do not set this for current hard gates.
- `HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM=1`: selects the old one-warp-per-16x8 Linear Tensor Core kernel for debugging. Do not set this for current hard gates.
- `HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES=1`: makes the tiny CUDA training fixture choose ragged Linear dimensions that require Tensor Core pad/crop edge handling.
- `HEIRLOOM_CUDA_FIXTURE_RAGGED_ATTENTION_TENSOR_CORES=1`: makes the tiny CUDA training fixture choose ragged attention dimensions that require Tensor Core attention boundary predicates and edge counters.
- `HEIRLOOM_EXPECT_TENSOR_CORE_PADDING=1`: makes the fixture validator require `tensor_core_pad_crop.status == "passed"` plus positive padded/remainder tile counts in train and resume reports. The raw `cuda_runtime.tensor_core_padded_tiles` and `cuda_runtime.tensor_core_remainder_tiles` counters remain for compatibility.
- `HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORE_PADDING=1`: makes the fixture validator require positive attention padded/remainder tile counters and zero attention scalar fallback counters in train and resume reports.
- `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1`: hard-fails `amp-bf16` when causal-attention QK/AV forward or score-grad/dQ/dK/dV backward matmuls cannot use the BF16 Tensor Core path. Current requirements are CUDA sm80+ and valid nonzero attention dimensions with `channels % n_heads == 0`; ragged `time` and `head_dim` edges are handled with boundary predicates. The softmax backward and transpose glue are still separate f32 CUDA kernels.

The wrapper writes `timings.json`, and `summary.json` includes model config, data hashes/token counts, gate threshold, train/eval/generation summaries, and per-stage timings.

`quick` mode runs:

```text
cargo fmt --all --check
cargo test --workspace --exclude heirloom-python
cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings
```

Set `HEIRLOOM_RUN_LEARNING_SANITY_FIXTURES=1` to add explicit Rust-native
`readiness validate-learning-sanity` passes for the ladder, LR/gradient sweep,
and longer 32K blend fixtures. The worker uploads the resulting reports under
`learning-sanity-fixtures/`, and `summary.json` records their URIs.

Set `HEIRLOOM_RUN_LEARNING_SANITY_SWEEP=1`, or use
`scripts/gcp/submit_vertex_learning_sanity_sweep.sh`, to run the live 4x A100
LR/grad-accumulation sweep. The wrapper defaults to a dense LR grid around the
current best point, `0.015,0.0125,0.01,0.0075`, crossed with grad accumulation
`1,2`. The worker uploads `learning-sanity-sweep/lr-recommendation.json` and
copies its contents into `summary.json` as `learning_sanity_sweep_recommendation`
so the next longer 32K run can consume the recommended LR directly. Set
`HEIRLOOM_LEARNING_SANITY_SWEEP_MIN_BEST_LOSS_REDUCTION` to make the validator
fail if the best run does not clear a required winner threshold.

Recorded improvement sweep: `heirloom-validate-quick-20260619-222355` / job
`2884384020337000448` passed on 4x A100 and recommended
`lr=0.015, grad_accumulation_steps=2` with `loss_reduction=0.797442`.

The Vertex quick gate excludes the optional `heirloom-python` PyO3 binding crate because some Vertex Python images expose a non-PIC Python archive that cannot link a `cdylib`. Local validation still runs `cargo test --workspace` and the Python parity path separately.

For scoped kernel-throughput validation jobs, set `HEIRLOOM_RUN_QUICK_CLIPPY=0` to keep the quick gate on `fmt` plus tests while avoiding unrelated lint failures after the CUDA/A100 evidence has already been uploaded. Leave it unset for the default full quick gate. The uploaded `summary.json` records `quick_clippy == "skipped"` and `quick_clippy_reason == "HEIRLOOM_RUN_QUICK_CLIPPY=0"` when this escape hatch is used.

For the full local gate on Vertex:

```bash
HEIRLOOM_VERTEX_VALIDATE_MODE=full zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

`full` mode runs `./scripts/validate.sh` inside the worker.

To consume the full approved 4x A100 80GB quota explicitly:

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4 \
HEIRLOOM_GPU_SMOKE_DEVICES=all \
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The script packages the current repo, uploads it to the existing artifact bucket, creates a temporary Vertex YAML, submits the custom job, deletes the temporary YAML after submission, and optionally streams logs. The YAML contains no secrets, but it is still treated as temporary launch state.

After a run, document:

- Vertex custom job id
- source package URI
- artifact prefix URI
- `summary.json` URI
- one `gpu-smoke-device-<n>.json` URI per smoked device
- `heirloom-gpu-topology.json`, `nvidia-smi-topo.txt`, `nvidia-smi-nvlink.txt`, and `nvidia-smi-gpu-bus-ids.csv`
- one `tensor-core-probe-device-<n>.json` URI per probed A100 device, when `HEIRLOOM_RUN_TENSOR_CORE_PROBE=1`
- diagnostic log URIs from `summary.json`, especially `cuda-storage-tests.txt`, fixture `run.log`, and TinyStories `run.log` when a gate fails
- non-secret pass/fail summary, including CUDA device names and max absolute errors

## 2026-08-08 committee-readiness validation attempt

Vertex job `2925376863247269888` packaged source revision
`24beb90559601434e904868e7b6a86bdfda74422` and terminated without retry. CUDA
device discovery, topology, smoke (`add_max_abs_error=0`,
`relu_max_abs_error=0`), and the BF16 Tensor Core probe
(`max_abs_error=0`) passed on one A100-SXM4-80GB. The job then failed before
the classified CUDA suite, microbenchmarks, or software gate because the
multi-device `gpu nccl-probe` command correctly rejected a one-device list.

The launcher was subsequently split into an explicit single-rank
`HEIRLOOM_RUN_NCCL_TESTS` lane and a multi-device `HEIRLOOM_RUN_NCCL_PROBE`
lane. It now rejects a multi-device probe with fewer than two requested
accelerators before uploading or submitting a job. The failed attempt is not
current-commit GPU validation evidence; its retained failure summary lives
under `heirloom/reference-runs/heirloom-validate-quick-20260808-174311/` in the
private artifact bucket.

## Boundaries

- GPU/Vertex validation is opt-in only.
- CPU tests and CI must not require cloud access.
- Do not add repo-local `.env` loading to these scripts.
- Do not pass API tokens unless a future GPU job genuinely requires them.
- Keep large public-data or model-download validation behind explicit env flags.
- Do not describe `gpu smoke` as GPU training or full runtime parity. It validates a real GPU kernel boundary only.
