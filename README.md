# Heirloom

Heirloom is a serious Rust prototype of a PyTorch-like tensor and reverse-mode autograd runtime. It is intentionally small and correctness-first. The default training runtime is CPU-only, while an opt-in CUDA storage/math/autograd path now reaches a tiny transformer LM smoke path and a narrow BF16 activation-rounding mode without pretending broad GPU training is done. The point is to expose the hard framework design pressure: shape/stride metadata, views, dtype/device boundaries, dispatch, autograd graph recording, module composition, serialization, reproducibility, and extension seams.

It is not PyTorch parity. The docs call out the gaps instead of hiding them.

It is also not performance-oriented yet: many kernels materialize logical strided data into temporary contiguous buffers before computing. That choice keeps correctness visible while making the CPU-kernel gap explicit.

## Quickstart

```text
cargo test
cargo run --bin train
cargo run --bin classify
cargo run --bin custom
cargo run --example microgpt_heirloom
cargo run --bin heirloom -- --help
```

For the full review gate:

```text
./scripts/validate.sh
```

## Implemented Surface

| Area | Current Status |
| --- | --- |
| Tensor metadata | Shape, strides, storage offset, dtype, device |
| Storage | CPU storage for f32, bf16, f64, i64, and bool; opt-in CUDA storage copies plus f32<->bf16 CUDA casts |
| Views | transpose, permute, narrow, expand, view, reshape, contiguous, checked non-diff as_strided |
| Ops | add, sub, mul, div, matmul, relu, softmax_dim, sum, mean, sum_dim, mean_dim |
| Transformer ops | batched matmul, embedding, gelu, layer_norm_last_dim, masked_fill, causal attention, argmax |
| Autograd | Reverse-mode graph, graph release/retain, non-leaf retain_grad, grad hooks, storage-aware gradient tensors, saved tensor version checks |
| NN | Linear, Embedding, LayerNorm, GELU, CausalSelfAttention, TransformerBlock, TinyTransformerLm, MSE, cross-entropy, SGD, AdamW |
| Runtime support | deterministic RNG, versioned BPE-like tokenizer, prepared token-data manifests, token dataset batching/state, LM eval/perplexity, typed `.npy`, state_dict save/load, LM checkpoints, explicit `train-lm --precision f32|bf16`, opt-in CUDA storage/math/autograd/optimizer smoke gates |
| Dispatch | builtin operator catalog with dtype/device resolution |
| Extensions | Rust custom unary ops with custom backward and a separate user registry; thin PyO3 parity-test bindings |
| Demos | regression training, classification training, custom autograd, tiny LM train/eval/generate CLI |

## Validation

The validation script runs:

```text
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run --bin train
cargo run --bin classify
cargo run --bin custom
cargo run --example microgpt_heirloom
cargo run --bin heirloom -- tokenizer train ...
cargo run --bin heirloom -- data prepare ...
cargo run --bin heirloom -- train-lm ...
cargo run --bin heirloom -- train-lm --precision bf16 ...
cargo run --bin heirloom -- train-lm --resume ...
cargo run --bin heirloom -- eval-lm ...
cargo run --bin heirloom -- eval-lm --precision bf16 ...
cargo run --bin heirloom -- generate ...
cargo run --bin heirloom -- generate --precision bf16 ...
```

The project-local `.venv` contains PyTorch, NumPy, pytest, Hypothesis, and maturin. Rust tests consume the checked fixture at `tests/fixtures/pytorch_parity.json`; refresh it manually with:

```text
.venv/bin/python tools/generate_pytorch_fixtures.py
```

For live Python-vs-PyTorch parity testing, build the thin PyO3 extension into the local venv and run pytest with:

```text
./scripts/python_parity.sh
```

The Python module is intentionally named `heirloom_py` and is a parity harness, not a production Python API. It exposes flat tensor constructors, metadata, materialization, gradients, backward, core ops, view ops, layer norm, embedding, cross-entropy, and causal attention.

To include that Python parity suite in the main validation script when the local venv is available:

```text
HEIRLOOM_RUN_PYTHON_PARITY=1 ./scripts/validate.sh
```

## Review Map

- Start with `ARCHITECTURE.md` for system design.
- Read `HARD_MODE.md` for non-parity gaps.
- Read `OPERATORS.md` for the builtin dispatch catalog.
- Read `EXTENDING.md` for custom autograd support.
- Read `DISPATCH.md` for the production dispatcher plan.
- Read `ALIASING.md` for the storage-aware gradient and view semantics plan.
- Read `scripts/gcp/README.md` for opt-in Vertex validation using the existing GCP setup.
- Read `runs/vertex-cuda-a100-20260603.md` for successful A100 PTX execution reference runs.
- Read `PROGRESS.md` for checkpoint history and validation notes.
- Inspect `src/tensor.rs` plus `src/tensor/autograd.rs` for the core Tensor/autograd split.
- Inspect `src/bin/heirloom.rs` for the tiny LM training/generation CLI.
- Inspect `examples/microgpt_heirloom.rs` for a compact Karpathy-inspired GPT training demo built on Heirloom tensors.

## Reproducibility

Local Docker validation is available with:

```text
./scripts/docker_validate.sh
```

Opt-in Vertex validation is documented in `scripts/gcp/README.md`. It uses the existing GCP project and artifact bucket, does not load repo `.env`, and is not part of the local CPU validation gate.

Opt-in CUDA kernel smoke validation is available with:

```text
./scripts/gpu_smoke.sh
```

That path dynamically loads the CUDA Driver API and runs real GPU kernels from `heirloom-kernels`. The tensor API also has explicit `Device::Cuda(id)`, `Tensor::cuda`, `Tensor::cpu`, CUDA storage round-trips, f32<->bf16 CUDA cast kernels, first Tensor-level CUDA kernels for contiguous same-shape f32 `add`/`sub`/`mul`/`div`/`relu`/`gelu`, `[rows, cols] + [cols]` f32 bias-add broadcasting, rank-2 f32 `matmul` including strided transposed-weight views, f32 embedding gather/scatter-add backward, all-element `sum`/`mean`, last-dimension f32 layer norm, rank-2 f32 cross-entropy loss with tensor targets, strict f32 fused causal attention with saved CUDA softmax weights, CUDA-resident backward for those ops plus cast/2D transpose/view graph nodes, f32 CUDA SGD, f32 CUDA AdamW with device-resident moment buffers, module-level `.to_device`, `train-lm --device cuda:0`, `train-lm --precision bf16` activation rounding, and CPU-compatible checkpoint save/load for CUDA models behind `HEIRLOOM_CUDA_TESTS=1`. `heirloom-kernels` now also dynamically loads NCCL, exposes compute-capability checks, owns a safe `NcclCommunicator`, can all-reduce f32 CUDA buffers behind `HEIRLOOM_NCCL_TESTS=1`, includes a BF16 `mma.sync` probe exposed as `heirloom gpu tensor-core-probe --device 0`, has a BF16 Tensor Core RHS-transposed GEMM for `Linear` projections with explicit device-side pad/crop handling for ragged dimensions, and has boundary-predicated Tensor Core attention matmuls for ragged `time`/`head_dim` edges. The Linear Tensor Core path defaults to a shared-memory staged CTA PTX kernel: four warps per block own a 32x16 output region, cooperatively stage a 32x16 A tile plus 16x16 RHS tile into shared memory for each K slice, and feed four `m16n8k16` MMA warp tiles. An opt-in eight-warp, 32x32, XOR-swizzled shared-memory CTA kernel is available behind `HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM=1` and has a recorded A100 fixture gate with positive wide/swizzled counters and zero staged/global/legacy GEMM calls. The previous global-memory-fed CTA kernel is retained behind `HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM=1`, and the old one-warp-per-output-tile kernel is retained behind `HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM=1`. The CUDA runtime now has first-pass stream/event execution, a per-thread caching allocator, a PTX module cache, JSON-reportable runtime counters, CTA GEMM tile/warp counters, shared/swizzled-stage counters, attention edge tile counters, and a named `tensor_core_pad_crop` report section that exposes pad/crop status, logical/padded shapes, tile counts, and fallbacks. The A100 throughput track has an explicit microbench command, `heirloom gpu tensor-core-microbench --report ...`, guarded request toggles for `HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM=1`, `HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1`, `HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_ATTENTION=1`, and `HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_ATTENTION=1`, plus hard-require flags `HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_LDMATRIX_GEMM=1` and `HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1` that refuse staged fallback. The GEMM `ldmatrix` and double-buffered `cp.async` tiers are A100 microbench-validated as guarded instruction-path kernels, but they are not yet default train-path production kernels. It also has a forward-only scalar-streaming flash-like BF16 causal attention path behind `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION=1` with `HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION=1` hard-require and `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING=1` event timing counters. A separate experimental Tensor Core tiled flash forward candidate is guarded by `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE=1`; it uses tiled QK/online-softmax/PV scheduling with MMA and compact row max/denom state. Separate attention-only Vertex job `6050468434448220160` validated this forward path on A100 for `batch=4, heads=16, time=512, head_dim=64`. Grad-required routing now has an additional experimental `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD=1` guard; for exact-tile sm80 shapes it saves compact row stats and runs no-`[B,H,T,T]` tiled Tensor Core backward kernels for QK recompute, dP, dQ, dK, and dV. Vertex job `894480179806601216` validated the guarded target 4x A100 DDP training shape with positive forward/backward MMA counters, CUDA-event timing, MFU reporting, and zero flash/materialized-reference fallback counters. `amp-bf16` has an explicit policy, per-op precision decisions for transformer-critical ops, strict unexpected host-staging rejection during training, reportable host-staging allowances for scalar logging/checkpoint/eval/generation, finite-check events, and a gateable `amp_bf16_validation` report section. Default train-path production routing, ragged flash backward, bank-conflict/occupancy tuning, tuned batched Tensor Core GEMM, and full production AMP parity are still not done.

Current A100 throughput status: `HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM=1` launches a guarded ldmatrix-A shared-memory MMA kernel, and `HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1` launches a guarded double-buffered `cp.async` global-to-shared copy pipeline feeding the same ldmatrix-A MMA path. GEMM-only Vertex job `9203551123560988672` microbench-validated both tiers on A100 with zero fallback counters; that job did not validate Tensor Core flash attention. Separate attention-only Vertex job `6050468434448220160` validated Tensor Core tiled flash forward for the target A100 shape with `attention.tensor_core_flash_forward.status == "ok"` and `passed == true`. Vertex job `894480179806601216` then validated the guarded training-path flash forward/backward route on 4x A100 DDP for per-rank `batch=4, time=512, heads=16, head_dim=64`, with positive QK/dP/dQ/dK/dV MMA counters and zero flash fallback, scalar fallback, hard-require failure, or materialized-reference attention counters. Vertex job `6192402191454568448` extended that evidence into the QB manifest v2 memory-transformer hard path via `scripts/gcp/submit_vertex_qb_memory_flash_gate.sh`: manifest v2 binary-shard streaming, `train-memory-lm`/resume/eval/generation, NCCL on 4x A100, sparse-row memory updates, SMFT row-mask evidence, Tensor Core flash forward/backward hard-required, and double-buffered `cp.async` GEMM hard-required all passed artifact validation. Vertex job `660398552299601920` then passed the scale/tuning wrapper `scripts/gcp/submit_vertex_qb_memory_flash_scale_gate.sh`, raising the train shape to `block=1024`, `d_model=1024`, `heads=16`, `head_dim=64`, and grad accumulation `2`; it recorded `65536` tokens seen, `5892.465383923754` tokens/sec, flash forward/backward `64/64`, total flash `9.16802978515625` us/token, `cp.async` GEMM executed `1008`, and zero flash/cp.async/materialized-reference fallbacks. Vertex job `7673373985324662784` then passed the isolated head-dim shape gate with `block=1024`, `d_model=1024`, `heads=8`, `head_dim=128`, and grad accumulation `2`; it recorded `5715.182698177378` tokens/sec, flash forward/backward `64/64`, total flash `14.773788452148438` us/token, `cp.async` GEMM executed `1008`, and zero flash/cp.async/materialized-reference fallbacks. The dedicated measured-throughput entrypoint is now `scripts/gcp/submit_vertex_qb_memory_throughput.sh`: it keeps the `block=1024`, `d_model=1024`, `heads=16`, `head_dim=64` QB memory shape, raises grad accumulation to `4`, warms up for 16 steps, measures the 128-step resumed train report, skips eval/generation, and is validated by `scripts/validate_qb_memory_throughput_artifacts.py`. Vertex job `2438906988738904064` passed that measured-throughput lane with `2097152` measured tokens, `6035.629795488428` tokens/sec, `dense_core_mfu_estimate=0.0015616325048544738`, flash at `9.051356315612793` us/token and only `0.05491975717462137` of forward/backward CUDA time, `cp.async` GEMM executed `32256`, and zero flash/cp.async/materialized-reference fallbacks. Attention ldmatrix/cp.async request toggles still report fallback for the materialized attention matmuls. Tensor Core flash attention remains guarded and exact-tile only; it is not default train-path production and ragged flash backward remains pending.

The opt-in CUDA CLI training fixture is:

For measured MFU work, `scripts/gcp/submit_vertex_qb_memory_throughput.sh` now
sets `HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE=release` by default. Release
Vertex job `1689902075711848448` is the current measured baseline:
`6078.807167660886` tokens/sec and
`dense_core_mfu_estimate=0.0015686277322282277`; the earlier job
`2438906988738904064` is retained as the initial dev-profile baseline. Short
diagnostic runs can set `HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING=1` to populate
per-tier GEMM elapsed-us counters, but those synchronized runs are for
attribution, not comparable MFU measurement. Diagnostic Vertex job
`7447965305537560576` passed that timing lane and showed timed `cp.async` GEMM
at only `0.019202696445407154` of forward/backward CUDA time, while flash was
`0.05486458994569623`. The follow-up launch-family diagnostic, Vertex job
`2782376829070082048`, passed with `--require-kernel-launch-families` and
attributed the same short measured window: `vector_elementwise` was `9270` of
`19942` launches (`46.48%`), `bias_2d` was `3936` launches (`19.74%`),
`tensor_core_gemm_cp_async` was `2016` launches (`10.11%`), and
`materialize_matrix_layout` was `1344` launches (`6.74%`). The next MFU lever
is fusion/reduction of the dominant small elementwise, bias, layout, layernorm,
and strided reference-matmul runtime kernels.
The first launch-reduction slice added a fused CUDA `f32 -> bf16 -> f32`
roundtrip for AMP activation rounding. Vertex job `7398284972148129792`
validated it on the same short 4x A100 lane: total rank-0 launches fell
`19942 -> 19302`, `vector_elementwise` fell `9270 -> 7990`, and the new
`bf16_roundtrip` family recorded `640` fused launches. This cleaned up launch
count and slightly reduced forward/backward CUDA time in the short lane
(`21703.435668945312 -> 21630.9130859375` ms), but it did not yet improve
end-to-end tokens/sec. The second launch-reduction slice fuses Linear bias into
the exact-tile double-buffered `cp.async` Tensor Core GEMM store path while
preserving bias gradients through a fused matmul+bias autograd node. Vertex job
`5917310979254779904` passed the same short 4x A100 lane: total measured
rank-0 launches fell `19302 -> 18630`, `bias_2d` fell `3936 -> 3264`, and
launched elements fell by `905969664`. Measured resume tokens/sec moved
`5842.299977713395 -> 5887.701015182823`, while dense-core MFU stayed
effectively flat (`0.0015595487947830517 -> 0.0015533740387336967`), so this is
another correct launch-count cleanup rather than a proven throughput win. The
third launch-reduction slice avoids BF16 layout materialization when CUDA
matmul backward operands are already contiguous row-major. Vertex job
`863075928694063104` passed the same short 4x A100 lane: total measured rank-0
launches fell `18630 -> 17958`, `materialize_matrix_layout` fell `1344 -> 672`,
and launched elements fell by another `905969664`. Measured resume tokens/sec
moved `5887.701015182823 -> 5977.653121722078`, while dense-core MFU stayed
effectively flat (`0.0015533740387336967 -> 0.001552803693424789`), so this is
again a correct launch-count cleanup rather than a proven MFU win. The next
launch-reduction slice pairs the two BF16 transposes that prepare the
weight-gradient Tensor Core GEMM in Linear backward. Vertex job
`5016520685036503040` passed the same short 4x A100 lane: total measured rank-0
launches fell `17958 -> 17286`, `bias_2d` fell `3264 -> 1920`, and the new
`transpose2d_pair_bf16` family recorded `672` paired launches. Forward/backward
CUDA elapsed moved `21724.873901367188 -> 21655.704345703125` ms, but
end-to-end tokens/sec moved `5977.653121722078 -> 5932.470353942246`, so this
is another launch-count cleanup rather than a proven MFU win. The next
high-count target experiment added an exact-tile normal-RHS Tensor Core GEMM
for Linear input-gradient backward so the remaining transposed weight layout
does not have to be materialized. Vertex job `4641877491034619904` validated
the primitive and route on the same short 4x A100 lane, but it is not the
default path: total measured rank-0 launches fell `17286 -> 16614` and
`materialize_matrix_layout` fell `672 -> 0`, while `cp.async` GEMM calls fell
`2016 -> 1344`, `tensor_core_gemm_normal_rhs_staged` appeared with `672`
calls, forward/backward CUDA elapsed worsened `21538.968505859375 ->
21819.369567871094` ms on rank 0, and aggregate tokens/sec moved
`5932.470353942246 -> 5418.437370814386`. The new route is therefore kept
behind `HEIRLOOM_CUDA_TENSOR_CORE_NORMAL_RHS_GEMM=1` until it has an
instruction-path `ldmatrix`/`cp.async` implementation. The next high-count
targets are a faster normal/strided RHS Tensor Core GEMM, strided
reference-matmul backward traffic, layernorm, and remaining elementwise chains.

```text
scripts/cuda_train_lm_fixture.sh
```

By default it targets `cuda:0`, trains a tiny LM through the real `train-lm --device` CLI, requires loss decrease, resumes from checkpoint, and validates the resumed step count. Set `HEIRLOOM_CUDA_FIXTURE_PRECISION=bf16` to exercise the BF16 activation-rounding path. A recorded single-A100 Vertex run of this fixture is tracked in `runs/vertex-cuda-a100-20260603.md`. Set `HEIRLOOM_CUDA_FIXTURE_DEVICE=cpu` to exercise its report/resume logic without a GPU.

The public-data manual gate is:

```text
./scripts/run_tinystories_valid.sh
```

That script downloads `TinyStories-valid.txt`, trains a 1024-token BPE-like tokenizer, prepares a manifest-backed train/valid token dataset, trains the default tiny decoder LM, writes `report.json`, evaluates heldout validation loss/perplexity into `eval.json`, generates sampled text with repetition controls into `generation.json`, and fails if loss does not drop by the configured threshold.

The opt-in small public-data CUDA reference wrapper is:

```text
./scripts/run_tinystories_cuda_reference.sh
```

It defaults to `cuda:0`, a 1 MiB slice of `TinyStories-valid.txt`, a smaller model, and JSON `summary.json`/train/eval/generation reports. Recorded single-A100 Vertex runs, including the full uncapped TinyStories-valid gate, are tracked in `runs/vertex-cuda-a100-20260603.md`.

Set `HEIRLOOM_TINYSTORIES_CUDA_MODE=smoke|reference|full` to choose the gate tier, `HEIRLOOM_TINYSTORIES_CUDA_PRECISION=bf16|amp-bf16` to choose the precision policy, and `HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl` to request the rank-per-GPU path. `full` removes the byte cap and uses the default 64-wide, 500-step TinyStories validation configuration; it remains opt-in because the CUDA kernels are correctness-first and still intentionally slow. The full 4x A100 TinyStories-valid pass is frozen in `MILESTONE_4X_TINYSTORIES.md`.

The QB-native data hard path is `./scripts/run_qb_data_hardpath.sh`. It prepares manifest v2 binary token shards from TinyStories plus deterministic QB trace fixtures and reports timing/MFU estimates. The default dense mode trains/resumes/evals/generates through the streaming loader; `HEIRLOOM_QB_DATA_HARDPATH_MODEL=memory` trains/resumes/evals/generates with `train-memory-lm`, `eval-memory-lm`, and `generate-memory-lm` on the same manifest v2 streaming substrate with memory selection/SMFT evidence. The full dense and memory 4x A100 passes are documented in `scripts/gcp/README.md`.

The governed corpus materializer is:

```text
cargo run --bin heirloom -- data materialize-blend \
  --corpus-blend corpus-blend.json \
  --tokenizer tokenizer.json \
  --out-dir materialized \
  --target-tokens 20000000000 \
  --mode full
```

It reads a `heirloom.corpus_blend` source-governance manifest, applies approved
license-status gates and deterministic curation filters, writes text shards plus
manifest v2 binary token shards, and records `curation-report.json`,
`source-index.json`, `selected-docs.jsonl`, and `tokenizer-sample-manifest.json`.
The curation report includes phase timings, aggregate scan/write throughput,
per-source source-file counts, scan/write throughput, rejection counts, and
limit flags. Long source scans can emit bounded heartbeat logs with
`--progress-every-records` and/or `--progress-every-bytes`.
Scale runs can also bound retained candidates with
`--candidate-retention-token-multiplier`, `--candidate-retention-min-docs`, and
`--candidate-prune-every`; a multiplier of `1.0` or higher keeps the same
best-score prefix needed to satisfy each source quota while pruning lower-score
candidates during scan.
Use `--candidate-text-mode rescan` to retain only candidate metadata during
scan; the writer then re-reads sources, hash-prefilters selected documents, and
tokenizes only selected hashes when emitting text and token shards.
Long scans can checkpoint the expensive candidate-selection phase with
`--checkpoint-dir`, `--checkpoint-every-records`, and
`--checkpoint-every-bytes`; restart with `--resume-checkpoint` to load
compatible per-source scan checkpoints and continue from the saved file
cursor. Checkpoint resume covers scan/candidate state; text and token shard
outputs are still written atomically by the successful materialization run.
`sample` mode is for bounded local gates; `full` mode requires selecting at
least 95% of the requested target tokens.
Every curation report also includes a `sizing` block with selected text bytes,
estimated token dtype, token payload bytes, train/valid token estimates, and
estimated text/token shard counts.

Measure native tokenizer encode throughput with:

```text
cargo run --bin heirloom -- tokenizer bench-encode \
  --tokenizer tokenizer.json \
  --input source-jsonl-or-directory \
  --max-bytes 67108864 \
  --iterations 3 \
  --report tokenizer-encode-bench.json
```

The report records tokenizer identity, digit-isolation/reserved-token settings,
sample size, warmup/encode timings, tokens/sec, bytes/sec, docs/sec, and
tokens/byte. It is intended as a local throughput guardrail for the
materializer, where tokenizer encode cost is currently the dominant scan
bucket.

Production source slices should live in GCS, not on a developer laptop. The
tokenizer hard path accepts local file paths, flat local source directories,
single-object `gs://` URIs, or flat `gs://` prefixes ending in `/` in
`HEIRLOOM_QB_TOKENIZER_DOLMA_PATH`,
`HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH`,
`HEIRLOOM_QB_TOKENIZER_OLMO3_PATH`,
`HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH`, and
`HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH`. `gs://` inputs are staged to the
Vertex worker's local disk before materialization; prefix inputs are copied into
flat local directories. The generated corpus-blend manifest records the
original GCS URI.

Create bounded private Hugging Face source slices with:

```text
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
```

The slicer uses `HF_TOKEN` from the environment, or from
`/Users/andrewverdiramo/Desktop/VECL-QB/.env` when needed, but never writes the
token into reports. Outputs are uncompressed JSONL with source-file metadata and
hashes. With `--shard-output-bytes`, `--out` is treated as a directory and the
slicer writes deterministic `*-part-NNNNN.jsonl` files plus a file-set hash in
the report. URL-list inputs such as Dolma's `urls/v1_7.txt` do not use or
transmit the HF token.

Stage locally available governed slices and write the inventory with:

```text
python3 scripts/stage_qb_source_slices.py --upload \
  --project project-49b1b523-d248-434f-bd4
```

See `QB_SOURCE_GOVERNANCE.md` for the source approval rule and current
external-source blockers.

The QB-native tokenizer hard path is:

```text
./scripts/run_qb_tokenizer_hardpath.sh
```

It trains a native Heirloom byte-level BPE tokenizer artifact v2 from a
corpus-blend manifest, validates the reserved-token registry, writes fertility
reports, materializes governed source samples into manifest v2 binary shards,
and runs memory train/resume/eval/generation. `smoke` mode uses non-TinyStories
local fixtures plus available VECL-QB traces; `full` mode requires
caller-provided approved Dolma/Nemotron/OLMo/QB source paths, defaults to a 32K
vocab, and targets 20B materialized tokens unless overridden.
New production tokenizer v2 artifacts use forced ASCII digit isolation and the
expanded 128-token reserved registry, so learned BPE merges do not absorb math,
code, or schema digits into inconsistent multi-character tokens.

## Suggested Review Questions

- Are view semantics honest enough, especially around expand, aliasing, and mutation?
- Are dtype promotion choices explicit and tested where they diverge from PyTorch?
- Are saved tensor version checks applied consistently to builtin, transformer, and custom autograd paths?
- Is the builtin operator registry a useful stepping stone toward a real dispatcher?
- Are extension points narrow but well-bounded, especially the separate custom registry versus builtin dispatch catalog boundary?

## Known Hard Gaps

Heirloom lacks broad CUDA math/autograd/model training, CUDA batched matmul, default train-path production routing for the A100 microbench-validated `ldmatrix`/`cp.async` GEMM instruction tiers, tuned/general Tensor Core BF16 kernels, real production AMP parity, multi-node distributed training, general CUDA broadcasting/view kernels, true backend dispatch keys, dynamic operator registration, full TensorIterator-style broadcasting, higher-order gradients, activation checkpointing, alias-graph-aware gradient accumulation beyond the current explicit training-loop accumulation path, a production Python API, production corpus/dataloader tooling beyond the first binary-shard hard path, and production serialization.

Those are not accidental omissions. They are tracked in `HARD_MODE.md`.
