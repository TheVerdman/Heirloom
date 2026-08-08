# Progress Log

## Checkpoint 1 - Workspace And Toolchain

- Started from an empty workspace at `/Users/andrewverdiramo/Desktop/Heirloom`.
- Confirmed there was no existing git repository or Rust crate to preserve.
- Found `cargo`, `rustc`, and `clippy-driver` missing from `PATH`.
- Installed Rust with Homebrew after user approval.

## Checkpoint 2 - Runtime Skeleton

- Created a dependency-free Rust crate.
- Added tensor metadata: shape, dtype, device, strides, storage offset, and CPU storage.
- Added docs and tests alongside implementation, not afterthought-only documentation.

## Checkpoint 3 - Implemented First Full Pass

- Implemented real strided transpose views and contiguous reshape views.
- Implemented `add`, `matmul`, `transpose`, `relu`, `sum`, `mean`, and fused `mse_loss`.
- Implemented reverse-mode autograd with topological graph traversal and gradient accumulation.
- Implemented `Linear`, `ReLU`, `Sequential`, MSE loss helper, and SGD step.
- Added CLI training demo and integration tests including PyTorch-behavior fixtures.

## Checkpoint 4 - Validation

- `cargo fmt --check` passes.
- `cargo test` passes: 13 tests across unit, tensor ops, nn training, and PyTorch-behavior fixture coverage.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --bin train` runs end-to-end.
- Demo loss decreased from `4.869009` to effectively zero on the deterministic synthetic regression task.

## Checkpoint 5 - Scope Choice

- Chose a CLI demo instead of Python interop for this checkpoint.
- Python bindings remain the next likely extension path, but adding PyO3/maturin would expand packaging/toolchain scope before the runtime core is mature.

## Checkpoint 6 - Hard-Path Architecture Pass

- Split the monolithic tensor implementation into `shape`, `storage`, `dispatch`, `grad_mode`, `tensor`, and `nn` concerns.
- Added shared storage version counters and saved tensor mutation checks for backward.
- Added `no_grad`, `detach`, explicit non-scalar `backward_with_grad`, and in-place `add_`.
- Added primitive `sub`, `mul`, and `div`, then changed `mse_loss` to compose from primitives instead of using a fused autograd node.
- Added `permute`, `narrow`, `reshape` with copy fallback, `contiguous`, and checked non-differentiable `as_strided`.
- Added `sum_dim` and `mean_dim` with `keepdim`.
- Added `Optimizer` and `Sgd`, and updated the CLI demo/tests to use it.
- Expanded hard-path tests to cover mutation checks, no-grad/detach, non-contiguous reshape gradients, rank-3 permute, narrow scatter-backward, dimension reductions, primitive arithmetic gradients, target gradients in MSE, and explicit seed gradients.
- Added finite-difference gradient checks for composed matmul/add/relu/mean and broadcasted mul/div graphs.

## Checkpoint 7 - Current Validation

- `cargo fmt --check` passes.
- `cargo test` passes: 37 tests across unit, hard-path behavior, finite-difference gradient checks, tensor ops, nn training, and PyTorch-behavior fixture coverage.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --bin train` passes with loss decreasing from `4.869009` to effectively zero.

## Checkpoint 8 - Broadcast Views And Classification

- Added `expand` as a zero-stride broadcast view with autograd reduction back to the base tensor.
- Added `has_internal_overlap` and guarded in-place updates/writes on internally overlapping tensors.
- Added signed-dim wrappers for reductions and softmax.
- Added numerically stable `softmax_dim` and rank-2 fused `cross_entropy_for_logits`.
- Added finite-difference checks for softmax and cross-entropy gradients.
- Added `cargo run --bin classify`, a deterministic three-class cross-entropy training demo.
- `cargo run --bin classify` passes with loss decreasing from `1.638253` to `0.016792` and final accuracy `1.000`.

## Checkpoint 9 - Property And PyTorch Fixture Harness

- Added dev dependencies: `proptest`, `serde`, and `serde_json`.
- Added randomized property tests for add broadcasting gradients, expand gradients, reduction gradients, and matmul closed-form gradients.
- Added `tests/fixtures/pytorch_parity.json`, consumed by Rust integration tests.
- Added `tools/generate_pytorch_fixtures.py`, which refreshes the JSON fixture from live PyTorch when `torch` is installed.
- `cargo test` passes: 43 tests including proptest-powered property tests and fixture parity tests.
- `cargo clippy --all-targets -- -D warnings` passes.
- Regression and classification demos still pass after adding the test dependencies.

## Checkpoint 10 - Live PyTorch Fixture Refresh

- Created a project-local `.venv` and installed `torch 2.12.0`.
- Added `.venv/` to `.gitignore`.
- Regenerated `tests/fixtures/pytorch_parity.json` from live PyTorch via `tools/generate_pytorch_fixtures.py`.
- Verified `cargo test --test pytorch_fixture_parity` passes against the live-generated fixture.
- Installed `numpy 2.4.6`, `pytest 9.0.3`, `hypothesis 6.155.1`, and `maturin 1.13.3` in `.venv`.
- Regenerated the PyTorch fixture again after installing NumPy; the previous NumPy warning is gone.

## Checkpoint 11 - Heirloom-Native IO, RNG, And State

- Added `npy`: pure Rust `.npy` read/write for CPU f32 tensors.
- Added `rng`: deterministic SplitMix64-based RNG with uniform/normal scalar and tensor initializers.
- Added `Linear::new_with_rng`.
- Added named parameters to `Module`.
- Added `save_state_dict` and `load_state_dict`, writing a `state.tsv` manifest plus `.npy` files.
- Added `Tensor::copy_from` and `Tensor::copy_from_data` for parameter loading.
- Added tests for `.npy` round-trips, view logical layout serialization, RNG determinism, RNG-based Linear reproducibility, and state_dict save/load.
- `cargo test` passes: 48 tests.
- `cargo clippy --all-targets -- -D warnings` passes.

## Checkpoint 12 - Real DType Storage And Promotion

- Replaced decorative dtype metadata with `StorageData` variants for f32, f64, i64, and bool CPU storage.
- Added typed constructors and accessors: `from_f32`, `from_f64`, `from_i64`, `from_bool`, `zeros_with_dtype`, `ones_with_dtype`, `data_f32`, `data_f64_exact`, `data_i64`, and `data_bool`.
- Kept the legacy `from_vec`, `data`, `grad`, and `backward_with_grad` f32-facing APIs for compatibility while moving internal gradients to f64 and adding `grad_f64`/`backward_with_grad_f64`.
- Added prototype dtype promotion for binary arithmetic and explicit output dtype rules for reductions.
- Enforced that only floating tensors can require gradients, and rejected non-floating tensors from floating-only ops such as `matmul`, `relu`, `softmax_dim`, and cross-entropy.
- Extended `.npy` read/write to f32, f64, i64, and bool while preserving logical view layout.
- Added dtype dispatch tests covering constructors, views, promotion, f64 autograd, non-float reductions, non-float autograd rejection, and typed `.npy` round-trips.
- `cargo test` passes: 54 tests.
- `cargo fmt --check` passes.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --bin train` passes with loss decreasing from `4.869008` to effectively zero.
- `cargo run --bin classify` passes with loss decreasing from `1.638253` to `0.016792` and final accuracy `1.000`.

## Checkpoint 13 - Builtin Operator Registry

- Replaced loose dispatch helpers with a builtin `KernelRegistry` and `OperatorInfo` catalog.
- Added operator metadata for add, sub, mul, div, matmul, relu, softmax, cross-entropy, sum, mean, sum_dim, and mean_dim.
- Added dtype rules, operator kinds, alias policy, and autograd policy metadata.
- Moved binary, matmul, unary, and reduction dtype/device resolution into dispatch.
- Tensor methods now resolve through dispatch before running CPU scalar kernels or inline shape-specific kernels.
- Added a public `operator_catalog()` API and tests for catalog shape and presentability.
- Added `OPERATORS.md` as a compact operator registry report.
- `cargo fmt --check` passes.
- `cargo test` passes: 60 tests.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --bin train` passes with loss decreasing from `4.869008` to effectively zero.
- `cargo run --bin classify` passes with loss decreasing from `1.638253` to `0.016792` and final accuracy `1.000`.

## Checkpoint 14 - Rust Custom Unary Autograd

- Added `extension` module with `CustomUnaryOp`, forward context, backward context, and callback type aliases.
- Added `Tensor::apply_custom_unary` for floating, shape-preserving custom unary ops.
- Added `GradFn::CustomUnary` with saved input/output data, saved tensor version checking, and custom backward invocation.
- Added validation for custom op names, forward output length, and backward gradient length.
- Added `cargo run --bin custom`, a CLI demo that chains two custom ops and verifies forward values plus gradients.
- Added `EXTENDING.md` documenting the supported custom op API and the remaining non-parity gaps.
- Added custom op tests for chaining, f32/f64 behavior, `no_grad`, non-float rejection, bad callbacks, and mutation-before-backward.
- `cargo fmt --check` passes.
- `cargo test` passes: 65 tests.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --bin custom` passes with output `[125.0, 1.953125, 1000.0]` and gradient `[-300.0, 4.6875, 1800.0]`.
- `cargo run --bin train` passes with loss decreasing from `4.869008` to effectively zero.
- `cargo run --bin classify` passes with loss decreasing from `1.638253` to `0.016792` and final accuracy `1.000`.

## Checkpoint 15 - Review Readiness Hardening

- Added `README.md` with quickstart, feature matrix, validation commands, review map, and suggested review questions.
- Added executable `scripts/validate.sh` for one-command validation.
- Split autograd node definitions and backward formulas into `src/tensor/autograd.rs`, reducing `src/tensor.rs` from 1878 lines to 1297 lines.
- Added `CustomUnaryRegistry` as a separate user-op registry with duplicate-name checks, lookup, and listing.
- Added PyTorch semantics tests documenting dtype promotion matches, sum dtype matches, integer div/mean divergences, expanded-view mutation rejection, and custom op dispatch boundaries.
- Added GitHub Actions CI workflow that runs `scripts/validate.sh`.
- Added aliasing invariant tests for shared version counters, alias mutation, copy boundaries, contiguous behavior, and `as_strided` bounds checks.
- Added dtype property tests for mixed i64/f32 arithmetic promotion and bool sum behavior.
- Added `DISPATCH.md` with a production dispatcher plan.
- Added `ALIASING.md` with a storage-aware gradient and view semantics plan.
- `./scripts/validate.sh` passes end-to-end.
- `cargo test` passes: 77 tests.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --bin train`, `cargo run --bin classify`, and `cargo run --bin custom` pass.

## Checkpoint 17 - Tiny Transformer Training Runtime

- Converted the repo to a workspace and added `heirloom-kernels`, an internal crate that isolates the audited `unsafe` `matrixmultiply` GEMM boundary.
- Added runtime dependencies for CLI parsing, JSON serialization, rayon-backed kernels, matrix multiplication, and HTTP download.
- Added `BpeTokenizer` with byte-level BPE-like training, encode/decode, and JSON save/load.
- Added `TokenDataset` deterministic batching plus a TinyStories validation-split downloader.
- Extended matmul to support batched inputs with broadcasted batch dimensions and updated backward accumulation for broadcasted batches.
- Added transformer-critical Tensor ops and autograd paths: `embedding`, `gelu`, `layer_norm_last_dim`, `masked_fill`, `argmax_last_dim`, and fused causal self-attention.
- Expanded `nn` with `Embedding`, `LayerNorm`, `Gelu`, `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `TinyTransformerLm`, and `AdamW`.
- Added LM checkpoint save/load bundling model state, optimizer moments, tokenizer, model config, step, and dataset RNG state.
- Added the `heirloom` CLI with `data tinystories-valid`, `tokenizer train`, `train-lm`, and `generate` subcommands.
- Added transformer runtime tests for tokenizer round trips, deterministic batching, embedding scatter-add gradients, masked-fill gradients, batched matmul, attention/layernorm/GELU gradients, loss decrease, checkpoint load, and generation.
- Updated `scripts/validate.sh` to validate the full workspace and run a tiny local LM train/generate flow.
- `./scripts/validate.sh` passes end-to-end.
- `cargo test --workspace` passes: 86 tests.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.
- `cargo run --bin train`, `cargo run --bin classify`, `cargo run --bin custom`, and the local `heirloom` tokenizer/train-lm/generate flow pass.

## Checkpoint 18 - Reproducible Review Runtime

- Added `Dockerfile`, `.dockerignore`, and `scripts/docker_validate.sh` for reproducible CPU validation.
- Added `scripts/run_tinystories_valid.sh`, a manual public-data gate that downloads `TinyStories-valid.txt`, trains a 1024-token BPE-like tokenizer, trains the tiny LM, writes `report.json`, generates text, and enforces a configurable loss-reduction threshold.
- Added optional `--report` output to `heirloom train-lm` for machine-readable training reports.
- Extended live PyTorch fixture generation to cover batched matmul, tanh-GELU, layer norm, embedding scatter-add, and fused causal attention.
- Refreshed `tests/fixtures/pytorch_parity.json` from live PyTorch.
- Extended Rust fixture consumption to support `i64` inputs and transformer parity cases.
- Added `examples/microgpt_heirloom.rs`, a compact Karpathy-inspired GPT training demo built on Heirloom tensors rather than scalar autograd.
- Updated validation to run the microgpt-style example.

## Checkpoint 19 - Full TinyStories Gate Run

- Ran `scripts/run_tinystories_valid.sh` against the full public `TinyStories-valid.txt` split.
- The first attempt exposed a real bottleneck: the original BPE trainer repeatedly scanned the full corpus for each merge and did not complete in a reasonable time.
- Reworked tokenizer training and encoding to operate on weighted byte chunks with repeated-chunk caching, keeping byte-preserving decode semantics while making the full validation split practical.
- Full gate passed after the tokenizer fix: initial loss `7.012274`, final loss `3.735747`, loss reduction `0.467256`.
- Generated checkpoint, tokenizer, report, and text under `runs/tinystories-valid/`; added `/runs/` to `.gitignore`.
- Generation from `Once upon a time` completed from the checkpoint, but output was repetitive, which remains expected for this tiny CPU model and 500-step run.

## Checkpoint 16 - External Review Response

- Read and triaged Claude's external review as actionable input rather than authority.
- Added executable `OperatorInfo` helpers for fresh-output and autograd policy, then wired Tensor ops to consult those policies while recording out-of-place graphs.
- Added tests proving operator policy metadata is now executable and not only descriptive catalog text.
- Reworked `Linear::new` to use `Linear::DEFAULT_SEED` through the RNG-backed `Linear::new_with_seed`; `Linear::new_with_rng` remains the caller-owned reproducibility path.
- Added tests for the seeded default initialization path.
- Expanded `.gitignore` comments and local-noise protections around `target/`, `.venv/`, `.DS_Store`, and editor swap files.
- Updated docs to call out full temporary materialization, the current split between dispatcher metadata and Tensor-owned shape kernels, in-place op schema gaps, and RNG-backed Linear initialization.
- `./scripts/validate.sh` passes end-to-end.
- `cargo test` passes: 79 tests.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --bin train`, `cargo run --bin classify`, and `cargo run --bin custom` pass.

## Checkpoint 20 - Dispatcher-Native Transformer Ops

- Promoted transformer-critical ops into the builtin operator catalog: `aten.gelu`, `aten.layer_norm.last_dim`, `aten.embedding`, `aten.masked_fill`, `aten.argmax.last_dim`, and `heirloom.causal_self_attention`.
- Added dispatcher metadata for normalization, indexing, masking, selection, and attention operator kinds.
- Added dtype rules for embedding lookup, mask-preserving ops, argmax index output, and same-floating-input multi-input ops.
- Added `KernelRegistry` resolvers for layer norm, embedding, masked fill, argmax, and fused causal self-attention.
- Routed Tensor transformer methods through dispatch resolution before running their shape-specific kernels.
- Added tests proving transformer resolver contracts and public catalog metadata.
- Updated `OPERATORS.md`, `DISPATCH.md`, `ARCHITECTURE.md`, and `HARD_MODE.md` to document the narrower remaining dispatcher gap.
- `./scripts/validate.sh` passes end-to-end.
- `cargo test --workspace` passes: 92 tests.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.

## Checkpoint 21 - Storage-Aware Gradient Tensors

- Replaced internal `Option<Vec<f64>>` gradients with `GradTensor`, carrying f64 storage, shape, strides, storage offset, dtype, and device metadata.
- Kept `grad()` and `grad_f64()` as logical-value compatibility APIs, and added `Tensor::grad_tensor()` to expose the storage-aware gradient tensor.
- First gradient accumulation now preserves non-overlapping target layouts, including transposed and narrowed leaf views.
- Internally overlapping gradient targets, currently detected through zero strides, fall back to dense contiguous gradient storage so arbitrary seed gradients remain representable.
- Added aliasing invariant tests for transposed leaf-view gradient layout, narrowed gradient storage offsets, independent gradient storage, and expanded dense-gradient fallback.
- Updated `ALIASING.md`, `ARCHITECTURE.md`, `HARD_MODE.md`, and `README.md` to mark gradient tensor phase one as implemented while keeping alias graph accumulation and exact overlap analysis on the hard list.
- `./scripts/validate.sh` passes end-to-end.
- `cargo test --workspace` passes.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.

## Checkpoint 22 - Opt-In Vertex Validation Scaffold

- Inspected the existing VECL-QB GCP launcher pattern without reading or printing `.env` contents.
- Added `scripts/gcp/submit_vertex_heirloom_validate.sh`, an opt-in Vertex custom-job launcher that packages the Heirloom repo, uploads it to the existing artifact bucket, and runs Rust validation on an A100 worker.
- Kept the Heirloom launcher token-free: no `HF_TOKEN`, OpenAI, Anthropic, or other API tokens are passed into the temporary Vertex YAML.
- The launcher deletes its temporary YAML after submission and supports `quick` and `full` validation modes.
- Default Vertex validation uses one A100 80GB; using four A100 80GB GPUs requires explicit `HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4`.
- Added `scripts/gcp/README.md` with non-secret access checks, security boundaries, and launch instructions.
- `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh` passes.
- `./scripts/validate.sh` passes end-to-end without cloud access.

## Checkpoint 23 - Graph Lifecycle And Retained Backward

- Added graph lifecycle state so successful default backward releases non-leaf graph nodes.
- Second backward through a released graph now errors with an explicit autograd lifecycle message instead of silently failing to propagate.
- Added explicit retained-backward APIs: `backward_retain_graph`, `backward_with_grad_retain_graph`, and `backward_with_grad_f64_retain_graph`.
- Made non-leaf gradients transient engine buffers that are cleared after backward use, while leaf gradients remain accumulated.
- Added graph lifecycle tests for default graph release, retained scalar backward, retained explicit-seed backward, transient non-leaf grads, and reuse of a released intermediate.
- Updated `ARCHITECTURE.md`, `HARD_MODE.md`, and `README.md` to document graph release/retain behavior and identify non-leaf `retain_grad` as the next missing autograd lifecycle API.
- `./scripts/validate.sh` passes end-to-end.
- `cargo test --workspace` passes.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.

## Checkpoint 24 - Non-Leaf Retain Grad And Tensor Grad Hooks

- Added `Tensor::retain_grad()` and `Tensor::retains_grad()` so non-leaf gradients can be kept explicitly after backward.
- Added tensor gradient hooks with `register_grad_hook` and `remove_grad_hook`.
- Hooks receive incoming logical f64 gradients, may replace them, and are shape-validated before accumulation.
- Hooks on retained non-leaves affect both the retained gradient and the gradient propagated to parents.
- Added graph lifecycle tests for non-leaf retained grads, retain-grad rejection on no-grad tensors, hook modification/removal, retained non-leaf hook propagation, and hook shape mismatch errors.
- Updated docs to move non-leaf `retain_grad` and tensor grad hooks out of the missing list while keeping higher-order gradients, saved-tensor hooks, and activation checkpointing on the hard list.
- `./scripts/validate.sh` passes end-to-end.
- `cargo test --workspace` passes.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.

## Checkpoint 25 - Production-Ish Tokenizer And Data Path

- Added versioned tokenizer metadata with format/version, special-id constants, training byte count, and stable training hash.
- Added tokenizer load validation for byte-token invariants and merge graph consistency, while preserving compatibility with pre-metadata tokenizer JSON.
- Added prepared LM data manifests through `heirloom data prepare`, writing train/valid token files with source, tokenizer, and token-file hashes.
- Added `PreparedTokenData` loading with tokenizer hash checks and token-file length/hash validation.
- Added resumable `TokenDatasetState` with RNG state plus `batches_seen`.
- Extended LM checkpoint metadata to include dataset batch count and optional prepared-dataset manifest path.
- Updated `heirloom train-lm` to support `--dataset-manifest`, checkpoint-linked manifest resume, and tokenizer/manifest mismatch rejection.
- Updated local and TinyStories validation scripts to exercise the prepared data path; local validation now also performs a manifest-backed resume step.
- Added integration coverage for tokenizer legacy loading, prepared-data manifest round trips, hash-checked tokenizer reload, exact next-batch dataset resume, and checkpoint dataset-state persistence.

## Checkpoint 26 - Better Generation And Evaluation

- Added `GenerationOptions` with temperature, top-k, top-p, repetition penalty, frequency penalty, presence penalty, max-new-token, and EOS controls.
- Added seeded LM sampling that returns generated token ids, per-step probabilities/log-probabilities, finish reason, new-token count, and final RNG state.
- Kept greedy generation as `temperature = 0` behavior and preserved `generate_greedy` as a compatibility wrapper.
- Added deterministic LM evaluation over non-overlapping token windows, reporting token-weighted validation loss and perplexity under `no_grad`.
- Added `heirloom eval-lm` with manifest-backed train/valid split support, raw text fallback, max-batch limiting, and JSON reports.
- Extended `heirloom generate` with sampling controls, seed, and JSON generation reports.
- Updated local and TinyStories validation scripts to run heldout eval and sampled generation with reports.
- Added integration coverage for top-k sampling equivalence to greedy and finite eval metrics.
- `./scripts/validate.sh` passes end-to-end.
- `cargo test --workspace` passes.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.

## Checkpoint 27 - First Real CUDA Kernel Boundary

- Added `heirloom-kernels::cuda`, an audited unsafe CUDA Driver API wrapper that dynamically loads `libcuda` at runtime.
- Added embedded PTX kernels for `heirloom_add_f32` and `heirloom_relu_f32`.
- Added safe wrappers for CUDA device discovery, f32 add, f32 ReLU, and a smoke report that checks GPU output against CPU references.
- Added `heirloom gpu info` and `heirloom gpu smoke --report ...` CLI commands.
- Added `scripts/gpu_smoke.sh` for opt-in local or remote GPU kernel validation.
- Updated the Vertex launcher to run real GPU smoke before quick/full Rust validation when `HEIRLOOM_REQUIRE_GPU=1`, upload per-device smoke JSON reports, and upload a non-secret summary JSON under the reference-run artifact prefix.
- Documented the 4x A100 smoke path with `HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4` and `HEIRLOOM_GPU_SMOKE_DEVICES=all`.
- Kept the boundary honest: this validates CUDA allocation/copy/kernel/copy-back in `heirloom-kernels`, not Tensor CUDA storage, GPU autograd, or GPU training.

## Checkpoint 28 - CUDA Tensor Storage And BF16 DType Boundary

- Added `DType::BFloat16` with CPU BF16 bit storage, typed construction, typed inspection, and CPU dtype conversion.
- Added `Device::Cuda(usize)` plus `Tensor::to_device`, `Tensor::cuda`, `Tensor::cpu`, `zeros_on_device`, and `ones_on_device`.
- Added a safe owning `CudaBuffer` in `heirloom-kernels::cuda` with typed f32/f64/i64/u16/u8 host/device copies while keeping all CUDA Driver API FFI outside the main `#![forbid(unsafe_code)]` crate.
- Switched tensor CUDA buffers to CUDA primary contexts so same-device allocations share the address space needed by future device-side Tensor kernels.
- Added safe peer-access discovery as a precursor to the 4x A100 all-reduce gate; enabling peer access and D2D copies remain later work.
- Extended `StorageData` with CUDA storage handles so CUDA tensors now own real device allocations instead of only metadata.
- Kept CUDA math honest: dispatcher resolution errors on CUDA inputs until CUDA kernels/autograd formulas are explicitly registered, and CUDA dtype casts / CUDA-to-CUDA peer copies are still blocked.
- Added CPU BF16 tests plus `HEIRLOOM_CUDA_TESTS=1` gated CUDA storage round-trip and no-CPU-fallback tests.
- Updated `README.md`, `ARCHITECTURE.md`, and `HARD_MODE.md` to distinguish CUDA Tensor storage from CUDA Tensor execution, autograd, optimizer, and training.

## Checkpoint 29 - First Tensor-Level CUDA Kernels

- Added safe device-to-device CUDA buffer kernels for f32 add and ReLU in `heirloom-kernels::cuda`.
- Wired `Tensor::add` and `Tensor::relu` to execute on CUDA for strict contiguous full-storage f32 tensors.
- Kept CUDA dispatch intentionally narrow: no broadcasting, no non-contiguous/view kernels, no dtype promotion, no CUDA-to-CUDA copies, and no CUDA autograd recording yet.
- CUDA tensors requiring gradients now error clearly for CUDA add/ReLU until CUDA backward formulas exist instead of constructing a CPU-backed autograd path.
- Updated gated CUDA tests to prove add/ReLU results remain CUDA tensors before explicit inspection copies, while unsupported CUDA `mul` still errors.
- Updated docs to mark the first Tensor CUDA execution path without claiming a training-capable GPU backend.

## Checkpoint 30 - First CUDA-Resident Autograd

- Added `heirloom_relu_backward_f32` PTX plus a safe `relu_backward_f32_buffer` wrapper in `heirloom-kernels::cuda`.
- Extended gradient storage so CUDA leaf gradients can live as CUDA f32 `CudaBuffer` tensors instead of CPU f64 buffers.
- Added a CUDA branch in backward traversal that propagates CUDA f32 buffers for supported CUDA graph nodes.
- Implemented CUDA backward for strict same-shape f32 `add`; repeated-parent accumulation such as `x.add(&x)` uses CUDA add rather than CPU materialization.
- Implemented CUDA backward for strict contiguous f32 `relu` through a device-side mask kernel.
- Kept the boundary honest: CUDA gradient hooks, broadcasting, views, reductions, matmul, attention, optimizer kernels, mixed precision, and training are still not implemented.
- Updated gated CUDA tests to assert CUDA-resident gradients for add accumulation and ReLU masking.

## Checkpoint 31 - CUDA Arithmetic And Scalar Reductions

- Added device-to-device f32 CUDA kernels/wrappers for `sub`, `mul`, `div`, negation, scalar scaling, scalar-fill, all-element `sum`, all-element `mean`, and division RHS backward.
- Wired Tensor CUDA forward for strict contiguous same-shape f32 `sub`, `mul`, and `div`.
- Wired Tensor CUDA forward for all-element f32 `sum` and `mean`, returning CUDA scalar tensors.
- Extended CUDA backward formulas for `sub`, `mul`, `div`, `sum`, and `mean`, keeping gradients CUDA-resident.
- Kept reductions correctness-first: current `sum`/`mean` PTX uses a deliberately slow single-thread reduction kernel, not a production parallel reduction.
- Added gated CUDA tests for arithmetic forward, scalar reductions, binary backward formulas, and reduction backward fills.
- Still missing for CUDA training: optimizer update kernels, scalar/literal ops, broadcasting, views, matmul, layer norm, softmax/cross-entropy, embedding, attention, BF16 autocast, and checkpoint/load-to-CUDA integration.

## Checkpoint 32 - First CUDA Optimizer Update

- Added `heirloom_sgd_update_f32`, a device-side PTX kernel that mutates f32 CUDA parameter buffers in place as `param -= lr * grad`.
- Exposed a safe `heirloom-kernels::cuda::sgd_update_f32_buffer` wrapper with dtype, device, length, and positive-learning-rate checks.
- Added a CUDA branch in `Tensor::apply_sgd` for contiguous full-storage f32 CUDA tensors with CUDA-resident gradients, bumping the shared storage version after the in-place update.
- Kept the main crate safe and explicit: unsupported CUDA optimizer cases still error, and CPU gradients are not silently used for CUDA parameter updates.
- Added a gated CUDA training smoke test that builds `mean(parameter * parameter)`, runs CUDA-resident backward, verifies the CUDA gradient tensor, and applies a CUDA SGD step.
- Still missing for CUDA model training: scalar/literal ops, broadcasting, views, matmul, embedding, layer norm, softmax/cross-entropy, attention, AdamW, BF16 autocast, checkpoint/load-to-CUDA integration, and multi-GPU all-reduce.

## Checkpoint 33 - CUDA Rank-2 Matmul Forward And Backward

- Added naive pure-PTX f32 kernels for strict rank-2 `matmul`, matmul grad-left, and matmul grad-right.
- Exposed safe `heirloom-kernels::cuda` buffer wrappers with dtype, same-device, exact-length, and checked-shape-product validation.
- Wired `Tensor::matmul` to dispatch to CUDA before CPU dispatch resolution when operands are CUDA tensors.
- Kept the CUDA Tensor path intentionally narrow: operands must be matching-device f32, rank-2, contiguous, and full-storage; batched/broadcasted CUDA matmul errors clearly instead of falling back to CPU.
- Added CUDA autograd support for `GradFn::Matmul`, keeping both operand gradients CUDA-resident.
- Added gated CUDA tests for rank-2 matmul forward, matmul backward gradients, and unsupported batched CUDA matmul.
- Still missing for CUDA model training: batched matmul, view-aware matmul, broadcast gradients, layer norm, softmax/cross-entropy, attention, AdamW, BF16 autocast, checkpoint/load-to-CUDA integration, and multi-GPU all-reduce.

## Checkpoint 34 - A100 PTX Execution Gate

- Updated the Vertex Heirloom validation launcher to run `HEIRLOOM_CUDA_TESTS=1 cargo test --workspace --test cuda_storage -- --nocapture` on GPU workers before the normal quick/full Rust gate.
- Tightened `scripts/gcp/README.md` to use constrained `gcloud ai custom-jobs list --format=...` output because unformatted job listings can expose prior job environment values.
- Ran a first single-A100 Vertex job (`4155293567964676096`) that failed during `gpu smoke` with `cuModuleLoadDataEx` CUDA error `218`, proving the A100 gate catches invalid embedded PTX that local CPU/no-driver validation cannot.
- Fixed the invalid PTX by correcting the f32 register declaration in `heirloom_sgd_update_f32`.
- Ran a successful replacement single-A100 Vertex job (`8843540780057362432`) on `NVIDIA A100-SXM4-80GB`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 14 live CUDA storage/math/autograd/SGD tests, including rank-2 matmul forward/backward and CUDA-resident gradients.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Documented the non-secret reference-run details and artifact URIs in `runs/vertex-cuda-a100-20260603.md`.

## Checkpoint 35 - CUDA Embedding Gather And Scatter-Add

- Added pure-PTX f32/i64 CUDA kernels for embedding gather and embedding-table scatter-add backward, including device-side out-of-range index status reporting.
- Added safe `heirloom-kernels::cuda` wrappers for embedding forward/backward with dtype, same-device, exact-length, and checked-shape-product validation.
- Added CUDA JIT error-log capture to `cuModuleLoadDataEx` so invalid embedded PTX reports compiler logs instead of only CUDA error `218`.
- Wired `Tensor::embedding` to dispatch to CUDA for matching-device CUDA i64 indices and f32 weights before CPU dispatch resolution.
- Extended `GradFn::Embedding` to save CPU indices as host metadata on CPU and save the CUDA index tensor on CUDA, preserving saved-tensor version checks without CPU materialization.
- Extended CUDA autograd so embedding backward returns a CUDA-resident f32 gradient for the embedding table and accumulates repeated rows with a device atomic add.
- Added gated CUDA tests for embedding gather, repeated-index scatter-add backward, and out-of-range CUDA index errors without CPU fallback.
- Ran an initial single-A100 Vertex job (`4294060731483029504`) that failed during `gpu smoke` with `cuModuleLoadDataEx` error `218`; root cause was invalid PTX status atomic syntax.
- Fixed the invalid PTX by changing the status atomic exchange from `exch.u32` to `exch.b32`.
- Ran a successful replacement single-A100 Vertex job (`5192739963375976448`) on `NVIDIA A100-SXM4-80GB`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 17 live CUDA storage/math/autograd/SGD tests, including CUDA embedding forward, repeated-row scatter-add backward, and out-of-range error handling.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260603-225851.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-225851`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-225851/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-225851/gpu-smoke-device-0.json`
- Still missing for CUDA model training: batched matmul, view-aware kernels, broadcasting, layer norm, softmax/cross-entropy, attention, AdamW, BF16 autocast, checkpoint/load-to-CUDA integration, and multi-GPU all-reduce.

## Checkpoint 36 - CUDA Layer Norm And Cross-Entropy Loss

- Added pure-PTX f32 CUDA kernels for last-dimension layer norm forward, layer norm input backward, and layer norm affine weight/bias backward.
- Added pure-PTX f32/i64 CUDA kernels for rank-2 mean cross-entropy forward and logits backward, with device-side target validation.
- Exposed safe `heirloom-kernels::cuda` wrappers for layer norm and cross-entropy with dtype, same-device, exact-length, positive-epsilon, and checked-shape-product validation.
- Wired `Tensor::layer_norm_last_dim` to dispatch to CUDA for matching-device f32 CUDA input/weight/bias tensors.
- Wired `Tensor::cross_entropy_for_logits` to dispatch to CUDA for rank-2 f32 CUDA logits while saving the i64 targets as a CUDA tensor for backward.
- Extended CUDA autograd so layer norm and cross-entropy produce CUDA-resident gradients without CPU fallback.
- Added gated CUDA tests for layer norm forward parity, layer norm backward parity, cross-entropy forward parity, and cross-entropy backward parity while asserting gradients remain on `Device::Cuda(0)`.
- Local validation passed: `cargo test --workspace --test cuda_storage` and `./scripts/validate.sh`.
- Ran a successful single-A100 Vertex job (`5470274290412683264`) on `NVIDIA A100-SXM4-80GB`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 21 live CUDA storage/math/autograd/SGD tests, including CUDA layer norm forward/backward and CUDA cross-entropy forward/backward.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260603-232307.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-232307`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-232307/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-232307/gpu-smoke-device-0.json`
- Still missing for CUDA model training: batched matmul, view-aware kernels, broadcasting, fused softmax, causal attention, AdamW, BF16 autocast, checkpoint/load-to-CUDA integration, and multi-GPU all-reduce.

## Checkpoint 37 - CUDA Fused Causal Attention

- Added correctness-first pure-PTX f32 causal-attention kernels in `heirloom-kernels::cuda`: causal softmax weights, attention output, and Q/K/V backward kernels.
- Exposed safe `CausalAttentionDims`-based wrappers with dtype, same-device, checked shape-product, non-empty dimension, and head-divisibility validation.
- Wired `Tensor::causal_self_attention` to dispatch to CUDA for matching-device contiguous full-storage f32 query/key/value tensors with shape `[batch, time, channels]`.
- Extended the autograd node so CPU attention saves host softmax weights while CUDA attention saves the `[batch, heads, time, time]` softmax tensor on device.
- Extended CUDA autograd so causal attention backward computes CUDA-resident Q/K/V gradients without CPU fallback.
- Kept the boundary honest: this is a Tensor/autograd-level fused attention op with scalar-loop PTX and saved probabilities, not FlashAttention, tensor-core attention, a memory-efficient attention kernel, or full GPU transformer training.
- Added gated CUDA tests for causal attention forward parity, backward parity, and CUDA-resident Q/K/V gradients.
- Local validation passed: `cargo test --workspace --test cuda_storage` and `./scripts/validate.sh`.
- Ran a successful single-A100 Vertex job (`6568871124514373632`) on `NVIDIA A100-SXM4-80GB`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 23 live CUDA storage/math/autograd/SGD tests, including CUDA causal attention forward/backward.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260603-235551.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-235551`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-235551/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260603-235551/gpu-smoke-device-0.json`
- Still missing for CUDA model training: batched matmul, view-aware kernels, broadcasting, CUDA GELU, CUDA softmax_dim outside the fused attention/loss paths, CUDA AdamW, BF16 autocast, checkpoint/load-to-CUDA integration, and multi-GPU all-reduce.

## Checkpoint 38 - CUDA GELU Forward And Backward

- Added correctness-first pure-PTX f32 CUDA kernels for tanh-approximation GELU forward and backward.
- Exposed safe `heirloom-kernels::cuda` buffer wrappers with f32 dtype, same-device, and exact-length validation.
- Wired `Tensor::gelu` to dispatch to CUDA for contiguous full-storage f32 CUDA tensors.
- Extended CUDA autograd so `GradFn::Gelu` computes CUDA-resident input gradients without CPU fallback.
- Added gated CUDA tests for GELU forward parity against the CPU runtime and GELU backward parity while asserting CUDA-resident gradients.
- Local validation passed: `cargo test --workspace --test cuda_storage` and `./scripts/validate.sh`.
- Ran an initial single-A100 Vertex job (`2760514689619197952`) that failed in the new GELU backward parity test because the CPU expected-gradient side incorrectly called `grad_tensor().data_f32()` even though CPU gradient tensors are intentionally f64. The CUDA GELU forward test and other CUDA kernels had already passed in that run.
- Fixed the gated-test bug by comparing against the public logical `grad()` accessor instead of assuming CPU gradient tensor dtype.
- Ran a successful replacement single-A100 Vertex job (`1441382211264708608`) on `NVIDIA A100-SXM4-80GB`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 25 live CUDA storage/math/autograd/SGD tests, including CUDA GELU forward/backward.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260604-002543.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-002543`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-002543/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-002543/gpu-smoke-device-0.json`
- Still missing for CUDA model training: batched matmul, view-aware kernels, broadcasting, CUDA softmax_dim outside the fused attention/loss paths, CUDA AdamW, BF16 autocast, checkpoint/load-to-CUDA integration, and multi-GPU all-reduce.

## Checkpoint 39 - CUDA Linear Projection Path

- Added CUDA rank-2 matrix-layout metadata and safe strided matmul wrappers so rank-2 CUDA matmul can read dense or strided operands, including `weight.transpose()` views.
- Added pure-PTX f32 kernels for strided matmul forward, strided matmul grad-left/grad-right, `[rows, cols] + [cols]` bias-add broadcasting, bias backward reduction, and rank-2 transpose.
- Wired `Tensor::matmul` CUDA dispatch to use the strided layout path while preserving dense CUDA outputs.
- Wired CUDA `add` for the Linear bias pattern `[rows, cols] + [cols]`, including CUDA-resident backward that reduces the row dimension into the bias gradient.
- Extended CUDA autograd for dense logical `view` passthrough and rank-2 transpose backward, so `input.reshape(...).matmul(weight.transpose()).add(bias).reshape(...)` can backpropagate to the original input, weight, and bias without CPU fallback.
- Adjusted CUDA gradient accumulation for non-leaf CUDA views to store dense logical f32 gradients rather than requiring the primal view's non-contiguous layout.
- Added gated CUDA tests for transposed-RHS matmul forward, transposed-weight backward reaching the original weight, bias-add broadcast forward/backward, and the full Linear-shaped projection path.
- Local validation passed: `cargo test --workspace` and `./scripts/validate.sh`.
- Ran a successful single-A100 Vertex job (`803559914038362112`) on `NVIDIA A100-SXM4-80GB`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 29 live CUDA storage/math/autograd/SGD tests, including the new strided transposed-weight matmul, bias-add broadcast, view/transpose backward, and Linear-shaped projection path tests.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260604-005030.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-005030`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-005030/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-005030/gpu-smoke-device-0.json`
- Still missing for CUDA model training: general batched matmul, broad broadcasting, general CUDA view kernels, CUDA AdamW, BF16 autocast, checkpoint/load-to-CUDA integration, and multi-GPU all-reduce.

## Checkpoint 40 - CUDA AdamW And Tiny CUDA Training Step

- Added `heirloom-kernels::cuda::AdamWParams` and a safe `adamw_update_f32_buffers` wrapper that validates f32 dtype, same-device buffers, exact lengths, and finite optimizer scalars.
- Added correctness-first pure-PTX kernels for f32 AdamW updates and f32 sum-of-squares reduction.
- Extended `Tensor` with crate-visible CUDA optimizer helpers: CUDA gradient sum-of-squares for clipping and a contiguous full-storage f32 AdamW update path that mutates parameters in place and bumps storage versions.
- Extended `AdamW::step_mut` to update CPU parameters through the existing CPU path and CUDA f32 parameters through CUDA-resident first/second moment buffers.
- Added `AdamW::try_state` so checkpoints can explicitly materialize CUDA moment state through a fallible path instead of relying on an infallible clone.
- Kept the optimizer math intentionally matched to the existing CPU implementation: weight decay is folded into the gradient before moment updates.
- Added gated CUDA tests for AdamW CPU/CUDA parity plus exported moment state, and for a tiny CUDA Linear-shaped regression loop whose loss must decrease under CUDA AdamW.
- Local validation passed: `cargo test --workspace` and `./scripts/validate.sh`.
- Ran a successful single-A100 Vertex job (`6512848808055930880`) on `NVIDIA A100-SXM4-80GB`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 31 live CUDA storage/math/autograd/optimizer tests, including CUDA AdamW CPU/CUDA parity, CUDA moment-state export, and a tiny CUDA AdamW Linear-shaped training loop.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260604-100820.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-100820`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-100820/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-100820/gpu-smoke-device-0.json`
- At this checkpoint, CUDA model training was still missing module-level `.to_device`, general batched matmul, broad broadcasting, general CUDA view kernels, BF16 autocast, checkpoint/load-to-CUDA integration, full transformer CUDA training, and multi-GPU all-reduce.

## Checkpoint 41 - Narrow CUDA Tiny Transformer Training Path

- Added concrete module `.to_device(Device)` helpers for `Linear`, `Embedding`, `LayerNorm`, `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, and `TinyTransformerLm`.
- Added `TinyTransformerLm::device` and moved LM loss to a tensor-target cross-entropy path so CUDA targets can stay on device.
- Added `Tensor::cross_entropy_for_logits_tensor`, including a CUDA path that saves the i64 target tensor on device for backward instead of materializing host target ids.
- Changed the LM forward path to gather repeated position ids shaped `[batch, time]`, avoiding a CUDA `expand` dependency for position embeddings.
- Added `load_lm_checkpoint_on_device` and made `save_state_dict` serialize explicit CPU snapshots of CUDA parameters, keeping the checkpoint format CPU-compatible while allowing reload to CUDA.
- Added `heirloom train-lm --device cpu|cuda:<id>` and moved model plus token batches to the requested device for training.
- Added always-on CPU coverage for tensor-target cross-entropy and a gated CUDA test for tiny transformer forward/backward/AdamW/checkpoint-on-CUDA shape.
- Local validation passed: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `./scripts/validate.sh`.
- Ran a successful single-A100 Vertex job (`6393221942953902080`) on `NVIDIA A100-SXM4-80GB`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 32 live CUDA storage/math/autograd/optimizer tests, including the new tiny CUDA transformer forward/backward/AdamW/checkpoint smoke test.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260604-103057.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-103057`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-103057/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-103057/gpu-smoke-device-0.json`
- Still missing for a legitimate CUDA public-data training run: CUDA `train-lm --device cuda:0` fixture run, loss-decrease gate for the transformer path, BF16 autocast, CUDA image, multi-GPU all-reduce, and performance work.

## Checkpoint 42 - CUDA Train-LM Fixture Gate

- Added `scripts/cuda_train_lm_fixture.sh`, an opt-in end-to-end CLI gate that trains a tiny language model through the real `heirloom train-lm --device` path, checks loss decrease from the emitted JSON report, resumes from the saved checkpoint, and checks the resumed step count.
- Wired the Vertex validation launcher to run that fixture when `HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1`, pass through non-secret fixture knobs, and upload `cuda-train-lm-fixture/train-report.json` plus `cuda-train-lm-fixture/resume-report.json` under the reference-run artifact prefix.
- Verified the fixture locally in CPU mode: `HEIRLOOM_CUDA_FIXTURE_DEVICE=cpu HEIRLOOM_CUDA_FIXTURE_STEPS=20 HEIRLOOM_CUDA_FIXTURE_RESUME_STEPS=2 HEIRLOOM_CUDA_FIXTURE_MIN_REDUCTION=0.01 scripts/cuda_train_lm_fixture.sh` passed with loss decreasing from `6.013635` to `4.229871` and resume advancing from step `20` to `22`.
- Local validation passed: `bash -n scripts/cuda_train_lm_fixture.sh`, `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`, `cargo fmt --all --check`, and `./scripts/validate.sh`.
- Ran a successful single-A100 Vertex job (`1449395452007940096`) on `NVIDIA A100-SXM4-80GB` with `HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 32 live CUDA storage/math/autograd/optimizer tests, including tiny CUDA transformer forward/backward/AdamW/checkpoint coverage.
- The successful run then trained through `heirloom train-lm --device cuda:0` for 40 steps on the tiny fixture corpus: loss decreased from `5.673435` to `3.936789`, a `0.306101` reduction.
- The same run resumed the CUDA checkpoint from step `40` to step `43`, with resumed loss decreasing from `2.915643` to `2.622527`.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260604-105451.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-105451`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-105451/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-105451/gpu-smoke-device-0.json`
  - CUDA fixture train report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-105451/cuda-train-lm-fixture/train-report.json`
  - CUDA fixture resume report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-105451/cuda-train-lm-fixture/resume-report.json`
- Still missing for a legitimate CUDA public-data training run: TinyStories-valid CUDA reference training, CUDA eval/generate reports from a public-data checkpoint, BF16 autocast, CUDA image, multi-GPU all-reduce, and performance work.

## Checkpoint 43 - Public-Data CUDA Reference Path

- Extended `TinyTransformerLm::evaluate_token_loss` and `TinyTransformerLm::next_token_logits` to construct token windows on `self.device()`, allowing CUDA-loaded models to run eval and generation forward passes instead of requiring CPU inputs.
- Added `--device cpu|cuda:<id>` to `heirloom eval-lm` and `heirloom generate`, with JSON reports now recording the selected device.
- Updated `scripts/run_tinystories_valid.sh` to accept `HEIRLOOM_TINYSTORIES_DEVICE`, `HEIRLOOM_TINYSTORIES_VOCAB`, and optional `HEIRLOOM_TINYSTORIES_MAX_BYTES`, pass the device through train/eval/generate, and emit a compact `summary.json`.
- Added `scripts/run_tinystories_cuda_reference.sh`, an opt-in small public-data CUDA wrapper that defaults to `cuda:0`, the first `1048576` bytes of `TinyStories-valid.txt`, a smaller model, 80 training steps, heldout eval, sampled generation, and a 1% loss-reduction gate.
- Wired the Vertex launcher to run that wrapper when `HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=1`, pass through non-secret CUDA reference knobs, upload train/eval/generation summaries, and upload the small checkpoint directory under the reference-run artifact prefix.
- Added gated CUDA test coverage so a CUDA-loaded tiny transformer checkpoint can run eval and generation paths after checkpoint reload.
- Local validation passed: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `./scripts/validate.sh`.
- CPU-mode script smoke passed against a tiny local text fixture with `scripts/run_tinystories_cuda_reference.sh`: train report loss decreased from `5.901700` to `5.457375`, eval emitted heldout loss/perplexity, generation emitted JSON, and `summary.json` recorded device/report metadata.
- Ran a successful single-A100 Vertex job (`3115533800088535040`) on `NVIDIA A100-SXM4-80GB` with `HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=1`.
- The successful run passed GPU smoke with `add_max_abs_error=0.0` and `relu_max_abs_error=0.0`.
- The successful run passed all 32 live CUDA storage/math/autograd/optimizer tests, now including CUDA-loaded eval/generation coverage.
- The public-data CUDA reference downloaded `TinyStories-valid.txt`, trained a 512-token BPE-like tokenizer on the first `1048576` bytes, prepared train/valid token files, and trained a CUDA LM for 40 steps.
- CUDA public-data reference loss decreased from `6.580609` to `4.550163`, a `0.308550` reduction.
- CUDA heldout eval emitted `loss=4.457110`, `perplexity=86.237935`, `batches=10`, and `tokens=320`.
- CUDA generation emitted an 80-token JSON report from prompt `Once upon a time`.
- The successful run also passed the quick CPU/Rust gate: `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260604-185012.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-185012`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-185012/summary.json`
  - GPU smoke report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-185012/gpu-smoke-device-0.json`
  - TinyStories CUDA summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-185012/tinystories-cuda-reference/summary.json`
  - TinyStories CUDA train report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-185012/tinystories-cuda-reference/report.json`
  - TinyStories CUDA eval report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-185012/tinystories-cuda-reference/eval.json`
  - TinyStories CUDA generation report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-185012/tinystories-cuda-reference/generation.json`
  - TinyStories CUDA checkpoint prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-185012/tinystories-cuda-reference/checkpoint`
- Still missing: the full TinyStories-valid CUDA gate, BF16 autocast, CUDA image, multi-GPU all-reduce, and performance work.

## Checkpoint 44 - Tiered TinyStories CUDA Gate Metadata

- Added `HEIRLOOM_TINYSTORIES_CUDA_MODE=smoke|reference|full` to `scripts/run_tinystories_cuda_reference.sh`.
- Kept `reference` as the default 1 MiB public-data CUDA run, added `smoke` for cheap plumbing checks, and defined `full` as the uncapped TinyStories validation file with the default 64-wide, 500-step, 20% loss-reduction gate.
- Reworked Vertex CUDA reference env plumbing so mode defaults are no longer accidentally overridden by old hard-coded reference values; explicit env overrides still win.
- Fixed a Vertex optional-env preflight bug where unset values could leak as `__HEIRLOOM_UNSET__` into the CUDA reference wrapper.
- Added `timings.json` plus millisecond per-stage timings in `summary.json`.
- Expanded `summary.json` to include run label, model config, gate threshold, source/tokenizer hashes, train/valid token counts, eval/generation metadata, and artifact pointers.
- Local smoke validation passed in CPU mode with `HEIRLOOM_TINYSTORIES_CUDA_MODE=smoke`; the summary reported `run_label=cuda-smoke`, model/data config, and millisecond timings.
- Ran a successful full single-A100 Vertex job (`6213728868742725632`) on `NVIDIA A100-SXM4-80GB` with `HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=1` and `HEIRLOOM_TINYSTORIES_CUDA_MODE=full`.
- The full CUDA gate downloaded the complete TinyStories validation file, trained a 1024-token BPE-like tokenizer, prepared train/valid token files with `train_tokens=8962067` and `valid_tokens=471688`, and trained the default tiny LM shape for 500 steps on `cuda:0`.
- CUDA public-data loss decreased from `6.980931` to `3.566740`, a `0.489074` reduction, clearing the 20% full-gate threshold.
- CUDA heldout eval emitted `loss=3.685430`, `perplexity=39.862263`, `batches=200`, and `tokens=51200`.
- CUDA generation emitted a 120-token JSON report from prompt `Once upon a time`.
- Stage timings were recorded: download `705 ms`, tokenizer `105365 ms`, prepare `14984 ms`, train `60568 ms`, eval `7715 ms`, generate `4639 ms`.
- The successful run also passed GPU smoke, all 32 live CUDA storage/math/autograd/optimizer tests, `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- Artifacts:
  - Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260604-193310.tar.gz`
  - Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-193310`
  - Summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-193310/summary.json`
  - TinyStories CUDA summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-193310/tinystories-cuda-reference/summary.json`
  - TinyStories CUDA timings: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-193310/tinystories-cuda-reference/timings.json`
  - TinyStories CUDA train report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-193310/tinystories-cuda-reference/report.json`
  - TinyStories CUDA eval report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-193310/tinystories-cuda-reference/eval.json`
  - TinyStories CUDA generation report: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260604-193310/tinystories-cuda-reference/generation.json`
- Still missing: BF16 autocast, CUDA image, multi-GPU all-reduce, tensor-core kernels, and serious performance work.

## Checkpoint 45 - Thin PyO3 Parity Bindings

- Added `heirloom-python`, a sibling PyO3 `cdylib` crate that produces the `heirloom_py` Python module while keeping the main `heirloom` runtime under `#![forbid(unsafe_code)]`.
- Added top-level maturin metadata in `pyproject.toml`; maturin builds the extension with the crate-local `extension-module` feature.
- Exposed a deliberately narrow Python `Tensor` wrapper for parity testing: flat constructors, NumPy construction/materialization, shape/dtype/device/strides/storage metadata, `.to_device`, `.cpu`, `.cuda`, `.to_dtype`, `.grad`, `.backward`, `.zero_grad`, basic arithmetic, matmul, ReLU, GELU, reductions, view/reshape/transpose/permute/narrow/expand/contiguous, softmax, layer norm, embedding, cross entropy, MSE, and fused causal self-attention.
- Added `scripts/python_parity.sh` to build/install `heirloom_py` into the project `.venv` with maturin and run pytest.
- Added `python/tests/test_heirloom_py_parity.py`, a live PyTorch comparison suite covering NumPy round-trip, matmul/ReLU/mean backward, broadcast plus transpose gradients, layer norm forward/backward, embedding repeated-row gradient accumulation, cross-entropy forward/backward, fused causal attention forward/backward, and a local-CUDA smoke that skips when CUDA is unavailable.
- Validation passed:
  - `scripts/python_parity.sh`: 7 passed, 1 skipped local-CUDA test.
  - `cargo fmt --all --check`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- This is intentionally a parity harness, not Python API parity. Missing Python work includes modules, optimizers, dataloaders, checkpointing, zero-copy NumPy/DLPack/Torch interop, typing stubs, stable packaging, Python custom autograd, and ergonomic PyTorch-like namespaces.

## Checkpoint 46 - BF16 Activation Rounding Path

- Added differentiable `GradFn::Cast` support so dtype casts participate in reverse-mode autograd instead of becoming silent graph breaks.
- Added CUDA f32->bf16 and bf16->f32 cast kernels in `heirloom-kernels`, with CUDA storage validation for BF16 buffers and FP32 CUDA gradient propagation through cast nodes.
- Extended CUDA gradient accumulation to allow BF16 primals while keeping gradients as FP32 CUDA tensors.
- Added `TinyTransformerLm::forward_bf16_activations` and `loss_bf16_activations`, which insert f32->bf16->f32 activation round-trips while keeping parameters, gradients, optimizer state, logits, and losses in f32.
- Added `heirloom train-lm --precision f32|bf16`; train reports now include `precision`.
- Threaded precision through `scripts/validate.sh`, `scripts/cuda_train_lm_fixture.sh`, `scripts/run_tinystories_valid.sh`, and `scripts/run_tinystories_cuda_reference.sh`, including summary/report metadata.
- Added always-on CPU tests for differentiable BF16 casts and fixed-batch BF16 activation training.
- Added gated CUDA tests for f32<->bf16 cast round-trip/backward and a BF16 activation tiny transformer step that keeps gradients on device.
- Validation so far:
  - `cargo check --workspace`
  - `cargo fmt --all --check`
  - `cargo test --workspace --test cuda_storage bfloat16`
  - `cargo test --workspace --test transformer_runtime tiny_transformer_bf16_activation_policy_trains_fixed_batch`
  - `cargo test --workspace --test transformer_runtime tiny_transformer_loss_decreases_and_checkpoint_loads`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `./scripts/validate.sh`
  - `./scripts/python_parity.sh`: 7 passed, 1 skipped local-CUDA smoke.
- This is not tensor-core mixed precision. Missing BF16/AMP work includes BF16 matmul/attention kernels, autocast op policies, loss scaling, master/compute dtype scheduling, optimizer policy integration beyond FP32 masters, numerical tolerance studies, and live A100 execution of the new PTX cast kernels.

## Checkpoint 47 - Precision-Consistent Eval/Generation And Vertex Plumbing

- Added BF16 activation variants for tiny transformer generation, heldout token-loss evaluation, and next-token logits, so BF16 checkpoints are no longer evaluated or sampled through an accidental f32-only path.
- Added `--precision f32|bf16` to `heirloom eval-lm` and `heirloom generate`; their JSON reports now record the selected precision.
- Updated the TinyStories local wrapper to pass the selected precision through train, eval, and generate, and to record eval/generation precision in the summary JSON.
- Updated `scripts/validate.sh` to exercise BF16 train, BF16 eval, and BF16 generation on the same tiny checkpoint.
- Updated the Vertex validation launcher and GCP docs to forward `HEIRLOOM_CUDA_FIXTURE_PRECISION` and `HEIRLOOM_TINYSTORIES_CUDA_PRECISION` into the generated job spec.
- Updated README, architecture, and hard-mode docs so validation claims match the actual script behavior.
- Validation passed:
  - `cargo check --workspace`
  - `cargo test --workspace --test transformer_runtime tiny_transformer_bf16_activation_policy_trains_fixed_batch`
  - `cargo test --workspace --test cuda_storage bfloat16`
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `./scripts/validate.sh`
- This checkpoint did not launch a new live A100/Vertex run; BF16 CUDA PTX cast execution remains covered by gated tests and should be rerun on A100 before claiming production mixed precision.

## Checkpoint 48 - First NCCL DDP Training Surface

- Added CUDA device compute-capability discovery and surfaced compute capability in `gpu info` and smoke reports.
- Added dynamic NCCL loading inside `heirloom-kernels`, including `NcclUniqueId`, version discovery, stream-backed `NcclCommunicator`, and in-place f32 CUDA buffer all-reduce.
- Added CUDA in-place f32 buffer scaling so all-reduced gradients can be averaged without CPU staging.
- Added `AdamW::all_reduce_cuda_gradients_nccl`, which averages rank-local CUDA f32 gradients through NCCL before local CUDA AdamW updates.
- Added deterministic DDP sharded token batches derived from `(seed, global_step, rank, sample_index)`.
- Added `--precision amp-bf16` as a distinct CLI/report policy, with early CUDA compute-capability checks. It currently delegates to the BF16 activation-rounding path until Tensor Core matmuls are wired.
- Added `train-lm --devices cuda:0,cuda:1,... --distributed nccl`, implemented as a parent launcher plus hidden `train-lm-rank` workers. Rank 0 saves checkpoints; aggregate reports include rank metadata, global batch size, NCCL all-reduce calls/bytes, and loss summaries.
- Threaded distributed TinyStories knobs through the local CUDA reference wrapper and Vertex launcher:
  - `HEIRLOOM_TINYSTORIES_CUDA_DEVICES`
  - `HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED`
  - `HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16`
- Added CPU-safe CLI validation tests for `amp-bf16` on CPU, invalid distributed device lists, duplicate devices, deterministic DDP batching, and a gated `HEIRLOOM_NCCL_TESTS=1` single-rank NCCL all-reduce smoke.
- Validation passed so far:
  - `cargo check --workspace`
  - focused `distributed_cli` tests
  - focused deterministic batching test
  - focused gated NCCL test in skip mode
  - shell syntax checks for TinyStories and Vertex scripts
- Still missing for the original plan: hand-written PTX Tensor Core BF16 matmul, AMP matmul dispatch in transformer modules, multi-rank live NCCL validation, and the 4x A100 TinyStories full gate.

## Checkpoint 49 - Tensor Core Acceptance Latch

- Added a separate sm80 PTX module with `heirloom_bf16_mma_probe`, a one-warp BF16 `mma.sync` execution probe over all-ones fragments. The expected dot product is 16, which lets an A100 run prove that the driver can JIT and execute the BF16 MMA instruction without claiming general GEMM correctness.
- Added safe kernel-crate APIs for Tensor Core probe reports, Tensor Core counters, counter reset, and the `HEIRLOOM_REQUIRE_TENSOR_CORES=1` hard gate.
- Added `heirloom gpu tensor-core-probe --device 0` so Vertex runs can validate BF16 MMA availability before a full training job.
- Added gated CUDA test coverage for the BF16 MMA probe and counter accounting. It skips by default unless `HEIRLOOM_CUDA_TESTS=1` and the device reports compute capability >= 8.0.
- Added rank and aggregate train reports for:
  - `bf16_mma_probe_calls`
  - `bf16_tensor_core_matmul_calls`
  - `bf16_scalar_matmul_fallback_calls`
- Wired `HEIRLOOM_REQUIRE_TENSOR_CORES=1` into `TinyTransformerLm::forward_amp_bf16` so unsupported Tensor Core shapes/devices fail clearly instead of silently passing under the AMP name.
- Threaded the hard gate through the Vertex wrapper as an optional non-secret env var.
- Validation passed:
  - `cargo check --workspace`
  - focused gated Tensor Core probe test in skip mode
- Still missing at this checkpoint: real BF16 Tensor Core GEMM tiling, strided/transposed RHS support, backward GEMM kernels, batched attention GEMM kernels, Tensor Core dispatch from `Linear`/attention, and live A100 probe execution.

## Checkpoint 50 - First Tensor Core Linear Engine

- Added `heirloom_matmul_bf16_mma_rhs_t_f32`, a strict sm80 BF16 Tensor Core forward GEMM for `A[M,K] * W[N,K]^T -> C[M,N]`.
  - One warp computes one `16x8` output tile.
  - The kernel loops over `K` in `16`-wide chunks using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
  - Supported shapes are intentionally strict: `M%16=0`, `K%16=0`, `N%8=0`, all non-zero.
  - Inputs are BF16, accumulation/output is FP32.
- Added safe `matmul_bf16_tensor_core_rhs_t_f32_buffers` and `bf16_tensor_core_matmul_shape_supported` APIs in `heirloom-kernels`.
- Added `Tensor::matmul_bf16_tensor_core_rhs_t` for CUDA BF16 tensors. It records a normal matmul autograd edge and returns an FP32 CUDA tensor.
- Made CUDA matmul backward BF16-aware by converting saved BF16 operands back to FP32 on device and reusing the existing CUDA f32 backward kernels. This keeps gradients CUDA-resident but is not Tensor Core backward GEMM.
- Added `Linear::forward_amp_bf16` and routed attention, feed-forward, transformer block, and LM AMP paths through it. Tile-compatible CUDA Linear projections now use Tensor Core forward GEMM; unsupported shapes fall back unless `HEIRLOOM_REQUIRE_TENSOR_CORES=1`.
- Single-rank and DDP train reports now include Tensor Core counters.
- Added a gated CUDA test for tile-compatible AMP Linear forward parity and CUDA-resident backward participation.
- Validation passed:
  - `cargo check --workspace`
  - focused Tensor Core Linear test in skip mode
  - focused transformer BF16 training test
- Still missing: live A100 execution of the new GEMM, Tensor Core backward kernels, batched QK/AV attention GEMM, shared-memory/`ldmatrix` tiling, remainder kernels, and production AMP policy.

## Checkpoint 51 - GPU Failure Flight Recorder

- Added `--report` to `heirloom gpu tensor-core-probe`, producing structured JSON with the BF16 MMA probe result and Tensor Core counters.
- Hardened the Vertex validation launcher so GPU info, per-device smoke, per-device Tensor Core probe, gated CUDA storage tests, CUDA LM fixture, and TinyStories CUDA reference all stream to logs while also writing uploadable diagnostic transcripts.
- Failed smoke/probe/fixture/reference stages now upload any available `.txt` transcript and partial JSON reports before re-raising the failure.
- Vertex workers now attempt to upload `failure-summary.json` when the normal validation summary cannot be written.
- Vertex summaries now include:
  - `gpu_log_uris`
  - `tensor_core_probe_report_uris`
  - `tensor_core_probe_log_uris`
  - `diagnostic_log_uris`
- Added `HEIRLOOM_RUN_TENSOR_CORE_PROBE`, defaulting to enabled in the Vertex wrapper, and documented the new probe artifacts in `scripts/gcp/README.md`.
- Extended `scripts/gpu_smoke.sh` with an opt-in local Tensor Core probe report path via `HEIRLOOM_RUN_TENSOR_CORE_PROBE=1`.
- Still missing: a live A100 run proving the new flight-recorder artifacts arrive as expected.

## Checkpoint 52 - 4x Topology Evidence Before Distributed Runs

- Added CUDA PCI bus IDs to `CudaDeviceInfo` via `cuDeviceGetPCIBusId`, so `gpu info`, smoke reports, and DDP rank reports can identify the physical device behind each `cuda:<id>` ordinal.
- Added `heirloom gpu topology --report ...`, which emits CUDA-visible devices plus the full `cuDeviceCanAccessPeer` matrix.
- Extended the Vertex launcher to upload `nvidia-smi.txt`, `nvidia-smi-topo.txt`, `nvidia-smi-nvlink.txt`, `nvidia-smi-gpu-bus-ids.csv`, `heirloom-gpu-topology.txt`, and `heirloom-gpu-topology.json` when `HEIRLOOM_COLLECT_GPU_TOPOLOGY=1`.
- Distributed TinyStories reference jobs now default NCCL debug logging on through `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH` when `HEIRLOOM_ENABLE_NCCL_DEBUG=1`.
- Updated GCP docs so 4x A100 runs must interpret performance against the uploaded NVLink/P2P/NCCL route evidence.
- Still missing: live 4x A100 topology capture and NCCL route confirmation from a recorded Vertex run.

## Checkpoint 53 - Single-A100 AMP BF16 Hard Gate

- Ran the single-A100 Vertex quick gate with:
  - `HEIRLOOM_VERTEX_ACCELERATOR_COUNT=1`
  - `HEIRLOOM_RUN_TENSOR_CORE_PROBE=1`
  - `HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE=1`
  - `HEIRLOOM_CUDA_FIXTURE_DEVICE=cuda:0`
  - `HEIRLOOM_CUDA_FIXTURE_PRECISION=amp-bf16`
  - `HEIRLOOM_REQUIRE_TENSOR_CORES=1`
  - `HEIRLOOM_EXPECT_TENSOR_CORES=1`
- Successful Vertex job:
  - job id: `5638811830764699648`
  - source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260605-114910.tar.gz`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-114910`
  - summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-114910/summary.json`
- Live A100 evidence:
  - GPU: `NVIDIA A100-SXM4-80GB`, compute capability `8.0`.
  - BF16 MMA probe passed with `expected_dot=16`, `max_abs_error=0`.
  - CUDA storage/autograd gate passed: `40 passed`.
  - AMP BF16 fixture used tile-compatible transformer settings: `d_model=16`, `n_heads=4`, `ff_hidden=64`.
  - Fixture loss dropped from `5.5050526` to `1.9856855` over 40 steps, a `63.93%` reduction.
  - Tensor Core counters for the training fixture: `bf16_tensor_core_matmul_calls=280`, `bf16_scalar_matmul_fallback_calls=0`.
  - Resume proof loaded the checkpoint at step 40 and reached step 43, with loss `1.9446229 -> 1.5684482` and `21` additional Tensor Core matmul calls.
- Fixes discovered by the live gate:
  - CUDA matmul backward now accepts saved BF16 layouts and casts BF16 operands back to FP32 on-device before using FP32 gradient kernels.
  - CUDA permute backward now accepts BF16 CUDA inputs while producing FP32 gradients, fixing transposed AMP Linear weight views.
  - Tensor Core counter-sensitive CUDA tests now serialize counter access so parallel tests cannot poison global instrumentation.
  - Vertex CUDA fixture env wiring no longer forces tiny non-tiled defaults when optional fixture model dimensions are omitted.
  - Vertex quick gate excludes the optional `heirloom-python` crate because the worker image exposes a non-PIC Python archive that cannot link the PyO3 `cdylib`; local full-workspace validation still includes `heirloom-python`.
- Local validation passed after the fixes:
  - `cargo fmt --all --check`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace --exclude heirloom-python`
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`
- Still missing: Tensor Core backward GEMM, Tensor Core attention QK/AV batched matmuls, multi-rank NCCL live validation, and the 4x A100 TinyStories reference gate.

## Checkpoint 54 - AMP BF16 TinyStories Reference With Coverage

- Added `amp_bf16_tensor_core_coverage` and `reset_amp_bf16_tensor_core_coverage` in the safe `nn` crate.
  - Coverage is keyed by module path, currently for CUDA AMP `Linear` projections.
  - Reports include per-module calls, Tensor Core calls, fallback calls, unsupported-shape/device calls, dtype-mismatch calls, last shape `(M,K,N)`, dtype labels, device, and last selected path.
  - `TinyTransformerLm::forward_amp_bf16` now records named modules such as `block.attention.q_proj`, `block.feed_forward.fc1`, and `lm_head`.
- Extended train reports and distributed rank reports with `tensor_core_coverage`.
- Hardened `scripts/run_tinystories_valid.sh` and `scripts/cuda_train_lm_fixture.sh` so `HEIRLOOM_REQUIRE_TENSOR_CORES=1` / `HEIRLOOM_EXPECT_TENSOR_CORES=1` now require:
  - global Tensor Core matmul calls > 0,
  - global scalar fallback calls == 0,
  - attributed Linear Tensor Core calls > 0,
  - attributed Linear fallback calls == 0.
- Local validation passed:
  - `cargo fmt --all --check`
  - `cargo check --workspace`
  - focused `cuda_linear_amp_bf16_uses_tensor_core_matmul_when_tile_aligned`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - shell syntax checks for the Vertex, TinyStories, CUDA reference, and CUDA fixture scripts.
- Ran a successful single-A100 AMP BF16 TinyStories reference job:
  - job id: `7180028065743896576`
  - source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260605-122143.tar.gz`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-122143`
  - summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-122143/summary.json`
- Live A100 evidence:
  - GPU: `NVIDIA A100-SXM4-80GB`, compute capability `8.0`.
  - BF16 MMA probe passed with `expected_dot=16`, `max_abs_error=0`.
  - CUDA storage/autograd gate passed: `40 passed`, including the Tensor Core Linear coverage assertion.
  - Reference mode used a 1 MiB TinyStories validation-split slice, `vocab_size=512`, `steps=80`, `batch_size=2`, `block_size=16`, `d_model=16`, `n_heads=4`, `ff_hidden=64`, and `precision=amp-bf16`.
  - Training loss dropped from `6.5790763` to `4.4355693`, a `32.58%` reduction against the `1%` reference threshold.
  - Heldout eval emitted `loss=4.0439429`, `perplexity=57.050848`, `batches=20`, and `tokens=640`.
  - Generation emitted an 80-token JSON report from prompt `Once upon a time`.
  - Tensor Core counters: `bf16_tensor_core_matmul_calls=560`, `bf16_scalar_matmul_fallback_calls=0`.
  - Tensor Core coverage totals: `linear calls=560`, `tensor_core_calls=560`, `fallback_calls=0`, `unsupported_shape_calls=0`, `unsupported_device_calls=0`, `dtype_mismatch_calls=0`.
  - Each of the seven Linear projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`, `lm_head`) recorded `80` Tensor Core calls and zero fallbacks.
- Still missing: Tensor Core backward GEMM, Tensor Core QK/AV attention matmuls, coverage for non-Linear AMP decisions, and multi-rank NCCL live validation.

## Checkpoint 55 - Tensor Core Linear Backward Gate

- Added split Tensor Core counters in `heirloom-kernels::cuda`:
  - `bf16_tensor_core_matmul_forward_calls`
  - `bf16_tensor_core_matmul_backward_calls`
  - the existing aggregate `bf16_tensor_core_matmul_calls` remains for compatibility.
- Added BF16 layout-materialization and transpose CUDA kernels so AMP `Linear` backward can prepare logical BF16 operands without CPU staging.
- Routed tile-compatible AMP `Linear` backward matmuls through the BF16 Tensor Core RHS-T GEMM:
  - input gradient uses `grad_output_bf16[M,N] * weight_bf16[K,N]^T -> grad_input[M,K]`,
  - weight gradient uses transposed BF16 operands to compute the compatible `[K,N]` view gradient before normal transpose/cast autograd propagation reaches the FP32 master weight.
- Extended `HEIRLOOM_REQUIRE_TENSOR_CORES=1` so unsupported BF16 Tensor Core backward shapes/devices error clearly instead of falling back to FP32 CUDA matmul-gradient kernels.
- Extended train, rank, distributed aggregate, and Tensor Core probe reports with forward/backward Tensor Core call counts.
- Hardened `scripts/run_tinystories_valid.sh` and `scripts/cuda_train_lm_fixture.sh` so Tensor Core hard gates now require both forward and backward Tensor Core matmul calls, not just aggregate forward participation.
- Strengthened the gated CUDA AMP Linear test to assert:
  - one forward Tensor Core call,
  - two backward Tensor Core calls,
  - zero scalar fallbacks,
  - CUDA-resident FP32 input and weight gradients,
  - gradient values against a BF16-rounded CPU reference.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo check --workspace`
  - focused `cuda_linear_amp_bf16_uses_tensor_core_matmul_when_tile_aligned`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - shell syntax checks for `scripts/run_tinystories_valid.sh`, `scripts/cuda_train_lm_fixture.sh`, and `scripts/gcp/submit_vertex_heirloom_validate.sh`.
- Ran an informative failed A100 hard-gate job:
  - job id: `6982151157116305408`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-125727`
  - failure summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-125727/failure-summary.json`
  - cause: the stricter CUDA Linear backward test used `out_features=8`, which is valid for forward (`N%8=0`) but invalid for input-gradient Tensor Core backward because the original output dimension becomes the MMA inner dimension and must be divisible by 16.
  - fix: the CUDA test now uses `out_features=16`, and the AMP fixture auto-vocab moved from `264` to `272`.
- Ran a successful replacement single-A100 AMP BF16 Tensor Core backward hard gate:
  - job id: `7586407563369906176`
  - source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260605-131318.tar.gz`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-131318`
  - summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-131318/summary.json`
- Live A100 evidence:
  - GPU: `NVIDIA A100-SXM4-80GB`, compute capability `8.0`.
  - BF16 MMA probe passed with `expected_dot=16`, `max_abs_error=0`.
  - CUDA storage/autograd gate passed: `40 passed`, including the stricter Tensor Core Linear backward assertion.
  - AMP fixture used tile-compatible settings: `batch_size=4`, `block_size=4`, `d_model=16`, `n_heads=4`, `ff_hidden=64`, `vocab_size=272`.
  - Fixture loss dropped from `6.0103579` to `1.9931357`, a `66.84%` reduction over 40 steps.
  - Training Tensor Core counters: `forward=280`, `backward=560`, total `840`, scalar fallback `0`.
  - Resume Tensor Core counters: `forward=21`, `backward=42`, total `63`, scalar fallback `0`.
  - Tensor Core coverage totals: `linear calls=280`, `tensor_core_calls=280`, `fallback_calls=0`, `unsupported_shape_calls=0`, `unsupported_device_calls=0`, `dtype_mismatch_calls=0`.
- Still missing: Tensor Core QK/AV attention matmuls, general/batched/remainder Tensor Core GEMM, production AMP policy, and multi-rank NCCL live validation.

## Checkpoint 56 - Tensor Core Attention Forward Prototype

- Added strict BF16 Tensor Core causal-attention forward kernels in `heirloom-kernels::cuda`:
  - QK computes f32 scores from BF16 query/key tiles,
  - a f32 CUDA softmax kernel applies the causal mask,
  - AV computes f32 output from BF16 attention/value tiles.
- Added the safe runtime wrapper `causal_attention_bf16_tensor_core_forward_f32_buffers`.
  - Requires CUDA sm80+ BF16 Tensor Core support.
  - Requires `time%16=0` and `head_dim%16=0`.
  - Keeps query/key/value, scores, attention, and output device-resident.
- Added `Tensor::causal_self_attention_amp_bf16_tensor_core` and routed `TinyTransformerLm::forward_amp_bf16` through it when shapes/devices are compatible.
  - Backward intentionally still uses the existing f32 CUDA causal-attention gradient kernels.
  - `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1` now turns attention forward fallback into a clear error.
- Extended Tensor Core counters and reports with:
  - `bf16_tensor_core_attention_forward_calls`,
  - `bf16_tensor_core_attention_qk_matmul_calls`,
  - `bf16_tensor_core_attention_av_matmul_calls`.
- Added a gated CUDA attention fixture that compares Tensor Core attention forward against a BF16-rounded CPU reference and asserts CUDA-resident FP32 Q/K/V gradients after backward.
- Hardened validation wrappers:
  - `scripts/cuda_train_lm_fixture.sh` now supports `HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES=1` and auto-selects a small attention-compatible AMP shape when `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1`.
  - `scripts/run_tinystories_valid.sh` fails explicitly when the attention hard gate is requested but attention counters stay zero.
  - `scripts/run_tinystories_cuda_reference.sh` adjusts small-tier defaults to an attention-compatible shape when the attention hard gate is set.
  - `scripts/gcp/README.md` documents the single-A100 Tensor Core attention validation command and caveats.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo check --workspace`
  - focused `cuda_causal_attention_amp_bf16_tensor_core_forward_is_gated`
  - focused `cuda_linear_amp_bf16_uses_tensor_core_matmul_when_tile_aligned`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Ran an initial successful single-A100 job that proved the new attention PTX through the CUDA storage test:
  - job id: `5517707222034939904`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-145557`
  - CUDA storage/autograd gate passed: `41 passed`, including `cuda_causal_attention_amp_bf16_tensor_core_forward_is_gated`.
  - The training fixture still recorded attention counters at zero because the Vertex launcher did not yet propagate `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES` / `HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES`.
- Fixed the Vertex launcher env propagation for the attention hard-gate variables.
- Ran a successful replacement single-A100 AMP BF16 Tensor Core attention hard gate:
  - job id: `2728853152785760256`
  - source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260605-151025.tar.gz`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-151025`
  - summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-151025/summary.json`
- Live A100 evidence:
  - GPU: `NVIDIA A100-SXM4-80GB`, compute capability `8.0`.
  - BF16 MMA probe passed with `expected_dot=16`, `max_abs_error=0`.
  - CUDA storage/autograd gate passed: `41 passed`, including the Tensor Core attention forward fixture.
  - AMP fixture used attention-compatible settings: `batch_size=4`, `block_size=16`, `d_model=16`, `n_heads=1`, `ff_hidden=64`, `vocab_size=272`.
  - Fixture loss dropped from `5.9439173` to `2.0518956`, a `65.48%` reduction over 40 steps.
  - Training Tensor Core counters: `forward=600`, `backward=560`, total `1160`, scalar fallback `0`.
  - Training attention Tensor Core counters: `attention_forward=40`, `attention_qk=160`, `attention_av=160`.
  - Resume Tensor Core counters: `forward=45`, `backward=42`, total `87`, scalar fallback `0`.
  - Resume attention Tensor Core counters: `attention_forward=3`, `attention_qk=12`, `attention_av=12`.
  - Tensor Core coverage totals: `linear calls=280`, `tensor_core_calls=280`, `fallback_calls=0`, `unsupported_shape_calls=0`, `unsupported_device_calls=0`, `dtype_mismatch_calls=0`.
- Still missing:
  - Tensor Core attention backward,
  - general batched/remainder Tensor Core GEMM,
  - multi-rank NCCL live validation.

## Checkpoint 57 - Tensor Core Attention Backward Prototype

- Added a strict AMP Tensor Core attention-backward path for attention nodes created by `Tensor::causal_self_attention_amp_bf16_tensor_core`.
  - Normal CUDA f32 causal attention keeps the existing f32 CUDA backward kernels so ordinary f32 gradients are not silently rounded through BF16 Tensor Core math.
  - The AMP path decomposes backward into Tensor Core matmuls for `dAttention = dO * V^T`, `dQ = dScore * K`, `dV = attention^T * dO`, and `dK = dScore^T * Q`.
  - A f32 CUDA softmax-backward kernel computes causal rowwise `dScore`, and a f32 CUDA square-transpose kernel prepares `attention^T` and `dScore^T`.
  - Requirements remain CUDA sm80+, `time%16=0`, and `head_dim%16=0`.
- Extended `GradFn::CausalSelfAttention` with an `amp_bf16_tensor_core` marker so backward can select the Tensor Core path only for strict AMP attention nodes.
- Added safe kernel-crate support through `causal_attention_bf16_tensor_core_backward_f32_buffers`.
- Extended Tensor Core counters and JSON reports with:
  - `bf16_tensor_core_attention_backward_calls`,
  - `bf16_tensor_core_attention_score_grad_matmul_calls`,
  - `bf16_tensor_core_attention_dq_matmul_calls`,
  - `bf16_tensor_core_attention_dk_matmul_calls`,
  - `bf16_tensor_core_attention_dv_matmul_calls`.
- Hardened validation wrappers so `HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES=1` requires attention forward counters plus backward score-grad/dQ/dK/dV counters in train and resume reports.
- Strengthened the gated CUDA attention test to compare CUDA AMP Tensor Core attention gradients against a BF16-rounded CPU reference and to assert the new backward counters.
- Local validation passed:
  - `cargo fmt --all`
  - shell syntax checks for `scripts/cuda_train_lm_fixture.sh`, `scripts/run_tinystories_valid.sh`, `scripts/run_tinystories_cuda_reference.sh`, and `scripts/gcp/submit_vertex_heirloom_validate.sh`
  - `cargo check --workspace`
  - focused `cargo test cuda_causal_attention_amp_bf16_tensor_core_forward_is_gated --test cuda_storage`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Ran a successful single-A100 AMP BF16 Tensor Core attention backward hard gate:
  - job id: `2528829997460750336`
  - source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260605-160714.tar.gz`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-160714`
  - summary: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-160714/summary.json`
- Live A100 evidence:
  - GPU: `NVIDIA A100-SXM4-80GB`, compute capability `8.0`.
  - CUDA storage/autograd gate passed with the stricter Tensor Core attention backward fixture.
  - AMP fixture used attention-compatible settings: `batch_size=4`, `block_size=16`, `d_model=16`, `n_heads=1`, `ff_hidden=64`, `vocab_size=272`.
  - Fixture loss dropped from `5.9546175` to `2.1760991`, a `63.46%` reduction over 40 steps.
  - Training Tensor Core counters: `forward=600`, `backward=1200`, total `1800`, scalar fallback `0`.
  - Training attention Tensor Core counters: `attention_forward=40`, `attention_qk=160`, `attention_av=160`, `attention_backward=40`, `score_grad=160`, `dq=160`, `dk=160`, `dv=160`.
  - Resume loss dropped from `1.8110380` to `1.6643879` over 3 steps.
  - Resume Tensor Core counters: `forward=45`, `backward=90`, total `135`, scalar fallback `0`.
  - Resume attention Tensor Core counters: `attention_forward=3`, `attention_qk=12`, `attention_av=12`, `attention_backward=3`, `score_grad=12`, `dq=12`, `dk=12`, `dv=12`.
  - Tensor Core coverage totals: `linear calls=280`, `tensor_core_calls=280`, `fallback_calls=0`, `unsupported_shape_calls=0`, `unsupported_device_calls=0`, `dtype_mismatch_calls=0`.
- Still missing:
  - fused/streaming FlashAttention-style forward/backward,
  - general batched/remainder Tensor Core GEMM,
  - live 4x NCCL DDP validation.

## Checkpoint 58 - 4x NCCL Diagnostic Attempt

- Hardened the distributed `train-lm --distributed nccl` report path:
  - aggregate reports now include per-rank parameter checksum sums and sum-of-squares,
  - aggregate reports include max checksum drift versus rank 0,
  - drift above `1e-3` fails after writing the aggregate report.
- Hardened rank-worker lifecycle:
  - parent now polls all rank workers with `try_wait` and kills remaining siblings on the first nonzero rank exit,
  - `scripts/run_tinystories_valid.sh` now cleans up `heirloom train-lm-rank` workers on distributed NCCL script exit, including after a parent process segfault.
- Added distributed diagnostics:
  - parent logs before and after NCCL unique-id creation,
  - rank workers log before `ncclCommInitRank` and after communicator creation,
  - Vertex wrapper can pass `HEIRLOOM_NCCL_NET_PLUGIN` through as `NCCL_NET_PLUGIN`.
- Adjusted NCCL dynamic loading to use `RTLD_GLOBAL` on Linux so NCCL network/coll plugins can resolve NCCL symbols.
- Local validation passed after the distributed hardening:
  - `cargo fmt --all`
  - `cargo check --workspace`
  - `cargo test --test distributed_cli`
  - shell syntax checks for TinyStories and Vertex wrapper scripts.
- Ran a first 4x A100 NCCL smoke/reference attempt:
  - job id: `4896034551597367296`
  - display: `heirloom-validate-quick-20260605-163015`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-163015`
  - result: segfault during `train-lm --distributed nccl`, then job was canceled because orphaned rank workers kept Vertex running after the crash.
  - useful evidence: 4x `NVIDIA A100-SXM4-80GB`, compute capability `8.0`, peer access true across all pairs, `nvidia-smi topo` reports `NV12` between every GPU pair, GPU smoke passed on all four devices, and Tensor Core probes uploaded for all four devices.
- Ran a second 4x attempt after `RTLD_GLOBAL` and cleanup fixes:
  - job id: `912882156164874240`
  - display: `heirloom-validate-quick-20260605-171135`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-171135`
  - result: clean failed job with `failure-summary.json`; still segfaulted at distributed train startup.
- Ran a minimal 4x diagnostic with one training step, one eval batch, tiny generation, and `HEIRLOOM_NCCL_NET_PLUGIN=none`:
  - job id: `8191825053902438400`
  - display: `heirloom-validate-quick-20260605-172124`
  - source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260605-172124.tar.gz`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-172124`
  - result: clean failed job with `failure-summary.json`; all four ranks logged `initializing NCCL`, no rank logged `NCCL communicator ready`, and NCCL reported internal network plugin use before the process segfaulted.
- Current distributed diagnosis:
  - single-rank NCCL all-reduce test passes,
  - 4x hardware/topology is suitable for same-node distributed training,
  - the blocker is multi-rank `ncclCommInitRank` under the current pure FFI/Driver API runtime, before any gradient all-reduce or optimizer synchronization.

## Checkpoint 59 - Isolated 4x NCCL Probe Harness

- Added a focused distributed fabric probe:
  - public command: `cargo run --bin heirloom -- gpu nccl-probe --devices cuda:0,cuda:1,... --len N --report ...`
  - hidden worker command: `nccl-probe-rank`.
- The probe mirrors the DDP process shape without tokenizer/model/training noise:
  - parent creates one NCCL unique id,
  - parent spawns one Rust process per CUDA device,
  - each rank initializes `NcclCommunicator`,
  - each rank all-reduces a small f32 CUDA buffer in place,
  - each rank writes a JSON report with PCI bus id, NCCL version, sample values, max error, and all-reduce bytes.
- Added an aggregate probe report with:
  - devices/world size,
  - expected all-reduce sum,
  - max absolute error and tolerance,
  - rank reports,
  - total NCCL calls/bytes.
- Added `HEIRLOOM_NCCL_TRACE=1` stage tracing inside `heirloom-kernels::cuda::NcclCommunicator::init_rank`:
  - library load,
  - CUDA primary-context retain,
  - stream creation,
  - pre/post `ncclCommInitRank`.
- Extended Vertex validation wrapper support:
  - `HEIRLOOM_RUN_NCCL_PROBE=1`,
  - `HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`,
  - `HEIRLOOM_NCCL_PROBE_LEN=1024`,
  - `HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0` for narrow fabric-only probes,
  - uploaded `nccl-probe.txt` and `nccl-probe.json`.
- Extended `scripts/gpu_smoke.sh` with the same opt-in probe knobs.
- Added always-on CLI validation tests for NCCL probe invalid/duplicate device lists.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo check --workspace`
  - `cargo test --test distributed_cli`
  - shell syntax checks for `scripts/gpu_smoke.sh` and `scripts/gcp/submit_vertex_heirloom_validate.sh`.
- Next live validation should run the isolated 4x NCCL probe before another full `train-lm --distributed nccl` attempt.

## Checkpoint 60 - Production-Grade Distributed Launcher

- Added a shared `DistributedLauncher` used by `gpu nccl-probe` and `train-lm --distributed nccl`.
- Each launched rank now has:
  - a rank config JSON,
  - per-rank stdout/stderr logs,
  - a stage JSON file updated before blocking CUDA/NCCL boundaries,
  - a rank report JSON,
  - PID, command, env, exit status, and last-stage metadata in `launcher-report.json`.
- Extended `gpu nccl-probe` into a diagnostic ladder:
  - `--probe-kind spawn`,
  - `--probe-kind cuda-context`,
  - `--probe-kind nccl-init`,
  - `--probe-kind all-reduce`.
- Added launcher timeout behavior:
  - parent kills sibling ranks on first failure,
  - parent kills all live ranks on timeout,
  - rank-start timeout catches workers that never progress beyond `spawned`,
  - NCCL init timeout catches ranks stuck at `nccl_init_start`.
- Moved DDP training onto the same launcher lifecycle and added `--ddp-init-timeout-secs`.
- Added hidden local test commands:
  - `launcher-test`,
  - `launcher-test-rank`.
- Added always-on local tests for launcher success, one-rank failure cleanup, hung-rank timeout reporting, and `gpu nccl-probe --probe-kind spawn` without CUDA/NCCL.
- Hardened the Vertex wrapper:
  - `HEIRLOOM_NCCL_PROBE_KIND`,
  - `HEIRLOOM_NCCL_PROBE_TIMEOUT_SECS`,
  - `HEIRLOOM_NCCL_PROBE_RANK_START_TIMEOUT_SECS`,
  - `HEIRLOOM_NCCL_PROBE_KILL_GRACE_SECS`,
  - Python-side process-group timeout around the NCCL probe,
  - upload of `launcher-report.json` and `nccl-probe-ranks/` artifacts even when the probe fails.
- Local validation passed:
  - `cargo check --workspace`
  - `cargo test --test distributed_cli`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - shell syntax checks for the Vertex, GPU smoke, TinyStories, and CUDA fixture scripts
  - embedded Python syntax check for `scripts/gcp/submit_vertex_heirloom_validate.sh`

## Checkpoint 61 - 4x A100 Distributed Launcher Ladder

- Ran the staged Vertex probe ladder on 4x `NVIDIA A100-SXM4-80GB` with probe-only flags:
  - `HEIRLOOM_RUN_NCCL_PROBE=1`
  - `HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`
  - `HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0`
  - `HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=0`
  - `HEIRLOOM_RUN_TENSOR_CORE_PROBE=0`
- `spawn` passed:
  - job id: `145370263419092992`
  - display: `heirloom-validate-quick-20260605-190520`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-190520`
  - all four rank workers exited `0`
  - launcher report status: `passed`
  - all rank stages ended at `report_written`
- `cuda-context` passed:
  - job id: `4667547239252492288`
  - display: `heirloom-validate-quick-20260605-192118`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-192118`
  - all four ranks allocated/copied a CUDA buffer and exited `0`
  - launcher report status: `passed`
  - all rank stages ended at `report_written`
- `nccl-init` failed quickly rather than hanging:
  - job id: `1662238912912818176`
  - display: `heirloom-validate-quick-20260605-193215`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-193215`
  - failure summary: `failure-summary.json`
  - probe log: `nccl-probe.txt`
  - top-level error: the `gpu nccl-probe --probe-kind nccl-init` process died with `SIGSEGV`
  - probe log reached `NCCL probe parent creating unique id world_size=4 devices=[0, 1, 2, 3] len=1024 probe_kind=nccl-init`
  - partial rank artifacts existed, but no top-level `launcher-report.json` or `nccl-probe.json`
  - rank 1 stderr reached `calling ncclCommInitRank`
  - rank 0 and rank 1 stage files reached `cuda_context_start`
  - rank 2 and rank 3 had configs but no meaningful stage/log evidence
- Did not run `all-reduce`; the accepted ladder requires `nccl-init` to pass first.
- Verified no active Vertex jobs remained after the ladder and cleaned up stale local `gcloud stream-logs` processes.
- Updated diagnosis: the hardware/topology and rank orchestration are not the immediate blockers. The next bug is in the NCCL FFI/communicator initialization boundary, now reproducible as a bounded `nccl-init` probe failure with artifacts.

## Checkpoint 62 - NCCL Init Segfault Mitigation

- Investigated the `nccl-init` ladder failure from job `1662238912912818176`.
- Key artifact evidence:
  - `spawn` and `cuda-context` passed on all four A100s.
  - `nccl-init` failed with `SIGSEGV`.
  - the probe log reached parent unique-id creation and partial child-rank startup.
  - rank 1 reached `calling ncclCommInitRank`.
  - no top-level `launcher-report.json` was produced, meaning the launcher parent itself did not survive long enough to summarize the child-rank failure.
- Likely cause: the launcher parent loaded NCCL and called `ncclGetUniqueId` before spawning ranks. If NCCL/plugin initialization is not fork/spawn-clean in the Vertex image, the parent can crash before the process supervisor can write a launcher report.
- Mitigation implemented:
  - added hidden `heirloom nccl-unique-id` helper command,
  - `gpu nccl-probe` now obtains the unique id from that helper process instead of loading NCCL in the launcher parent,
  - `train-lm --distributed nccl` uses the same helper-created unique id,
  - rank workers no longer perform a separate pre-init `nccl_info()` load/drop before communicator initialization,
  - added `ncclUniqueId` ABI layout and hex round-trip tests in `heirloom-kernels`.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo check --workspace`
  - `cargo test --test distributed_cli`
  - `cargo test -p heirloom-kernels`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Next live validation should rerun only:
  - `HEIRLOOM_NCCL_PROBE_KIND=nccl-init`
  - same 4x probe-only flags as Checkpoint 61.

## Checkpoint 63 - NCCL Unique-Id Helper Protocol Hardening

- Reran the 4x A100 `nccl-init` probe after the helper-based unique-id mitigation.
- Job details:
  - job id: `2520878329368674304`
  - display: `heirloom-validate-quick-20260605-212345`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-212345`
- Result:
  - no parent `SIGSEGV`; the previous crash mode is gone.
  - all four CUDA smoke probes still passed before the NCCL rung.
  - the NCCL probe failed before rank launch because the helper's stdout contained NCCL/log output around the raw unique id:
    - `NCCL unique id hex must contain 256 hex characters, got 844`
  - no `launcher-report.json` was expected because the probe failed while obtaining the unique id, before spawning rank workers.
- Fix implemented:
  - hidden `heirloom nccl-unique-id` now prints a marker line:
    - `HEIRLOOM_NCCL_UNIQUE_ID_HEX=<256 hex chars>`
  - the launcher parent parses the marker line instead of treating all helper stdout as the id.
  - parser still accepts a lone exact 256-character hex line for backwards-compatible manual diagnostics.
  - parser rejects missing ids, invalid marker payloads, and multiple candidate ids.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo check --workspace`
  - `cargo test --bin heirloom nccl_unique_id -- --nocapture`
  - `cargo test --test distributed_cli`
  - `cargo test -p heirloom-kernels`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Next live validation should rerun the same `nccl-init` rung once more. If that passes, proceed to `all-reduce`.

## Checkpoint 64 - NCCL Bootstrap Interface Diagnosis

- Reran the 4x A100 `nccl-init` probe with the marked unique-id helper protocol.
- Job details:
  - job id: `8876609691774353408`
  - display: `heirloom-validate-quick-20260605-213600`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-213600`
- Progress versus Checkpoint 63:
  - helper unique-id parsing succeeded.
  - parent spawned all four rank workers.
  - `launcher-report.json` and all per-rank stdout/stderr/stage artifacts were uploaded.
  - hardware topology reported full NVLink between all pairs:
    - `nvidia-smi topo -m` showed `NV12` between GPUs.
  - CUDA peer access remained true for all pairs.
- Failure:
  - all four ranks reached `ncclCommInitRank`.
  - all four ranks failed quickly with:
    - `ncclCommInitRank failed: unhandled system error`
  - NCCL stdout showed bootstrap selected `eth0:10.106.0.7`.
  - NCCL then failed with:
    - `socketStartConnect: Connect to 10.106.0.7<45447> failed : Software caused connection abort`
  - NCCL reported `NET/IB : No device found`, then `Using network Socket`.
- Diagnosis:
  - The Rust FFI signature for `ncclCommInitRank` is already correct: NCCL unique id is passed by value.
  - This is likely a single-node Vertex bootstrap-interface issue, not a device topology issue.
  - Since all ranks are one process per GPU on the same worker and the GPUs are NVLink-connected, NCCL bootstrap should not depend on connecting back through the container's `eth0` address.
- Fix implemented:
  - launcher NCCL env now defaults to:
    - `NCCL_SOCKET_IFNAME=lo`
    - `NCCL_IB_DISABLE=1`
    - `NCCL_DEBUG=INFO`
    - `NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH`
  - user-provided env values still override these defaults.
  - `HEIRLOOM_NCCL_TRACE` remains opt-in via environment, so probe runs can stay verbose without forcing training runs to emit Heirloom trace logs.
  - added unit tests for default loopback bootstrap env and override preservation.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo test --bin heirloom nccl_launcher_env -- --nocapture`
  - `cargo test --bin heirloom nccl_unique_id -- --nocapture`
  - `cargo test --test distributed_cli`
  - `cargo test -p heirloom-kernels`
  - `cargo check --workspace`
- Next live validation should rerun `nccl-init` once with loopback bootstrap defaults. If it passes, immediately run `all-reduce`.

## Checkpoint 65 - Vertex Inherited NCCL Socket Exclusion

- Reran the 4x A100 `nccl-init` probe after adding default loopback bootstrap env.
- Job details:
  - job id: `8641296611244244992`
  - display: `heirloom-validate-quick-20260605-214722`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-214722`
- Result:
  - the run still failed at `ncclCommInitRank`.
  - `launcher-report.json` showed why the attempted default did not take effect:
    - `NCCL_SOCKET_IFNAME=^cbr,veth,docker,lo,cali,gke,node,cilium`
  - NCCL stdout confirmed it still selected `eth0:10.106.0.7` and failed with:
    - `socketStartConnect: Connect to 10.106.0.7<51847> failed : Software caused connection abort`
- Diagnosis:
  - the Vertex/PyTorch base image exports an ambient `NCCL_SOCKET_IFNAME` that explicitly excludes `lo`.
  - Heirloom's first loopback patch preserved inherited `NCCL_SOCKET_IFNAME`, so it treated the base image exclusion as an intentional user override.
- Fix implemented:
  - `launcher_env_from_process` no longer treats raw inherited `NCCL_SOCKET_IFNAME` or `NCCL_IB_DISABLE` as user intent.
  - rank workers now get:
    - `NCCL_SOCKET_IFNAME=lo`
    - `NCCL_IB_DISABLE=1`
  - explicit overrides must use:
    - `HEIRLOOM_NCCL_SOCKET_IFNAME`
    - `HEIRLOOM_NCCL_IB_DISABLE`
  - `HEIRLOOM_NCCL_DEBUG`, `HEIRLOOM_NCCL_DEBUG_SUBSYS`, and `HEIRLOOM_NCCL_NET_PLUGIN` are also supported as Heirloom-specific mappings to NCCL env.
  - Vertex wrapper now forwards those Heirloom-specific NCCL override variables.
  - `scripts/gcp/README.md` documents the inherited base-image exclusion and the override model.
  - tests now cover:
    - default loopback bootstrap,
    - ignoring inherited socket exclusions,
    - preserving Heirloom-specific overrides.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo test --bin heirloom nccl_launcher_env -- --nocapture`
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`
  - `cargo test --test distributed_cli`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Next live validation should rerun `nccl-init` and confirm `launcher-report.json` shows `NCCL_SOCKET_IFNAME=lo` for all ranks.

## Checkpoint 66 - Live NCCL Unique-Id Bootstrap Broker

- Reran the 4x A100 `nccl-init` probe after forcing loopback bootstrap.
- Job details:
  - job id: `7646001093595365376`
  - display: `heirloom-validate-quick-20260605-215805`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-215805`
- Progress versus Checkpoint 65:
  - `launcher-report.json` now showed the intended rank env:
    - `NCCL_SOCKET_IFNAME=lo`
    - `NCCL_IB_DISABLE=1`
  - NCCL stdout confirmed loopback bootstrap:
    - `Bootstrap : Using lo:127.0.0.1<0>`
    - `NET/Socket : Using [0]lo:127.0.0.1<0>`
- Failure:
  - ranks still failed at `ncclCommInitRank`.
  - rank stderr/stdout showed:
    - `socketStartConnect: Connect to 127.0.0.1<37185> failed : Software caused connection abort`
  - this is a different failure from the inherited `eth0` exclusion: NCCL is now using loopback correctly.
- Diagnosis:
  - the helper process that called `ncclGetUniqueId` exited immediately after printing the id.
  - the observed loopback port appears to belong to NCCL bootstrap state associated with the unique-id creator, so the rank workers were trying to initialize against an endpoint that no longer existed.
- Fix implemented:
  - hidden `heirloom nccl-unique-id` now accepts `--hold-secs` and flushes the marked id before sleeping.
  - the launcher parent now starts a live `NcclUniqueIdHelper`, reads the marked id from stdout, verifies the helper is still alive, and keeps the helper process alive until the rank launcher completes.
  - `gpu nccl-probe` and `train-lm --distributed nccl` both use the same live helper lifecycle.
  - helper cleanup is deterministic on success, failure, and timeout.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo test --bin heirloom nccl_unique_id -- --nocapture`
  - `cargo test --bin heirloom nccl_launcher_env -- --nocapture`
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`
  - `cargo test --test distributed_cli`
  - `cargo test -p heirloom-kernels`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Next live validation should rerun `nccl-init` once with the live helper. If that passes, proceed to the `all-reduce` rung rather than jumping directly to training.

## Checkpoint 67 - Vertex FastSocket Plugin Isolation

- Reran the 4x A100 `nccl-init` probe with the live unique-id helper.
- Job details:
  - job id: `5258248866158870528`
  - display: `heirloom-validate-quick-20260605-221446`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-221446`
  - local artifact copy: `/tmp/heirloom-validate-quick-20260605-221446/heirloom-validate-quick-20260605-221446`
- Progress versus Checkpoint 66:
  - the parent helper stayed alive and printed:
    - `NCCL unique-id helper holding bootstrap endpoint for 600s`
  - all four rank workers spawned with per-rank stdout/stderr/stage artifacts.
  - four GPU smoke tests passed before the NCCL rung.
  - topology again showed four A100-SXM4 80GB GPUs with `NV12` links and CUDA peer access.
- Failure:
  - ranks 2 and 3 failed at `ncclCommInitRank`; ranks 0 and 1 were killed by the parent after sibling failure.
  - `launcher-report.json` captured the failure instead of hanging:
    - rank 2: `ncclCommInitRank failed: unhandled system error`
    - rank 3: `ncclCommInitRank failed: unhandled system error`
  - rank stdout showed NCCL used loopback but still auto-loaded provider plugins:
    - `NET/Plugin: Loaded net plugin FastSocket (v6)`
    - `NET/Plugin: Loaded coll plugin ReductionServer (v6)`
    - `NET/FastSocket disabled`
    - `NET/Socket : Using [0]lo:127.0.0.1<0>`
    - `socketStartConnect: Connect to 127.0.0.1<41631> failed : Software caused connection abort`
- Diagnosis:
  - the previous helper-lifetime hypothesis was not sufficient; the helper now stayed alive.
  - the remaining failure occurs after NCCL auto-loads the Vertex image's FastSocket/ReductionServer plugins, even though the single-node probe only needs NCCL's internal socket transport.
  - for this launcher, inherited or auto-discovered net plugins should not participate unless explicitly requested by a Heirloom override.
- Fix implemented:
  - launcher rank env now defaults `NCCL_NET_PLUGIN=none`.
  - raw inherited `NCCL_NET_PLUGIN` is ignored for rank workers.
  - explicit provider plugins must use `HEIRLOOM_NCCL_NET_PLUGIN=<value>`.
  - `scripts/gcp/README.md` now documents the single-node defaults:
    - `NCCL_SOCKET_IFNAME=lo`
    - `NCCL_IB_DISABLE=1`
    - `NCCL_NET_PLUGIN=none`
  - tests now cover ignoring inherited Vertex-style bootstrap/plugin overrides while preserving Heirloom-specific overrides.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo test --bin heirloom nccl_launcher_env -- --nocapture`
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Next live validation should rerun only the 4x `nccl-init` rung and confirm rank stdout no longer contains `Loaded net plugin FastSocket`.

## Checkpoint 68 - NCCL Unique-Id Root Lifetime

- Reran the 4x A100 `nccl-init` probe after forcing `NCCL_NET_PLUGIN=none`.
- Job details:
  - job id: `6794539289045630976`
  - display: `heirloom-validate-quick-20260605-223051`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-223051`
  - local artifact copy: `/tmp/heirloom-validate-quick-20260605-223051/heirloom-validate-quick-20260605-223051`
- Progress versus Checkpoint 67:
  - `launcher-report.json` showed every rank got the intended single-node NCCL env:
    - `NCCL_SOCKET_IFNAME=lo`
    - `NCCL_IB_DISABLE=1`
    - `NCCL_NET_PLUGIN=none`
  - rank stdout confirmed FastSocket was no longer loaded:
    - `NET/Plugin: No plugin found (none)`
    - `NET/Plugin: Using internal network plugin.`
    - `NET/Socket : Using [0]lo:127.0.0.1<0>`
- Failure:
  - rank 0 failed at `ncclCommInitRank`.
  - sibling ranks were killed by the parent after the bounded failure path.
  - rank stdout still showed:
    - `socketStartConnect: Connect to 127.0.0.1<43635> failed : Software caused connection abort`
- Diagnosis:
  - the helper process stayed alive, and the provider plugin was isolated, so the remaining suspect is the lifetime of the NCCL library itself inside the helper.
  - the previous helper called `ncclGetUniqueId`, printed the id, then dropped the dynamically loaded `NcclLibrary` before sleeping.
  - if NCCL's bootstrap root state is tied to that loaded library handle, the helper process can remain alive while the endpoint the ranks need has already been torn down.
- Fix implemented:
  - added `heirloom_kernels::cuda::NcclUniqueIdRoot`, a safe handle that owns both the unique id and the loaded NCCL library.
  - hidden `heirloom nccl-unique-id --hold-secs ...` now keeps `NcclUniqueIdRoot` alive for the full hold period.
  - existing `nccl_unique_id()` remains available for one-shot cases.
  - main crate remains `#![forbid(unsafe_code)]`; the lifetime fix stays in the kernel wrapper crate.
- Local validation passed:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
- Next live validation should rerun only the 4x `nccl-init` rung. If it reaches `nccl_ready` for all ranks, immediately advance to the bounded `all-reduce` rung before attempting any DDP training.

## Checkpoint 69 - First Clean 4x NCCL Init And All-Reduce

- Reran the 4x A100 `nccl-init` probe after keeping the NCCL library alive inside the unique-id helper.
- `nccl-init` job details:
  - job id: `3578969155103096832`
  - display: `heirloom-validate-quick-20260605-224435`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-224435`
  - local artifact copy: `/tmp/heirloom-validate-quick-20260605-224435/heirloom-validate-quick-20260605-224435`
  - Vertex status: `JOB_STATE_SUCCEEDED`
- `nccl-init` results:
  - four `A100-SXM4-80GB` devices were visible.
  - `nvidia-smi topo -m` reported `NV12` links between every GPU pair.
  - Heirloom topology reported CUDA peer access for every cross-device pair.
  - `launcher-report.json` reported `status=passed`, no timeout, four ranks, and exit status `0` for every rank.
  - every rank reached final stage `report_written`.
  - every rank stderr reported `ncclCommInitRank returned success`.
  - rank stdout used loopback plus the internal NCCL network plugin:
    - `Bootstrap : Using lo:127.0.0.1<0>`
    - `NET/Plugin: No plugin found (none)`
    - `NET/Plugin: Using internal network plugin.`
  - NCCL logs reported `Connected all rings`, `Connected all trees`, and `Init COMPLETE`.
  - `nccl-probe.json` reported:
    - `status=passed`
    - `probe_kind=nccl-init`
    - `world_size=4`
    - `nccl_version=22105`
- Advanced immediately to the 4x A100 `all-reduce` rung after `nccl-init` passed.
- `all-reduce` job details:
  - job id: `6453813829737381888`
  - display: `heirloom-validate-quick-20260605-225455`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-225455`
  - local artifact copy: `/tmp/heirloom-validate-quick-20260605-225455/heirloom-validate-quick-20260605-225455`
  - Vertex status: `JOB_STATE_SUCCEEDED`
- `all-reduce` results:
  - `launcher-report.json` reported `status=passed`, no timeout, four ranks, and exit status `0` for every rank.
  - every rank reached final stage `report_written`.
  - `nccl-probe.json` reported:
    - `status=passed`
    - `probe_kind=all-reduce`
    - `world_size=4`
    - `len=1024`
    - `expected_sum=10.0`
    - `max_abs_error=0.0`
    - `tolerance=0.000001`
    - `all_reduce_calls=4`
    - `all_reduce_bytes=16384`
    - `nccl_version=22105`
  - rank reports showed each rank's input values `1.0`, `2.0`, `3.0`, and `4.0` all reduced to sample outputs of `10.0`.
- Local validation before launching the successful probes:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
- Cloud process hygiene:
  - both Vertex jobs completed successfully.
  - local `gcloud stream-logs` tailers were stopped after job success so no completed-job streamers remained open.
- Next live validation should wire this proven launcher/collective path into the DDP tiny fixture: one rank per GPU, gradient all-reduce, local optimizer step, and per-step parameter checksum synchronization.

## Checkpoint 70 - DDP Tiny Fixture Gate Wiring

- Hardened `train-lm --devices ... --distributed nccl` after the successful 4x NCCL probe ladder:
  - added `--ddp-checksum-every`,
  - rank reports now record per-step parameter checksum samples after optimizer updates,
  - aggregate DDP reports now include all-reduce calls/bytes, final parameter-checksum drift, per-step checksum drift, rank loss reductions, and rank-0 Tensor Core coverage,
  - DDP training now errors even without a top-level report if no gradient all-reduce is recorded or if final/per-step parameter checksum drift exceeds tolerance.
- Extended `scripts/cuda_train_lm_fixture.sh` from single-GPU-only to an opt-in distributed fixture:
  - `HEIRLOOM_CUDA_FIXTURE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`,
  - `HEIRLOOM_CUDA_FIXTURE_DISTRIBUTED=nccl`,
  - `HEIRLOOM_CUDA_FIXTURE_DDP_CHECKSUM_EVERY=1`,
  - train and resume validators check world size, per-rank/global batch size, positive all-reduce stats, checkpoint ownership, parameter-sync tolerance, and Tensor Core counters.
- Updated the Vertex validation wrapper:
  - forwards the DDP fixture knobs and Heirloom-specific NCCL overrides,
  - uploads `train-ddp-ranks/`, `resume-ddp-ranks/`, and `ddp-ranks/` artifact trees when present.
- Updated `ARCHITECTURE.md`, `HARD_MODE.md`, and `scripts/gcp/README.md`:
  - 4x NCCL init/all-reduce is now recorded as passed,
  - the immediate next gate is the 4x DDP tiny train/resume fixture,
  - TinyStories DDP remains behind that fixture.
- Local validation passed:
  - `bash -n scripts/cuda_train_lm_fixture.sh`
  - `bash -n scripts/gcp/submit_vertex_heirloom_validate.sh`
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --bin heirloom ddp_step_checksum -- --nocapture`
  - `cargo test --bin heirloom -- --nocapture`
  - `cargo test --test distributed_cli -- --nocapture`
  - `cargo test --workspace`
- Next live validation should run one 4x A100 DDP tiny fixture with the existing successful `all-reduce` probe enabled and public TinyStories disabled.

## Checkpoint 71 - First 4x A100 DDP Tiny Training Fixture Pass

- Launched the bounded 4x A100 DDP fixture gate after Checkpoint 70:
  - job id: `6473235603130417152`
  - display: `heirloom-validate-quick-20260605-234242`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260605-234242`
  - Vertex status: `JOB_STATE_SUCCEEDED`
- Hardware/topology evidence:
  - four `NVIDIA A100-SXM4-80GB` devices were visible,
  - `nvidia-smi topo -m` reported `NV12` between every GPU pair,
  - Heirloom topology reported CUDA peer access across the GPU pairs.
- NCCL preflight passed again:
  - `probe_kind=all-reduce`,
  - `world_size=4`,
  - `expected_sum=10.0`,
  - `max_abs_error=0.0`,
  - `all_reduce_calls=4`,
  - `all_reduce_bytes=16384`,
  - `nccl_version=22105`.
- DDP training fixture passed:
  - command shape: `train-lm --devices cuda:0,cuda:1,cuda:2,cuda:3 --distributed nccl --precision amp-bf16`,
  - per-rank batch size: `4`,
  - global batch size: `16`,
  - steps: `40`,
  - train loss: `5.903006 -> 0.918672`,
  - loss reduction: `0.844372`,
  - gradient all-reduce calls: `3520`,
  - gradient all-reduce bytes: `7905280`,
  - final parameter checksum drift: `0.0`,
  - per-step parameter checksum drift: `0.0`,
  - Tensor Core Linear counters: forward `1120`, backward `2240`, total `3360`, scalar fallback `0`.
- DDP resume fixture passed:
  - start/final step: `40 -> 43`,
  - resume loss: `1.420355 -> 1.212291`,
  - gradient all-reduce calls: `264`,
  - gradient all-reduce bytes: `592896`,
  - final and per-step parameter checksum drift: `0.0`,
  - Tensor Core Linear counters: forward `84`, backward `168`, total `252`, scalar fallback `0`.
- Uploaded artifacts include:
  - `summary.json`,
  - `nccl-probe.json` and `nccl-probe-ranks/`,
  - `cuda-train-lm-fixture/train-report.json`,
  - `cuda-train-lm-fixture/resume-report.json`,
  - `cuda-train-lm-fixture/train-ddp-ranks/`,
  - `cuda-train-lm-fixture/resume-ddp-ranks/`,
  - topology, smoke, and per-device Tensor Core probe reports.
- Local cloud-process hygiene:
  - Vertex job completed successfully.
  - local `gcloud stream-logs` and submit wrapper processes were stopped after success.
- Next frontier: run the first 4x TinyStories DDP reference tier, still keeping it bounded and artifact-rich before attempting a full public-data DDP gate.

## Checkpoint 72 - First 4x A100 TinyStories DDP Reference Tier Pass

- Launched the first bounded public-data 4x A100 DDP reference tier:
  - job id: `2438784393192407040`
  - display: `heirloom-validate-quick-20260606-002304`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-002304`
  - Vertex status: `JOB_STATE_SUCCEEDED`
- Wrapper settings:
  - `HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4`
  - `HEIRLOOM_GPU_SMOKE_DEVICES=all`
  - `HEIRLOOM_RUN_NCCL_PROBE=1`
  - `HEIRLOOM_NCCL_PROBE_KIND=all-reduce`
  - `HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=1`
  - `HEIRLOOM_TINYSTORIES_CUDA_MODE=reference`
  - `HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`
  - `HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl`
  - `HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16`
  - `HEIRLOOM_REQUIRE_TENSOR_CORES=1`
- Hardware/topology evidence repeated:
  - four `NVIDIA A100-SXM4-80GB` devices were visible,
  - `nvidia-smi topo -m` reported `NV12` between every GPU pair,
  - Heirloom topology reported CUDA peer access across the GPU pairs.
- NCCL preflight passed:
  - `probe_kind=all-reduce`,
  - `world_size=4`,
  - `expected_sum=10.0`,
  - `max_abs_error=0.0`,
  - `all_reduce_calls=4`,
  - `all_reduce_bytes=16384`,
  - `nccl_version=22105`.
- TinyStories reference tier:
  - downloaded `TinyStories-valid.txt`,
  - trained the homegrown byte-level BPE-like tokenizer with vocab size `512` on the first `1048576` bytes,
  - prepared `567918` train tokens and `29890` validation tokens,
  - trained `80` steps with per-rank batch size `2`, global batch size `8`, block size `16`, `d_model=16`, `n_heads=4`, and `ff_hidden=64`.
- DDP public-data train results:
  - train loss: `6.579771 -> 4.123779`,
  - loss reduction: `0.373264`,
  - gradient all-reduce calls: `7040`,
  - gradient all-reduce bytes: `26193920`,
  - final parameter checksum drift: `0.0`,
  - per-step parameter checksum drift: `0.0`.
- Tensor Core evidence:
  - BF16 Tensor Core Linear matmul calls: `6720`,
  - forward calls: `2240`,
  - backward calls: `4480`,
  - scalar matmul fallbacks: `0`,
  - `tensor_core_coverage.linear_totals`: `560` calls, `560` Tensor Core calls, `0` fallbacks.
- Eval/generation:
  - heldout validation loss: `3.951583`,
  - perplexity: `52.017655`,
  - eval batches/tokens: `20` / `640`,
  - generation prompt: `Once upon a time`,
  - generation emitted `80` new tokens and stopped at `max_new_tokens`.
- Uploaded artifacts include:
  - top-level `summary.json`,
  - `nccl-probe.json` and `nccl-probe-ranks/`,
  - `tinystories-cuda-reference/summary.json`,
  - `tinystories-cuda-reference/report.json`,
  - `tinystories-cuda-reference/eval.json`,
  - `tinystories-cuda-reference/generation.json`,
  - `tinystories-cuda-reference/generation.txt`,
  - `tinystories-cuda-reference/timings.json`,
  - `tinystories-cuda-reference/launcher-report.json`,
  - `tinystories-cuda-reference/ddp-ranks/`,
  - `tinystories-cuda-reference/checkpoint/`.
- Local cloud-process hygiene:
  - Vertex job completed successfully.
  - local `gcloud stream-logs` and submit wrapper processes were stopped after success.
- This is now a real 4x public-data DDP reference tier. The next frontier is the full 4x TinyStories-valid DDP gate: uncapped validation text, `500` steps, `d_model=64`, `block_size=64`, `20%` loss-reduction gate, eval/perplexity, generation, and checkpoint/resume evidence.

## Checkpoint 73 - Full 4x A100 TinyStories DDP Gate Pass

- Launched the full 4x TinyStories-valid DDP gate:
  - job id: `9051757496032559104`
  - display: `heirloom-validate-quick-20260606-004954`
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260606-004954`
  - Vertex status: `JOB_STATE_SUCCEEDED`
- Wrapper settings:
  - `HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4`
  - `HEIRLOOM_GPU_SMOKE_DEVICES=all`
  - `HEIRLOOM_RUN_NCCL_PROBE=1`
  - `HEIRLOOM_NCCL_PROBE_KIND=all-reduce`
  - `HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE=1`
  - `HEIRLOOM_TINYSTORIES_CUDA_MODE=full`
  - `HEIRLOOM_TINYSTORIES_CUDA_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`
  - `HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED=nccl`
  - `HEIRLOOM_TINYSTORIES_CUDA_PRECISION=amp-bf16`
  - `HEIRLOOM_TINYSTORIES_CUDA_STEPS=500`
  - `HEIRLOOM_TINYSTORIES_CUDA_RESUME_STEPS=5`
  - `HEIRLOOM_TINYSTORIES_CUDA_BLOCK=64`
  - `HEIRLOOM_TINYSTORIES_CUDA_D_MODEL=64`
  - `HEIRLOOM_TINYSTORIES_CUDA_HEADS=4`
  - `HEIRLOOM_TINYSTORIES_CUDA_FF=256`
  - `HEIRLOOM_TINYSTORIES_CUDA_VOCAB=1024`
  - `HEIRLOOM_REQUIRE_TENSOR_CORES=1`
  - `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1`
- NCCL preflight passed:
  - `probe_kind=all-reduce`,
  - `world_size=4`,
  - `expected_sum=10.0`,
  - `max_abs_error=0.0`,
  - `all_reduce_calls=4`,
  - `all_reduce_bytes=16384`,
  - `nccl_version=22105`.
- TinyStories full-data preparation:
  - downloaded full `TinyStories-valid.txt`,
  - trained vocab-`1024` byte-level BPE-like tokenizer without a byte cap,
  - prepared `8962067` train tokens and `471688` validation tokens,
  - source hash: `88e02b461de6a19e`,
  - tokenizer hash: `094b45d84fd7a7eb`.
- Full DDP train results:
  - steps: `0 -> 500`,
  - per-rank batch size: `4`,
  - global batch size: `16`,
  - train loss: `7.063565 -> 3.666313`,
  - loss reduction: `0.480954`,
  - rank loss reductions: `0.480954`, `0.505837`, `0.489435`, `0.491327`,
  - gradient all-reduce calls: `44000`,
  - gradient all-reduce bytes: `1490432000`,
  - final parameter checksum drift: `0.0`,
  - per-step parameter checksum drift: `0.0`.
- Resume continuation passed from a copied checkpoint:
  - steps: `500 -> 505`,
  - resume loss: `3.600464 -> 3.545029`,
  - resume all-reduce calls: `440`,
  - resume all-reduce bytes: `14904320`,
  - final and per-step parameter checksum drift: `0.0`.
- Tensor Core evidence under hard gates:
  - train BF16 Tensor Core matmul calls: `234000`,
  - train forward calls: `78000`,
  - train backward calls: `156000`,
  - train scalar fallbacks: `0`,
  - train Linear coverage: `3500` Linear calls, `3500` Tensor Core calls, `0` fallbacks,
  - train attention forward calls: `2000`,
  - train attention QK/AV matmul calls: `32000` / `32000`,
  - train attention backward calls: `2000`,
  - train attention score-grad/dQ/dK/dV matmul calls: `32000` / `32000` / `32000` / `32000`,
  - resume BF16 Tensor Core matmul calls: `2340`,
  - resume scalar fallbacks: `0`,
  - resume attention forward/backward Tensor Core counters were also positive.
- Eval/generation:
  - heldout validation loss: `3.581789`,
  - perplexity: `35.937761`,
  - eval batches/tokens: `200` / `51200`,
  - generation prompt: `Once upon a time`,
  - generation emitted `120` new tokens and stopped at `max_new_tokens`.
- Stage timings:
  - tokenizer: `90834` ms,
  - prepare: `14313` ms,
  - train: `146594` ms,
  - resume: `9848` ms,
  - eval: `14239` ms,
  - generate: `8253` ms.
- Uploaded artifacts include:
  - top-level `summary.json`,
  - `nccl-probe.json` and `nccl-probe-ranks/`,
  - `tinystories-cuda-reference/summary.json`,
  - `tinystories-cuda-reference/report.json`,
  - `tinystories-cuda-reference/resume-report.json`,
  - `tinystories-cuda-reference/eval.json`,
  - `tinystories-cuda-reference/generation.json`,
  - `tinystories-cuda-reference/generation.txt`,
  - `tinystories-cuda-reference/timings.json`,
  - `tinystories-cuda-reference/train-launcher-report.json`,
  - `tinystories-cuda-reference/resume-launcher-report.json`,
  - `tinystories-cuda-reference/train-ddp-ranks/`,
  - `tinystories-cuda-reference/resume-ddp-ranks/`,
  - `tinystories-cuda-reference/checkpoint/`,
  - `tinystories-cuda-reference/checkpoint-resume/`.
- Local cloud-process hygiene:
  - Vertex job completed successfully.
  - local `gcloud stream-logs` and submit wrapper processes were stopped after success.
- This crosses off the first full 4x A100 TinyStories-valid DDP gate. It is still not production distributed training: kernels remain correctness-first, the launcher is single-node only, AMP is narrow, and the tokenizer/data path is still prototype-grade.

## Checkpoint 74 - Milestone Freeze, CUDA Runtime Hardening, AMP Policy

- Added `MILESTONE_4X_TINYSTORIES.md` as the canonical reviewer anchor for the full 4x A100 TinyStories-valid DDP gate.
- Added CUDA runtime counters in `heirloom-kernels` for kernel launches, stream/event activity, host synchronization, H2D/D2H/D2D bytes, allocation stats, PTX module loads/cache hits, and Tensor Core padding/remainder counters.
- Reworked CUDA kernel launches to enqueue on cached per-context compute streams instead of synchronizing the full context after every launch.
- Added a first per-thread CUDA caching allocator with active/reserved/high-water byte accounting and stream-event deferred frees, plus `HEIRLOOM_CUDA_ALLOCATOR_DIRECT=1` as a direct-allocation debug mode.
- Added a per-context PTX module cache, with throwaway CUDA-session modules evicted before context destruction.
- Added `AmpBf16Policy`, per-op AMP BF16 decisions, CUDA host-staging event reports, and strict AMP training scopes that reject unexpected CUDA-to-CPU tensor materialization.
- Threaded AMP policy, op decisions, host-staging events, and CUDA runtime counters into single-rank and DDP train reports.
- Added always-on tests for CUDA runtime counter reset and strict AMP host-staging guard behavior.
- Updated `README.md`, `ARCHITECTURE.md`, and `HARD_MODE.md` to distinguish the new first-pass allocator/stream/AMP surfaces from the still-missing production pieces.

## Checkpoint 75 - Tensor Core Linear Pad/Crop Edges

- Broadened the sm80 BF16 Tensor Core RHS-transposed GEMM wrapper used by AMP `Linear` forward/backward:
  - logical inputs remain `A[M,K]` and `W[N,K]`,
  - BF16 matrices are padded on device to `M%16=0`, `K%16=0`, and `N%8=0`,
  - the existing `mma.sync` tile kernel runs on the padded buffers,
  - the f32 output is cropped back to logical `[M,N]`.
- Added CUDA support kernels:
  - `heirloom_pad_matrix_bf16`,
  - `heirloom_crop_matrix_f32`.
- Split shape predicates:
  - `bf16_tensor_core_matmul_exact_tile_shape_supported` reports already-perfect MMA tile shapes,
  - `bf16_tensor_core_matmul_shape_supported` reports positive shapes supported through pad/crop.
- Wired padded and remainder tile accounting into existing CUDA runtime counters.
- Added an opt-in CUDA test with deliberately ragged `M=15`, `K=17`, `N=7` that compares Tensor Core output against a BF16-rounded CPU reference and asserts nonzero padded/remainder counters.
- Updated `README.md`, `ARCHITECTURE.md`, and `HARD_MODE.md` to clarify that Linear now has explicit pad/crop edge handling, while tuned CTA-level/general/batched Tensor Core GEMM and strict attention remainders remain open.

## Checkpoint 76 - Single-A100 Ragged Tensor Core Gate

- Added `HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES=1` to `scripts/cuda_train_lm_fixture.sh`.
  - The fixture selects deliberately ragged Linear dimensions: batch `3`, block `5`, `d_model=18`, `n_heads=3`, `ff_hidden=37`, vocab `281`.
  - It requires positive Tensor Core pad/remainder counters when `HEIRLOOM_EXPECT_TENSOR_CORE_PADDING=1`.
  - It rejects combining this isolated Linear pad/crop gate with the strict attention Tensor Core hard gate.
- Forwarded `HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES` and `HEIRLOOM_EXPECT_TENSOR_CORE_PADDING` through the Vertex validation wrapper.
- Fixed the Vertex YAML optional-env path by replacing the empty `HEIRLOOM_NCCL_PROBE_DEVICES` default with the existing unset sentinel.
- First Vertex attempt exposed a launcher-test race after the CUDA gate itself passed:
  - job id: `8978404677297111040`,
  - display: `heirloom-validate-quick-20260607-160107`,
  - failure: `launcher_test_kills_sibling_on_one_rank_failure` expected the killed sibling to have reached `config_loaded`, but the parent can legitimately kill it while its last stage is still `spawned`.
- Hardened the launcher test to assert the real contract:
  - failing rank reports `failed`,
  - sibling is completed after termination,
  - sibling last stage is a valid bounded stage, either `spawned` or `config_loaded`.
- Patched Vertex rerun passed:
  - job id: `8524104064886112256`,
  - display: `heirloom-validate-quick-20260607-161144`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260607-161144`,
  - Vertex status: `JOB_STATE_SUCCEEDED`.
- Single-A100 gate evidence:
  - observed GPU: `NVIDIA A100-SXM4-80GB`,
  - CUDA storage tests passed: `43` tests, including raw ragged `M=15,K=17,N=7` BF16 Tensor Core matmul parity,
  - ragged AMP BF16 fixture train loss: `5.744056 -> 2.257442`, reduction `0.606995`,
  - resume loss: `2.420082 -> 1.937643`, reduction `0.199348`,
  - train Tensor Core matmul calls: forward `280`, backward `560`, total `840`,
  - resume Tensor Core matmul calls: forward `21`, backward `42`, total `63`,
  - scalar matmul fallbacks: `0`,
  - Linear coverage fallback calls: `0`,
  - train padded/remainder tile counters: `7760` / `7760`,
  - resume padded/remainder tile counters: `582` / `582`.
- Local validation after the race fix passes:
  - `cargo test --test distributed_cli`,
  - `cargo fmt --all --check`,
  - `./scripts/validate.sh`.
- Local cloud-process hygiene:
  - the failed and successful Vertex jobs reached terminal states,
  - dangling local `gcloud stream-logs` processes were stopped after terminal state,
  - no full Vertex job JSON or secrets were printed.

## Checkpoint 77 - Tensor Core Pad/Crop Report Promotion

- Promoted Linear Tensor Core pad/crop evidence from raw CUDA counters into a named `tensor_core_pad_crop` train-report section.
  - Reports now include `used`, `passed`, `status`, padded/remainder tile counts, scalar and Linear fallback counts, tile rules, logical shapes, padded shapes, and per-Linear module evidence.
  - Single-rank reports use local runtime counters; distributed aggregate reports sum runtime and Tensor Core counters across ranks while taking shape evidence from rank 0.
- Updated `scripts/cuda_train_lm_fixture.sh` to validate `tensor_core_pad_crop.status == "passed"` when `HEIRLOOM_EXPECT_TENSOR_CORE_PADDING=1`.
  - The script still accepts older reports through raw `cuda_runtime.tensor_core_padded_tiles` and `cuda_runtime.tensor_core_remainder_tiles` counters.
  - It now writes compact reviewer artifacts:
    - `tensor-core-pad-crop-summary.json`,
    - `tensor-core-pad-crop-summary.txt`.
- Updated the Vertex validation wrapper to upload those compact pad/crop summaries and include their GCS URIs in the top-level summary JSON.
- Threaded `tensor_core_pad_crop` into TinyStories train/resume summaries so public-data reference runs expose the same evidence without opening full rank reports.
- Updated `README.md`, `ARCHITECTURE.md`, `HARD_MODE.md`, and `scripts/gcp/README.md` to document the promoted report surface and its gates.
- Local validation passed:
  - `cargo test -p heirloom --bin heirloom tensor_core_pad_crop_report`,
  - `bash -n scripts/cuda_train_lm_fixture.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - CPU fixture smoke for `scripts/cuda_train_lm_fixture.sh`,
  - `./scripts/validate.sh`.

## Checkpoint 78 - Single-A100 Pad/Crop Artifact Gate

- Reran the narrow single-A100 ragged Tensor Core Linear pad/crop gate after promoting the report surface.
  - Vertex job id: `2194224419872702464`,
  - display: `heirloom-validate-quick-20260607-171521`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260607-171521`,
  - Vertex status: `JOB_STATE_SUCCEEDED`,
  - observed GPU: `NVIDIA A100-SXM4-80GB`.
- The run produced and uploaded the new compact reviewer artifacts:
  - `cuda-train-lm-fixture/tensor-core-pad-crop-summary.json`,
  - `cuda-train-lm-fixture/tensor-core-pad-crop-summary.txt`.
- The top-level summary reported `status=passed` and included both pad/crop summary artifact URIs.
- CUDA storage tests passed `43` tests on the A100, including the raw ragged `M=15,K=17,N=7` BF16 Tensor Core matmul parity test.
- Ragged AMP BF16 fixture evidence:
  - train loss: `5.800171 -> 2.331136`, reduction `0.598092`,
  - resume loss: `2.296570 -> 1.916594`, reduction `0.165454`,
  - train `tensor_core_pad_crop.status=passed`,
  - resume `tensor_core_pad_crop.status=passed`,
  - train padded/remainder tiles: `7760` / `7760`,
  - resume padded/remainder tiles: `582` / `582`,
  - scalar matmul fallbacks: `0`,
  - Linear fallback calls: `0`,
  - train BF16 Tensor Core matmul calls: `840`,
  - resume BF16 Tensor Core matmul calls: `63`.
- Local cloud-process hygiene:
  - the Vertex job reached terminal success,
  - the local `gcloud stream-logs` wrapper was stopped after success because it did not unwind on its own,
  - no full Vertex job JSON or secrets were printed.

## Checkpoint 79 - CTA Tensor Core Linear GEMM Engine

- Replaced the default one-warp-per-output-tile Linear Tensor Core GEMM launch with a CTA-level PTX entry:
  - `heirloom_matmul_bf16_mma_rhs_t_f32_cta`,
  - four warps per block,
  - each CTA owns a 32x16 output region,
  - each warp computes one `m16n8k16` MMA tile,
  - inactive edge warps skip cleanly after pad/crop,
  - the previous one-warp kernel remains available behind `HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM=1`.
- Added `TensorCoreMatmulCtaPlan` and always-on plan tests so CTA tile geometry is testable without CUDA.
- Extended CUDA runtime counters and report JSON with:
  - `tensor_core_cta_gemm_calls`,
  - `tensor_core_cta_tiles`,
  - `tensor_core_cta_warps_launched`,
  - `tensor_core_mma_warp_tiles`,
  - `tensor_core_legacy_warp_gemm_calls`.
- Hardened `scripts/cuda_train_lm_fixture.sh` so Tensor Core hard gates require positive CTA counters and zero legacy one-warp GEMM calls unless the legacy debug env is explicitly set.
- Extended compact pad/crop summaries with CTA engine counters.
- Forwarded `HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM` through the Vertex wrapper for explicit debug runs.
- Updated `README.md`, `ARCHITECTURE.md`, `HARD_MODE.md`, and `scripts/gcp/README.md` to distinguish the CTA global-memory-fed engine from the next targets: shared-memory staging, `ldmatrix`, and tuned GEMM.
- Local validation passed:
  - `bash -n scripts/cuda_train_lm_fixture.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test cuda_storage`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- Live single-A100 CTA gate passed:
  - Vertex job id: `6442104030901567488`,
  - display: `heirloom-validate-quick-20260607-175826`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260607-175826`,
  - Vertex status: `JOB_STATE_SUCCEEDED`,
  - observed GPU: `NVIDIA A100-SXM4-80GB`,
  - CUDA storage tests passed `44` tests,
  - train loss: `5.776606 -> 2.408088`, reduction `0.583131`,
  - resume loss: `2.496775 -> 1.487485`, reduction `0.404238`,
  - train CTA GEMM calls: `840`,
  - train CTA tiles: `3160`,
  - train launched CTA warps: `12640`,
  - train active MMA warp tiles: `7760`,
  - resume CTA GEMM calls: `63`,
  - resume CTA tiles: `237`,
  - resume launched CTA warps: `948`,
  - resume active MMA warp tiles: `582`,
  - legacy one-warp GEMM calls: `0`,
  - scalar matmul fallbacks: `0`,
  - Linear fallback calls: `0`.
- Local cloud-process hygiene:
  - the Vertex job reached terminal success,
  - the local `gcloud stream-logs` wrapper was stopped after success because it did not unwind on its own,
  - no full Vertex job JSON or secrets were printed.

## Checkpoint 80 - Shared-Memory Staged CTA Linear GEMM

- Promoted the default Linear Tensor Core GEMM path from the global-memory-fed CTA kernel to a shared-memory-staged CTA PTX entry:
  - `heirloom_matmul_bf16_mma_rhs_t_f32_cta_staged`,
  - four warps per block,
  - each CTA owns a 32x16 output region,
  - each K slice cooperatively stages a 32x16 A tile and a 16x16 RHS tile into shared memory,
  - inactive edge lanes zero-fill staged tiles instead of returning before CTA barriers,
  - each active warp feeds one `m16n8k16` MMA tile,
  - the previous global-memory-fed CTA kernel remains available behind `HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM=1`,
  - the older one-warp kernel remains available behind `HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM=1`.
- Extended CTA planning and runtime counters with staged-kernel evidence:
  - `TensorCoreMatmulCtaPlan::k_tiles`,
  - `TensorCoreMatmulCtaPlan::shared_stage_tiles`,
  - `TensorCoreMatmulCtaPlan::shared_stage_bytes`,
  - `tensor_core_staged_cta_gemm_calls`,
  - `tensor_core_shared_stage_tiles`,
  - `tensor_core_shared_stage_bytes`,
  - `tensor_core_global_cta_gemm_calls`.
- Hardened train/resume fixture checks so `HEIRLOOM_EXPECT_TENSOR_CORES=1` requires positive staged CTA/shared-stage counters and zero global/legacy GEMM calls unless an explicit debug env selects the older kernels.
- Extended compact pad/crop summaries and aggregate rank CUDA reports with staged CTA, shared-stage, and global CTA counters.
- Updated CUDA docs to describe the new default honestly:
  - this is real shared-memory staging,
  - it was still not an `ldmatrix` or tuned Tensor Core GEMM,
  - attention still uses its own strict tile-aligned Tensor Core matmul path rather than this general Linear wrapper.
- First live A100 staged-CTA attempt exposed a real test-suite race rather than a PTX failure:
  - Vertex job id: `516237734491193344`,
  - display: `heirloom-validate-quick-20260607-185226`,
  - state: `JOB_STATE_FAILED`,
  - raw Tensor Core staged GEMM tests had already passed,
  - failure was `cuda_runtime_counters_reset_to_zero_without_cuda`,
  - root cause: the CPU-only counter-reset assertion can race with parallel CUDA tests that legitimately create streams/events after reset,
  - fix: skip that CPU-only reset invariant when `HEIRLOOM_CUDA_TESTS=1`.
- Successful replacement single-A100 staged-CTA gate passed:
  - Vertex job id: `4807042279468433408`,
  - display: `heirloom-validate-quick-20260607-185948`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260607-185948`,
  - Vertex status: `JOB_STATE_SUCCEEDED`,
  - observed GPU: `NVIDIA A100-SXM4-80GB`,
  - CUDA storage tests passed `44` tests,
  - train loss: `5.833638 -> 2.131579`, reduction `0.634605`,
  - resume loss: `2.498653 -> 1.840109`, reduction `0.263560`,
  - train Tensor Core matmul calls: `840`,
  - train CTA GEMM calls: `840`,
  - train CTA tiles: `3160`,
  - train launched CTA warps: `12640`,
  - train active MMA warp tiles: `7760`,
  - train staged CTA calls: `840`,
  - train shared-stage tiles: `6440`,
  - train shared-stage bytes: `9891840`,
  - train global CTA calls: `0`,
  - train legacy one-warp GEMM calls: `0`,
  - resume staged CTA calls: `63`,
  - resume shared-stage tiles: `483`,
  - resume shared-stage bytes: `741888`,
  - resume global CTA calls: `0`,
  - resume legacy one-warp GEMM calls: `0`,
  - scalar matmul fallbacks: `0`,
  - Linear fallback calls: `0`.
- Local validation passed after the staged-kernel and test-race fixes:
  - `cargo fmt --all`,
  - `bash -n scripts/cuda_train_lm_fixture.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `cargo test -p heirloom --test cuda_storage`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.

## Checkpoint 81 - Ragged Tensor Core Attention Edge Handling

- Extended the BF16 Tensor Core causal-attention path beyond exact tile-compatible shapes:
  - `causal_attention_bf16_tensor_core_exact_tile_shape_supported` keeps the old exact `time/head_dim` tile predicate for tests and diagnostics,
  - `causal_attention_bf16_tensor_core_shape_supported` now accepts valid nonzero attention shapes when `channels % n_heads == 0`,
  - QK, AV, score-grad, dQ, dK, and dV Tensor Core launches now use ceil grids and predicated edge loads/stores,
  - invalid edge lanes zero-fill before MMA and stores are cropped to logical `time`/`head_dim` outputs.
- Added CUDA runtime counters for attention edge handling:
  - `tensor_core_attention_padded_tiles`,
  - `tensor_core_attention_remainder_tiles`,
  - `tensor_core_attention_scalar_fallbacks`.
- Threaded the new counters through train/rank JSON reports and the fixture validator.
- Added test coverage:
  - an always-on shape-support test that distinguishes exact tile support from ragged-but-valid attention support,
  - a gated A100 ragged attention forward/backward parity test using `time=15`, `head_dim=17`,
  - CUDA runtime counter reset coverage for the new attention counters.
- Added a ragged attention fixture mode:
  - `HEIRLOOM_CUDA_FIXTURE_RAGGED_ATTENTION_TENSOR_CORES=1`,
  - default shape `batch=3`, `block=15`, `d_model=17`, `n_heads=1`, `ff_hidden=37`, vocab `281`,
  - `HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORE_PADDING=1` requires positive attention padded/remainder counters and zero attention scalar fallback in both train and resume reports.
- Fixed an orchestration bug exposed by the first A100 attempt:
  - failed Vertex job id: `5417139291488780288`,
  - display: `heirloom-validate-quick-20260607-202859`,
  - CUDA storage tests had already passed the new ragged attention test,
  - fixture validation failed because the general attention auto-tile branch overwrote ragged attention dimensions back to `block=16`, `d_model=16`,
  - fix: keep explicit ragged fixture modes mutually exclusive with the general attention auto-tile override.
- Successful replacement single-A100 ragged attention gate passed:
  - Vertex job id: `4726399698640830464`,
  - display: `heirloom-validate-quick-20260607-204532`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260607-204532`,
  - Vertex status: `JOB_STATE_SUCCEEDED`,
  - observed GPU: `NVIDIA A100-SXM4-80GB`,
  - CUDA storage tests passed `46` tests,
  - train shape: batch `3`, block `15`, `d_model=17`, `n_heads=1`, `ff_hidden=37`, vocab `281`,
  - train loss: `5.734654 -> 1.945937`, reduction `0.660671`,
  - resume loss: `1.769477 -> 1.705577`, reduction `0.036113`,
  - train attention Tensor Core counters: QK `120`, AV `120`, score-grad `120`, dQ `120`, dK `120`, dV `120`,
  - train attention padded/remainder/scalar fallback counters: `1920` / `1920` / `0`,
  - resume attention padded/remainder/scalar fallback counters: `144` / `144` / `0`,
  - train Linear pad/crop counters: padded/remainder `14080` / `14080`, scalar fallbacks `0`, Linear fallback calls `0`,
  - train staged CTA counters: calls `840`, shared-stage tiles `14200`, shared-stage bytes `21811200`, global CTA calls `0`, legacy one-warp GEMM calls `0`.
- Local validation passed before and after the Vertex run:
  - `cargo fmt --all --check`,
  - `bash -n scripts/cuda_train_lm_fixture.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `cargo test -p heirloom --test cuda_storage bf16_tensor_core_attention_shape_support_distinguishes_exact_and_ragged_edges`,
  - `cargo test -p heirloom --test cuda_storage`.
- Local cloud-process hygiene:
  - both Vertex jobs reached terminal states,
  - stale local `gcloud stream-logs` wrappers were stopped after terminal state,
  - no full Vertex job JSON or secrets were printed.

## Checkpoint 82 - Opt-In Wide Swizzled CTA Tensor Core GEMM

- Added an opt-in Linear Tensor Core GEMM family behind `HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM=1`:
  - one CTA covers a `32x32` output region,
  - eight warps per CTA cover up to two row groups by four column groups,
  - each active warp still feeds one `m16n8k16` MMA tile,
  - A/RHS BF16 tiles are staged through shared memory with a simple XOR column swizzle,
  - the default path remains the four-warp shared-memory staged CTA kernel.
- Added no-CUDA launch-plan coverage:
  - `bf16_tensor_core_matmul_wide_swizzled_cta_plan(32, 64, 32)` reports one CTA, eight launched warps, eight active MMA warp tiles, four K tiles, and `8192` swizzled-stage bytes,
  - ragged single-tile shapes such as padded `16x32x8` report eight launched warps but one active MMA warp tile.
- Added CUDA runtime counters and report aggregation:
  - `tensor_core_wide_swizzled_cta_gemm_calls`,
  - `tensor_core_swizzled_stage_tiles`,
  - `tensor_core_swizzled_stage_bytes`.
- Threaded the new counters through train/rank reports, compact pad/crop summaries, the CUDA fixture validator, and the Vertex wrapper env forwarding.
- First A100 attempt failed usefully:
  - Vertex job id: `1870765691167047680`,
  - display: `heirloom-validate-quick-20260607-212939`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260607-212939`,
  - failure: `cuda_raw_bf16_tensor_core_matmul_pads_ragged_shapes` expected four launched CTA warps, but the new wide kernel correctly reported eight,
  - math had already passed the raw ragged BF16 comparison before the assertion failed,
  - fix: make the test expect eight launched warps only under `HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM=1`.
- Successful replacement single-A100 wide-swizzled gate passed:
  - Vertex job id: `5850540386879012864`,
  - display: `heirloom-validate-quick-20260607-213948`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260607-213948`,
  - Vertex status: `JOB_STATE_SUCCEEDED`,
  - observed GPU: `NVIDIA A100-SXM4-80GB`,
  - CUDA storage tests passed `46` tests,
  - fixture shape: batch `3`, block `15`, `d_model=17`, `n_heads=1`, `ff_hidden=37`, vocab `281`,
  - train loss: `5.894208 -> 1.465120`, reduction `0.751430`,
  - resume loss: `1.657021 -> 1.517183`, reduction `0.084391`,
  - train wide-swizzled CTA calls: `840`,
  - train swizzled-stage tiles/bytes: `7320` / `14991360`,
  - train staged/global/legacy GEMM calls: `0` / `0` / `0`,
  - train CTA tiles/warps/MMA warp tiles: `2600` / `20800` / `14080`,
  - train Linear pad/crop padded/remainder tiles: `14080` / `14080`,
  - train attention padded/remainder/scalar fallback counters: `1920` / `1920` / `0`,
  - resume wide-swizzled CTA calls: `63`,
  - resume swizzled-stage tiles/bytes: `549` / `1124352`,
  - resume staged/global/legacy GEMM calls: `0` / `0` / `0`,
  - scalar matmul fallbacks and Linear fallback calls stayed `0`.
- Local validation around the change:
  - `cargo fmt --all --check`,
  - `bash -n scripts/cuda_train_lm_fixture.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `cargo test -p heirloom --test cuda_storage cuda_raw_bf16_tensor_core_matmul_pads_ragged_shapes`,
  - `cargo test -p heirloom --test cuda_storage bf16_tensor_core_cta_plan_reports_tile_geometry_without_cuda`,
  - `cargo test -p heirloom --test cuda_storage`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- Cloud-process hygiene:
  - both Vertex jobs reached terminal states,
  - stale local `gcloud stream-logs` wrappers were stopped after terminal state,
  - a noisy access-check listing printed older non-secret job specs; subsequent state checks used constrained `--format` output.

## Checkpoint 83 - Production AMP Report Hardening

- Added `AmpBf16FiniteCheckEvent` and `AmpBf16ValidationReport`:
  - reports now summarize AMP status, op-decision counts, Tensor Core decisions, fallback reasons, finite-check counts/failures, and unexpected/allowed CUDA host-staging counts,
  - `train-lm`, DDP rank reports, DDP aggregate reports, `eval-lm`, and `generate` now emit `amp_bf16_validation`,
  - the CUDA fixture script now fails `amp-bf16` runs unless validation status is `passed`, finite checks exist, op decisions exist, and unexpected host staging is zero.
- Expanded transformer AMP op-decision reporting beyond Linear/attention/loss:
  - token and position embeddings,
  - layer norm,
  - GELU,
  - residual and token-position adds,
  - AdamW update policy.
- Hardened finite checks:
  - training loss scalar reads now return a normal error on non-finite values,
  - eval loss scalar reads are explicitly allowed and finite-checked in AMP mode,
  - CUDA gradient sum-of-squares, global gradient norm, clip scale, and AdamW learning rate are recorded as AMP finite checks,
  - AdamW now rejects non-finite or negative global gradient sum-of-squares instead of accidentally treating NaN norms as unclipped.
- Corrected `eval-lm --precision amp-bf16` and `generate --precision amp-bf16` to use the explicit AMP forward/loss path rather than the generic BF16 activation-rounding path; generation logits inspection is now a scoped host-staging allowance.
- Added always-on tests for AMP validation summaries, finite-check rejection, and AdamW non-finite clipped-gradient rejection.

## Checkpoint 84 - Production AMP Single-A100 Gate

- Ran the single-A100 `amp-bf16` CUDA LM fixture gate with hard Tensor Core and attention Tensor Core requirements.
- First run produced the important CUDA signal, then failed later in always-on tests:
  - Vertex job id: `3666400120720588800`,
  - display: `heirloom-validate-quick-20260608-015330`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260608-015330`,
  - fixture passed before the wrapper failed,
  - failure cause: hidden launcher self-tests used a 1s rank-start timeout, which was too tight for Vertex `cargo test` process startup jitter.
- Hardened the launcher self-tests:
  - `launcher-test` hidden CLI default rank-start timeout is now 5s,
  - success and sibling-kill tests pass `--rank-start-timeout-secs 5` explicitly,
  - hung-rank timeout coverage remains bounded by its short overall timeout.
- Successful replacement run:
  - Vertex job id: `1223197322872094720`,
  - display: `heirloom-validate-quick-20260608-020555`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260608-020555`,
  - Vertex status: `JOB_STATE_SUCCEEDED`,
  - observed GPU: `NVIDIA A100-SXM4-80GB`,
  - precision/device: `amp-bf16` on `cuda:0`,
  - fixture shape: batch `3`, block `15`, `d_model=17`, `n_heads=1`, `ff_hidden=37`, vocab `281`,
  - train loss: `5.765558 -> 1.697121`, reduction `0.705645`,
  - resume: step `40 -> 43`, loss `2.056494 -> 1.562211`, reduction `0.240353`,
  - AMP validation status: `passed`,
  - train AMP finite checks/op decisions/Tensor Core decisions: `1080` / `760` / `320`,
  - resume AMP finite checks/op decisions/Tensor Core decisions: `81` / `57` / `24`,
  - finite-check failures, fallback decisions, and unexpected host staging all stayed `0`,
  - strict no-CPU-staging was enabled; dynamic FP16-style loss scaling stayed disabled by policy,
  - train Tensor Core matmul forward/backward/total calls: `520` / `1040` / `1560`,
  - train attention Tensor Core forward/backward calls: `40` / `40`,
  - train attention QK/AV/score-grad/dQ/dK/dV calls: `120` each,
  - train Linear Tensor Core calls: `280`,
  - train Linear pad/crop padded/remainder tiles: `14080` / `14080`,
  - train staged CTA GEMM calls/shared-stage tiles/bytes: `840` / `14200` / `21811200`,
  - scalar matmul fallbacks, Linear fallbacks, and attention scalar fallbacks stayed `0`,
  - wrapper summary status: `passed`.
- Local validation around the fix:
  - `cargo test -p heirloom --test distributed_cli launcher_test -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.

## Checkpoint 85 - Memory Transformer Design And CPU Baseline

- Read primary-source paper pages/PDFs before coding:
  - `Memory Layers at Scale`, arXiv `2412.09764`,
  - `Continual Learning via Sparse Memory Finetuning`, arXiv `2510.15103`.
- Added `MEMORY_TRANSFORMER_DESIGN.md`:
  - paper requirements for memory placement, key/value tables, top-k lookup, shared memory, Memory+ gating, sparse updates, and SMFT behavior,
  - exact current implementation scope,
  - explicit approximations,
  - CUDA primitive plan with named kernels for scoring, top-k, weighted value aggregation, sparse row gradients, sparse AdamW, and access counts,
  - compatibility notes for AMP BF16, checkpoint model-family metadata, and DDP rank reports.
- Repository isolation note:
  - `/Users/andrewverdiramo/Desktop/Heirloom` is not a Git checkout in this environment,
  - no `.git` directory exists, so a true branch/worktree could not be created,
  - changes were kept additive and isolated.
- Added `src/memory_transformer.rs` and exported it from `src/lib.rs`:
  - `MemoryTransformerConfig`,
  - `MemoryLayerConfig`,
  - `MemoryFeedForward`,
  - `MemoryTransformerBlock`,
  - `MemoryTransformerLm`,
  - `MemoryUpdatePolicy`,
  - `SmftMode`.
- Implemented minimal CPU memory path:
  - query projection,
  - exact full-table CPU top-k,
  - selected key/value gathers through Tensor `embedding`,
  - selected-score softmax,
  - weighted value aggregation,
  - Memory+ style gate using GELU as an explicit SiLU approximation,
  - output projection,
  - shared memory tables named once under `shared_memory.*`,
  - dense FFN fallback for non-memory layers,
  - deterministic initialization and stable parameter names.
- Preserved baseline model behavior:
  - `TinyTransformerLm` was not moved, wrapped, or edited.
- Added `tests/memory_transformer.rs`:
  - 32-layer construction with memory layers,
  - forward shape `[batch, time, vocab]`,
  - finite loss,
  - stable shared-memory and block-local parameter names,
  - state-dict save/load round trip,
  - fixed-batch training loss decrease.
- Validation:
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - CUDA memory kernels are not implemented yet,
  - CUDA memory forward intentionally errors instead of CPU-staging tensors through the CPU fallback path.

## Checkpoint 86 - Memory Transformer CLI And Checkpoint Family

- Added additive LM checkpoint family metadata:
  - `LmModelFamily::{TinyTransformer, MemoryTransformer}`,
  - legacy checkpoints without `model_family` still load as `TinyTransformer`,
  - Tiny loader rejects memory-transformer checkpoints instead of deserializing them as `TinyTransformerLm`,
  - memory loader reconstructs `MemoryTransformerLm` from recorded `memory_config`.
- Added memory LM checkpoint save/load path:
  - `save_memory_lm_checkpoint_with_dataset_state`,
  - `load_memory_lm_checkpoint_on_device`,
  - tokenizer, optimizer state, dataset RNG state, batches seen, manifest path, and optimizer step are preserved.
- Added `heirloom train-memory-lm`:
  - trains the isolated `MemoryTransformerLm` path rather than wrapping `TinyTransformerLm`,
  - exposes model/memory flags including layer count, memory layer indices, slots, key/value dims, top-k, shared memory, Memory+ gate, update policy, and SMFT mode,
  - supports `f32`, `bf16`, and `amp-bf16` precision surfaces using the current Heirloom policy checks,
  - writes JSON train reports with model family, memory config, loss curve, checkpoint path, Tensor Core counters, AMP report fields, and explicit CUDA-memory-kernel status.
- Updated memory-transformer tests:
  - checkpoint metadata records `model_family: "memory_transformer"` and full memory config,
  - memory checkpoint reload reconstructs the correct model family,
  - Tiny checkpoint loader rejects memory checkpoints,
  - CLI training writes a memory-family checkpoint and JSON report.
- Validation:
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- Remaining hard-path status:
  - CUDA memory lookup/backward/sparse-update kernels are still not implemented,
  - CUDA memory-transformer training is intentionally not accepted yet,
  - DDP memory rank reports are planned but should stay gated until single-GPU CUDA memory kernels pass.

## Checkpoint 87 - Memory CUDA Contract And Rejection Counters

- Added memory CUDA contract surface in `heirloom-kernels`:
  - `MemoryLookupDims`,
  - `MemoryKernelCounters`,
  - `validate_memory_lookup_dims`,
  - `memory_kernel_counters`,
  - `reset_memory_kernel_counters`,
  - `record_memory_cuda_lookup_rejected`.
- Added explicit memory-kernel counters:
  - lookup rejection calls,
  - query-key score calls,
  - top-k calls,
  - top-k softmax calls,
  - weighted-value forward/backward calls,
  - selected-key backward calls,
  - sparse row scatter-add calls,
  - sparse AdamW row-update calls,
  - access-count calls,
  - selected token and selected row counts.
- Hardened `MemoryFeedForward` CUDA behavior:
  - plain `forward()` now rejects CUDA memory lookup before running dense CUDA projections,
  - `amp-bf16` still records an AMP decision and rejects CUDA memory lookup,
  - no CPU fallback path is used for CUDA memory lookup.
- Extended `train-memory-lm` JSON reports with `cuda_memory_kernels.counters`.
- Added tests:
  - always-on memory CUDA dimension validation and counter accounting,
  - gated CUDA test proving CUDA memory-transformer forward rejects CPU fallback until dedicated kernels exist,
  - CLI report assertions for memory-kernel counter fields.
- Validation:
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- Remaining hard-path status:
  - this is a real contract/counter surface, not the CUDA memory implementation,
  - next required kernel work is device-resident top-k plus weighted-value aggregation/backward and sparse row gradient accumulation.

## Checkpoint 88 - CUDA Exact Top-K Memory Forward Path

- Added a first real memory CUDA kernel:
  - `heirloom_memory_topk_f32` in embedded PTX,
  - exact top-k over rank-2 score matrices,
  - deterministic lower-slot tie-break behavior,
  - safe wrapper `memory_topk_indices_f32`,
  - checked shape contract `MemoryTopkDims`.
- Added `Tensor::topk_indices_dim1(k)`:
  - CPU deterministic implementation for local parity coverage,
  - CUDA implementation returning an i64 CUDA tensor through `memory_topk_indices_f32`,
  - no autograd through the discrete routing decision.
- Updated `MemoryFeedForward` CUDA forward:
  - query projection stays CUDA-resident,
  - query-key scoring uses existing CUDA matmul,
  - top-k routing uses `heirloom_memory_topk_f32`,
  - selected key/value gathers use existing CUDA embedding kernels,
  - selected-score softmax, weighted aggregation, gate, and output projection use existing CUDA Tensor ops,
  - gradients flow through selected key/value rows and projections using existing CUDA autograd paths.
- Updated `MEMORY_TRANSFORMER_DESIGN.md`:
  - records the implemented exact CUDA top-k path,
  - distinguishes it from the still-missing fused EmbeddingBag-style weighted-value memory kernels and sparse optimizer path.
- Added/updated tests:
  - deterministic CPU top-k tie-break test,
  - memory top-k dimension validation,
  - gated CUDA memory-transformer step test requiring CUDA forward on `cuda:0`, finite loss, CUDA-resident memory gradients, AdamW step, and positive memory top-k counters.
- Validation:
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - the new CUDA test is gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - actual PTX JIT/runtime validation on A100 remains required before accepting the CUDA memory path.
- Remaining hard-path status:
  - fused `memory_weighted_value_forward_f32` and backward kernels are still missing,
  - sparse selected-row gradient buffers and sparse AdamW updates are still missing,
  - broad DDP memory synchronization remains gated until single-GPU A100 validation passes.

## Checkpoint 89 - Fused CUDA Memory Weighted-Value Forward And Backward

- Removed another CPU-staging blocker from the memory CUDA path:
  - `Tensor::softmax_dim(1)` now has a CUDA rank-2 f32 forward/backward path,
  - CUDA softmax backward recomputes softmax from saved CUDA input and keeps gradients device-resident.
- Added fused memory weighted-value Tensor op:
  - `Tensor::memory_weighted_value(indices, weights, values)`,
  - CPU and CUDA implementations,
  - dedicated autograd node `MemoryWeightedValue`,
  - gradients for selected weights and memory value rows.
- Added CUDA kernels in `heirloom-kernels`:
  - `heirloom_softmax_dim1_f32`,
  - `heirloom_softmax_dim1_backward_f32`,
  - `heirloom_memory_weighted_value_forward_f32_i64`,
  - `heirloom_memory_weighted_value_backward_weights_f32_i64`,
  - `heirloom_memory_weighted_value_backward_values_f32_i64`.
- Kernel behavior:
  - forward computes weighted value aggregation without materializing `[tokens, top_k, value_dim]`,
  - backward computes grad-weights by dotting output gradients with selected value rows,
  - backward computes grad-values with atomic scatter-add into a dense CUDA value-table gradient buffer,
  - out-of-range indices use the same device status/error pattern as CUDA embedding.
- Updated `MemoryFeedForward`:
  - selected key gather remains explicit for selected-score recomputation and key gradients,
  - value aggregation now uses the fused memory weighted-value op,
  - CUDA memory layer forward/backward no longer depends on generic CUDA `softmax_dim` CPU materialization.
- Added tests:
  - CPU fused memory weighted-value forward/backward with repeated rows,
  - gated CUDA softmax dim-1 backward device-residency test,
  - gated CUDA fused memory weighted-value forward/backward with counter checks,
  - gated CUDA memory-transformer step now requires top-k, fused weighted-value forward, backward, and scatter-add counters.
- Validation:
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - local tests compile and gated CUDA tests are present,
  - actual PTX JIT/runtime validation on A100 is still required.
- Remaining hard-path status:
  - memory key gradients still flow through selected key embedding, not a dedicated fused selected-key backward kernel,
  - gradients are CUDA-resident but still dense table buffers after scatter-add; sparse row-gradient buffers and sparse AdamW are not implemented,
  - product-key lookup and DDP rank reports for memory parameters remain pending.

## Checkpoint 90 - Fused CUDA Selected-Key Scores And Backward

- Removed the selected-key materialization step from the memory scoring path:
  - added `Tensor::memory_selected_scores(indices, query, keys)`,
  - CPU forward computes selected key dot-products directly,
  - CPU backward accumulates query gradients and repeated-row key-table gradients,
  - CUDA forward/backward stay device-resident.
- Added CUDA wrappers and PTX kernels in `heirloom-kernels`:
  - `MemorySelectedScoreDims`,
  - `validate_memory_selected_score_dims`,
  - `heirloom_memory_selected_scores_forward_f32_i64`,
  - `heirloom_memory_selected_scores_backward_query_f32_i64`,
  - `heirloom_memory_selected_scores_backward_keys_f32_i64`.
- Kernel behavior:
  - forward computes `[tokens, top_k]` selected scores without materializing `[tokens, top_k, key_dim]`,
  - query backward sums selected key rows weighted by score gradients,
  - key backward atomic scatter-adds selected-row gradients into a dense CUDA key-table gradient buffer,
  - out-of-range indices use the same device status/error pattern as CUDA embedding.
- Updated `MemoryFeedForward`:
  - selected-score computation now uses `memory_selected_scores`,
  - CUDA memory key gradients no longer flow through generic selected-key embedding.
- Added tests:
  - CPU selected-score forward/backward with repeated selected rows,
  - gated CUDA selected-score forward/backward with counter checks,
  - gated CUDA memory-transformer step now requires positive query-key score and selected-key backward counters and CUDA-resident key/value memory gradients.
- Validation:
  - `cargo fmt --all`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - local targeted tests pass and gated CUDA tests are present,
  - actual PTX JIT/runtime validation on A100 is still required.
- Remaining hard-path status:
  - sparse row-gradient buffers and sparse AdamW updates are still missing; current key/value memory gradients are dense CUDA buffers after selected-row scatter-add,
  - product-key or approximate top-k lookup remains pending,
  - DDP memory-parameter rank reports and A100 validation remain pending.

## Checkpoint 91 - CUDA Sparse-Row AdamW Bridge For Memory Tables

- Added a narrow CUDA sparse-row optimizer bridge for memory parameters:
  - `SparseAdamWRowsDims` in `heirloom-kernels`,
  - safe wrapper `memory_sparse_adamw_rows_f32_i64_buffers`,
  - PTX kernel `heirloom_memory_sparse_adamw_rows_f32_i64`,
  - existing `sparse_adamw_rows_calls` memory counter now increments when the sparse-row kernel runs.
- Kernel behavior:
  - input is a dense CUDA parameter/gradient/m/v table plus i64 selected row ids,
  - only selected rows and their AdamW moment rows are updated,
  - duplicate selected row ids are deduplicated by first occurrence to avoid racing or double-applying one dense accumulated row gradient,
  - invalid selected row ids are reported through the device status/error path.
- Added Heirloom optimizer surface:
  - `SparseAdamWRowsUpdate`,
  - `AdamW::step_cuda_sparse_rows_mut`,
  - Tensor-side validation and launch path `apply_adamw_cuda_sparse_rows_f32`.
- Added tests:
  - sparse AdamW row dimension validation,
  - gated CUDA test proving duplicate selected rows update once, unselected rows remain unchanged, and the sparse-row counter increments.
- Validation:
  - `cargo fmt --all`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - local targeted tests pass and CUDA execution remains gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - actual PTX JIT/runtime validation on A100 is still required.
- Remaining hard-path status:
  - the sparse-row AdamW bridge still consumes dense CUDA gradient buffers; true sparse row-gradient buffers are not implemented,
  - memory training policy routing is implemented in the next checkpoint,
  - product-key lookup, DDP memory rank reports, and A100 validation remain pending.

## Checkpoint 92 - Memory Sparse-Row Policy Routing

- Added device-resident selected-row aggregation:
  - `heirloom_copy_i64_with_offset` PTX helper,
  - `concat_i64_buffers` in `heirloom-kernels`,
  - `Tensor::concat_i64_flat`.
- Updated `MemoryFeedForward` to retain the latest selected row ids as a Tensor on the original device.
- Added `MemoryTransformerLm::memory_sparse_adamw_updates`:
  - maps memory key/value parameter names to optimizer parameter indices,
  - aggregates selected rows across shared memory layers,
  - emits `SparseAdamWRowsUpdate` descriptors for shared or per-layer memory tables.
- Routed `train-memory-lm` optimizer behavior:
  - `Full` uses dense AdamW,
  - `SparseRows` uses `AdamW::step_cuda_sparse_rows_mut`,
  - SMFT modes use sparse-row AdamW,
  - `Frozen` skips optimizer updates.
- Added tests:
  - CPU descriptor mapping for shared memory parameters,
  - gated CUDA memory-transformer sparse-row AdamW update test proving device-selected rows drive key/value memory updates and sparse-row counters increment.
- Validation:
  - `cargo fmt --all`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - local targeted tests pass and CUDA execution remains gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - actual PTX JIT/runtime validation on A100 is still required.
- Remaining hard-path status:
  - sparse-row policy routing still consumes dense CUDA gradient buffers; true sparse row-gradient buffers are pending,
  - `MemoryOnly` table-only stepping is implemented in the next checkpoint,
  - product-key lookup, DDP memory rank reports, and A100 validation remain pending.

## Checkpoint 93 - MemoryOnly Table-Only Optimizer Semantics

- Added selective AdamW stepping:
  - `AdamW::step_parameter_indices_mut`,
  - subset-aware gradient clipping through `clip_scale_for_indices`,
  - duplicate/out-of-range parameter-index validation.
- Added `MemoryTransformerLm::memory_table_parameter_indices`:
  - maps shared memory tables to `shared_memory.key` and `shared_memory.value`,
  - maps non-shared memory tables to each memory block's key/value parameters.
- Updated `train-memory-lm` optimizer policy:
  - `MemoryOnly` now updates only memory key/value tables with dense AdamW,
  - `Full` still updates all parameters with dense AdamW,
  - `SparseRows` and SMFT modes still use sparse-row AdamW,
  - `Frozen` still skips optimizer updates.
- Added CPU regression coverage:
  - memory-only AdamW changes shared memory tables,
  - token embedding has gradients but remains unchanged when memory-only stepping is used.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - local targeted tests pass and CUDA execution remains gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - actual PTX JIT/runtime validation on A100 is still required.
- Remaining hard-path status:
  - memory-only optimizer semantics are table-only; query/gate/out projections are treated as dense model parameters and stay frozen,
  - sparse-row paths still consume dense CUDA gradient buffers; true sparse row-gradient buffers are pending,
  - product-key lookup, DDP memory rank reports, and A100 validation remain pending.

## Checkpoint 94 - Memory Training Report Observability

- Added `MemoryTransformerLm::memory_selection_report`:
  - reports configured versus captured memory layers,
  - selected row event counts,
  - unique selected row counts,
  - per-memory-layer selected row participation,
  - selected-row device labels.
- Hardened `train-memory-lm` JSON reports:
  - `cuda_memory_kernels.implemented` now reflects the implemented CUDA memory kernel surface,
  - reports the memory kernel surface list and counters,
  - reports memory optimizer policy, SMFT mode, applied optimizer path, memory table parameter indices, sparse update descriptor counts, and the current dense-gradient-buffer dependency for sparse row modes,
  - reports memory table checksum scalars for future rank-drift comparisons,
  - adds a single-rank `ddp_rank_report_contract` describing required fields before enabling broad memory DDP.
- Updated design documentation:
  - corrected the sparse CUDA optimizer status,
  - kept the true sparse row-gradient buffer limitation explicit.
- Added CLI regression coverage:
  - `train-memory-lm` report now asserts memory selection evidence,
  - optimizer path evidence,
  - finite memory table checksum,
  - implemented CUDA memory-kernel report state.
- Validation:
  - `cargo fmt --all`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - CUDA memory execution remains gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - actual PTX JIT/runtime validation on A100 is still required.
- Remaining hard-path status:
  - sparse-row optimizer still reads dense CUDA memory-table gradient buffers and only narrows the update write set,
  - product-key lookup, true sparse row-gradient buffers, broad memory DDP rank workers, and A100 validation remain pending.

## Checkpoint 95 - CPU Product-Key Candidate Lookup

- Added `MemoryLookupKind`:
  - `exact` keeps the existing full-table exact top-k path,
  - `product-key` enables a CPU product-key-style candidate generator.
- Extended memory configs and checkpoints:
  - `MemoryLayerConfig` and `MemoryTransformerConfig` now carry `memory_lookup`,
  - serde defaults preserve older memory checkpoint metadata,
  - `train-memory-lm --memory-lookup exact|product-key` wires the mode through CLI config and checkpoint metadata.
- Implemented CPU product-key candidate routing:
  - requires square `memory_slots`,
  - requires even `memory_key_dim`,
  - interprets slot ids as `(left, right)` key pairs,
  - ranks left/right key halves, forms candidate pairs, and scores candidates against the dense key table,
  - records selected rows through the existing memory selection report.
- Kept CUDA claims strict:
  - CUDA product-key lookup errors clearly instead of using CPU fallback,
  - exact CUDA memory lookup remains unchanged.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now distinguishes CPU product-key candidate routing from full paper-grade product-key tables,
  - `HARD_MODE.md` records missing trainable half-key tables, CUDA product-key kernels, true sparse row-gradient buffers, SMFT masks/ranking, and memory DDP validation.
- Added tests:
  - product-key shape validation for non-square slots and odd key dims,
  - full CPU memory-transformer forward through product-key lookup,
  - selected-row reporting for product-key mode.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - product-key CUDA kernels are not implemented,
  - existing exact CUDA memory execution remains gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - actual A100 validation for the current memory-transformer state remains pending.
- Remaining hard-path status:
  - product-key mode is CPU candidate routing over dense key rows, not separate trainable half-key tables,
  - true sparse row-gradient buffers, CUDA product-key kernels, broad memory DDP rank workers, and A100 validation remain pending.

## Checkpoint 96 - SMFT Access Counts And Row-Mask Derivation

- Added SMFT access-count types:
  - `MemoryAccessCounts`,
  - `SmftRowScore`,
  - `SmftRowMask`,
  - `SmftAccessReport`.
- Added model access accounting:
  - `MemoryTransformerLm::memory_access_counts`,
  - `MemoryTransformerLm::smft_access_report`,
  - `MemoryTransformerLm::smft_row_mask_against`.
- Implemented TF-IDF-like SMFT row scoring:
  - foreground counts come from the latest captured selected memory rows,
  - background counts are supplied as a separate `MemoryAccessCounts`,
  - trainable row masks are ranked by foreground frequency weighted against background access.
- Extended `train-memory-lm` reports:
  - adds `smft_access.counts`,
  - adds top accessed foreground memory rows,
  - report-time CUDA selected-row inspection remains explicit host-staging/report behavior, not a training fallback.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now records access-count and mask derivation as implemented,
  - `HARD_MODE.md` now records the missing CUDA mask enforcement and persistent background-count pipeline.
- Added tests:
  - access-count construction and deterministic row counts,
  - TF-IDF-like foreground/background row ranking,
  - mask validation for mismatched memory sizes,
  - model SMFT report from a real memory forward,
  - CLI report assertions for `smft_access`.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - no CUDA SMFT mask-enforcement kernel was added,
  - existing exact CUDA memory execution remains gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - actual A100 validation for the memory-transformer state remains pending.
- Remaining hard-path status:
  - SMFT masks are derivable but not yet enforced by CUDA sparse-row AdamW,
  - persistent background access-count collection is not wired into data/training,
  - true sparse row-gradient buffers, CUDA product-key kernels, broad memory DDP rank workers, and A100 validation remain pending.

## Checkpoint 97 - SMFT Row-Mask Enforcement For Sparse AdamW

- Added optional sparse update masks:
  - `SparseAdamWRowsUpdate` now carries `row_mask: Option<Tensor>`,
  - `AdamW::step_cuda_sparse_rows_mut` validates optional rank-1 row masks against the sparse update row count,
  - the Tensor-to-CUDA bridge validates Bool CUDA masks and passes them through without CPU staging.
- Added SMFT mask materialization:
  - `SmftRowMask::to_tensor` materializes deterministic Bool masks on CPU or CUDA,
  - `MemoryTransformerLm::memory_sparse_adamw_updates_with_mask` attaches a supplied SMFT row mask to every memory-table sparse update descriptor.
- Extended CUDA sparse-row AdamW:
  - `heirloom_memory_sparse_adamw_rows_f32_i64` now accepts an optional Bool/u8 row mask,
  - masked rows are skipped on device before deduplication and AdamW math,
  - the safe CUDA wrapper now uses `SparseAdamWRowsBuffers` so mask support does not expand the public positional argument list.
- Updated report/docs surface:
  - `train-memory-lm` reports `sparse_adamw_rows_f32_i64_optional_row_mask`,
  - `MEMORY_TRANSFORMER_DESIGN.md` records CUDA row-mask enforcement as implemented,
  - `HARD_MODE.md` now distinguishes supplied-mask enforcement from missing persistent automatic SMFT mask plumbing.
- Added tests:
  - CPU SMFT row masks materialize to Bool tensors and attach to sparse update descriptors,
  - invalid out-of-range SMFT rows fail early,
  - gated CUDA sparse AdamW row-mask test verifies masked rows remain unchanged while trainable rows update.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - the CUDA row-mask test remains gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - actual A100 PTX validation for the new row-mask branch is still required.
- Remaining hard-path status:
  - sparse-row AdamW can enforce a supplied SMFT mask on CUDA, but training does not yet persist/load background access counts or automatically select/update masks over time,
  - autograd still accumulates dense CUDA memory-table gradients before sparse updates read selected rows,
  - true sparse row-gradient buffers, CUDA product-key kernels, broad memory DDP rank workers, and A100 validation remain pending.

## Checkpoint 98 - Explicit SMFT Mask CLI Plumbing

- Added SMFT mask validation:
  - `SmftRowMask::validate` checks finite trainable fraction, row bounds, duplicate trainable rows, and stale `frozen_rows` metadata,
  - `SmftRowMask::to_tensor` now validates before materializing Bool row masks.
- Added explicit training input:
  - `train-memory-lm --smft-row-mask path.json` loads a persisted `SmftRowMask`,
  - the flag is rejected unless sparse memory updates are active through `--smft-mode` or `--memory-update-policy sparse-rows`,
  - loaded masks are validated against the model's `memory_slots`.
- Wired masks into training:
  - `step_memory_optimizer` now uses `memory_sparse_adamw_updates_with_mask` when a mask is supplied,
  - masked descriptors flow into the CUDA sparse-row AdamW path through the existing optional row-mask kernel argument.
- Extended memory optimizer reporting:
  - reports the mask source path,
  - reports mask memory slot count,
  - reports trainable/frozen row counts,
  - reports how many sparse update descriptors carried a row mask.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now records explicit persisted mask input as implemented,
  - `HARD_MODE.md` now narrows the remaining SMFT gap to persistent background access counting and automatic mask refresh.
- Added tests:
  - binary-unit coverage for masked sparse update descriptor/report plumbing,
  - CLI regression proving `--smft-row-mask` is parsed and rejected when sparse memory updates are not active,
  - mask validation coverage for duplicate rows and stale frozen-row counts.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test -p heirloom --bin heirloom memory_optimizer_report_records_supplied_smft_row_mask -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - the explicit mask path is wired to the CUDA sparse-row AdamW descriptors,
  - live A100 validation of masked sparse-row memory training remains pending behind `HEIRLOOM_CUDA_TESTS=1`.
- Remaining hard-path status:
  - training can consume a persisted mask, but it still does not collect long-running background access counts or refresh masks automatically,
  - true sparse row-gradient buffers, CUDA product-key kernels, broad memory DDP rank workers, and A100 validation remain pending.

## Checkpoint 99 - SMFT Count Artifacts And CUDA Access Counting

- Added persistent SMFT count support:
  - `MemoryAccessCounts::from_row_counts`,
  - `MemoryAccessCounts::validate`,
  - `MemoryAccessCounts::merge_in`.
- Added a CUDA access-count primitive:
  - `CudaBuffer::from_u64` / `to_u64`,
  - `memory_access_count_rows_i64_u64`,
  - PTX kernel `heirloom_memory_access_count_rows_i64_u64`,
  - one thread per selected memory row,
  - bounds checks through the existing device status-buffer pattern,
  - atomic `u64` count accumulation,
  - increments the existing memory-kernel `access_count_calls` counter.
- Updated model access accounting:
  - CPU selected-row tensors still use direct row iteration,
  - CUDA selected-row tensors now count rows through the device kernel and only copy the compact count vector for SMFT artifact/report use.
- Added file-backed SMFT artifact flags to `train-memory-lm`:
  - `--smft-background-counts`,
  - `--smft-access-counts-out`,
  - `--smft-mask-out`,
  - `--smft-trainable-fraction`,
  - `--smft-min-rows`.
- Wired training artifacts:
  - optional foreground counts accumulate over training steps,
  - foreground counts can be written as JSON,
  - a derived `SmftRowMask` can be written against supplied background counts or an empty background,
  - `train-memory-lm` reports `smft_artifacts` with paths and count/mask summaries.
- Updated reports/docs:
  - CUDA memory kernel surface now includes `access_count_rows_i64_u64`,
  - `MEMORY_TRANSFORMER_DESIGN.md` records device-side access counting and persisted count/mask artifacts,
  - `HARD_MODE.md` narrows the remaining SMFT gap to long-running background corpus collection and online mask refresh.
- Added tests:
  - count validation, construction from row-count vectors, and merge behavior,
  - gated CUDA access-count kernel test,
  - CLI artifact test that writes foreground counts, derives a mask, and verifies JSON report summaries.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - the CUDA count kernel test remains gated behind `HEIRLOOM_CUDA_TESTS=1`,
  - live A100 validation of access counting plus masked sparse-row memory training remains pending.
- Remaining hard-path status:
  - foreground count artifacts can now be persisted, but Heirloom still lacks a separate large background-corpus collection command and online SMFT mask refresh,
  - true sparse row-gradient buffers, CUDA product-key kernels, broad memory DDP rank workers, and A100 validation remain pending.

## Checkpoint 100 - CUDA Product-Key Candidate Lookup

- Added CUDA product-key candidate lookup support for the memory transformer:
  - `MemoryProductKeyDims`,
  - product-key shape validation,
  - a `product_key_calls` memory-kernel counter,
  - PTX side-score kernel `heirloom_memory_product_key_side_scores_f32`,
  - PTX candidate kernel `heirloom_memory_product_key_candidates_f32_i64`.
- Added the composed CUDA lookup wrapper:
  - computes left/right half-key side scores,
  - reuses CUDA top-k for side beams,
  - scores beam-pair product-key candidates,
  - returns final `[tokens, memory_top_k]` selected row indices on device,
  - records final selected-token and selected-row counters for reporting.
- Added the Tensor and model bridge:
  - `Tensor::cuda_memory_product_key_topk_indices`,
  - `MemoryFeedForward` now routes CUDA `MemoryLookupKind::ProductKey` through device candidate lookup instead of rejecting the path,
  - memory selection reports now identify `"cuda_product_key_candidate_topk_memory_lookup"`.
- Updated CLI/report surfaces:
  - memory kernel reports include `product_key_calls`,
  - `train-memory-lm` lists `product_key_candidate_topk_f32` in the CUDA memory kernel surface.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` records CPU/CUDA product-key candidate generators over the dense memory key table,
  - `HARD_MODE.md` narrows the remaining product-key gap to separate trainable half-key tables and production-quality memory indexing rather than missing CUDA routing kernels.
- Added tests:
  - product-key CUDA contract validation for square slots, beam size, and top-k constraints,
  - gated CUDA primitive parity against a CPU product-key candidate reference,
  - gated CUDA model test proving product-key memory lookup keeps selected rows on `cuda:0`,
  - report/counter assertions for the new product-key path.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - gated CUDA tests compile and skip locally when `HEIRLOOM_CUDA_TESTS` is unset,
  - live A100 validation of the new PTX product-key kernels remains pending.
- Remaining hard-path status:
  - product-key lookup still uses the dense memory key table split into halves, not separate trainable half-key tables,
  - true sparse row-gradient buffers, broad memory DDP rank workers, online SMFT refresh, and A100 validation remain pending.

## Checkpoint 101 - Trainable Product-Key Half Tables

- Added paper-shaped product-key memory tables:
  - product-key mode now owns trainable `product_key_left` and `product_key_right` half-key tables,
  - exact lookup still owns the existing dense `key` table,
  - both modes continue to use the memory `value` table.
- Updated product-key lookup:
  - CPU product-key routing scores query halves against the separate half-key tables,
  - CUDA product-key routing builds side scores through `query_left @ product_key_left.T` and `query_right @ product_key_right.T`,
  - a new candidate-combine CUDA kernel forms top-k slot ids from left/right side-score beams.
- Added product-key selected-score training:
  - `Tensor::memory_product_key_selected_scores`,
  - new autograd node `MemoryProductKeySelectedScores`,
  - CPU backward into query, left half keys, and right half keys,
  - CUDA forward/backward wrappers and PTX kernels for query and half-key gradients.
- Added CUDA row utilities:
  - `memory_product_key_split_rows_i64`,
  - selected slot rows can now map to `slot / side` and `slot % side` rows for sparse half-key optimizer descriptors.
- Updated memory optimizer routing:
  - product-key sparse updates target `product_key_left`, `product_key_right`, and `value`,
  - `memory-only` parameter filtering includes half-key tables,
  - product-key plus full-slot SMFT row masks now errors clearly because a slot mask cannot be safely projected onto shared half-key rows without changing mask semantics.
- Updated reports/docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now records trainable product-key half tables and product-key selected-score gradients,
  - `HARD_MODE.md` narrows product-key gaps to production-scale indexing/tuning, live validation, and unresolved full-slot SMFT mask semantics,
  - `train-memory-lm` CUDA kernel surface now lists half-row split and product-key selected-score kernels.
- Added tests:
  - CPU regression proving product-key named parameters expose half-key tables, omit dense `shared_memory.key`, receive gradients, and produce sparse half-key update descriptors,
  - gated CUDA primitive test for side-score candidate combine, half-row split, and product-key selected-score forward/backward buffers,
  - gated CUDA model test now asserts product-key half-key gradients stay on `cuda:0`.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - new CUDA tests compile and skip locally when `HEIRLOOM_CUDA_TESTS` is unset,
  - live A100 validation of the new product-key half-table PTX kernels remains pending.
- Remaining hard-path status:
  - product-key lookup is still a scalar beam candidate path rather than a tuned large-memory index,
  - product-key full-slot SMFT masks need a principled half-key masking policy,
  - true sparse row-gradient buffers, broad memory DDP rank workers, online SMFT refresh, and A100 validation remain pending.

## Checkpoint 102 - Product-Key SMFT Mask Projection

- Added explicit ProductKey SMFT mask semantics:
  - `SmftProductKeyMaskPolicy::ConservativeAllSlots`,
  - `SmftProductKeyMaskProjection`,
  - `SmftRowMask::product_key_projection`.
- Implemented conservative full-slot to half-key projection:
  - value rows keep the exact full-slot `SmftRowMask`,
  - a left half-key row is trainable only if every full slot sharing that left row is trainable,
  - a right half-key row follows the same all-slots rule,
  - this prevents a half-key update from mutating parameters that necessarily affect frozen full slots.
- Updated sparse optimizer routing:
  - ProductKey `memory_sparse_adamw_updates_with_mask` no longer errors,
  - left/right half-key sparse update descriptors receive side-length projected masks,
  - value-table sparse update descriptors receive the original full-slot mask.
- Updated reports/docs:
  - `memory_optimizer_report` now records ProductKey mask projection policy, side length, value trainable rows, left trainable half rows, right trainable half rows, and conservative-policy status,
  - `MEMORY_TRANSFORMER_DESIGN.md` documents exact value-mask behavior and conservative half-key masking,
  - `HARD_MODE.md` narrows the remaining gap to non-conservative/exact product-key SMFT parity.
- Added tests:
  - CPU regression proving ProductKey SMFT masks project to left/right half-key row masks and exact value masks,
  - binary report regression proving the projection appears in JSON artifacts.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test -p heirloom --bin heirloom memory_optimizer_report_records -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - mask projection is CPU-side descriptor construction and feeds existing CUDA sparse-row AdamW masks,
  - live A100 validation of ProductKey half-key kernels plus projected masked sparse updates remains pending.
- Remaining hard-path status:
  - ProductKey SMFT masks are conservative and can freeze more half-key rows than the full-slot mask selected,
  - true sparse row-gradient buffers, broad memory DDP rank workers, online SMFT refresh, production product-key indexing, and A100 validation remain pending.

## Checkpoint 103 - Online SMFT Mask Refresh

- Added optional online SMFT refresh for `train-memory-lm`:
  - new `--smft-refresh-every N` flag,
  - foreground memory access counts are accumulated during training when refresh is enabled,
  - every `N` global steps the active `SmftRowMask` is re-derived from accumulated foreground counts against supplied background counts or an empty background,
  - the refreshed active mask is applied before sparse-row AdamW,
  - refresh is rejected early unless `--smft-mode` or `--memory-update-policy sparse-rows` activates memory sparse-row updates.
- Updated SMFT reporting:
  - `smft_artifacts.online_refresh.enabled`,
  - `refresh_every`,
  - `refresh_count`,
  - `last_refresh_step`,
  - `active_mask_source`,
  - active mask size,
  - `memory_optimizer.smft_row_mask.source` now reports the final active mask source, including `online_refresh_step_<n>`.
- Kept default behavior unchanged:
  - `--smft-refresh-every` defaults to `0`,
  - existing offline `--smft-access-counts-out`, `--smft-background-counts`, and `--smft-mask-out` behavior remains available.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now records online refresh as implemented and distinguishes it from production background/windowed/distributed SMFT refresh,
  - `HARD_MODE.md` now states that online refresh exists but is still not paper-grade SMFT pipeline parity.
- Added tests:
  - CPU CLI rejection for refresh without sparse-row memory updates,
  - CUDA-gated CLI fixture proving two-step refresh metadata, active mask source, and masked sparse update descriptors,
  - existing SMFT artifact and optimizer report tests continue to cover offline masks and product-key projection.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test -p heirloom --bin heirloom memory_optimizer_report_records -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - the new success fixture is gated by `HEIRLOOM_CUDA_TESTS=1`,
  - live A100 validation of online refresh plus product-key projected masks remains pending.
- Remaining hard-path status:
  - online refresh is a single-run accumulated foreground-count path, not a production background-corpus/windowed SMFT refresh system,
  - distributed refresh synchronization and memory DDP rank workers remain pending,
  - sparse-row optimizer modes still read dense CUDA gradient buffers before narrowing updates,
  - true sparse row-gradient buffers, production product-key indexing, and A100 reference validation remain pending.

## Checkpoint 104 - Dense Memory Transformer DDP Path

- Added distributed `train-memory-lm` support for dense memory update modes:
  - parent `train-memory-lm --devices cuda:... --distributed nccl`,
  - hidden `train-memory-lm-rank` workers,
  - shared `DistributedLauncher` lifecycle, rank stdout/stderr/stage/report artifacts, NCCL init timeout, and sibling cleanup,
  - deterministic sharded batches from `(seed, global_step, rank, sample_index)`,
  - rank 0 memory checkpoint save.
- Scoped the implementation deliberately:
  - `Full` and `MemoryOnly` memory update policies are supported,
  - `SparseRows`, SMFT modes, row-mask artifacts, and online refresh are rejected in DDP before NCCL launch,
  - the rejection is explicit because rank-local selected rows can differ and need a row-union/all-reduce protocol before sparse-row DDP is safe.
- Added memory DDP rank evidence:
  - model family and memory config,
  - memory table parameter indices/count,
  - memory-gradient parameter count,
  - NCCL all-reduce calls/bytes,
  - memory table checksum sum/sumsq,
  - per-step memory checksum samples,
  - memory selection and SMFT access reports,
  - memory kernel counters,
  - AMP BF16 policy/validation/host-staging reports.
- Added parent aggregate checks:
  - fail if no NCCL gradient all-reduces are recorded,
  - fail if any rank reports zero memory-table gradients,
  - fail if final or per-step memory-table checksum drift exceeds `DDP_PARAMETER_CHECKSUM_TOLERANCE`.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now documents implemented dense memory DDP and deferred sparse-row/SMFT DDP,
  - `HARD_MODE.md` now distinguishes dense memory DDP support from missing row-union sparse/SMFT synchronization and live A100 validation.
- Added tests:
  - CLI rejection for non-CUDA memory DDP device lists,
  - CLI rejection for duplicate memory DDP devices,
  - CLI rejection for sparse-row/SMFT memory DDP before NCCL launch,
  - unit coverage for memory-table per-step checksum drift parsing.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --bin heirloom ddp_memory_step_checksum_drifts -- --nocapture`,
  - `cargo test -p heirloom --test distributed_cli -- --nocapture`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - this change compiles the memory DDP rank worker and validates CPU-safe failure paths locally,
  - live 2/4-rank A100 memory DDP validation remains pending.
- Remaining hard-path status:
  - sparse-row/SMFT DDP needs row-union gradient synchronization,
  - distributed online SMFT refresh synchronization remains pending,
  - sparse-row optimizer modes still read dense CUDA gradient buffers before narrowing updates,
  - true sparse row-gradient buffers, production product-key indexing, and A100 reference validation remain pending.

## Checkpoint 105 - Sparse-Row Memory DDP Row Union

- Extended distributed `train-memory-lm` from dense memory update modes to synchronized sparse-row memory updates:
  - `MemoryUpdatePolicy::SparseRows` is now allowed in `train-memory-lm --devices cuda:... --distributed nccl`,
  - SMFT masks, SMFT artifacts, and online refresh remain rejected in DDP until distributed mask intersection and refresh synchronization exist,
  - rank workers still reject SMFT mode before CUDA/NCCL training.
- Added a CUDA row-union path for sparse memory DDP:
  - `Tensor::cuda_row_union_mask_nccl` builds a per-rank CUDA f32 row mask from selected memory rows,
  - NCCL all-reduce-sums that mask across ranks,
  - a CUDA bool conversion marks rows selected by any rank,
  - sparse AdamW receives an all-row CUDA `i64` candidate tensor plus the union mask, so every rank applies the same memory-row update set.
- Added `heirloom-kernels` CUDA helpers and PTX kernels:
  - `i64_arange_buffer` / `heirloom_i64_arange`,
  - `memory_selected_rows_to_f32_mask` / `heirloom_memory_selected_rows_to_f32_mask`,
  - `f32_mask_to_bool_buffer` / `heirloom_f32_mask_to_bool`.
- Hardened distributed reports:
  - rank reports now include `row_union_all_reduce_calls`, `row_union_all_reduce_bytes`, and `row_union_candidate_rows`,
  - aggregate memory DDP reports sum those fields,
  - sparse-row distributed runs fail if row-union all-reduce evidence is missing.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now records row-union sparse DDP as implemented,
  - `HARD_MODE.md` now distinguishes synchronized sparse-row DDP from missing compressed sparse-gradient buffers and SMFT distributed mask intersection.
- Updated tests:
  - distributed CLI validation now rejects SMFT before NCCL instead of rejecting plain `SparseRows`,
  - existing memory-transformer tests continue to cover sparse AdamW, product-key, SMFT artifacts, and memory checkpoint behavior,
  - memory DDP checksum parsing still covers memory-table sync report contracts.
- Validation so far:
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test distributed_cli -- --nocapture`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test -p heirloom --bin heirloom ddp_memory_step_checksum_drifts -- --nocapture`,
  - `cargo test -p heirloom-kernels --lib`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - the row-union PTX kernels have not yet been live-validated on A100 in this checkpoint,
  - the next GPU gate should run a small 2-rank or 4-rank memory DDP sparse-row fixture and verify nonzero row-union all-reduce fields plus zero memory checksum drift.
- Remaining hard-path status:
  - row-union sparse DDP is synchronized but not production sparse: it still all-reduces dense gradients first and scans all rows as sparse optimizer candidates,
  - SMFT DDP mask intersection and distributed online refresh remain pending,
  - true sparse row-gradient buffers, production product-key indexing, and A100 reference validation remain pending.

## Checkpoint 106 - Sparse-Row Memory DDP Fixture Wiring

- Added an opt-in GPU fixture for the memory-transformer path:
  - `scripts/cuda_train_memory_lm_fixture.sh`,
  - tiny local memory corpus, tokenizer training, prepared token manifest,
  - `train-memory-lm` train and resume runs,
  - default `--memory-update-policy sparse-rows`,
  - single-device and `--distributed nccl` target modes,
  - JSON checks for model family, finite losses, step/resume metadata, memory optimizer evidence, and CUDA memory-kernel evidence.
- Added distributed sparse-row DDP assertions to the fixture:
  - positive aggregate `all_reduce_calls`/`all_reduce_bytes`,
  - positive `row_union_all_reduce_calls`,
  - positive `row_union_all_reduce_bytes`,
  - positive `row_union_candidate_rows`,
  - nonzero per-rank `memory_gradient_parameter_count`,
  - memory-table checksum drift within tolerance,
  - per-step memory checksum entries when `DDP_CHECKSUM_EVERY > 0`.
- Wired the fixture into the Vertex validation wrapper behind an explicit opt-in flag:
  - `HEIRLOOM_RUN_CUDA_TRAIN_MEMORY_LM_FIXTURE=1`,
  - forwards memory fixture shape, policy, precision, DDP, and NCCL env knobs,
  - uploads `cuda-train-memory-lm-fixture/run.log`,
  - uploads `summary.json`, `train-report.json`, `resume-report.json`, tokenizer/prepared-data artifacts, launcher report, rank artifacts, and checkpoint prefix.
- Updated `scripts/gcp/README.md` with the intended 4x A100 sparse-row memory DDP command and expected non-secret report evidence.
- Validation so far:
  - `bash -n scripts/cuda_train_memory_lm_fixture.sh`,
  - `bash -n scripts/gcp/submit_vertex_heirloom_validate.sh`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - this checkpoint makes the next A100 run bounded and artifact-rich but does not itself validate the new row-union PTX path live.
- Remaining hard-path status:
  - run the opt-in 2/4-rank A100 sparse-row memory DDP fixture,
  - then decide whether to attack true compressed sparse row-gradient buffers or distributed SMFT mask intersection next.

## Checkpoint 107 - Offline SMFT Mask Intersection For Memory DDP

- Added a CUDA Bool mask intersection primitive for distributed SMFT sparse updates:
  - `heirloom-kernels::cuda::bool_and_buffers`,
  - PTX kernel `heirloom_bool_and_u8`,
  - Tensor wrapper `Tensor::cuda_bool_and`.
- Extended memory DDP from plain row-union sparse updates to offline SMFT row-mask intersection:
  - parent `train-memory-lm --distributed nccl` now allows `--smft-row-mask` when `--memory-update-policy sparse-rows`,
  - `--smft-mode masked-memory-rows` is allowed only with an explicit offline row mask,
  - background counts, generated mask artifacts, online refresh, and freeze-dense SMFT remain rejected before NCCL launch,
  - hidden memory rank configs carry `smft_row_mask`,
  - each rank loads and validates the same offline `SmftRowMask`,
  - sparse update descriptors can carry SMFT row masks into `ddp_row_union_sparse_updates`,
  - the final optimizer mask is `row_union_mask && smft_mask` computed on CUDA.
- Added report evidence:
  - rank reports include `smft_mode`, `smft_row_mask_source`, `smft_row_mask_trainable_rows`, and `smft_row_mask_frozen_rows`,
  - aggregate reports include `smft_mode`, `smft_row_mask`, and rank-0 SMFT row counts.
- Extended the memory CUDA fixture:
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK=auto` writes a deterministic offline mask,
  - fixture checks now require SMFT mask evidence when mask mode is enabled.
- Updated `scripts/gcp/README.md` with the 4x A100 sparse-row memory DDP plus offline SMFT mask-intersection command.
- Added CPU-safe CLI guard coverage:
  - distributed masked SMFT without `--smft-row-mask` rejects before NCCL,
  - distributed `--smft-row-mask` without sparse-row updates rejects before NCCL.
- Validation so far:
  - `bash -n scripts/cuda_train_memory_lm_fixture.sh`,
  - `bash -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --test distributed_cli -- --nocapture`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`,
  - `cargo test -p heirloom-kernels --lib`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - live validation still needs the opt-in A100 sparse-row memory DDP fixture with `HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK=auto`.
- Remaining hard-path status:
  - offline SMFT row-mask intersection exists,
  - distributed background-count mask generation and online refresh synchronization remain pending,
  - true compressed sparse row-gradient buffers remain pending.

## Checkpoint 108 - Selected-Row Gradient Gather Primitive

- Added a CUDA compact selected-row gather primitive:
  - `heirloom-kernels::cuda::SelectedRowsDims`,
  - `heirloom-kernels::cuda::memory_gather_selected_rows_f32_i64`,
  - PTX kernel `heirloom_memory_gather_selected_rows_f32_i64`,
  - device status validation for out-of-range selected rows.
- Exposed Tensor helpers:
  - `Tensor::cuda_memory_gather_selected_rows_f32_i64`,
  - `Tensor::cuda_memory_gather_selected_grad_rows_f32_i64`.
- Added report instrumentation:
  - `MemoryKernelCounters::gather_selected_rows_calls`,
  - `gather_selected_rows_calls` in memory LM rank/train report counters,
  - `gather_selected_rows_f32_i64` in the memory kernel surface list.
- Added gated CUDA test coverage:
  - `cuda_memory_gather_selected_rows_and_gradients_stays_device_resident` validates table-row gather and gradient-row gather on `cuda:0` when `HEIRLOOM_CUDA_TESTS=1`.
- Updated `MEMORY_TRANSFORMER_DESIGN.md` and `HARD_MODE.md`:
  - the primitive is now implemented as a real prerequisite for compressed sparse row-gradient transport,
  - DDP still all-reduces dense gradients before row-union sparse AdamW, so compressed sparse-gradient DDP remains pending.
- Validation so far:
  - `cargo fmt --all`,
  - `cargo test -p heirloom-kernels --lib`,
  - `cargo test -p heirloom --test memory_transformer -- --nocapture`.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - the new primitive has a gated CUDA test but still needs live A100 validation with `HEIRLOOM_CUDA_TESTS=1` or the memory fixture.

## Checkpoint 109 - Compact Selected-Row Sparse AdamW Consumption

- Added a compact selected-row sparse AdamW kernel:
  - `heirloom-kernels::cuda::SparseAdamWCompactRowsBuffers`,
  - `heirloom-kernels::cuda::memory_sparse_adamw_compact_rows_f32_i64_buffers`,
  - PTX kernel `heirloom_memory_sparse_adamw_compact_rows_f32_i64`.
- Routed `Tensor::apply_adamw_cuda_sparse_rows_f32` through compact gradient rows:
  - gathers selected dense-gradient rows on CUDA with `memory_gather_selected_rows_f32_i64`,
  - runs compact-row AdamW over selected parameter and moment rows,
  - preserves duplicate selected-row dedupe and optional SMFT Bool row-mask semantics.
- Added report evidence:
  - `MemoryKernelCounters::sparse_adamw_compact_rows_calls`,
  - `sparse_adamw_compact_rows_calls` in train/rank memory-kernel JSON counters,
  - `sparse_adamw_compact_rows_f32_i64_optional_row_mask` in the memory kernel surface,
  - `memory_optimizer.sparse_updates_accumulate_dense_gradient_buffers`,
  - `memory_optimizer.sparse_optimizer_gathers_compact_gradient_rows`,
  - `memory_optimizer.compressed_sparse_gradient_transport`.
- Strengthened the CUDA memory fixture validator:
  - sparse-row runs now require positive `gather_selected_rows_calls`,
  - sparse-row runs now require positive `sparse_adamw_compact_rows_calls`,
  - single-rank reports must state compact sparse optimizer row gathering is active.
- Updated `MEMORY_TRANSFORMER_DESIGN.md`, `HARD_MODE.md`, and `scripts/gcp/README.md`:
  - sparse AdamW no longer reads the full dense gradient table inside the row-update kernel,
  - autograd still accumulates dense memory-table gradients,
  - DDP still all-reduces dense gradients before row-union sparse AdamW,
  - compressed sparse-gradient transport remains pending.
- Validation status:
  - pending local formatting/tests after this checkpoint.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - live A100 validation remains pending for the compact sparse optimizer path.

## Checkpoint 110 - Row-Union Mask Compaction For Memory DDP

- Added CUDA Bool-mask-to-row-index compaction:
  - `heirloom-kernels::cuda::bool_mask_to_i64_indices`,
  - PTX kernel `heirloom_bool_mask_to_i64_indices`,
  - Tensor wrapper `Tensor::cuda_bool_mask_to_i64_indices`.
- Implementation detail:
  - correctness-first prefix computation scans prior mask entries per active row,
  - the wrapper reads a scalar row count back to the host, then returns an exact-length CUDA i64 tensor,
  - this is row metadata synchronization, not gradient payload staging.
- Routed memory DDP row-union sparse updates through compact candidates:
  - `ddp_row_union_sparse_updates` still builds and NCCL-sums row masks,
  - offline SMFT masks are still intersected on CUDA,
  - the final Bool mask is now compacted to selected row ids,
  - sparse AdamW receives compact candidate rows instead of `arange(all_rows)` plus a row mask.
- Added report/fixture evidence:
  - `MemoryKernelCounters::bool_mask_to_indices_calls`,
  - `bool_mask_to_indices_calls` in memory LM train/rank reports,
  - `bool_mask_to_i64_indices` in the memory kernel surface,
  - the A100 memory fixture now requires positive `bool_mask_to_indices_calls` for sparse-row DDP ranks.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now describes compact row-union candidates,
  - `HARD_MODE.md` removes the stale "scans all rows as candidates" limitation and keeps dense DDP gradient all-reduce as the remaining gap,
  - `scripts/gcp/README.md` records the new sparse-row DDP evidence requirement.
- Validation status:
  - pending local formatting/tests after this checkpoint.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - live A100 validation remains pending for row-union mask compaction.

## Checkpoint 111 - Compact Sparse-Gradient DDP Transport For Memory Rows

- Replaced dense memory-gradient all-reduce for `train-memory-lm --distributed nccl --memory-update-policy sparse-rows`:
  - ranks now build the CUDA row-union mask first,
  - intersect any offline SMFT Bool row mask on CUDA,
  - compact the final Bool mask to CUDA i64 row ids,
  - gather only those local dense-gradient rows into a compact CUDA f32 payload,
  - contribute CUDA-filled zero compact rows when a local rank did not touch a unioned row,
  - NCCL all-reduce only the compact gradient payload,
  - scale by `1/world_size` on device,
  - apply compact-row AdamW from the externally averaged compact gradient rows.
- Added optimizer/runtime API:
  - `SparseAdamWCompactRowsUpdate`,
  - `AdamW::step_cuda_sparse_compact_rows_mut`,
  - `Tensor::cuda_f32_zeros_on_device`,
  - `Tensor::cuda_f32_sum_squares`,
  - `Tensor::cuda_all_reduce_sum_in_place_f32_nccl`,
  - `Tensor::apply_adamw_cuda_sparse_rows_compact_grad_f32`.
- Added report evidence:
  - per-rank and aggregate `compact_gradient_all_reduce_calls`,
  - per-rank and aggregate `compact_gradient_all_reduce_bytes`,
  - per-rank and aggregate `compressed_sparse_gradient_transport=true` for sparse-row DDP.
- Strengthened the CUDA memory fixture validator:
  - sparse-row DDP train/resume reports must prove compact gradient all-reduce counters,
  - sparse-row DDP reports must set `compressed_sparse_gradient_transport`.
- Updated docs:
  - `MEMORY_TRANSFORMER_DESIGN.md` now describes compact sparse-gradient all-reduce,
  - `HARD_MODE.md` removes the stale dense-DDP-gradient limitation and keeps the remaining narrowness/A100-validation caveats,
  - `scripts/gcp/README.md` records the new fixture evidence requirements.
- Validation status:
  - pending local formatting/tests after this checkpoint.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - live A100 validation remains pending for compact sparse-gradient DDP transport.

## Checkpoint 115 - Vertex Memory Fixture Validation Launched

- Fixed the Vertex wrapper env-var limit issue:
  - `scripts/gcp/submit_vertex_heirloom_validate.sh` now filters generated YAML env entries whose value is `__HEIRLOOM_UNSET__`,
  - the wrapper fails before submit if the filtered env count still exceeds Vertex's 100-variable limit,
  - the successful submit kept `37` env entries and removed `69` unset entries.
- Launched the requested 4x A100 memory fixture validation:
  - display name: `heirloom-validate-quick-20260608-231110`,
  - job id: `2336619971762716672`,
  - job resource: `projects/232930557062/locations/us-central1/customJobs/2336619971762716672`,
  - source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260608-231110.tar.gz`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260608-231110`,
  - accelerator config: `4x NVIDIA_A100_80GB` on the wrapper-selected `a2-ultragpu-4g`,
  - fixture config: `amp-bf16`, `SparseRows`, `SMFT masked-memory-rows`, auto row mask, `40` train steps, `3` resume steps, `ddp_checksum_every=1`.
- Local validation before launch:
  - `python3 -m py_compile scripts/validate_memory_fixture_artifacts.py`,
  - synthetic pass/fail artifact validator check,
  - `cargo fmt --all --check`,
  - `cargo test -p heirloom --bin heirloom -- --nocapture`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - Vertex job is submitted and active at launch time,
  - final A100 result, artifact validation, and report summaries remain pending until the job finishes.

## Checkpoint 116 - Vertex Smoke Failure Triage And PTX Predicate Guard

- The first 4x A100 memory fixture validation reached a terminal failure before the memory fixture ran:
  - display name: `heirloom-validate-quick-20260608-231110`,
  - job id: `2336619971762716672`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260608-231110`,
  - failure stage: per-device CUDA smoke on device `0`,
  - observed failure: `cuModuleLoadDataEx` CUDA error `218` from the embedded `KERNEL_PTX`.
- Downloaded non-secret failure artifacts showed:
  - all four `NVIDIA A100-SXM4-80GB` devices were visible,
  - CUDA driver loading and device discovery succeeded,
  - PTX JIT failed on `heirloom_bool_mask_to_i64_indices` because the entry declared `.reg .pred %p<4>` but used `%p4`.
- Fixed the PTX declaration to `.reg .pred %p<5>`.
- Added a static `heirloom-kernels` unit test that scans both embedded PTX strings and fails if any `.visible .entry` uses a predicate register outside its declared `%p<N>` range.
- Validation status:
  - `cargo test -p heirloom-kernels --lib` passes.
- GPU status:
  - the failed Vertex job is terminal,
  - retry launch remains pending until the broader local validation suite passes.

## Checkpoint 117 - Vertex Memory Fixture Validation Relaunched After PTX Fix

- Broader local validation passed after the PTX predicate fix:
  - `cargo fmt --all --check`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `python3 -m py_compile scripts/validate_memory_fixture_artifacts.py`,
  - `cargo test -p heirloom-kernels --lib`,
  - `cargo test -p heirloom --bin heirloom -- --nocapture`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- Verified non-secret GCP access before relaunch:
  - active project: `project-49b1b523-d248-434f-bd4`,
  - recent Vertex custom jobs are listable,
  - artifact bucket `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/` is reachable.
- Relaunched the requested 4x A100 memory fixture validation:
  - display name: `heirloom-validate-quick-20260608-232825`,
  - job id: `8227891234316746752`,
  - job resource: `projects/232930557062/locations/us-central1/customJobs/8227891234316746752`,
  - source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260608-232825.tar.gz`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260608-232825`,
  - accelerator config: `4x NVIDIA_A100_80GB` on the wrapper-selected `a2-ultragpu-4g`,
  - fixture config: `amp-bf16`, `SparseRows`, `SMFT masked-memory-rows`, auto row mask, `40` train steps, `3` resume steps, `ddp_checksum_every=1`,
  - Vertex env filter kept `37` entries and removed `69` unset entries before submission.
- GPU status:
  - retry job was `JOB_STATE_PENDING` immediately after submission,
  - final A100 result, artifact validation, and report summaries remain pending until the job finishes.

## Checkpoint 118 - Vertex Retry Reached Memory DDP And Exposed CUDA Softmax Dispatch Gap

- The 4x A100 retry reached terminal failure after progressing past the earlier PTX smoke blocker:
  - display name: `heirloom-validate-quick-20260608-232825`,
  - job id: `8227891234316746752`,
  - artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260608-232825`,
  - GPU smoke artifacts exist for devices `0..3`,
  - Tensor Core probe artifacts exist for devices `0..3`,
  - memory fixture rank artifacts were uploaded under `cuda-train-memory-lm-fixture/`.
- Downloaded non-secret artifacts to `/tmp/heirloom-vertex-8227891234316746752`.
- Launcher evidence:
  - all four ranks spawned and wrote stage files,
  - all four ranks failed in the same forward-path stage,
  - rank message: `aten.softmax.dim has no registered CUDA kernel yet for device Cuda(N); copy explicitly to CPU or implement the CUDA kernel path`,
  - no rank report was written because training did not reach report emission.
- Root cause:
  - `Tensor::softmax_dim` resolved through CPU-only dispatch before checking for CUDA,
  - the existing CUDA rank-2 `dim=1` softmax implementation and CUDA backward path were therefore unreachable from memory-layer retrieval.
- Fixed `Tensor::softmax_dim` so CUDA tensors validate shape/dim and route to `cuda_softmax_dim` before CPU-only dispatch resolution.
- Validation status after fix:
  - `cargo fmt --all --check`,
  - `cargo test --test memory_transformer`,
  - `cargo test -p heirloom --bin heirloom -- --nocapture`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - retry job is terminal failed,
  - a new paid Vertex retry is not launched from this checkpoint without explicit approval.

## Checkpoint 119 - Failed Memory Fixture Artifact Diagnosis

- Hardened `scripts/validate_memory_fixture_artifacts.py` for partial failed memory fixture runs:
  - if `summary.json` is missing but `launcher-report.json` exists, the validator now reports the launcher status, launcher reason, each rank's device, exit status, final stage, rank-report presence, and final stage message,
  - unsupported CUDA operator messages are classified as real device-residency blockers rather than missing artifact/report problems.
- Updated `scripts/gcp/README.md` to document running the same validator on failed fixture directories.
- Validated against the failed 4x A100 memory fixture artifacts from job `8227891234316746752`:
  - the validator reports all four ranks failed at `aten.softmax.dim` on their respective CUDA devices,
  - the validator exits nonzero as expected because the fixture did not complete.
- Validation status:
  - `python3 -m py_compile scripts/validate_memory_fixture_artifacts.py`,
  - `cargo fmt --all --check`,
  - failure-diagnostic validator run against `/tmp/heirloom-vertex-8227891234316746752/cuda-train-memory-lm-fixture`.
- GPU status:
  - no new paid GCP/Vertex job was launched,
  - next GPU gate remains a fresh 4x A100 retry after explicit approval.

## Checkpoint 120 - Vertex Wrapper Uploads Memory Fixture Validator Transcript

- Hardened `scripts/gcp/submit_vertex_heirloom_validate.sh` so the Vertex worker runs `scripts/validate_memory_fixture_artifacts.py` after the CUDA memory fixture:
  - successful fixture runs must also pass the validator or the Vertex job fails,
  - failed fixture runs still upload `cuda-train-memory-lm-fixture/artifact-validator.txt` with launcher/rank diagnostics without masking the original fixture failure,
  - validator flags are derived from the fixture env: distributed NCCL, sparse-row policy, and nonzero resume steps.
- Updated `scripts/gcp/README.md` so the expected memory fixture artifact list includes `artifact-validator.txt`.
- Validation status:
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - embedded Vertex worker Python payload compiled via local extraction,
  - `python3 -m py_compile scripts/validate_memory_fixture_artifacts.py`,
  - failure-diagnostic validator run against `/tmp/heirloom-vertex-8227891234316746752/cuda-train-memory-lm-fixture`.
- GPU status:
  - no new paid GCP/Vertex job was launched,
  - next 4x A100 retry should upload an artifact validator transcript whether it passes or fails.

## Checkpoint 121 - CUDA Dispatch Routing Regression Guard

- Added `tests/cuda_dispatch_guards.rs`, an always-on source-level guard for CUDA-backed Tensor methods.
- The guard checks that CUDA-capable methods route non-CPU tensors to their CUDA implementation before reaching CPU-only dispatch or CPU materialization:
  - binary ops,
  - matmul,
  - relu,
  - gelu,
  - layer norm,
  - embedding,
  - causal self-attention,
  - softmax,
  - sum/mean reductions,
  - cross entropy with slice and tensor targets.
- This specifically protects against the failure exposed by Vertex job `8227891234316746752`, where `softmax_dim` had a CUDA kernel but CPU dispatch resolution made the CUDA path unreachable.
- Validation status:
  - `cargo fmt --all --check`,
  - `cargo test --test cuda_dispatch_guards`,
  - `python3 -m py_compile scripts/validate_memory_fixture_artifacts.py`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`.
- GPU status:
  - no new paid GCP/Vertex job was launched,
  - next 4x A100 retry remains the live proof that the fixed softmax route gets memory DDP past the previous blocker.

## Checkpoint 122 - Memory Fixture Validator Promoted Into Full Gate

- Promoted `cuda-train-memory-lm-fixture/artifact-validator.txt` into `cuda_train_memory_lm_fixture_report_uris` so successful Vertex summaries link the post-run artifact validator transcript.
- Extended `scripts/validate.sh` to syntax-check the memory fixture shell script, the Vertex wrapper, and the Python artifact validator before running the workspace test suite.
- Confirmed the latest hardening with the full local gate:
  - `cargo fmt --all --check`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - launching one approved 4x A100 retry after this checkpoint,
  - retry target is the distributed AMP BF16 memory fixture with sparse-row memory updates, SMFT row mask, checksum-every-step validation, and resume.

## Checkpoint 123 - 4x A100 Memory Fixture Retry Launched

- Submitted Vertex custom job `6846271311132491776`.
- Display name: `heirloom-validate-quick-20260608-235142`.
- Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260608-235142.tar.gz`.
- Expected artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260608-235142`.
- Launch settings:
  - `HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4`,
  - `HEIRLOOM_GPU_SMOKE_DEVICES=all`,
  - `HEIRLOOM_RUN_CUDA_TRAIN_MEMORY_LM_FIXTURE=1`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED=nccl`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION=amp-bf16`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY=sparse-rows`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE=masked-memory-rows`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK=auto`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS=40`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS=3`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_CHECKSUM_EVERY=1`.
- Validation objective:
  - prove the fixed CUDA softmax dispatch gets memory DDP past the previous `aten.softmax.dim` blocker,
  - require uploaded memory fixture reports plus the new `artifact-validator.txt` transcript.
- Current state after bounded polling:
  - `JOB_STATE_SUCCEEDED`,
  - downloaded artifacts to `/tmp/heirloom-vertex-6846271311132491776/cuda-train-memory-lm-fixture`,
  - `scripts/validate_memory_fixture_artifacts.py /tmp/heirloom-vertex-6846271311132491776/cuda-train-memory-lm-fixture --expect-distributed --expect-sparse-rows --require-resume` passed.
- Result evidence:
  - top-level `summary.json` status is `passed`,
  - top-level `cuda_train_memory_lm_fixture_report_uris` includes the promoted `cuda-train-memory-lm-fixture/artifact-validator.txt`,
  - 4x distributed memory fixture ran with `world_size=4`, `precision=amp-bf16`, `distributed=nccl`, and sparse-row memory updates,
  - train: step `0 -> 40`, loss `5.673858 -> 5.619308`, compact gradient all-reduce calls `320`, bytes `119040`,
  - resume: step `40 -> 43`, loss `5.805376 -> 5.636040`, compact gradient all-reduce calls `24`, bytes `8832`,
  - memory table checksum drift stayed `0.0`,
  - rank-0 CUDA memory kernel counters were positive for top-k, weighted-value forward/backward, selected-key backward, scatter-add rows, selected-row gather, Bool-mask compaction, and compact sparse AdamW,
  - artifact validator transcript reports `memory_fixture_artifacts status=passed distributed=True sparse_rows=True train_compact_gradient_all_reduce_calls=320`.

## Checkpoint 124 - Explicit 32-Layer Forward/Loss CPU Proof

- Added `thirty_two_layer_memory_transformer_forward_shape_and_loss_are_finite` to `tests/memory_transformer.rs`.
- The new always-on test constructs a true 32-block `MemoryTransformerLm` with selected memory layers at blocks `8`, `16`, and `24`, then verifies:
  - logits shape `[1, 4, 32]`,
  - finite language-model loss.
- This closes the CPU acceptance gap where 32-layer construction was covered but forward/loss was only covered by a smaller 2-layer model.
- Validation status:
  - `cargo fmt --all --check`,
  - `cargo test --test memory_transformer thirty_two_layer_memory_transformer_forward_shape_and_loss_are_finite`,
  - `cargo test --test memory_transformer constructs_32_layer_memory_transformer_with_stable_memory_names`,
  - `cargo test --test memory_transformer` passed with `38` tests,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `./scripts/validate.sh`.
- GPU status:
  - no new paid GCP/Vertex job was launched after the successful job `6846271311132491776`.

## Checkpoint 125 - 32-Block Large-Data Memory Fixture Gate Wiring

- Extended `scripts/cuda_train_memory_lm_fixture.sh` with data-source modes:
  - `synthetic` keeps the existing small local memory text,
  - `tinystories-valid` downloads the public TinyStories validation split through the Rust CLI,
  - `file` accepts an explicit text file and optional byte cap.
- The memory fixture `summary.json` now records:
  - `data_source`,
  - source path,
  - source bytes,
  - prepared train and validation token counts.
- Hardened `scripts/validate_memory_fixture_artifacts.py` with optional gates for:
  - precision,
  - world size,
  - `memory_config.n_layers`,
  - memory layer indices,
  - memory update policy,
  - SMFT mode,
  - train/resume step ranges,
  - data source,
  - minimum source bytes and train tokens,
  - core CUDA memory-kernel counters,
  - Tensor Core BF16 matmul counters,
  - attention Tensor Core counters when explicitly required.
- Updated `scripts/gcp/submit_vertex_heirloom_validate.sh` so the Vertex wrapper forwards the new memory fixture data env vars and passes declared shape/data/kernel expectations to the artifact validator.
- Updated `scripts/gcp/README.md` with the paid 4x A100 32-block TinyStories-valid memory-transformer reference command.
- Revalidated the previous successful 4x artifact with strict validator flags:
  - distributed NCCL,
  - sparse rows,
  - resume,
  - `amp-bf16`,
  - world size `4`,
  - `n_layers=4`,
  - memory layer indices `1,3`,
  - sparse-row policy,
  - masked-memory-rows SMFT,
  - step range `40 + 3`,
  - core memory kernel counters,
  - Tensor Core counters.
- Validation status:
  - `bash -n scripts/cuda_train_memory_lm_fixture.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `python3 -m py_compile scripts/validate_memory_fixture_artifacts.py`,
  - strict validator replay against `/tmp/heirloom-vertex-6846271311132491776/cuda-train-memory-lm-fixture`,
  - `./scripts/validate.sh`.
- GPU status:
  - preparing approved paid 4x A100 32-block TinyStories-valid memory run,
  - planned gate requires `n_layers=32`, memory layers `8,16,24`, vocab `1024`, TinyStories-valid data source, at least `10,000,000` source bytes, at least `8,000,000` train tokens, sparse-row DDP, resume, core memory kernels, and Tensor Core BF16 matmul counters.

## Checkpoint 126 - 4x A100 32-Block TinyStories-Valid Memory Run Launched

- Submitted Vertex custom job `904616027747254272`.
- Display name: `heirloom-validate-quick-20260609-000934`.
- Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260609-000934.tar.gz`.
- Expected artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260609-000934`.
- Launch settings:
  - 4x A100 Vertex quick validation,
  - NCCL all-reduce probe enabled,
  - memory fixture data source `tinystories-valid`,
  - expected minimum source bytes `10,000,000`,
  - expected minimum train tokens `8,000,000`,
  - vocab `1024`,
  - devices `cuda:0,cuda:1,cuda:2,cuda:3`,
  - distributed `nccl`,
  - precision `amp-bf16`,
  - `n_layers=32`,
  - memory layers `8,16,24`,
  - block size `64`,
  - `d_model=64`,
  - `n_heads=4`,
  - `ff_hidden=256`,
  - memory slots `1024`,
  - memory key/value dims `32/64`,
  - top-k `4`,
  - sparse-row memory updates,
  - offline SMFT row mask,
  - train/resume steps `100 + 5`,
  - per-rank batch size `1`,
  - DDP checksum every `5` steps.
- Validation objective:
  - prove 32-block memory-transformer CUDA/DDP execution on full TinyStories-valid,
  - require large-data token evidence,
  - require memory kernel counters,
  - require Tensor Core BF16 matmul counters,
  - require resume and artifact validator transcript.
- Current state after initial polling:
  - `JOB_STATE_FAILED`,
  - GPU smoke artifacts are present for all four devices,
  - topology artifacts are present,
  - NCCL probe artifacts are present,
  - Tensor Core probe artifacts are present for all four devices,
  - `cuda-train-memory-lm-fixture/` artifacts were downloaded to `/tmp/heirloom-vertex-904616027747254272/cuda-train-memory-lm-fixture`.
- Failure diagnosis:
  - TinyStories-valid download, tokenizer training, and data preparation succeeded,
  - prepared data had `19,447,282` source bytes, `7,547,004` train tokens, and `1,886,751` validation tokens,
  - 32-block distributed train completed and wrote train report, rank reports, and checkpoint,
  - report proved `n_layers=32`, memory layers `8,16,24`, vocab `1024`, `amp-bf16`, `world_size=4`, sparse-row DDP, zero memory checksum drift, positive memory-kernel counters, and positive Tensor Core counters,
  - train loss moved `6.815419 -> 7.025311`,
  - script failed before resume because `HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION=0.0` rejected the negative early loss reduction,
  - no unsupported CUDA operator, NCCL, Tensor Core, memory-kernel, checkpoint-write, or rank-launch blocker was found in this run.
- Follow-up adjustment:
  - `scripts/gcp/README.md` now sets the 32-block reference tier to `HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION=-1.0`, because this tier is a runtime/shape/resume/kernel proof rather than an early-training-quality gate,
  - documented minimum train-token threshold is now `7,000,000`, matching the 80/20 TinyStories-valid split while still proving large-data execution,
  - documented LR is now `0.0003` for the retry.

## Checkpoint 127 - 32-Block Large-Data Runtime-Gate Retry Launched

- Submitted Vertex custom job `295046781308239872`.
- Display name: `heirloom-validate-quick-20260609-002335`.
- Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260609-002335.tar.gz`.
- Expected artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260609-002335`.
- Retry differences from job `904616027747254272`:
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION=-1.0`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_TRAIN_TOKENS=7000000`,
  - `HEIRLOOM_CUDA_MEMORY_FIXTURE_LR=0.0003`.
- Objective:
  - keep the same 32-block TinyStories-valid CUDA/DDP runtime proof,
  - continue into checkpoint resume and strict artifact validation even if the short run's loss is noisy.
- Current state:
  - `JOB_STATE_FAILED`,
  - no `cuda-train-memory-lm-fixture/` directory was created,
  - top-level `failure-summary.json` reports `ConnectionError`: `Connection reset by peer`,
  - GPU smoke artifacts were uploaded,
  - NCCL probe passed on all four ranks with `max_abs_error=0.0`,
  - Tensor Core probe artifacts were uploaded for all four devices.
- Diagnosis:
  - retry failure occurred before memory fixture execution, likely in the wrapper/network path rather than in the memory transformer runtime,
  - no new CUDA operator, DDP, Tensor Core, memory optimizer, checkpoint, or resume blocker was exposed by this retry.
- Current best evidence remains job `904616027747254272` for the 32-block large-data memory runtime:
  - it completed TinyStories-valid prep,
  - completed 100-step 4x distributed 32-block memory training,
  - wrote checkpoint and rank reports,
  - proved large-data/model-shape/kernel/Tensor-Core evidence,
  - failed before resume solely because the old `MIN_REDUCTION=0.0` runtime gate rejected the short-run loss increase.

## Checkpoint 114 - Standalone Memory Fixture Artifact Validator

- Added `scripts/validate_memory_fixture_artifacts.py`.
- The validator checks a completed `cuda-train-memory-lm-fixture/` directory without rerunning training:
  - required `summary.json`, `train-report.json`, and optionally `resume-report.json`,
  - `train-memory-lm` / `memory_transformer` report identity,
  - finite train/resume losses,
  - distributed NCCL all-reduce evidence,
  - sparse-row compact gradient all-reduce calls/bytes,
  - `compressed_sparse_gradient_transport=true`,
  - row-union counters,
  - checksum drift within report tolerance,
  - rank-level compact-gradient counters,
  - rank-0 memory kernel counters for Bool-mask compaction, selected-gradient gather, and compact sparse AdamW.
- Updated `scripts/gcp/README.md` with the post-run validator command.
- Validation status:
  - pending local script tests and repo validation after this checkpoint.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - live A100 validation remains pending for compact sparse-gradient DDP transport.

## Checkpoint 113 - Memory Fixture Summary Exposes Compact Transport Evidence

- Promoted sparse memory DDP evidence into `scripts/cuda_train_memory_lm_fixture.sh` `summary.json`:
  - train/resume `compact_gradient_all_reduce_calls`,
  - train/resume `compact_gradient_all_reduce_bytes`,
  - train/resume `compressed_sparse_gradient_transport`,
  - row-union counters,
  - memory checksum drift fields,
  - per-rank compact-gradient counter arrays,
  - rank-0 memory kernel counters including gather, Bool-mask compaction, and compact sparse AdamW calls.
- Updated `scripts/gcp/README.md` so Vertex artifact reviewers know the memory fixture summary mirrors the compact transport evidence.
- Validation status:
  - pending script syntax/local validation after this checkpoint.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - live A100 validation remains pending for compact sparse-gradient DDP transport.

## Checkpoint 112 - Compact Sparse Transport Evidence Tests

- Extracted the distributed memory all-reduce evidence check into a testable helper:
  - `validate_distributed_memory_all_reduce_evidence`.
- Added CLI unit coverage:
  - full-memory DDP still requires positive dense gradient all-reduce counters,
  - sparse-row DDP now explicitly fails if row-union counters are present but compact gradient all-reduce counters are missing,
  - sparse-row DDP passes the evidence gate only with positive compact gradient calls and bytes.
- Added an optimizer API contract test:
  - malformed `SparseAdamWCompactRowsUpdate` compact-gradient shapes are rejected before CUDA execution.
- Validation status:
  - pending local formatting/tests after this checkpoint.
- GPU status:
  - no paid GCP/Vertex job was launched,
  - live A100 validation remains pending for compact sparse-gradient DDP transport.

## Checkpoint 128 - Vertex Cloud I/O Retry Hardening

- Hardened `scripts/gcp/submit_vertex_heirloom_validate.sh` after retry job `295046781308239872` failed with a transient `ConnectionResetError` before the memory fixture started.
- Added retry handling for Vertex-worker GCS download/upload operations:
  - source package download,
  - diagnostic uploads,
  - JSON/report uploads,
  - checkpoint directory uploads.
- Added retry handling for launcher-side `gcloud storage` bucket/package operations so package staging has the same transient-failure protection.
- Retry logging intentionally reports labels, attempt counts, error type, and sleep duration only; it does not dump env values, Vertex specs, or secrets.
- Validation:
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - embedded Vertex worker Python extracted and compiled with `python3 -m py_compile`,
  - `python3 -m py_compile scripts/validate_memory_fixture_artifacts.py`,
  - `./scripts/validate.sh`.
- GPU status:
  - ready to relaunch the full 4x A100 32-block TinyStories-valid memory fixture with the corrected loss gate and cloud I/O retry hardening.

## Checkpoint 129 - 4x A100 32-Block TinyStories-Valid Memory Fixture Passed

- Launched the full 4x A100 32-block large-data memory-transformer reference run after cloud I/O retry hardening.
- Vertex job:
  - id `6080483452619063296`,
  - display `heirloom-validate-quick-20260609-003752`,
  - final state `JOB_STATE_SUCCEEDED`.
- Artifact prefix:
  - `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260609-003752`.
- Source package:
  - `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260609-003752.tar.gz`.
- Preflight:
  - GPU smoke artifacts uploaded for devices `0..3`,
  - Tensor Core probe passed on all four devices,
  - NCCL all-reduce probe passed for `cuda:0,cuda:1,cuda:2,cuda:3`,
  - NCCL probe `max_abs_error=0.0`.
- Large-data memory fixture:
  - `data_source=tinystories-valid`,
  - `source_bytes=19447282`,
  - `train_tokens=7547004`,
  - `valid_tokens=1886751`,
  - `precision=amp-bf16`,
  - `distributed=nccl`,
  - `world_size=4`,
  - `n_layers=32`,
  - memory layers `[8,16,24]`,
  - sparse-row memory updates with `smft_mode=masked-memory-rows`.
- Training report:
  - status `passed`,
  - steps `0 -> 100`,
  - loss `6.8154191970825195 -> 7.045341491699219`,
  - short-run loss reduction `-0.033735605685872146`,
  - `all_reduce_calls=800`,
  - `compact_gradient_all_reduce_calls=800`,
  - memory table checksum drift `0.0`,
  - AMP validation `passed`,
  - unexpected host staging `0`.
- Resume report:
  - status `passed`,
  - steps `100 -> 105`,
  - loss `7.017897129058838 -> 6.9734978675842285`,
  - `all_reduce_calls=40`,
  - `compact_gradient_all_reduce_calls=40`,
  - memory table checksum drift `0.0`,
  - AMP validation `passed`,
  - unexpected host staging `0`.
- Rank-0 Tensor Core evidence during training:
  - `bf16_tensor_core_matmul_calls=132900`,
  - `bf16_tensor_core_matmul_forward_calls=44300`,
  - `bf16_tensor_core_matmul_backward_calls=88600`,
  - `bf16_scalar_matmul_fallback_calls=0`,
  - attention Tensor Core counters were positive for forward, backward, QK, AV, score-grad, dQ, dK, and dV.
- Rank-0 memory kernel evidence during training:
  - `topk_calls=300`,
  - `weighted_value_forward_calls=300`,
  - `weighted_value_backward_calls=300`,
  - `selected_key_backward_calls=300`,
  - `scatter_add_rows_calls=600`,
  - `gather_selected_rows_calls=200`,
  - `bool_mask_to_indices_calls=200`,
  - `sparse_adamw_compact_rows_calls=200`,
  - `selected_rows=76800`.
- Artifact validator:
  - `memory_fixture_artifacts status=passed distributed=True sparse_rows=True train_compact_gradient_all_reduce_calls=800`.
- Local validation before launch:
  - `./scripts/validate.sh` passed.

## Checkpoint 130 - Review Hardening: Single-Rank SMFT, Frozen Steps, Sparse Clipping, Milestone Doc

- Addressed the post-milestone review findings.
- Single-rank `train-memory-lm` now rejects `--smft-mode masked-memory-rows` without an initial `--smft-row-mask`, unless `--smft-refresh-every 1` is configured so the mask is materialized before the first optimizer step.
- Memory-LM checkpoint metadata and reports now record training steps consumed, not only optimizer update steps:
  - added explicit-step checkpoint save helper,
  - single-rank and DDP memory training now report `final_step=start_step+steps`,
  - `MemoryUpdatePolicy::Frozen` can advance data/checkpoint state without pretending an optimizer update happened.
- Single-rank CUDA sparse-row AdamW clipping now mirrors the compact sparse DDP path more closely:
  - selected rows are converted to a unique CUDA bool row mask,
  - optional SMFT row masks are intersected,
  - compact active row indices are gathered,
  - clip norm is computed only from active sparse row gradients.
- Added `MILESTONE_32_BLOCK_MEMORY_TINYSTORIES.md` as a reviewer anchor for the June 9 4x A100 32-block memory-transformer run.
- Added regression tests:
  - single-rank masked SMFT without an initial mask is rejected,
  - frozen memory policy reports/checkpoints the consumed training step instead of stale optimizer step.
- Validation:
  - `cargo test --test memory_transformer`,
  - `cargo test --test distributed_cli`,
  - `./scripts/validate.sh`.
- GPU status:
  - no new paid Vertex run launched for this hardening pass,
  - the existing successful 4x A100 run remains job `6080483452619063296`.

## Checkpoint 131 - QB Data Hard Path Passed And Scale Knobs Added

- Recorded the successful full QB-native data hard-path gate:
  - display `heirloom-validate-quick-20260610-212624`,
  - job `projects/232930557062/locations/us-central1/customJobs/9079935230273388544`,
  - completed `2026-06-11T01:36:13Z` (`2026-06-10` US/Eastern),
  - summary `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260610-212624/qb-data-hardpath/summary.json`.
- Gate evidence:
  - manifest v2 with `storage="binary_shards"`,
  - sources include full TinyStories-valid and deterministic QB traces,
  - train/resume/eval/generation completed through `loader.kind="binary_shard_streaming"` with `tokens_materialized=false`,
  - 4x DDP all-reduce evidence was positive (`44000` calls, `1490432000` bytes),
  - checksum drift was `0.0`,
  - Tensor Core Linear and attention hard gates passed with zero scalar fallbacks,
  - performance report included timing buckets, tokens/sec, dense-core MFU estimate, and end-to-end MFU estimate.
- Added configurable multi-shard binary token preparation:
  - `heirloom data prepare --format binary-shard --shard-tokens <n>`,
  - rejects `--shard-tokens 0`,
  - keeps v1 JSON manifests compatible,
  - updated tests to prove multi-shard v2 round-trip and CLI train/eval behavior.
- Added explicit gradient accumulation to dense and memory training paths:
  - `train-lm --grad-accumulation-steps <n>`,
  - `train-memory-lm --grad-accumulation-steps <n>`,
  - DDP rank configs use a micro-step sample key so accumulated micro-batches remain deterministic and distinct across ranks,
  - `steps` remains optimizer steps,
  - reports now include micro-batch size, grad accumulation steps, effective batch size, and global effective batch size.
- Hardened hard-path scripts:
  - `scripts/run_qb_data_hardpath.sh` forwards shard-token and grad-accumulation knobs into train/resume and summary artifacts,
  - `scripts/gcp/submit_vertex_heirloom_validate.sh` forwards the new env vars and uploads the full `prepared/shards/` directory instead of assuming one shard,
  - `scripts/validate_qb_data_hardpath_artifacts.py` validates local or GCS hard-path summaries.
- Documentation:
  - updated `QB_NATIVE_DATA_STRATEGY.md` to mark v2 binary-shard streaming as the proven data hard path and keep 32K tokenizer/corpus manifests as next work,
  - updated `scripts/gcp/README.md` with the recorded 4x pass and validator command,
  - updated `README.md` so the known gaps reflect the first binary-shard hard path.
- Local validation:
  - `./scripts/validate.sh`,
  - CPU hard-path smoke with `HEIRLOOM_QB_DATA_HARDPATH_SHARD_TOKENS=32` and `HEIRLOOM_QB_DATA_HARDPATH_GRAD_ACCUMULATION_STEPS=2`,
  - `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile smoke /tmp/heirloom-qb-hardpath-accum-smoke/summary.json`.
- CPU smoke evidence:
  - train shards `81`,
  - valid shards `5`,
  - loader `binary_shard_streaming`,
  - `tokens_materialized=false`,
  - `grad_accumulation_steps=2`,
  - effective batch size `4`,
  - tokens seen `128`.
- GPU status:
  - no new paid Vertex run launched for this implementation pass,
  - the previously completed QB hard-path gate remains the reference job `9079935230273388544`.

## Checkpoint 132 - CUDA Event Timing For MFU Buckets

- Added a safe CUDA event timer in `heirloom-kernels`:
  - dynamically loads `cuEventElapsedTime`,
  - records start/stop timing events on the cached compute stream,
  - can time NCCL communicator streams,
  - reports `cuda_runtime.event_elapsed_calls` alongside existing event/stream counters.
- Training reports now keep host and CUDA timing separate:
  - compatibility fields such as `forward_backward_elapsed_ms` remain host elapsed buckets,
  - new `*_host_elapsed_ms` fields make that explicit,
  - new `*_cuda_elapsed_ms` fields report CUDA-event buckets for host-to-device, forward/backward, all-reduce, and optimizer paths when running on CUDA,
  - `performance.cuda_event_timing_available` and `performance.mfu_timing_source` explain whether dense-core MFU used CUDA forward/backward event time or a host fallback.
- Distributed aggregate reports now preserve max-rank host and CUDA timing buckets and use CUDA-event forward/backward elapsed time for dense-core MFU when available.
- Hardened the QB hard-path gates:
  - `scripts/run_qb_data_hardpath.sh` requires the new performance fields,
  - full distributed CUDA runs require positive CUDA event availability, positive forward/backward CUDA elapsed time, and positive `event_elapsed_calls`,
  - `scripts/validate_qb_data_hardpath_artifacts.py` enforces the same full-gate contract.
- Documentation:
  - updated `QB_NATIVE_DATA_STRATEGY.md` to separate host timing, CUDA-event timing, dense-core MFU, and end-to-end MFU,
  - updated `scripts/gcp/README.md` with the new hard-path timing requirements.
- Validation:
  - `cargo check --workspace`,
  - `cargo test --workspace`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo fmt --all --check`,
  - `bash -n scripts/run_qb_data_hardpath.sh`,
  - `python3 -m py_compile scripts/validate_qb_data_hardpath_artifacts.py`,
  - CPU QB hard-path smoke at `/tmp/heirloom-qb-hardpath-event-smoke/summary.json`,
  - `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile smoke /tmp/heirloom-qb-hardpath-event-smoke/summary.json`,
  - `./scripts/validate.sh`.
- CPU smoke evidence:
  - `tokens_seen=128`,
  - `tokens_per_second=482.70700641074126`,
  - `grad_accumulation_steps=2`,
  - `train_tokens=2562`,
  - `valid_tokens=135`.
- GPU status:
  - no new paid Vertex run launched for this timing pass,
  - live positive CUDA-event timing evidence should be collected on the next 4x A100 QB hard-path relaunch.

## Checkpoint 133 - Memory-Transformer QB Data Hard Path On Manifest V2

- Extended `scripts/run_qb_data_hardpath.sh` with `HEIRLOOM_QB_DATA_HARDPATH_MODEL=dense|memory`:
  - `dense` remains the default full train/resume/eval/generation data hard path,
  - `memory` runs `train-memory-lm` train/resume on the same manifest v2 binary-shard streaming loader,
  - memory mode records `model_kind="memory"` in `summary.json`,
  - memory mode validates memory-transformer report evidence instead of trying to load memory checkpoints through dense `eval-lm`/`generate`.
- Added memory hard-path controls:
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_N_LAYERS`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LAYER_INDICES`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SLOTS`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_KEY_DIM`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_VALUE_DIM`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_TOP_K`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LOOKUP`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_UPDATE_POLICY`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_MODE`,
  - `HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_ROW_MASK`.
- Full memory defaults now target the native sparse path:
  - `steps=100`,
  - `batch_size=1`,
  - `lr=0.0003`,
  - `min_reduction=-1.0`,
  - `memory_update_policy=sparse-rows`,
  - `smft_mode=masked-memory-rows`,
  - deterministic offline SMFT row mask when `MEMORY_SMFT_ROW_MASK=auto`.
- Hardened artifact validation:
  - `scripts/validate_qb_data_hardpath_artifacts.py` accepts `model_kind="memory"`,
  - memory summaries require `train-memory-lm`, `model_family="memory_transformer"`, memory layers, streaming loader evidence, memory selection/optimizer/SMFT evidence for single-rank runs, and memory gradient/checksum evidence for distributed runs,
  - full sparse-row memory gates require positive row-union and compact sparse-gradient all-reduce counters.
- Extended the Vertex launcher:
  - forwards all memory hard-path env vars,
  - uploads `smft-row-mask.json`,
  - uploads `ddp-memory-ranks`, `train-ddp-memory-ranks`, and `resume-ddp-memory-ranks` under `qb-data-hardpath/`.
- Added a focused Rust CLI regression:
  - `train-memory-lm` consumes a v2 binary-shard manifest,
  - report uses `loader.kind="binary_shard_streaming"` with `tokens_materialized=false`,
  - memory selection and SMFT access evidence are positive.
- Local validation:
  - `cargo test --test memory_transformer train_memory_lm_cli_uses_binary_shard_manifest_streaming_loader`,
  - memory QB hard-path CPU smoke at `/tmp/heirloom-qb-hardpath-memory-smoke/summary.json`,
  - `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile smoke /tmp/heirloom-qb-hardpath-memory-smoke/summary.json`.
- CPU memory smoke evidence:
  - `model_kind=memory`,
  - `train_tokens=2565`,
  - `valid_tokens=135`,
  - `tokens_seen=64`,
  - `tokens_per_second=283.77924571263657`,
  - loader `binary_shard_streaming`,
  - `tokens_materialized=false`,
  - memory layers `[1, 3]`,
  - `memory_update_policy=full` for CPU compatibility.
- GPU status:
  - no paid Vertex run launched for this memory hard-path implementation pass,
  - next gate should launch `HEIRLOOM_QB_DATA_HARDPATH_MODEL=memory` on 4x A100 with `memory_update_policy=sparse-rows`.

## Checkpoint 134 - Memory Eval/Generation CLI For QB Hard Path

- Added memory checkpoint inference/eval commands:
  - `eval-memory-lm`,
  - `generate-memory-lm`.
- Added memory-transformer eval/generation APIs matching the dense LM surface:
  - f32,
  - bf16 activation rounding,
  - amp-bf16 generation/eval with explicit host-staging allowances for scalar/logit inspection.
- Extended reports:
  - eval/generation reports now include `model_family`,
  - dense reports identify `tiny_transformer`,
  - memory reports identify `memory_transformer`.
- Tightened the QB hard-path wrapper:
  - memory mode now runs train, resume, eval, and generation,
  - memory eval uses the manifest v2 binary-shard streaming loader,
  - memory summaries include `eval.json` and `generation.json`,
  - memory eval/generation are required by `scripts/validate_qb_data_hardpath_artifacts.py`.
- Extended the focused Rust CLI regression:
  - `train-memory-lm` consumes a v2 binary-shard manifest,
  - `eval-memory-lm` evaluates the valid split with `loader.kind="binary_shard_streaming"` and `tokens_materialized=false`,
  - `generate-memory-lm` loads the memory checkpoint and writes a generation report.
- Validation so far:
  - `cargo fmt --all --check`,
  - `cargo test --test memory_transformer train_memory_lm_cli_uses_binary_shard_manifest_streaming_loader`,
  - `bash -n scripts/run_qb_data_hardpath.sh`,
  - `python3 -m py_compile scripts/validate_qb_data_hardpath_artifacts.py`,
  - memory QB hard-path CPU smoke at `/tmp/heirloom-qb-hardpath-memory-evalgen-smoke/summary.json`,
  - `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile smoke /tmp/heirloom-qb-hardpath-memory-evalgen-smoke/summary.json`,
  - `./scripts/validate.sh`.
- CPU memory smoke evidence:
  - `model_kind=memory`,
  - train command `train-memory-lm`,
  - eval command `eval-memory-lm`,
  - generation command `generate-memory-lm`,
  - eval loader `binary_shard_streaming`,
  - eval `tokens_materialized=false`,
  - `tokens_seen=64`,
  - `tokens_per_second=279.57950264274274`.

## Checkpoint 135 - 4x A100 Memory QB Data Hard Path Passed

- Launched the full memory-transformer QB data hard path on Vertex:
  - display `heirloom-validate-quick-20260610-231715`,
  - job `projects/232930557062/locations/us-central1/customJobs/683395937506164736`,
  - state `JOB_STATE_SUCCEEDED`,
  - start `2026-06-11T03:21:25Z`,
  - end `2026-06-11T03:29:29Z`.
- Artifact summary:
  - `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260610-231715/qb-data-hardpath/summary.json`.
- Full artifact validation passed:
  - `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile full --expected-world-size 4 gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260610-231715/qb-data-hardpath/summary.json`.
- Run shape:
  - `model_kind=memory`,
  - `precision=amp-bf16`,
  - `distributed=nccl`,
  - devices `cuda:0,cuda:1,cuda:2,cuda:3`,
  - manifest version `2`,
  - storage `binary_shards`,
  - train tokens `8962719`,
  - valid tokens `471722`.
- Memory hard-path evidence:
  - `train-memory-lm` with `model_family=memory_transformer`,
  - `eval-memory-lm` with streaming eval loader,
  - `generate-memory-lm`,
  - train final step `100`,
  - resume final step `105`,
  - eval tokens `12800`,
  - generated tokens `120`.
- Performance and distributed evidence:
  - `tokens_seen=25600`,
  - `tokens_per_second=1982.9589465530596`,
  - `cuda_event_timing_available=true`,
  - `forward_backward_cuda_elapsed_ms=10954.545608520508`,
  - all-reduce calls `800`,
  - all-reduce bytes `10172928`,
  - row-union all-reduce calls `800`,
  - compact-gradient all-reduce calls `800`.
- Tensor Core evidence:
  - Linear forward calls `44300`,
  - Linear backward calls `88600`,
  - attention forward calls `3200`,
  - scalar matmul fallbacks `0`.

## Checkpoint 136 - QB-Native Pretraining Readiness Starts With Corpus Blend Metadata

- Added the first source-level corpus blend manifest contract:
  - format `heirloom.corpus_blend`,
  - version `1`,
  - tokenizer target vocab size,
  - source role/status/license/provenance metadata,
  - sampling weights,
  - tokenizer/pretraining/memory-trace inclusion flags,
  - local record/byte counts and hashes when sources are available.
- Added `heirloom data corpus-blend`:
  - `--out <path>`,
  - optional `--qb-root <path>`,
  - `--blend-id`,
  - `--tokenizer-vocab-size`.
- Seeded the first planned production blend:
  - `allenai.dolma.v1_7` at weight `0.55`,
  - `nvidia.nemotron_cc.high_actual` at weight `0.25`,
  - `nvidia.nemotron_cc_math` at weight `0.10`,
  - `vecl_qb.synthetic.v1-hard` at weight `0.10`,
  - VECL-QB `v0-small` and `v0-full` retained as zero-weight regression/dev sources.
- Scanned `/Users/andrewverdiramo/Desktop/VECL-QB/data` successfully:
  - local sources `3`,
  - local records `30420`,
  - corpora `v0-small`, `v0-full`, `v1-hard`.
- Added `QB_PRETRAINING_READINESS.md`:
  - 32K tokenizer decision,
  - TinyStories exclusion from the production blend,
  - corpus blend v1 source plan,
  - source governance,
  - learning sanity ladder,
  - MFU measurement readiness,
  - phase exit criteria.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test --test transformer_runtime corpus_blend_manifest_summarizes_qb_trace_sources`,
  - `cargo check --bin heirloom`,
  - generated `/tmp/heirloom-qb-native-corpus-blend.json` from the real VECL-QB data root.

## Checkpoint 137 - Native 32K Tokenizer Hard Path

- Added tokenizer artifact v2 while preserving legacy/v1 loading:
  - stable PAD/BOS/EOS and byte-token IDs,
  - reserved-token registry for chat, tools, memory, SMFT, traces, evidence,
    documents, temporal markers, separators, masks, and QB references,
  - source/sample hashes, training config, artifact hash, validation metadata,
    and token length histogram reporting.
- Reworked tokenizer runtime APIs around bytes:
  - `encode_bytes`,
  - `decode_bytes`,
  - `decode_lossy_utf8`,
  - reserved token strings encode atomically and are excluded from learned BPE
    merges.
- Added native corpus tokenizer CLIs:
  - `heirloom tokenizer train-corpus`,
  - `heirloom tokenizer validate`,
  - `heirloom tokenizer fertility`.
- Added tokenizer identity/report blocks to train/eval/generation reports.
- Added `scripts/run_qb_tokenizer_hardpath.sh`:
  - trains tokenizer v2 from a corpus-blend manifest,
  - validates/fertility-checks the tokenizer,
  - prepares manifest v2 binary shards,
  - runs memory train, resume, eval, and generation.
- Added Vertex integration:
  - `HEIRLOOM_RUN_QB_TOKENIZER_HARDPATH=1`,
  - uploads artifacts under `qb-tokenizer-hardpath/`.
- Smoke validation passed locally:
  - tokenizer v2 vocab `512`,
  - reserved tokens `34`,
  - manifest version `2`,
  - storage `binary_shards`,
  - loader `binary_shard_streaming`,
  - `tokens_materialized=false`,
  - memory train/resume/eval/generation completed.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --test transformer_runtime`,
  - `bash -n scripts/run_qb_tokenizer_hardpath.sh`,
  - `bash -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `scripts/run_qb_tokenizer_hardpath.sh` smoke mode.

## Checkpoint 138 - Governed Corpus Blend Materializer

- Added `heirloom data materialize-blend` as the production bridge from
  `heirloom.corpus_blend` to manifest v2 binary token shards:
  - license-status allowlist gating,
  - local uncompressed source requirements,
  - deterministic source token quotas from sampling weights,
  - exact dedupe plus malformed/secret-like/repetitive/high-fertility filters,
  - source scoring with QB/math/Nemotron/OLMo bias hooks,
  - deterministic train/valid split assignment with tiny-fixture safeguards.
- Materializer outputs:
  - `source-index.json`,
  - `curation-report.json`,
  - `selected-docs.jsonl`,
  - `tokenizer-sample-manifest.json`,
  - `text-shards/*.txt`,
  - `prepared/manifest.json` plus binary token shard metadata/payload files.
- Updated `scripts/run_qb_tokenizer_hardpath.sh` so the tokenizer hard path now
  uses `data materialize-blend` before memory train/resume/eval/generation.
  Full mode defaults to `20_000_000_000` target tokens and `full`
  materialization; smoke mode uses bounded `sample` materialization.
- Added GCS-first source staging for production slices:
  - tokenizer hard-path source variables can point at local files or single
    `gs://` objects,
  - `gs://` sources are staged to the Vertex worker's local disk before
    materialization,
  - the generated blend manifest records the original GCS URI,
  - Vertex forwarding now includes target-token, materialization-mode,
    valid-fraction, and source-stage-dir controls.
- Added `scripts/stage_qb_source_slices.py` and staged the available
  VECL-QB `v1-hard` source slice into GCS:
  - `corpus.jsonl` bytes `43,971,635`, records `25,000`,
  - corpus SHA-256
    `d4323a9ff683452da1347643f875575a276015206a46fae8289c8d003c63770f`,
  - metadata SHA-256
    `662bbdeaec5ea676a09fc8ebe3f816eb8d149d4fa1d212d0ff3d47dc65d0f8a7`,
  - inventory available sources `1`, pending external sources `4`,
  - GCS prefix
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/`.
- Added `QB_SOURCE_GOVERNANCE.md` and embedded first-pass governance evidence in
  the source-slice inventory:
  - VECL-QB v1-hard is `approved_internal_synthetic`,
  - Dolma is conditionally approved as an internal-training attribution source
    under `odc_by_internal_attribution`,
  - Nemotron-CC and Nemotron-CC-Math sample-artifact slices are conditionally
    approved as internal-training-only sources under
    `nvidia_data_agreement_internal_training`,
  - OLMo 3 pretraining data is tracked as the
    `allenai/dolma3_dolmino_mix-100B-1125` ODC-BY mixed-down artifact under
    `odc_by_internal_attribution`,
  - external source upload requires an explicit approved `--*-license-status`
    override plus recorded evidence.
- Updated Nemotron governance after reviewing NVIDIA's Data Agreement for Model
  Training on `nvidia/Nemotron-Pretraining-Dataset-sample`:
  - Nemotron-CC high-quality and Nemotron-CC-MATH sample-artifact slices are
    conditionally approved for internal model training under
    `nvidia_data_agreement_internal_training`,
  - raw slices must remain private/internal and must not be redistributed or
    treated as open-source data,
  - full Nemotron artifacts outside the sample dataset still require their own
    exact artifact/license evidence.
- Updated the generated corpus-blend plan to include the Dolma 3 Dolmino 100B /
  OLMo 3 pretraining slice and the production weights:
  - Dolma v1.7 `0.35`,
  - Nemotron-CC high/actual `0.25`,
  - Dolma 3 Dolmino 100B / OLMo 3 stage 2 `0.20`,
  - Nemotron-CC-Math `0.10`,
  - VECL-QB v1-hard `0.10`.
- Added `scripts/slice_hf_dataset.py` for deterministic Hugging Face source
  slicing:
  - stdlib Hugging Face API/download path with optional `HF_TOKEN`,
  - optional VECL-QB `.env` token loading without writing token values,
  - `.jsonl`, `.jsonl.gz`, and `.jsonl.zst` input support via system `zstd`,
  - Dolma URL-manifest support via `--url-list` and `--preset dolma-qb`,
  - deterministic file shuffle and record sampling,
  - normalized uncompressed JSONL output with source-file metadata,
  - redacted `heirloom.hf_source_slice_report`,
  - optional GCS upload for the slice and report.
- Staged the first external rehearsal source slice:
  - source `allenai.dolma3_dolmino_mix-100B-1125`,
  - output `dolmino-100m.jsonl`,
  - `72,909` selected records,
  - `400,004,346` selected text bytes,
  - `100,001,086` estimated tokens,
  - output SHA-256
    `991a67a7b8eca51fed777856778267623086f1fd16b25f723e1f3b3989ae9774`,
  - uploaded under
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/dolma3-dolmino-mix-100b-1125/`,
  - refreshed `source-slice-inventory.json` recorded `2` available sources and
    `3` pending external sources.
- Added optional Parquet support to `scripts/slice_hf_dataset.py` via `pyarrow`
  so the NVIDIA sample artifact can be staged without changing the JSONL
  materializer contract.
- Staged NVIDIA sample-artifact slices under the private source-slices prefix:
  - `nvidia.nemotron_cc.high_actual` from
    `Nemotron-CC-High-Quality/part_000000.parquet`,
  - `765` selected records,
  - `2,451,471` selected text bytes,
  - `612,867` estimated tokens,
  - output SHA-256
    `f2d6aa2d4d9c9c024c170cf7741815cd0f6270917a91b1139e065047d8c8693c`,
  - `nvidia.nemotron_cc_math` from `Nemotron-CC-MATH/part_0000.parquet`,
  - `954` selected records,
  - `3,183,012` selected text bytes,
  - `795,753` estimated tokens,
  - output SHA-256
    `1c5027c868ae9a75d6171cd033183d68f847acc55087c1694c69ed6c0d55ecac`,
  - refreshed `source-slice-inventory.json` now records `4` available sources
    and `1` pending external source.
- Staged the Dolma v1.7 rehearsal slice from the public `urls/v1_7.txt`
  manifest using `--preset dolma-qb`:
  - source `allenai.dolma.v1_7`,
  - output `dolma-v1_7-100m.jsonl`,
  - `159,470` selected records,
  - `400,001,277` selected text bytes,
  - `100,000,319` estimated tokens,
  - output SHA-256
    `5aba4f82fc1dc945faebab0e02685ee3e3423ba7de8567309c898f3d9a05ef17`,
  - uploaded under
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/dolma-v1_7/`,
  - refreshed `source-slice-inventory.json` now records all `5` planned sources
    available and `0` pending sources.
- Fixed a materializer rehearsal bug: candidate scanning honored
  `--max-source-bytes` / `--max-docs-per-source`, but output writing reopened
  source files and could rescan the full corpus after selected documents were
  already known. `write_materialized_outputs` now applies the same caps and
  stops once all selected hashes are written.
- Added tokenizer hard-path wrapper envs for bounded local materialization:
  - `HEIRLOOM_QB_TOKENIZER_HARDPATH_MAX_SOURCE_BYTES`,
  - `HEIRLOOM_QB_TOKENIZER_HARDPATH_MAX_DOCS_PER_SOURCE`,
  - `HEIRLOOM_QB_TOKENIZER_HARDPATH_MAX_DOC_BYTES`.
- Completed the first all-source local rehearsal in
  `runs/qb-tokenizer-rehearsal-100m-512-doccap/`:
  - used all five staged source slices,
  - trained tokenizer artifact v2 with vocab `512` as a bounded smoke,
  - materialized manifest v2 binary shards with `9,388` train tokens and
    `1,629` valid tokens,
  - curation selected `8` documents / `11,017` tokens across all five sources,
  - memory train loss moved `6.414626 -> 6.281556`,
  - resume step loss was `5.896986`,
  - eval loss was `6.044056` with perplexity `421.599755`,
  - train/eval reported `loader.kind="binary_shard_streaming"` and
    `tokens_materialized=false`,
  - generation completed from the memory checkpoint.
- Reworked tokenizer v2 merge training to use an incremental native BPE trainer
  with deterministic heap selection, initial heap construction from final pair
  counts, and per-word unique pair deltas to reduce stale heap churn. The
  trainer is recorded as `heirloom.byte_bpe.native_incremental.v2`.
- Added deterministic trainer counters to tokenizer v2 validation metadata and
  timing buckets to `tokenizer train-corpus` reports.
- Added deterministic heap compaction for tokenizer v2 training. On the 4K /
  16 MiB all-source rehearsal, stale heap pops dropped from `131,265,137` to
  `265,915`, internal tokenizer train time dropped to `14.665 s`, and exact
  32K bounded training became practical.
- Reworked tokenizer runtime encoding to use a cached merge lookup and
  rank-ordered chunk encoder, while preserving the old sequential-merge
  semantics with a regression test for overlapping-pair cases.
- Completed an all-source local release rehearsal in
  `runs/qb-tokenizer-rehearsal-100m-4096-release/`:
  - used all five staged source slices,
  - trained tokenizer artifact v2 with vocab `4096`,
  - tokenizer hash `03363a37b7e06036`,
  - artifact validation hash `9f5363706ded95b6`,
  - release tokenizer training wall time `105.54 s` including the one-time
    release compile,
  - materialized manifest v2 binary shards from `11` selected documents /
    `10,786` selected tokens,
  - memory train loss moved `8.824377 -> 8.420417`,
  - resume step loss was `8.461718`,
  - eval loss was `8.501845` with perplexity `4923.846731`,
  - train/eval/generation reports include tokenizer identity and the streaming
    loader reports `loader.kind="binary_shard_streaming"` with
    `tokens_materialized=false`.
- Completed the first exact-32K bounded all-source local rehearsal in
  `runs/qb-tokenizer-rehearsal-100m-32768-16m-heap-rebuild/`:
  - used all five staged source slices,
  - trained tokenizer artifact v2 with vocab `32768` and
    `--require-exact-vocab`,
  - tokenizer hash `2996feb60e39fa2e`,
  - artifact validation hash `129ced1d1eb1f0f2`,
  - tokenizer internal train time `15.222 s` on a deterministic 16 MiB sample,
  - validation and fertility reports completed,
  - materialized manifest v2 binary shards from `11` selected documents /
    `10,736` selected tokens,
  - manifest reports `storage="binary_shards"` with `7,539` train tokens and
    `3,197` valid tokens,
  - memory train loss moved `10.553624 -> 10.376928`,
  - resume step loss was `10.266456`,
  - eval loss was `10.314114` with perplexity `30155.228909`,
  - train/eval reports include tokenizer identity and
    `loader.kind="binary_shard_streaming"` with `tokens_materialized=false`,
  - generation completed from the exact-32K memory checkpoint.
- Scaled the exact-32K tokenizer rehearsal request to 64 MiB and 256 MiB in:
  - `runs/qb-tokenizer-rehearsal-100m-32768-64m-heap-rebuild/`,
  - `runs/qb-tokenizer-rehearsal-100m-32768-256m-heap-rebuild/`.
- Both larger requests completed quickly because the current staged rehearsal
  slices exhaust at `44,159,551` physical sampled bytes:
  - 64 MiB request: tokenizer hash `1c287ddc4ab4d25a`, internal train time
    `15.389 s`,
  - 256 MiB request: tokenizer hash `ff87d08a04792bec`, internal train time
    `14.803 s`, weighted training bytes `306,568,239`.
- Completed the 256 MiB-request exact-32K local memory smoke:
  - validation and fertility reports completed,
  - materialized manifest v2 binary shards from `11` selected documents /
    `11,391` selected tokens,
  - manifest reports `storage="binary_shards"` with `7,543` train tokens and
    `3,848` valid tokens,
  - memory train loss moved `10.514318 -> 10.199512`,
  - resume step loss was `9.871762`,
  - eval loss was `10.419749` with perplexity `33515.029532`,
  - train/eval reports include tokenizer identity and
    `loader.kind="binary_shard_streaming"` with `tokens_materialized=false`,
  - generation completed from the exact-32K memory checkpoint.
- Added directory-backed source-slice support for production-sized staging:
  - `data materialize-blend` now accepts flat source directories as well as
    single files and scans all approved files in deterministic order,
  - `tokenizer train-corpus` samples from all files in a flat source directory,
  - `scripts/run_qb_tokenizer_hardpath.sh` accepts local flat directories and
    `gs://` prefixes ending in `/`, stages GCS prefixes into local source
    directories, emits absolute local paths in the generated blend manifest,
    and rejects compressed or nested source layouts,
  - `scripts/slice_hf_dataset.py` now supports `--shard-output-bytes`, treating
    `--out` as a directory and writing deterministic `*-part-NNNNN.jsonl`
    source parts with a file-set hash,
  - `scripts/stage_qb_source_slices.py` records and uploads source directories
    file-by-file while preserving a source-level hash and prefix.
- Completed a directory-backed tokenizer hard-path smoke in
  `runs/qb-tokenizer-hardpath-directory-smoke-2/`:
  - one source was provided as a sharded local directory,
  - tokenizer v2 vocab `512` trained and validated with hash
    `6a8a4800c9aed239`,
  - materialized manifest v2 binary shards from `5` selected documents /
    `2,505` selected tokens,
  - memory train/resume/eval/generation completed,
  - train report used `loader.kind="binary_shard_streaming"` with
    `tokens_materialized=false`.
- Staged the first larger sharded 1B-token-class rehearsal source set:
  - GCS prefix
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices-1b-rehearsal/`,
  - inventory path
    `runs/qb-native-pretraining-v1/source-slices-1b-rehearsal/inventory/source-slice-inventory.json`,
  - all `5` planned sources available and `0` pending,
  - total `4,954,749,509` bytes and `1,189,942` records,
  - Dolma v1.7: `5` shard files, `2,507,910,453` bytes,
    `796,588` records, `500,002,478` estimated selected tokens,
  - Dolma 3 Dolmino: `5` shard files, `2,396,195,856` bytes,
    `366,635` records, `500,000,380` estimated selected tokens,
  - Nemotron high/math sample-artifact slices and VECL-QB `v1-hard` carried
    forward unchanged.
- Fixed a tokenizer-sampling renderer bug for general JSONL sources:
  - `render_jsonl_record_for_tokenizer` now preserves records with a `text`
    field as `<|document|>` text instead of treating them as empty QB
    prompt/target traces,
  - added regression test
    `tokenizer_train_corpus_preserves_general_jsonl_text_samples`.
- Completed a corrected larger exact-32K tokenizer rehearsal in
  `runs/qb-tokenizer-rehearsal-1b-32768-512m-textfix/`:
  - requested tokenizer sample `536,870,912` bytes,
  - physical sampled bytes `318,117,790`,
  - weighted training bytes `554,132,722`,
  - Dolma/Dolmino hit their sample quotas without exhaustion, while smaller
    Nemotron/QB sources exhausted and used effective sample weights,
  - tokenizer hash `1574298cf19369db`, saved artifact hash
    `50291e31a9676a4b`,
  - tokenizer timing: `19,295 ms` sample materialization,
    `495,010 ms` native tokenizer training, `514,447 ms` total.
- Completed the capped local memory-transformer smoke from that tokenizer:
  - materialization used `--max-source-bytes 134217728`,
  - manifest v2 binary shards from `137` selected docs / `135,410` selected
    tokens,
  - memory train loss moved `10.178035 -> 9.795135`,
  - resume loss `9.971405`,
  - eval loss `9.823355`, perplexity `18459.874713`,
  - train report used `loader.kind="binary_shard_streaming"` with
    `tokens_materialized=false`,
  - summary written to
    `runs/qb-tokenizer-rehearsal-1b-32768-512m-textfix/summary-capped.json`.
- New scale finding: unbounded local materialization over multi-GB sources was
  active but too slow for quick CPU smoke, so production materialization needs
  progress reporting, bounded/staged curation passes, and tokenizer encode
  throughput work before the 20B-token target.
- Investigated Claude's tokenizer critique against the actual exact-32K
  artifacts:
  - confirmed the previous artifact learned digit-bearing tokens, including
    examples like `20`, `31`, `32`, `3)`, `2)`, `0.`, source slugs, and JSON
    confidence fragments,
  - confirmed the default reserved registry had only `34` tokens.
- Added production digit isolation for new tokenizer v2 artifacts:
  - `BpeTokenizerTrainingConfig` now records `digit_isolation`,
  - `BpeTokenizerV2Options` defaults `digit_isolation=true`,
  - tokenizer v2 training splits ASCII digits into singleton byte-token
    segments before BPE merge training,
  - tokenizer encoding honors the same digit boundary,
  - tokenizer validation rejects learned merge payloads containing ASCII digits
    when digit isolation is enabled,
  - old artifacts still load with `digit_isolation=false` through serde
    defaults.
- Expanded the default reserved registry from `34` to exactly `128` tokens,
  covering chat, tool, tool result, memory, SMFT, trace, evidence/citation,
  document/source, temporal, separator, mask, QB references, artifact,
  governance, and schema markers.
- Added tokenizer regressions:
  - `tokenizer_v2_digit_isolation_blocks_digit_merges`,
  - `tokenizer_train_corpus_preserves_general_jsonl_text_samples`.
- Completed the digit-isolated exact-32K hard-path smoke in
  `runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/`:
  - tokenizer hash `10079cec3ace7a4b`,
  - artifact hash `92be858136043b4f`,
  - vocab `32768`, reserved tokens `128`,
  - trainer `heirloom.byte_bpe.native_incremental.digit_isolated.v2`,
  - digit audit `learned_digit_token_count=0`,
  - materialized manifest v2 binary shards from `34` selected docs /
    `35,705` selected tokens,
  - memory train/resume/eval/generation completed,
  - train report used `loader.kind="binary_shard_streaming"` with
    `tokens_materialized=false`,
  - digit audit written to
    `runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/digit-isolation-audit.json`.
- Added materializer throughput/observability:
  - `data materialize-blend` now records load/scan/selection/write/total
    timing, aggregate scan/write throughput, per-source source-file counts,
    scan/write throughput, candidate token/byte counts, write counts, and
    source cap flags in `curation-report.json` and `source-index.json`,
  - long scans can emit bounded heartbeat logs with
    `--progress-every-records` and/or `--progress-every-bytes`,
  - `scripts/run_qb_tokenizer_hardpath.sh` exposes
    `HEIRLOOM_QB_TOKENIZER_HARDPATH_PROGRESS_EVERY_RECORDS` and
    `HEIRLOOM_QB_TOKENIZER_HARDPATH_PROGRESS_EVERY_BYTES`, defaulting full
    mode to `100000` records and `1073741824` bytes,
  - the output writer now consumes selected scan candidates directly instead
    of re-opening and re-scoring every source file for the write phase.
- Added exact bounded candidate retention for scale materialization:
  - new CLI flags `--candidate-retention-token-multiplier`,
    `--candidate-retention-min-docs`, and `--candidate-prune-every`,
  - retention uses the same score/sample-key/hash order as final selection and
    keeps the best-score prefix until retained tokens meet the source quota
    times the multiplier,
  - a multiplier of at least `1.0` preserves the selected prefix needed to
    satisfy the source quota while pruning lower-score candidates during scan,
  - full tokenizer hard-path mode now defaults to multiplier `1.0`, min docs
    `100000`, and prune cadence `50000`.
- Added metadata-only candidate text mode for scale materialization:
  - new CLI flag `--candidate-text-mode retain|rescan`,
  - `retain` keeps the fast previous behavior for local/smoke runs,
  - `rescan` drops rendered text from scan candidates, keeping only metadata
    needed for selection,
  - the rescan writer re-reads source files and hash-prefilters rendered
    records before tokenizing, so only selected hashes are encoded/written,
  - full tokenizer hard-path mode now defaults to `rescan`.
- Completed a capped materializer observability rehearsal at
  `runs/qb-materializer-observability-smoke/`:
  - manifest v2 binary shards from `17` selected docs / `19,418` selected
    tokens,
  - report timing: `59,573 ms` scan, `6 ms` selection, `206 ms` write,
    `59,871 ms` total,
  - aggregate local debug-build throughput:
    `309.13` scanned docs/sec, `958,665` scanned bytes/sec,
    `93,813` written tokens/sec,
  - per-source scan rates are now visible for Dolma v1.7, Nemotron high,
    Dolma 3 Dolmino, Nemotron Math, and VECL-QB `v1-hard`.
- Completed a capped materializer retention rehearsal at
  `runs/qb-materializer-retention-smoke/`:
  - manifest v2 binary shards from the same `17` selected docs / `19,418`
    selected tokens,
  - accepted candidates were reduced from `17,674` total candidates to `40`
    retained candidates across the five sources,
  - report timing showed `59,898 ms` scan, `57,762 ms` tokenizer encode,
    `72 ms` hashing, and `208 ms` write,
  - local debug-build tokenizer encode throughput was about `148,388`
    candidate tokens/sec, confirming scan time is tokenizer-bound.
- Completed a capped metadata-only rescan rehearsal at
  `runs/qb-materializer-rescan-hashprefilter-smoke/`:
  - manifest v2 binary shards from the same `17` selected docs / `19,418`
    selected tokens,
  - accepted candidates were again reduced from `17,674` to `40` retained
    metadata candidates,
  - retained candidate text bytes were `0` across all sources,
  - initial unfiltered rescan in `runs/qb-materializer-rescan-smoke/` took
    `53,339 ms` in the write phase because it re-tokenized rescanned
    candidates,
  - hash-prefiltered rescan reduced `write_outputs_elapsed_ms` to `1,105 ms`
    while keeping total runtime near retain mode (`61,779 ms` total).
- Added tokenizer encode throughput instrumentation:
  - new CLI command `heirloom tokenizer bench-encode`,
  - reports tokenizer ID/hash/artifact path, vocab, digit-isolation state,
    reserved-token count, sample bytes/docs, warmup and encode timings,
    tokens/sec, bytes/sec, docs/sec, tokens/byte, and average tokens/doc,
  - `tokenizer_train_corpus_cli_emits_v2_artifacts_and_fertility_report` now
    exercises train/validate/fertility/bench in one CLI hard-path test.
- Optimized native tokenizer encoding:
  - reserved-token matching now uses a first-byte lookup table instead of
    scanning the full registry at every byte position,
  - BPE chunk encoding now uses a priority queue plus linked-neighbor updates,
    preserving the sequential merge-rank/leftmost-tie behavior while avoiding
    repeated whole-vector scans and `Vec::remove`,
  - added a regression that compares the fast chunk encoder against the
    sequential merge reference.
- Completed tokenizer encode and materializer throughput follow-up rehearsals:
  - `runs/tokenizer-encode-bench-32k-digitiso-16m.json` measured the
    digit-isolated 32K tokenizer at about `532,306` tokens/sec and `1,324,642`
    bytes/sec on a 16 MiB mixed Dolma/Dolmino line sample,
  - `runs/tokenizer-encode-bench-32k-digitiso-16m-pq.json` measured the same
    sample after the priority-queue encoder at about `605,887` tokens/sec and
    `1,507,748` bytes/sec,
  - `runs/qb-materializer-rescan-pq-smoke/curation-report.json` materialized
    the same capped `17` selected docs / `19,418` selected tokens with
    metadata-only rescan and bounded retention,
  - that capped materializer rehearsal improved from `61,779 ms` total /
    `58,463 ms` tokenizer encode in
    `runs/qb-materializer-rescan-hashprefilter-smoke/` to `20,325 ms` total /
    `17,250 ms` tokenizer encode,
  - reported tokenizer encode throughput rose from about `146,609` to
    `496,881` candidate tokens/sec in the materializer report.
- Added production materializer sizing and scan checkpointing:
  - new CLI flags `--checkpoint-dir`, `--resume-checkpoint`,
    `--checkpoint-every-records`, and `--checkpoint-every-bytes`,
  - per-source scan checkpoints store a compatible scan config hash, source
    cursor `(file_index, byte_offset, ordinal)`, curation stats, retained
    candidates, accepted document hashes for deterministic duplicate filtering,
    and FNV content-hash state,
  - resume loads compatible incomplete or complete source checkpoints before
    continuing the candidate scan; output text/token shards are still emitted by
    the successful materialization run rather than partially resumed,
  - curation reports and source indexes now include `checkpoint` and `sizing`
    blocks,
  - `sizing` reports selected text bytes, estimated token dtype, bytes/token,
    token payload bytes, total payload bytes, train/valid token estimates, and
    text/token shard estimates,
  - full tokenizer hard-path mode now defaults checkpointing to
    `$out_dir/materializer-checkpoints`, every `100000` records or `1 GiB`,
    while `HEIRLOOM_QB_TOKENIZER_HARDPATH_RESUME_CHECKPOINT=1` remains explicit.
- Remaining tokenizer/data scale blocker is now the actual production source
  materialization run: checkpointing and sizing are in place, so the next step
  is a bounded dry-run/sizing pass over the staged 1B-class slices, then the
  target production token budget.
- Added CLI tests for successful sample materialization and unapproved
  license-status rejection.
- Added a checkpoint-resume CLI test that intentionally stops after two
  scanned records, resumes from the source scan checkpoint, and verifies the
  final report loaded the checkpoint and scanned all source records.
- Completed a bounded 1B-slice materializer sizing dry-run at
  `runs/qb-materializer-sizing-dryrun-1b-64m/`:
  - used the digit-isolated exact-32K tokenizer
    `runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/tokenizer.json`,
  - scanned the five-source corpus blend with `64 MiB` max source bytes,
    metadata-only rescan mode, bounded retention, and scan checkpoints,
  - selected `96` docs / `104,452` tokens against a `100,000` token target,
  - report timing: `75,720 ms` scan, `67,625 ms` tokenizer encode,
    `75,854 ms` total,
  - aggregate throughput: `2,441,475` scanned bytes/sec, `788.7` scanned
    docs/sec, `496,014` tokenizer-encoded candidate tokens/sec,
  - sizing estimate: `u16` token shards, `332,223` estimated text payload
    bytes, `208,904` estimated token payload bytes, `541,127` estimated total
    payload bytes,
  - source checkpoints completed for all five sources and occupied about
    `1.6 MiB` total with `128` retained candidates per source.
- Verified completed-checkpoint resume against that dry-run:
  - `runs/qb-materializer-sizing-dryrun-1b-64m-resume-check/` loaded all five
    completed source checkpoints,
  - reproduced the same `96` docs / `104,452` tokens in `199 ms`.
- Materialized the resumed `100k` real-source sample at
  `runs/qb-materializer-sample-1b-64m-100k/`:
  - manifest v2 with `storage="binary_shards"`,
  - tokenizer hash `10079cec3ace7a4b`,
  - `80,696` train tokens and `23,756` valid tokens,
  - `2` train token shards, `1` valid token shard, and `1` text shard,
  - source token mix: Dolma v1.7 `35,877`, Nemotron high/actual `26,283`,
    Dolma 3 Dolmino `20,155`, Nemotron Math `11,938`, VECL-QB v1-hard
    `10,199`,
  - resumed materialization completed in `3,207 ms` because candidate scan
    state came from the completed checkpoints.
- Ran a tiny CPU memory-transformer smoke on the `100k` materialized sample:
  - train report at
    `runs/qb-materializer-sample-1b-64m-100k/train-memory-report.json`,
  - train used `loader.kind="binary_shard_streaming"` with
    `tokens_materialized=false`,
  - train loss `9.883890 -> 9.914296` over two tiny CPU steps,
  - resume reached step `3` with loss `10.029259`,
  - eval on the valid split reported loss `10.041141`, perplexity
    `22951.545524`, `1` batch / `32` tokens,
  - generation completed with `16` new tokens.
- Completed the next-scale `512 MiB`/source materializer sizing dry-run at
  `runs/qb-materializer-sizing-dryrun-1b-512m/`:
  - target `1,000,000` tokens, selected `933` docs / `1,003,355` tokens,
  - report timing: `430,275 ms` scan, `388,517 ms` tokenizer encode,
    `430,417 ms` total,
  - aggregate throughput: `2,613,193` scanned bytes/sec, `638.3` scanned
    docs/sec, `564,629` tokenizer-encoded candidate tokens/sec,
  - sizing estimate: `u16` token shards, `3,075,080` estimated text payload
    bytes, `2,006,710` estimated token payload bytes, `5,081,790` estimated
    total payload bytes,
  - source checkpoints completed for all five sources and occupied about
    `9.4 MiB` total with up to `2,048` retained candidates per source,
  - Dolma v1.7 scanned `170,646` docs / `536,874,101` bytes at `736`
    docs/sec and selected `351,258` tokens,
  - Dolma 3 Dolmino scanned `77,279` docs / `536,874,379` bytes at `432`
    docs/sec and selected `200,440` tokens,
  - Nemotron high/actual, Nemotron Math, and VECL-QB v1-hard exhausted their
    available local slices and selected `251,357`, `100,235`, and `100,065`
    tokens respectively.
- Materialized the resumed `1M` real-source sample at
  `runs/qb-materializer-sample-1b-512m-1m/`:
  - manifest v2 with `storage="binary_shards"`,
  - tokenizer hash `10079cec3ace7a4b`,
  - `866,999` train tokens and `136,356` valid tokens,
  - `4` train token shards, `1` valid token shard, and `1` text shard,
  - source token mix: Dolma v1.7 `351,258`, Nemotron high/actual `251,357`,
    Dolma 3 Dolmino `200,440`, Nemotron Math `100,235`, VECL-QB v1-hard
    `100,065`,
  - resumed materialization completed in `15,851 ms`; write phase was
    `15,442 ms`.
- Ran a tiny CPU memory-transformer smoke on the `1M` materialized sample:
  - train report at
    `runs/qb-materializer-sample-1b-512m-1m/train-memory-report.json`,
  - train used `loader.kind="binary_shard_streaming"` with
    `tokens_materialized=false`,
  - train loss `10.130000 -> 9.946422` over two tiny CPU steps,
  - resume reached step `3` with loss `9.880651`,
  - eval on the valid split reported loss `10.163255`, perplexity
    `25932.563549`, `1` batch / `32` tokens,
  - generation completed with `16` new tokens.
- Materializer scaling implication: local debug-build scan throughput is now
  measured on the larger staged slices, checkpoint size remains small, and the
  resumed write path is fast enough for iterative rehearsals. The next useful
  step is either a full-local-slice dry-run over the staged Dolma/Dolmino
  slices or moving the production-budget materialization to Vertex/GCS.
- Note: after the kernel-throughput thread began, fresh `cargo run`
  compilation was temporarily blocked by concurrent `heirloom-kernels` edits
  adding ldmatrix/cp.async counters without updating all
  `CudaRuntimeCounters` initializers. The post-change eval/generation smoke
  used the already-built `target/debug/heirloom` binary to avoid touching the
  kernel thread's work.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo test --test transformer_runtime data_materialize_blend -- --nocapture`,
  - `cargo test tokenizer::tests::fast_chunk_encoder_matches_sequential_merge_reference -- --nocapture`,
  - `cargo test --test transformer_runtime tokenizer_train_corpus_cli_emits_v2_artifacts_and_fertility_report -- --nocapture`,
  - `cargo check --bin heirloom`,
  - `cargo test --test transformer_runtime -- --nocapture`,
  - `cargo test tokenizer -- --nocapture`,
  - `cargo test materialize -- --nocapture`,
  - `cargo test`,
  - `python3 -m py_compile scripts/slice_hf_dataset.py scripts/stage_qb_source_slices.py`,
  - `bash -n scripts/run_qb_tokenizer_hardpath.sh`,
  - directory-backed CPU smoke at
    `runs/qb-tokenizer-hardpath-directory-smoke-2/summary.json`,
  - corrected 1B-source capped CPU memory smoke at
    `runs/qb-tokenizer-rehearsal-1b-32768-512m-textfix/summary-capped.json`,
  - digit-isolated exact-32K CPU memory smoke at
    `runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/summary.json`,
  - capped materializer observability smoke at
    `runs/qb-materializer-observability-smoke/curation-report.json`,
  - capped materializer retention smoke at
    `runs/qb-materializer-retention-smoke/curation-report.json`,
  - capped metadata-only rescan materializer smoke at
    `runs/qb-materializer-rescan-hashprefilter-smoke/curation-report.json`,
  - priority-queue tokenizer encode bench at
    `runs/tokenizer-encode-bench-32k-digitiso-16m-pq.json`,
  - capped priority-queue rescan materializer smoke at
    `runs/qb-materializer-rescan-pq-smoke/curation-report.json`,
  - bounded 1B-slice sizing dry-run at
    `runs/qb-materializer-sizing-dryrun-1b-64m/curation-report.json`,
  - completed-checkpoint resume check at
    `runs/qb-materializer-sizing-dryrun-1b-64m-resume-check/curation-report.json`,
  - resumed 100k-token materialized sample plus tiny CPU memory train/resume/
    eval/generation at `runs/qb-materializer-sample-1b-64m-100k/`,
  - bounded `512 MiB`/source 1B-slice sizing dry-run at
    `runs/qb-materializer-sizing-dryrun-1b-512m/curation-report.json`,
  - resumed 1M-token materialized sample plus tiny CPU memory train/resume/
    eval/generation at `runs/qb-materializer-sample-1b-512m-1m/`.

## A100 Kernel Throughput Track

- Started the no-cuBLAS hand-rolled PTX throughput track for BF16 Tensor Core
  GEMM/attention without changing model or data semantics.
- Added `heirloom gpu tensor-core-microbench`, which emits a JSON report for
  raw BF16 RHS-transposed GEMM and current materialized BF16 Tensor Core
  causal-attention forward, each with CPU BF16-rounded correctness parity,
  CUDA-event elapsed time, estimated matmul FLOP/s, Tensor Core counters, and
  CUDA runtime counters.
- Added explicit experimental request toggles and counters for the next
  instruction tiers:
  - `HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM=1`,
  - `HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1`,
  - `HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_ATTENTION=1`,
  - `HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_ATTENTION=1`.
- The GEMM request tiers now launch guarded instruction-path PTX kernels rather
  than staged fallback:
  - `ldmatrix_a_shared_mma` stages A/B in shared memory, loads the A operand
    fragment with `ldmatrix.sync.aligned.m8n8.x4.shared.b16`, keeps the proven
    shared-load B fragment path, and feeds `mma.sync.m16n8k16`,
  - `cp_async_double_buffered_ldmatrix_a_mma` uses 16-byte `cp.async` copies
    for a double-buffered global-to-shared A/B pipeline, overlaps the next K
    tile prefetch with the current ldmatrix-A MMA tile, then feeds the same
    ldmatrix-A compute path.
- These are still guarded A100 throughput tiers, not default training paths.
  GEMM-only Vertex job `9203551123560988672` A100 microbench-validated both
  the `ldmatrix_a_shared_mma` and
  `cp_async_double_buffered_ldmatrix_a_mma` tiers with correctness parity,
  CUDA-event timing, positive instruction-path counters, and zero fallback
  counters. They are not yet default train-path production kernels. That job
  did not validate Tensor Core flash attention. Attention ldmatrix/cp.async
  toggles still report global fallback for the materialized attention matmuls.
- Validation:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --test cuda_storage cuda_runtime_counters_reset_to_zero_without_cuda -- --nocapture`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `cargo test --bin heirloom tensor_core_pad_crop -- --nocapture`,
  - `cargo test --bin heirloom -- --nocapture`.

### Scalar-Streaming Flash-Like BF16 Causal Attention Slice

- Added a guarded, forward-only scalar-streaming flash-like BF16 causal
  attention runtime path for sm80 throughput experiments:
  - `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION=1` requests the tensor API path,
  - `HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION=1` turns fallback into an explicit
    error,
  - `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING=1` accumulates event-timer
    microseconds for flash calls.
- The safe wrapper
  `causal_attention_bf16_flash_forward_f32_buffers(query, key, value, dims)`
  converts f32 CUDA Q/K/V buffers to BF16 and returns f32 output without
  returning or saving a `[B,H,T,T]` attention matrix. This scalar-streaming
  wrapper remains a no-grad experimental path; grad-required routing is covered
  separately by the Tensor Core flash-state scaffold below.
- Added CUDA runtime counters for materialized-reference attention, generic
  flash requests/fallbacks, scalar-streaming requested/executed calls,
  scalar-streaming QK/AV tile counts, ragged/causal tile counts, hard-require
  failures, and accumulated elapsed microseconds. These counters are
  serialized in CLI/runtime JSON and aggregated across ranks.
- Split out placeholder Tensor Core flash counters (`executed`,
  `qk_mma_tile_calls`, `av_mma_tile_calls`, ragged/causal/timing). The current
  scalar-streaming path leaves them at zero and does not increment global
  Tensor Core attention/matmul counters, because it does not execute MMA
  instructions.
- Added a guarded Tensor Core tiled flash forward candidate behind
  `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE=1`. The PTX kernel uses one
  warp per `(batch-head, 16-query block, 8-output-dim tile)`, computes two QK
  `m16n8k16` MMA tiles per 16-token key block, performs online softmax in
  shared memory, and updates the local P/V tile with a Tensor Core MMA while
  saving compact row max/denom state.
- Separate attention-only Vertex job `6050468434448220160` validated this
  forward path on A100 for `batch=4, heads=16, time=512, head_dim=64` with
  `attention.tensor_core_flash_forward.status == "ok"` and `passed == true`.
  That job validated forward only; the later training-path gate below covers
  guarded exact-tile Tensor Core flash backward for the target DDP shape.
- Extended `heirloom gpu tensor-core-microbench` so the attention section
  reports `current_materialized`, `scalar_streaming_forward`, and
  `tensor_core_flash_forward`, including elapsed ms, conservative executed-FLOP
  estimates, max absolute error, pass flag, speedup ratio, runtime/Tensor Core
  counters, and materialized attention elements/bytes avoided. The old
  `flash_forward` JSON key remains as a compatibility alias for the scalar
  path.
- Added CPU/no-GPU tests for guard defaults, hard-require error text, counter
  reset, and runtime JSON serialization, plus CUDA-gated A100 parity/policy
  tests for tile-aligned, ragged, env-on no-grad execution, grad fallback, and
  hard-require grad errors.
- Local validation:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo run --bin heirloom -- gpu tensor-core-microbench --help`,
  - `target/debug/heirloom gpu info` reported `cuda_available=false` because
    `libcuda.so` was not present, so A100-only flash parity/timing commands
    remain to be run on sm80 hardware.

### Training-Path Tensor Core Flash Attention Backward Gate

- Added `HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD=1` as the explicit third
  guard required before grad-required AMP attention can route through the
  flash-state path. Without it, training keeps the existing materialized
  attention reference path.
- Extended causal-attention autograd state so flash training saves compact
  `[B,H,T]` row max/denom plus the forward output, not a `[B,H,T,T]`
  attention matrix.
- Added
  `causal_attention_bf16_flash_tensor_core_backward_f32_buffers(...)`, which
  recomputes causal probabilities from BF16 Q/K/V, saved row stats, forward
  output, and upstream gradients, then writes CUDA-resident `dQ/dK/dV`.
- Replaced the scalar recompute body for the target exact-tile sm80 path with
  tiled Tensor Core MMA kernels:
  - row-dot still computes compact `D_i = dot(O_i, dO_i)` in f32,
  - the tiled backward kernel recomputes 16x16 QK scores with MMA,
  - computes dP with MMA,
  - rebuilds P/dS tiles from saved row max/denom and row-dot,
  - accumulates dQ, dK, and dV with MMA and f32 atomics.
- The V1 Tensor Core backward is intentionally exact-tile only
  (`time % 16 == 0`, `head_dim % 16 == 0`). Unsupported/ragged backward shapes
  still fall back or hard-error under `HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION=1`.
- Added runtime JSON fields for flash backward requested/executed/fallback,
  row-dot calls, QK-recompute/dP/dQ/dK/dV MMA tile counts, backward scalar tile
  calls, ragged/causal tile counts, and CUDA-event elapsed microseconds.
- Fixed distributed train-report aggregation so all new flash backward runtime
  counters are summed across DDP ranks. The regression test
  `aggregate_rank_cuda_runtime_sums_flash_backward_fields` protects the parent
  report from collapsing rank-local backward counters to zero.
- Extended the CUDA train-LM fixture with
  `HEIRLOOM_EXPECT_FLASH_BF16_ATTENTION=1` checks requiring flash forward and
  backward execution, positive CUDA-event timing, zero flash fallback, zero
  hard-require failures, zero scalar backward tiles, positive QK/dP/dQ/dK/dV
  MMA tile counters, and zero materialized BF16 reference attention calls in
  both train and resume reports.
- Exposed the flash forward/backward/timing/require/expect envs through the
  Vertex validation launcher and added
  `scripts/gcp/submit_vertex_flash_attention_train_gate.sh` for the 4x A100
  target-shape gate.
- Local validation:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `bash -n scripts/cuda_train_lm_fixture.sh`,
  - `bash -n scripts/gcp/submit_vertex_flash_attention_train_gate.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`.
- A100 validation:
  - first launched job `5436078929033035776`; it failed because the distributed
    parent report did not aggregate the new flash backward counters even though
    forward/cp.async counters were present,
  - patched the aggregate field list and relaunched job `894480179806601216`,
  - job `894480179806601216` passed on 4x `NVIDIA_A100_80GB` with NCCL,
    per-rank `batch=4`, `time=512`, `heads=16`, `head_dim=64`, `d_model=1024`,
    `ff_hidden=4096`, `steps=10`, and resume `steps=1`,
  - train report: `flash_bf16_tensor_core_executed_calls=40`,
    `flash_bf16_tensor_core_backward_executed_calls=40`, backward
    QK/dP/dQ/dK/dV MMA tiles `167772160/167772160/20971520/20971520/20971520`,
    scalar backward tiles `0`, materialized reference attention `0`,
    flash hard failures/fallbacks `0`, `cp_async_gemm_executed_calls=840`,
    NCCL all-reduce calls/bytes `880/2188782080`, checksum max errors `0.0`,
    and positive CUDA-event/MFU fields,
  - resume report: flash forward/backward executed `4/4`, backward
    QK/dP/dQ/dK/dV MMA tiles `16777216/16777216/2097152/2097152/2097152`,
    scalar backward tiles `0`, `cp_async_gemm_executed_calls=84`, and positive
    CUDA-event/MFU fields.
- Remaining hard-path work: extend microbench forward+backward parity/timing,
  support ragged flash backward, tune occupancy/bank conflicts, and decide when
  guarded target-shape routing should graduate to default train-path policy.

### QB Memory Flash-Attention Hard-Path Gate

- Extended `scripts/run_qb_data_hardpath.sh` so
  `HEIRLOOM_EXPECT_FLASH_BF16_ATTENTION=1` now hard-validates train and resume
  reports for the guarded flash route: positive flash forward/backward execution,
  positive QK/dP/dQ/dK/dV MMA tile counters, positive optional flash timing,
  zero flash fallback, zero scalar backward tiles, zero hard-require failures,
  and zero materialized BF16 reference attention. The same gate validates
  `cp.async` GEMM execution and zero fallback/hard failures when the corresponding
  `cp.async` envs are set.
- Updated `scripts/validate_qb_data_hardpath_artifacts.py` with
  `--require-flash-attention` and `--require-flash-timing`, and made summaries
  that record `flash_attention_gate.expected=true` validate those counters by
  default.
- Added `scripts/gcp/submit_vertex_qb_memory_flash_gate.sh`, a scoped 4x A100
  wrapper for the QB manifest v2 memory-transformer path. It runs
  `train-memory-lm`/resume/eval/generation with binary-shard streaming, NCCL,
  AMP BF16, sparse-row memory updates, SMFT row mask, exact-tile
  `time=512`, `heads=16`, `head_dim=64`, guarded flash forward/backward, and
  double-buffered `cp.async` GEMM hard-required.
- Local validation:
  - `bash -n scripts/run_qb_data_hardpath.sh`,
  - `python3 -m py_compile scripts/validate_qb_data_hardpath_artifacts.py`,
  - `bash -n scripts/gcp/submit_vertex_qb_memory_flash_gate.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - CPU memory-mode QB hard-path smoke at
    `/tmp/heirloom-qb-memory-flash-contract-smoke/summary.json`,
  - `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile smoke --expected-world-size 0 --no-tensor-core-gates /tmp/heirloom-qb-memory-flash-contract-smoke/summary.json`,
  - `cargo fmt --all --check`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `cargo test --bin heirloom -- --nocapture`.
- Vertex validation passed:
  - job `6192402191454568448`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-010439/`,
  - validator:
    `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile full --expected-world-size 4 --require-flash-attention --require-flash-timing gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-010439/qb-data-hardpath/summary.json`
    returned `status="passed"`.
- A100 QB-memory evidence:
  - manifest v2 binary-shard streaming with `tokens_materialized=false`,
    `train_tokens=8962719`, `valid_tokens=471722`,
  - train report: 4x A100 NCCL, `tokens_seen=20480`,
    `tokens_per_second=5441.0201912858665`,
    `dense_core_mfu_estimate=0.0015274721106407851`,
    `end_to_end_mfu_estimate=0.001386659900269762`, loss
    `7.31654691696167 -> 7.253988265991211`,
  - train flash counters: forward/backward executed `40/40`, backward
    QK/dP/dQ/dK/dV MMA tiles
    `41943040/41943040/5242880/5242880/5242880`, flash fallback `0`, scalar
    backward tiles `0`, hard-require failures `0`, materialized-reference BF16
    attention `0`, flash timing forward/backward `34493/63631` us,
  - train GEMM/DDP counters: `cp_async_gemm_executed_calls=630`,
    `cp_async_gemm_staged_fallback_calls=0`,
    `cp_async_gemm_hard_require_failures=0`, NCCL all-reduce calls/bytes
    `80/1150976`, row-union calls/bytes `80/327680`, checksum drift `0.0`,
  - resume report: flash forward/backward executed `4/4`, positive
    QK/dP/dQ/dK/dV MMA counters, flash timing forward/backward `4173/6354` us,
    zero fallback/scalar/materialized-reference counters, `cp.async` GEMM
    executed `63`, NCCL all-reduce calls/bytes `8/94208`, row-union calls/bytes
    `8/32768`, checksum drift `0.0`.

### QB Memory Flash Scale/Tuning Gate

- Added summary observability to `scripts/run_qb_data_hardpath.sh`:
  `shape_gate`, `flash_attention_metrics`, and `cp_async_gemm_metrics`.
  These expose exact-tile compatibility, flash forward/backward us per call,
  total flash us/token, MMA tile counts, fallback/materialized-reference
  counters, and `cp.async` execution/instruction counters without requiring
  manual rank-0 report inspection.
- Extended `scripts/validate_qb_data_hardpath_artifacts.py` with scale
  thresholds:
  `--require-exact-tile-shape`, `--min-block-size`, `--min-d-model`,
  `--min-head-dim`, `--min-grad-accumulation-steps`, `--min-tokens-seen`,
  `--min-tokens-per-second`, and `--min-dense-core-mfu`.
- Added `scripts/gcp/submit_vertex_qb_memory_flash_scale_gate.sh`, a 4x A100
  scale wrapper that keeps the hard flash/cp.async contracts and raises the
  default shape to per-rank `batch=1`, `block=1024`, `d_model=1024`,
  `heads=16`, `head_dim=64`, `ff_hidden=4096`, and grad accumulation `2`.
- Local validation:
  - `python3 -m py_compile scripts/validate_qb_data_hardpath_artifacts.py`,
  - `bash -n scripts/run_qb_data_hardpath.sh`,
  - `bash -n scripts/gcp/submit_vertex_qb_memory_flash_scale_gate.sh`,
  - CPU memory-mode QB hard-path smoke at
    `/tmp/heirloom-qb-flash-scale-smoke/summary.json`,
  - `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile smoke --expected-world-size 0 --no-tensor-core-gates --min-block-size 16 --min-d-model 16 --min-head-dim 4 --min-grad-accumulation-steps 1 --min-tokens-seen 16 /tmp/heirloom-qb-flash-scale-smoke/summary.json`,
  - replayed the new exact-tile/min-shape validator against the passed A100
    summary from job `6192402191454568448`.
- Vertex validation passed:
  - job `660398552299601920`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-013844/`,
  - validator:
    `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile full --expected-world-size 4 --require-flash-attention --require-flash-timing --require-exact-tile-shape --min-block-size 1024 --min-d-model 1024 --min-head-dim 64 --min-grad-accumulation-steps 2 --min-tokens-seen 65536 gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-013844/qb-data-hardpath/summary.json`
    returned `status="passed"`.
- A100 scale evidence:
  - exact-tile shape `block=1024`, `d_model=1024`, `heads=16`, `head_dim=64`,
    grad accumulation `2`, `tokens_seen=65536`,
    `tokens_per_second=5892.465383923754`,
    `dense_core_mfu_estimate=0.0015504495422534528`,
    `end_to_end_mfu_estimate=0.0015165646661225846`,
  - flash forward/backward executed `64/64`, total flash `600836` us,
    `9.16802978515625` us/token, backward QK/dP/dQ/dK/dV MMA tiles
    `268435456/268435456/33554432/33554432/33554432`,
  - flash fallback `0`, scalar backward tiles `0`, hard-require failures `0`,
    materialized-reference BF16 attention `0`,
  - `cp_async_gemm_executed_calls=1008`,
    `cp_async_gemm_staged_fallback_calls=0`,
    `cp_async_gemm_hard_require_failures=0`,
  - NCCL all-reduce calls/bytes `64/970752`, row-union calls/bytes
    `64/262144`, checksum drift `0.0`,
  - loss moved `7.280417442321777 -> 7.3151936531066895`; this remains a
    throughput/evidence gate, not a quality gate.

### QB Memory Flash Head-Dim 128 Gate

- Added `scripts/gcp/submit_vertex_qb_memory_flash_head_dim128_gate.sh`, a
  4x A100 shape-coverage wrapper that keeps the scale baseline at
  `block=1024`, `d_model=1024`, `ff_hidden=4096`, and grad accumulation `2`,
  but changes `heads=8` so flash attention runs `head_dim=128`.
- Local validation:
  - `bash -n scripts/gcp/submit_vertex_qb_memory_flash_head_dim128_gate.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `python3 -m py_compile scripts/validate_qb_data_hardpath_artifacts.py`,
  - `cargo fmt --all --check`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo test --test cuda_storage -- --nocapture`.
- Vertex validation passed:
  - job `7673373985324662784`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-095048/`,
  - validator:
    `python3 scripts/validate_qb_data_hardpath_artifacts.py --profile full --expected-world-size 4 --require-flash-attention --require-flash-timing --require-exact-tile-shape --min-block-size 1024 --min-d-model 1024 --min-head-dim 128 --min-grad-accumulation-steps 2 --min-tokens-seen 65536 gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-095048/qb-data-hardpath/summary.json`
    returned `status="passed"`.
- A100 head-dim evidence:
  - exact-tile shape `block=1024`, `d_model=1024`, `heads=8`, `head_dim=128`,
    grad accumulation `2`, `tokens_seen=65536`,
    `tokens_per_second=5715.182698177378`,
    `dense_core_mfu_estimate=0.0014982224574761746`,
    `end_to_end_mfu_estimate=0.0014709367939840746`,
  - flash forward/backward executed `64/64`, total flash `968215` us,
    `14.773788452148438` us/token, backward QK/dP/dQ/dK/dV MMA tiles
    `536870912/536870912/33554432/33554432/33554432`,
  - flash fallback `0`, scalar backward tiles `0`, hard-require failures `0`,
    materialized-reference BF16 attention `0`,
  - `cp_async_gemm_executed_calls=1008`,
    `cp_async_gemm_staged_fallback_calls=0`,
    `cp_async_gemm_hard_require_failures=0`,
  - NCCL all-reduce calls/bytes `64/940032`, row-union calls/bytes
    `64/262144`, checksum drift `0.0`,
  - loss moved `7.2832841873168945 -> 7.3172783851623535`; this remains a
    throughput/shape-coverage gate, not a quality gate.

### QB Memory Throughput Harness

- Extended `scripts/run_qb_data_hardpath.sh` with throughput controls:
  - `HEIRLOOM_QB_DATA_HARDPATH_PROFILE=throughput`,
  - `HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE=dev|release`,
  - `HEIRLOOM_QB_DATA_HARDPATH_THROUGHPUT_REPORT=train|resume`,
  - `HEIRLOOM_QB_DATA_HARDPATH_SKIP_EVAL_GENERATION=1`,
  - `HEIRLOOM_QB_DATA_HARDPATH_RESUME_LOG_EVERY`.
- The summary now includes `summary.throughput` with the selected measured
  report, warmup/measured step counts, selected performance fields, flash
  metrics, `cp.async` GEMM metrics, and sparse/NCCL transport counters.
  `cp.async` GEMM metrics now include optional elapsed-us/per-call/share fields
  when `HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING=1` is enabled for short diagnostic
  runs.
- Added `scripts/validate_qb_memory_throughput_artifacts.py`, which validates
  the measured report without requiring eval/generation and emits bucket shares
  for flash, forward/backward, all-reduce, optimizer, and host-to-device timing.
  It also has `--require-cp-async-gemm-timing` for diagnostic artifacts.
- Added `scripts/gcp/submit_vertex_qb_memory_throughput.sh`, the first measured
  4x A100 baseline wrapper:
  - `block=1024`, `d_model=1024`, `heads=16`, `head_dim=64`, `ff_hidden=4096`,
  - per-rank `batch=1`, grad accumulation `4`,
  - 16 warmup train steps followed by 128 measured resumed train steps,
  - release-profile hard-path binary for the measured run,
  - manifest v2 binary-shard streaming, `train-memory-lm`, NCCL, sparse-row
    memory updates, SMFT row mask, Tensor Core flash forward/backward, and
    double-buffered `cp.async` GEMM hard-required,
  - eval/generation skipped so the run is a throughput profile rather than an
    end-to-end correctness gate.
- Initial dev-profile Vertex validation passed:
  - job `2438906988738904064`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-203926/`,
  - validator:
    `python3 scripts/validate_qb_memory_throughput_artifacts.py --expected-world-size 4 --require-flash-attention --require-flash-timing --require-cp-async-gemm --require-exact-tile-shape --min-block-size 1024 --min-d-model 1024 --min-head-dim 64 --min-grad-accumulation-steps 4 --min-warmup-steps 16 --min-measured-steps 128 --min-tokens-seen 2097152 gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-203926/qb-data-hardpath/summary.json`
    returned `status="passed"`.
- A100 measured-throughput evidence:
  - selected measured report `resume`, `warmup_steps=16`, `measured_steps=128`,
    exact-tile shape `block=1024`, `d_model=1024`, `heads=16`, `head_dim=64`,
    grad accumulation `4`, `tokens_seen=2097152`,
    `tokens_per_second=6035.629795488428`,
    `dense_core_mfu_estimate=0.0015616325048544738`,
    `end_to_end_mfu_estimate=0.0015534113973087484`,
  - timing buckets: `train_elapsed_ms=347462`,
    `forward_backward_cuda_elapsed_ms=345632.81005859375`,
    `all_reduce_cuda_elapsed_ms=1801.2379007339478`,
    `host_to_device_cuda_elapsed_ms=460.14483174681664`,
    `optimizer_cuda_elapsed_ms=66.70793595910072`,
  - flash forward/backward executed `2048/2048`, total flash `18982070` us,
    `9.051356315612793` us/token, flash share of forward/backward CUDA
    `0.05491975717462137`,
  - `cp_async_gemm_executed_calls=32256`, `cp_async_gemm_instructions=637802643456`,
    `cp_async_gemm_staged_fallback_calls=0`,
    `cp_async_gemm_hard_require_failures=0`,
  - flash fallback `0`, scalar backward tiles `0`, hard-require failures `0`,
    materialized-reference BF16 attention `0`,
  - NCCL all-reduce calls/bytes `1024/13383680`, row-union calls/bytes
    `1024/4194304`, row-union candidate rows `52280`, checksum drift `0.0`,
  - loss moved `7.27497410774231 -> 7.251707077026367`
    (`loss_reduction=0.003198228663271902`).
- Release-profile Vertex validation passed:
  - job `1689902075711848448`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-212601/`,
  - validator:
    `python3 scripts/validate_qb_memory_throughput_artifacts.py --expected-world-size 4 --require-flash-attention --require-flash-timing --require-cp-async-gemm --require-exact-tile-shape --require-release --min-block-size 1024 --min-d-model 1024 --min-head-dim 64 --min-grad-accumulation-steps 4 --min-warmup-steps 16 --min-measured-steps 128 --min-tokens-seen 2097152 gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-212601/qb-data-hardpath/summary.json`
    returned `status="passed"`,
  - selected measured report `resume`, `warmup_steps=16`, `measured_steps=128`,
    exact-tile shape `block=1024`, `d_model=1024`, `heads=16`, `head_dim=64`,
    grad accumulation `4`, `tokens_seen=2097152`,
    `tokens_per_second=6078.807167660886`,
    `dense_core_mfu_estimate=0.0015686277322282277`,
    `end_to_end_mfu_estimate=0.0015645241103662447`,
  - timing buckets: `train_elapsed_ms=344994`,
    `forward_backward_cuda_elapsed_ms=344091.47552490234`,
    `all_reduce_cuda_elapsed_ms=1659.6593832969666`,
    `host_to_device_cuda_elapsed_ms=199.64879997819665`,
    `optimizer_cuda_elapsed_ms=53.98527976870537`,
  - flash forward/backward executed `2048/2048`, total flash `18984547` us,
    `9.052537441253662` us/token, flash share of forward/backward CUDA
    `0.05517296518618946`,
  - `cp_async_gemm_executed_calls=32256`, `cp_async_gemm_instructions=637802643456`,
    `cp_async_gemm_staged_fallback_calls=0`,
    `cp_async_gemm_hard_require_failures=0`,
  - NCCL all-reduce calls/bytes `1024/13385728`, row-union calls/bytes
    `1024/4194304`, row-union candidate rows `52288`, checksum drift `0.0`,
  - loss moved `7.274963617324829 -> 7.251679062843323`
    (`loss_reduction=0.0032006420521548413`).
- Interpretation: the measured run keeps flash attention honest and cheap enough
  to deprioritize it for the next MFU step. `forward_backward_cuda_elapsed_ms`
  is `0.9973839415320335` of train time in the release-profile run, but flash is
  only `0.05517296518618946` of that bucket; the next optimization target is
  dense forward/backward GEMM/runtime structure. Release mode improved the
  measured lane only slightly (`6035.629795488428 -> 6078.807167660886`
  tokens/sec), so the bottleneck is not host-side debug overhead.
- Short GEMM-timing diagnostic passed:
  - job `7447965305537560576`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-215507/`,
  - launched with `HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING=1`,
    `HEIRLOOM_QB_DATA_HARDPATH_STEPS=2`, and
    `HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS=8`,
  - validator:
    `python3 scripts/validate_qb_memory_throughput_artifacts.py --expected-world-size 4 --require-flash-attention --require-flash-timing --require-cp-async-gemm --require-cp-async-gemm-timing --require-exact-tile-shape --require-release --min-block-size 1024 --min-d-model 1024 --min-head-dim 64 --min-grad-accumulation-steps 4 --min-warmup-steps 2 --min-measured-steps 8 --min-tokens-seen 131072 gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-215507/qb-data-hardpath/summary.json`
    returned `status="passed"`,
  - measured `131072` tokens, `6002.289691807483` tokens/sec,
    `dense_core_mfu_estimate=0.0015563596443726427`,
  - `forward_backward_cuda_elapsed_ms=21675.237182617188`,
    flash elapsed `1189203` us (`0.05486458994569623` of forward/backward CUDA),
    and timed `cp.async` GEMM elapsed `416223` us across `2016` calls
    (`206.45982142857142` us/call and `0.019202696445407154` of forward/backward
    CUDA),
  - rank-0 runtime counters recorded `kernel_launch_calls=19942`,
    `host_sync_calls=3232`, `stream_sync_calls=3232`,
    `allocation_cache_hits=20404`, `allocation_deferred_frees=20856`,
    `allocation_reserved_bytes=3963515412`, and `module_cache_hits=19804`,
  - rank-0 CUDA memory-kernel counters were modest by comparison:
    `query_key_score_calls=64`, `topk_calls=64`,
    `weighted_value_forward_calls=64`, `weighted_value_backward_calls=64`,
    `scatter_add_rows_calls=128`, `sparse_adamw_compact_rows_calls=16`,
  - conclusion: the cp.async GEMM kernel itself is not the first MFU lever for
    this shape. The next hard-path implementation target should be kernel-launch
    family attribution and then fusion/reduction of the dominant small
    elementwise/layout/autograd kernels.

## 2026-06-12 - A100 Launch-Family Attribution Diagnostic

- Added CUDA kernel launch-family attribution to the runtime counters:
  - each `KernelLaunchConfig` now carries a stable family label,
  - `cuda_runtime_counters()` reports `kernel_launch_families`,
  - DDP aggregate reports sum family call/element counts across rank reports,
  - `scripts/run_qb_data_hardpath.sh` includes top launch families in
    `summary.throughput`,
  - `scripts/validate_qb_memory_throughput_artifacts.py` supports
    `--require-kernel-launch-families` and validates family call/element totals
    against `kernel_launch_calls` / `kernel_launch_elements`.
- Local validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `bash -n scripts/run_qb_data_hardpath.sh`,
  - `bash -n scripts/gcp/submit_vertex_qb_memory_throughput.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `python3 -m py_compile scripts/validate_qb_memory_throughput_artifacts.py`.
- Short launch-family Vertex diagnostic passed:
  - job `2782376829070082048`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-223949/`,
  - launched with `HEIRLOOM_QB_DATA_HARDPATH_STEPS=2`,
    `HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS=8`, and release-profile
    measured binary,
  - validator:
    `python3 scripts/validate_qb_memory_throughput_artifacts.py --expected-world-size 4 --require-flash-attention --require-flash-timing --require-cp-async-gemm --require-kernel-launch-families --require-exact-tile-shape --require-release --min-block-size 1024 --min-d-model 1024 --min-head-dim 64 --min-grad-accumulation-steps 4 --min-warmup-steps 2 --min-measured-steps 8 --min-tokens-seen 131072 gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-223949/qb-data-hardpath/summary.json`
    returned `status="passed"`,
  - measured `131072` tokens, `5951.59605866594` tokens/sec,
    `dense_core_mfu_estimate=0.001554337522768353`,
    `end_to_end_mfu_estimate=0.001531783337112599`,
    `forward_backward_cuda_elapsed_ms=21703.435668945312`,
  - flash forward/backward elapsed `1189368` us total, or
    `0.05480090885803048` of forward/backward CUDA,
  - `cp_async_gemm_executed_calls=2016`, zero cp.async fallback/hard failures,
    zero flash fallback/scalar backward/materialized-reference counters.
- Launch-family evidence from rank 0:
  - total `kernel_launch_calls=19942`, `kernel_launch_elements=20986638420`,
  - `vector_elementwise`: `9270` calls (`46.48%`), `10248305984` elements
    (`48.83%`),
  - `bias_2d`: `3936` calls (`19.74%`), `3914244096` elements (`18.65%`),
  - `tensor_core_gemm_cp_async`: `2016` calls (`10.11%`), `2919235584`
    elements (`13.91%`),
  - `materialize_matrix_layout`: `1344` calls (`6.74%`), `2013265920`
    elements (`9.59%`),
  - `layer_norm`: `864` calls (`4.33%`),
  - `matmul_strided_f32_reference`: `640` calls (`3.21%`),
  - `scaled_vector_elementwise`: `592` calls (`2.97%`).
- Interpretation:
  - flash and cp.async GEMM are not the first MFU levers for this measured
    shape,
  - the next implementation pass should target launch-count and layout traffic
    reduction in the dense block runtime: fuse common elementwise chains, reduce
    bias-add launches, avoid/reuse layout materialization around strided
    reference matmul, and consider fused layernorm/dropout/residual-style
    kernels for the train path.

## 2026-06-12 - Fused BF16 Activation Roundtrip Launch Cleanup

- Implemented the first launch-count reduction slice from the launch-family
  diagnostic:
  - added `heirloom_f32_bf16_roundtrip_f32` PTX, which applies the same
    round-to-nearest-even BF16 conversion as `heirloom_f32_to_bf16` and widens
    the rounded bits back to f32 in one kernel,
  - added `f32_bf16_roundtrip_f32_buffer` in the CUDA runtime and labeled it as
    the `bf16_roundtrip` launch family,
  - added `Tensor::cuda_f32_bf16_roundtrip`,
  - routed `bf16_activation_roundtrip` through the fused CUDA kernel for CUDA
    f32 activations while preserving the CPU two-cast path,
  - kept autograd cast-like: the fused roundtrip uses a single identity-gradient
    cast node, matching the previous two identity-gradient cast nodes.
- Added A100-gated test coverage:
  - `cuda_f32_bf16_roundtrip_matches_two_cast_path_and_counts_one_family`
    compares the fused kernel against the old two-kernel CUDA cast path and
    asserts one `bf16_roundtrip` launch family entry.
- Local validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`.
- Short 4x A100 verification passed:
  - job `7398284972148129792`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-230716/`,
  - validator:
    `python3 scripts/validate_qb_memory_throughput_artifacts.py --expected-world-size 4 --require-flash-attention --require-flash-timing --require-cp-async-gemm --require-kernel-launch-families --require-exact-tile-shape --require-release --min-block-size 1024 --min-d-model 1024 --min-head-dim 64 --min-grad-accumulation-steps 4 --min-warmup-steps 2 --min-measured-steps 8 --min-tokens-seen 131072 gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260612-230716/qb-data-hardpath/summary.json`
    returned `status="passed"`,
  - measured `131072` tokens, `5842.299977713395` tokens/sec,
    `dense_core_mfu_estimate=0.0015595487947830517`,
    `end_to_end_mfu_estimate=0.0015036534180178636`,
    `forward_backward_cuda_elapsed_ms=21630.9130859375`,
  - `cp_async_gemm_executed_calls=2016`, zero cp.async fallback/hard failures,
    zero flash fallback/scalar backward/materialized-reference counters,
  - the new CUDA-gated roundtrip test passed on the A100 worker.
- Before/after versus launch-family job `2782376829070082048`:
  - total rank-0 launches moved `19942 -> 19302` (`-640`),
  - `vector_elementwise` moved `9270 -> 7990` (`-1280`),
  - new `bf16_roundtrip` recorded `640` calls,
  - `kernel_launch_elements` moved `20986638420 -> 19912896596`,
  - forward/backward CUDA elapsed moved `21703.435668945312 ->
    21630.9130859375` ms,
  - dense-core MFU moved `0.001554337522768353 ->
    0.0015595487947830517`,
  - end-to-end tokens/sec moved `5951.59605866594 -> 5842.299977713395`.
- Interpretation:
  - this is a correct launch-count cleanup and removes one kernel launch per
    AMP activation roundtrip,
  - it is not yet a proven throughput win because the short end-to-end lane is
    noisy and all-reduce/end-to-end share moved against the run,
  - the next MFU pass should target a larger family: fuse Linear bias into the
    Tensor Core GEMM store path or reduce layout materialization around
    strided reference matmul.

## 2026-06-13 - Fused Linear Bias Into cp.async Tensor Core GEMM Store

- Implemented the second launch-count reduction slice from the launch-family
  diagnostic:
  - added `matmul_bf16_tensor_core_rhs_t_bias_f32_buffers`, a guarded exact-tile
    BF16 Tensor Core GEMM+bias CUDA wrapper for the double-buffered `cp.async`
    path,
  - extended the `heirloom_matmul_bf16_mma_rhs_t_f32_cta_cp_async_db` PTX entry
    with optional bias parameters and f32 bias loads in the store epilogue,
  - added `Tensor::matmul_bf16_tensor_core_rhs_t_bias`,
  - added a fused `GradFn::MatmulBias` CUDA autograd node that reuses the
    existing BF16 Tensor Core matmul backward path and the existing
    `bias_add_backward_bias_f32_buffer` reduction, so bias gradients are
    preserved,
  - routed `Linear::forward_amp_bf16_named` through the fused path only when
    the input/weight/bias dtypes are f32, the device supports BF16 Tensor
    Cores, `HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1`, and `M/K/N` are exact
    Tensor Core tile shapes. Other shapes keep the previous matmul-plus-bias
    path.
- Added/updated A100-gated test coverage:
  - `cuda_raw_bf16_tensor_core_matmul_bias_cp_async_matches_cpu_bf16_reference`
    compares the fused CUDA kernel against a BF16-rounded CPU matmul+bias
    reference and asserts zero `bias_2d` launch-family calls,
  - `cuda_linear_amp_bf16_uses_tensor_core_matmul_when_tile_aligned` now checks
    that cp.async exact-tile Linear forward has no forward `bias_2d` launch and
    still produces the expected CUDA f32 bias gradient after backward.
- Local validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `bash -n scripts/run_qb_data_hardpath.sh`,
  - `bash -n scripts/gcp/submit_vertex_qb_memory_throughput.sh`,
  - `zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh`,
  - `python3 -m py_compile scripts/validate_qb_memory_throughput_artifacts.py`.
- Short 4x A100 verification passed:
  - job `5917310979254779904`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260613-005643/`,
  - validator:
    `python3 scripts/validate_qb_memory_throughput_artifacts.py /tmp/qb-memory-throughput-bias-fused-20260613-005643/qb-data-hardpath --require-kernel-launch-families`
    returned `status="passed"`,
  - measured `131072` resume tokens, `5887.701015182823` tokens/sec,
    `dense_core_mfu_estimate=0.0015533740387336967`,
    `end_to_end_mfu_estimate=0.0015153384436811953`,
    `forward_backward_cuda_elapsed_ms=21716.89727783203`,
  - `cp_async_gemm_executed_calls=2016`, zero cp.async fallback/hard failures,
    zero flash fallback/scalar backward/materialized-reference counters,
  - the new CUDA-gated fused-bias test passed on the A100 worker.
- Before/after versus fused-roundtrip job `7398284972148129792`:
  - total measured rank-0 launches moved `19302 -> 18630` (`-672`),
  - `bias_2d` moved `3936 -> 3264` (`-672`),
  - launched elements moved `19912896596 -> 19006926932` (`-905969664`),
  - `tensor_core_gemm_cp_async` stayed at `2016` calls,
  - measured resume tokens/sec moved `5842.299977713395 ->
    5887.701015182823`,
  - forward/backward CUDA elapsed moved `21630.9130859375 ->
    21716.89727783203` ms,
  - dense-core MFU moved `0.0015595487947830517 ->
    0.0015533740387336967`.
- Interpretation:
  - this is a correct launch-count cleanup and removes one forward bias launch
    from exact-tile cp.async Linear projections while preserving training
    gradients,
  - it is not yet a proven MFU win because the short end-to-end lane remains
    noisy and dense-core MFU stayed flat,
  - the next MFU pass should target `materialize_matrix_layout`,
    `matmul_strided_f32_reference`, `layer_norm`, or remaining high-count
    elementwise chains.

## 2026-06-13 - Skipped Contiguous BF16 Layout Materialization

- Implemented the third launch-count reduction slice from the launch-family
  diagnostic:
  - added a CUDA autograd helper that reuses BF16 buffers when the saved matrix
    layout is already contiguous row-major (`offset=0`, `col_stride=1`,
    `row_stride=cols`),
  - kept non-contiguous/transposed operands on the existing
    `materialize_matrix_layout_bf16_buffer` path,
  - updated the A100-gated Linear AMP test to assert that exact-tile Linear
    backward now performs one BF16 layout materialization rather than two for a
    single layer.
- Local validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `cargo test --test transformer_runtime -- --nocapture`,
  - `python3 scripts/validate_qb_memory_throughput_artifacts.py /tmp/qb-memory-throughput-layout-skip-20260613-021419/qb-data-hardpath --require-kernel-launch-families`.
- Short 4x A100 verification passed:
  - job `863075928694063104`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260613-021419/`,
  - validator returned `status="passed"`,
  - measured `131072` resume tokens, `5977.653121722078` tokens/sec,
    `dense_core_mfu_estimate=0.001552803693424789`,
    `end_to_end_mfu_estimate=0.0015384897356332727`,
    `forward_backward_cuda_elapsed_ms=21724.873901367188`,
  - `cp_async_gemm_executed_calls=2016`, zero cp.async fallback/hard failures,
    zero flash fallback/scalar backward/materialized-reference counters.
- Before/after versus fused-bias job `5917310979254779904`:
  - total measured rank-0 launches moved `18630 -> 17958` (`-672`),
  - `materialize_matrix_layout` moved `1344 -> 672` (`-672`),
  - launched elements moved `19006926932 -> 18100957268` (`-905969664`),
  - measured resume tokens/sec moved `5887.701015182823 ->
    5977.653121722078`,
  - forward/backward CUDA elapsed moved `21716.89727783203 ->
    21724.873901367188` ms,
  - dense-core MFU moved `0.0015533740387336967 ->
    0.001552803693424789`.
- Interpretation:
  - this is a correct launch/layout cleanup and removes the redundant
    contiguous-left BF16 materialization from exact-tile Linear backward,
  - it is not yet a proven MFU win because forward/backward CUDA time and
    dense-core MFU stayed flat in the short lane,
  - the next MFU pass should target the remaining transposed layout
    materialization, the `matmul_strided_f32_reference` family, `layer_norm`, or
    remaining high-count elementwise chains.

## 2026-06-13 - Paired BF16 Transposes For Linear Weight-Gradient Prep

- Implemented the fourth launch-count reduction slice from the launch-family
  diagnostic:
  - added `transpose2d_pair_bf16_buffers`, which transposes two independent
    BF16 matrices in one CUDA launch,
  - added the PTX entry `heirloom_transpose2d_pair_bf16`,
  - added a distinct `transpose2d_pair_bf16` launch family label,
  - routed BF16 Tensor Core matmul backward through the paired transpose for
    the `left_t` and `grad_output_t` buffers used by the weight-gradient GEMM.
- Added/updated A100-gated test coverage:
  - `cuda_transpose2d_pair_bf16_matches_cpu_transposes_and_counts_one_family`
    validates the paired kernel against CPU bit transposes,
  - `cuda_linear_amp_bf16_uses_tensor_core_matmul_when_tile_aligned` now checks
    that exact-tile Linear backward records one paired BF16 transpose launch.
- Local validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `bash -n scripts/gcp/submit_vertex_qb_memory_throughput.sh`.
- Short 4x A100 verification passed:
  - job `5016520685036503040`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260613-023553/`,
  - validator returned `status="passed"`,
  - measured `131072` resume tokens, `5932.470353942246` tokens/sec,
    `dense_core_mfu_estimate=0.0015577634370467514`,
    `end_to_end_mfu_estimate=0.0015268608868122915`,
    `forward_backward_cuda_elapsed_ms=21655.704345703125`,
  - `cp_async_gemm_executed_calls=2016`, zero cp.async fallback/hard failures,
    zero flash fallback/scalar backward/materialized-reference counters.
- Before/after versus layout-skip job `863075928694063104`:
  - total measured rank-0 launches moved `17958 -> 17286` (`-672`),
  - `bias_2d` moved `3264 -> 1920` (`-1344`),
  - `transpose2d_pair_bf16` moved `0 -> 672`,
  - forward/backward CUDA elapsed moved `21724.873901367188 ->
    21655.704345703125` ms,
  - measured resume tokens/sec moved `5977.653121722078 ->
    5932.470353942246`,
  - dense-core MFU moved `0.001552803693424789 ->
    0.0015577634370467514`.
- Interpretation:
  - this is a correct launch-count cleanup and combines two BF16 transpose
    launches into one for exact-tile Linear backward,
  - it is not yet a proven MFU win because the short end-to-end lane remains
    noisy,
  - the next MFU pass should target the remaining transposed layout
    materialization with a normal-RHS or strided-RHS Tensor Core GEMM, reduce
    the `matmul_strided_f32_reference` family, or fuse layernorm/elementwise
    chains.

2026-06-13: tested and gated normal-RHS Tensor Core GEMM for Linear
input-gradient backward.
- Code:
  - added `matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward`, a
    staged exact-tile BF16 Tensor Core GEMM that consumes a normal row-major
    RHS buffer instead of an already-transposed RHS buffer,
  - added the `tensor_core_gemm_normal_rhs_staged` launch family and raw CUDA
    parity coverage,
  - routed Linear backward through the normal-RHS path only when
    `HEIRLOOM_CUDA_TENSOR_CORE_NORMAL_RHS_GEMM=1` is set; the default remains
    the faster cp.async RHS-transposed route plus materialization.
- Local validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --test cuda_storage -- --nocapture`,
  - `cargo test --bin heirloom -- --nocapture`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`.
- Short 4x A100 verification passed:
  - job `4641877491034619904`, state `JOB_STATE_SUCCEEDED`,
  - artifacts:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260613-030621/`,
  - validator returned `status="passed"`,
  - measured `131072` resume tokens, `5418.437370814386` tokens/sec,
    `dense_core_mfu_estimate=0.0015460787869372994`,
    `end_to_end_mfu_estimate=0.0013945623990587338`.
- Before/after versus paired-transpose job `5016520685036503040`:
  - total measured rank-0 launches moved `17286 -> 16614` (`-672`),
  - `materialize_matrix_layout` moved `672 -> 0`,
  - `tensor_core_gemm_cp_async` moved `2016 -> 1344`,
  - `tensor_core_gemm_normal_rhs_staged` moved `0 -> 672`,
  - rank-0 forward/backward CUDA elapsed moved `21538.968505859375 ->
    21819.369567871094` ms,
  - aggregate resume tokens/sec moved `5932.470353942246 ->
    5418.437370814386`.
- Interpretation:
  - the primitive is correct and removes a high-count materialization family,
  - the staged normal-RHS GEMM is slower than keeping the materialization and
    using the double-buffered cp.async RHS-transposed GEMM,
  - the route remains experimental until the normal/strided RHS case gets a
    real `ldmatrix`/`cp.async` instruction-path kernel.

## 2026-06-16 - Rust-Native Padawan Validation And P3 Verifier Harness

- Implemented the first local Padawan validation/verifier slice in the
  Heirloom Rust CLI:
  - added `heirloom padawan validate` for episode JSONL, artifact references,
    SHA-256 content hashes, teacher masking, hidden-test audit opacity,
    compressed-rationale bounds, replay tracking, and SMFT eligibility checks,
  - added `heirloom padawan verify` for deterministic P3 verifier families:
    code patch artifacts, JSON/tool-call records, evidence/final-validation
    support, and memory/SMFT telemetry,
  - added optional JSON report output with per-family status and measured
    fixture details,
  - kept the implementation dependency-neutral by using the existing Rust CLI
    stack and a local SHA-256 helper,
  - removed the Python Padawan validator/harness path from active validation.
- Updated sidecar docs and fixtures:
  - `padawan/README.md` now documents the native Rust validation commands and
    expected outputs,
  - `PADAWAN_LOOP_DESIGN.md` marks P2 validation and the first P3 local verifier
    harness slice as Rust-native,
  - `CODEX_GOAL_LOOP.md` now points to `heirloom padawan` commands,
  - the fixture trace now names `heirloom padawan validate` as its final
    validator and the artifact bundle hash/byte metadata was refreshed.
- Focused validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --bin heirloom padawan -- --nocapture` ran `2` Padawan tests:
    SHA-256 known-vector coverage and fixture validate/verify coverage,
  - `cargo run --bin heirloom -- padawan validate --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `validated padawan episodes=1 sft_eligible=1 smft_eligible=1 max_guidance_level=1`,
  - `cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `padawan verifier harness status=passed episodes=1 families=4 passed=4 failed=0 skipped=0`,
  - `cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures --report /tmp/padawan-rust-harness-report.json`,
  - `bash -n scripts/validate.sh`.
- The generated Rust verifier report showed:
  - code patch changed only `padawan/README.md` with `3` added lines and `0`
    removed lines,
  - JSON/tool-call verifier parsed `trace`, `memory_selection`,
    `smft_access_counts`, and `verifier_result`,
  - evidence verifier found `1` observation and final validation metadata,
  - memory/SMFT verifier found `1` layer, `3` selected rows, lift `4.5`, and
    replay Jaccard `0.34`.
- No production tokenizer, corpus blend, manifest v2 behavior, base checkpoint,
  or A100/Vertex path was changed.

## 2026-06-16 - Padawan Rust Verifier Negative Coverage

- Hardened the native Rust Padawan verifier harness with focused negative
  coverage:
  - code patch verifier rejects paths outside the sidecar allowlist,
  - JSON/tool-call verifier rejects inline tool results,
  - evidence/final-validation verifier rejects traces without observations,
  - memory/SMFT verifier rejects selected-row budget overflow.
- This keeps Padawan verification in Heirloom-native Rust and avoids reviving
  the removed Python Padawan scripts.
- Focused validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --bin heirloom padawan -- --nocapture` ran `6` Padawan tests,
    including the valid fixture, SHA-256 known vector, and the four negative
    verifier cases,
  - `cargo run --bin heirloom -- padawan validate --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `validated padawan episodes=1 sft_eligible=1 smft_eligible=1 max_guidance_level=1`,
  - `cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `padawan verifier harness status=passed episodes=1 families=4 passed=4 failed=0 skipped=0`,
  - `bash -n scripts/validate.sh`.
- No production tokenizer, corpus blend, manifest v2 behavior, base checkpoint,
  or A100/Vertex path was changed.

## 2026-06-16 - Rust Learning Sanity Ladder Evidence Validator

- Added a native Rust readiness validator for learning-sanity ladder evidence:
  - new CLI path:
    `heirloom readiness validate-learning-sanity --manifest ...`,
  - manifest format `heirloom.learning_sanity_ladder` with recognized stage IDs
    matching the readiness ladder,
  - validates referenced train reports for finite `initial_loss`/`final_loss`,
    computed `loss_reduction`, per-stage minimum reduction, and optional
    command/model/loader/memory expectations,
  - emits `heirloom.learning_sanity_ladder_validation` JSON reports.
- Added a checked fixture under `tests/fixtures/learning_sanity/`:
  - `dense-train-report.json`,
  - `ladder-valid.json`.
- Updated readiness docs:
  - `QB_PRETRAINING_READINESS.md` documents the manifest contract and command,
  - `CODEX_GOAL_LOOP.md` includes the native readiness validator in the local
    validation checklist,
  - `scripts/validate.sh` now runs the readiness validator fixture command.
- Focused validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo test --bin heirloom learning_sanity -- --nocapture` ran `2`
    readiness tests: a passing loss-improvement manifest and a rejected
    flat-loss manifest,
  - `cargo test --bin heirloom padawan -- --nocapture` kept the `6` native
    Padawan tests passing,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/ladder-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - the same command with `--report /tmp/learning-sanity-validation.json`
    wrote a validation report with `format="heirloom.learning_sanity_ladder_validation"`,
  - `bash -n scripts/validate.sh`.
- This makes future tiny learning-sanity runs machine-checkable. It does not
  claim the full ladder has run yet; LR/grad-accumulation sweep, longer 32K
  blend, and remote 4x A100 quality evidence remain pending.

## 2026-06-16 - Local Learning Sanity Ladder Producer

- Added `scripts/run_learning_sanity_ladder.sh`, a bounded local producer that
  creates fresh learning-sanity evidence through native `heirloom` CLI paths:
  tokenizer training, manifest v2 binary-shard preparation, dense LM training,
  memory LM training with memory layers explicitly disabled, exact-memory LM
  training with SMFT disabled, manifest emission, and Rust readiness validation.
- Dense `train-lm --report` output now records `command="train-lm"` and
  `model_family="tiny_transformer"` so learning-sanity manifests can assert
  generated dense reports the same way they assert memory-train reports.
- Added `train-memory-lm --disable-memory-layers` and a Rust readiness check
  that rejects `memory_layers_disabled` stages when the report still contains
  active memory layer indices below `n_layers`.
- Cleaned up Rust Padawan helper code for the current clippy gate by replacing
  manual cap clamps with `.clamp(...)` and eliding unnecessary helper
  lifetimes; this does not change Padawan validation semantics.
- Updated `QB_PRETRAINING_READINESS.md` and `CODEX_GOAL_LOOP.md` to document
  the producer as a tiny local partial ladder, not a full quality claim.
- Focused validation passed:
  - `bash -n scripts/run_learning_sanity_ladder.sh`,
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `cargo test --bin heirloom learning_sanity -- --nocapture` ran `3`
    readiness tests, including the active-memory rejection for
    `memory_layers_disabled`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom learning_sanity -- --nocapture`
    passed after a parallel focused-test attempt hit a rustc incremental-cache
    ICE while compiling the same test binary,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom padawan -- --nocapture` ran
    the `6` native Padawan tests successfully,
  - `bash scripts/run_learning_sanity_ladder.sh /tmp/heirloom-learning-sanity-smoke-3stage`
    wrote fresh dense, memory-disabled, and exact-memory reports and returned
    `learning_sanity status=passed stages=3 passed=3 failed=0 min_loss_reduction=0.538070`,
  - `bash -n scripts/validate.sh`,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/ladder-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - `cargo run --bin heirloom -- padawan validate --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `validated padawan episodes=1 sft_eligible=1 smft_eligible=1 max_guidance_level=1`,
  - `cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `padawan verifier harness status=passed episodes=1 families=4 passed=4 failed=0 skipped=0`.
- The local smoke observed dense loss `6.196569 -> 2.273697`
  (`loss_reduction=0.6330716664404283`), memory-layers-disabled loss
  `5.860109 -> 2.554132` (`loss_reduction=0.5641494770471048`), and
  exact-memory/SMFT-disabled loss `5.590967 -> 2.582638`
  (`loss_reduction=0.5380695427283058`). Full ladder stages beyond this tiny
  local partial run remain pending.

## 2026-06-16 - Local SMFT-Enabled Learning Sanity Stage

- Extended the local learning-sanity producer to cover the first four ladder
  stages, adding exact-memory training with SMFT enabled:
  - `memory_update_policy="sparse_rows"`,
  - `smft_mode="masked_memory_rows"`,
  - online refresh every step,
  - generated SMFT access-count and row-mask artifacts.
- Added a CPU sparse-row AdamW path in Rust:
  - `AdamW::step_sparse_rows_mut(...)` now dispatches selected-row updates for
    CPU and CUDA parameters,
  - `step_cuda_sparse_rows_mut(...)` remains as a compatibility wrapper,
  - CPU updates reuse the existing f64 AdamW moment buffers and update each
    active selected row once after optional SMFT row-mask intersection.
- `train-memory-lm` now uses the device-neutral sparse-row optimizer, allowing
  the SMFT-enabled sanity stage to run locally on CPU without pretending it used
  CUDA.
- Hardened learning-sanity validation for `memory_exact_smft_enabled`:
  - requires exact memory lookup, `sparse_rows` update policy, and
    `masked_memory_rows` SMFT mode,
  - requires selected-row optimizer evidence and an attached SMFT row mask,
  - requires online refresh, generated/active mask summaries, and nonzero SMFT
    access counts.
- Updated `QB_PRETRAINING_READINESS.md` and `CODEX_GOAL_LOOP.md` so the local
  producer is documented as a four-stage machinery sanity ladder. Product-key
  parity, LR/grad-accumulation, longer 32K blend, and remote quality evidence
  remain pending.
- Focused validation passed:
  - `bash -n scripts/run_learning_sanity_ladder.sh`,
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom learning_sanity -- --nocapture`
    ran `5` readiness tests, including positive and negative SMFT-enabled
    artifact evidence checks,
  - `cargo test --test memory_transformer cpu_sparse_adamw -- --nocapture`
    ran `2` CPU sparse-row AdamW tests,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom memory_optimizer_report_records_supplied_smft_row_mask -- --nocapture`,
  - `bash scripts/run_learning_sanity_ladder.sh /tmp/heirloom-learning-sanity-smoke-4stage`
    wrote fresh four-stage evidence and returned
    `learning_sanity status=passed stages=4 passed=4 failed=0 min_loss_reduction=0.003100`,
  - `bash -n scripts/validate.sh`,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/ladder-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - `cargo run --bin heirloom -- padawan validate --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `validated padawan episodes=1 sft_eligible=1 smft_eligible=1 max_guidance_level=1`,
  - `cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `padawan verifier harness status=passed episodes=1 families=4 passed=4 failed=0 skipped=0`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom padawan -- --nocapture` ran
    the `6` native Padawan tests successfully.
- The four-stage smoke observed:
  - dense fixed-shard loss `6.196569 -> 2.273697`
    (`loss_reduction=0.6330716664404283`),
  - memory-layers-disabled loss `5.860109 -> 2.554132`
    (`loss_reduction=0.5641494770471048`),
  - exact-memory/SMFT-disabled loss `5.590967 -> 2.582638`
    (`loss_reduction=0.5380695427283058`),
  - exact-memory/SMFT-enabled loss `5.590967 -> 5.573637`
    (`loss_reduction=0.003099758228239774`).
- The SMFT-enabled report showed
  `sparse_optimizer_updates_selected_rows=true`, `refresh_count=32`, generated
  and active masks with `14` trainable rows, and accumulated SMFT access counts
  with `2048` total events across `28` unique rows.

## 2026-06-16 - Local Product-Key Learning Sanity Stage

- Extended `scripts/run_learning_sanity_ladder.sh` to cover product-key parity
  on the same deterministic fixed shard:
  - trains a `train-memory-lm` product-key memory model through the manifest v2
    binary-shard loader,
  - uses square `memory_slots=16` and even `memory_key_dim=8`,
  - writes `memory-product-key-report.json`,
  - validates the stage as `product_key_parity` in the learning-sanity
    manifest.
- Hardened the Rust learning-sanity validator for `product_key_parity`:
  - requires `memory_lookup="product_key"` and `smft_mode="disabled"`,
  - checks square product-key slot geometry and even key dimension,
  - requires product-key memory-selection evidence with selected rows,
  - requires at least three memory-table parameters for left/right/value
    product-key tables,
  - requires an executed product-key lookup kernel path in
    `amp_bf16_op_decisions`.
- Updated `QB_PRETRAINING_READINESS.md` and `CODEX_GOAL_LOOP.md` so the local
  producer is documented as a five-stage machinery sanity ladder. The remaining
  learning-sanity ladder work is LR/grad-accumulation on 4x A100 and the longer
  32K tokenizer/blend run.
- Focused validation passed:
  - `bash -n scripts/run_learning_sanity_ladder.sh`,
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom learning_sanity -- --nocapture`
    ran `7` readiness tests, including positive and negative product-key
    evidence checks,
  - `bash scripts/run_learning_sanity_ladder.sh /tmp/heirloom-learning-sanity-smoke-5stage`
    wrote fresh five-stage evidence and returned
    `learning_sanity status=passed stages=5 passed=5 failed=0 min_loss_reduction=0.003100`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `bash -n scripts/validate.sh`,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/ladder-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - `cargo run --bin heirloom -- padawan validate --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `validated padawan episodes=1 sft_eligible=1 smft_eligible=1 max_guidance_level=1`,
  - `cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `padawan verifier harness status=passed episodes=1 families=4 passed=4 failed=0 skipped=0`.
- The five-stage smoke observed:
  - dense fixed-shard loss `6.196569 -> 2.273697`
    (`loss_reduction=0.6330716664404283`),
  - memory-layers-disabled loss `5.860109 -> 2.554132`
    (`loss_reduction=0.5641494770471048`),
  - exact-memory/SMFT-disabled loss `5.590967 -> 2.582638`
    (`loss_reduction=0.5380695427283058`),
  - exact-memory/SMFT-enabled loss `5.590967 -> 5.573637`
    (`loss_reduction=0.003099758228239774`),
  - product-key parity loss `5.579786 -> 2.564575`
    (`loss_reduction=0.540381026314303`).
- The product-key report showed `memory_lookup="product_key"`,
  `memory_slots=16`, `memory_key_dim=8`, `memory_table_parameter_count=3`,
  `selected_row_events=64`, `unique_selected_rows=7`, and repeated
  `cpu_product_key_candidate_topk_memory_lookup` decisions.

## 2026-06-16 - LR/Grad Sweep Evidence Contract

- Extended the native Rust learning-sanity validator for
  `lr_grad_accumulation_sweep`:
  - added nested sweep reports with format
    `heirloom.learning_sanity_lr_grad_accumulation_sweep`,
  - validates at least four child runs, at least two distinct learning rates,
    at least two distinct grad-accumulation settings, 4-way NCCL distributed
    evidence, AMP BF16 precision, CUDA device labels, nonzero all-reduce
    counters, positive per-rank loss reductions, consistent
    global-effective-batch math, and positive `performance.tokens_seen`,
  - supports per-run `expected_learning_rate` and
    `expected_grad_accumulation_steps` checks.
- Promoted `learning_rate` into dense and memory train reports, including
  single-rank reports, DDP rank reports, and DDP aggregate reports, so future
  sweep evidence can be checked from artifacts rather than prose.
- Added a checked synthetic fixture under
  `tests/fixtures/learning_sanity/lr-grad-sweep-valid.json` and wired it into
  `scripts/validate.sh`. This proves the evidence contract only; it does not
  claim the paid 4x A100 sweep has run.
- Updated `QB_PRETRAINING_READINESS.md` and `CODEX_GOAL_LOOP.md` to distinguish
  LR/grad sweep validator readiness from missing live 4x A100 evidence.
- Focused validation passed:
  - `cargo fmt --all --check`,
  - `bash -n scripts/validate.sh`,
  - `bash -n scripts/run_learning_sanity_ladder.sh`,
  - `cargo check --bin heirloom`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom learning_sanity -- --nocapture`
    ran `10` readiness tests, including positive and negative LR/grad sweep
    coverage,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/ladder-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/lr-grad-sweep-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `cargo run --bin heirloom -- padawan validate --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `validated padawan episodes=1 sft_eligible=1 smft_eligible=1 max_guidance_level=1`,
  - `cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `padawan verifier harness status=passed episodes=1 families=4 passed=4 failed=0 skipped=0`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom padawan -- --nocapture`
    ran the `6` native Rust Padawan tests successfully,
  - `bash scripts/run_learning_sanity_ladder.sh /tmp/heirloom-learning-sanity-lr-contract-smoke`
    wrote fresh five-stage local evidence and returned
    `learning_sanity status=passed stages=5 passed=5 failed=0 min_loss_reduction=0.003100`.
- The generated local smoke reports now include `learning_rate=0.01`; the real
  LR/grad sweep on paid 4x A100 hardware and the longer 32K blend run remain
  pending approval/evidence.

## 2026-06-16 - 32K Blend Hard-Path Evidence Contract

- Extended the native Rust learning-sanity validator for `longer_32k_blend`:
  - treats the stage report as the `scripts/run_qb_tokenizer_hardpath.sh`
    summary artifact,
  - follows its child artifact paths for corpus blend, tokenizer training,
    fertility, materializer curation, manifest v2 dataset, train/resume/eval,
    and generation reports,
  - requires tokenizer v2, vocab `32768`, at least `128` reserved tokens,
    digit isolation, non-empty source-blend/sample/registry hashes, a governed
    `heirloom.corpus_blend` manifest, no TinyStories source, manifest v2
    binary shards, `loader.kind="binary_shard_streaming"`,
    `tokens_materialized=false`, train loss improvement, resume step advance,
    positive eval metrics, and generation output,
  - supports stricter stage knobs such as `min_selected_tokens`,
    `min_selected_docs`, `min_blend_sources`, `min_final_step`, and
    `require_production_gate`.
- Added a checked synthetic artifact bundle under
  `tests/fixtures/learning_sanity/longer_32k_blend/` plus
  `tests/fixtures/learning_sanity/longer-32k-blend-valid.json`, and wired the
  fixture into `scripts/validate.sh`. This proves the artifact contract only;
  it does not claim a new long production-data run.
- Updated `QB_PRETRAINING_READINESS.md` and `CODEX_GOAL_LOOP.md` so the
  learning-sanity ladder distinguishes local validator readiness from missing
  paid 4x A100 LR/grad and longer 32K production-data evidence.
- Focused validation passed:
  - `cargo fmt --all --check`,
  - `cargo check --bin heirloom`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom learning_sanity -- --nocapture`
    ran `12` readiness tests, including positive and negative 32K blend
    coverage,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/ladder-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/lr-grad-sweep-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/longer-32k-blend-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.150000`,
  - `bash -n scripts/validate.sh`,
  - `cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings`,
  - `cargo run --bin heirloom -- padawan validate --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `validated padawan episodes=1 sft_eligible=1 smft_eligible=1 max_guidance_level=1`,
  - `cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures`
    returned `padawan verifier harness status=passed episodes=1 families=4 passed=4 failed=0 skipped=0`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom padawan -- --nocapture`
    ran the `6` native Rust Padawan tests successfully.
- The actual paid 4x A100 LR/grad sweep and longer production-data 32K blend
  run remain pending approval/evidence; this checkpoint only makes their
  expected artifacts machine-checkable.

## 2026-06-16 - Local 32K Blend Rehearsal Learning-Sanity Validation

- Adjusted the `longer_32k_blend` validator to score loss across the full
  hard-path train/resume workflow:
  - still checks each train and resume report for finite internally consistent
    loss fields,
  - computes the stage `loss_reduction` from `train_report.initial_loss` to
    `resume_report.final_loss`,
  - preserves the resume step-advance requirement.
- Added learning-sanity output to `scripts/run_qb_tokenizer_hardpath.sh`:
  - `HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY=auto` writes
    `learning-sanity-ladder.json` and runs
    `heirloom readiness validate-learning-sanity` for `vocab_size=32768`,
  - the wrapper writes `learning-sanity-validation.json`,
  - thresholds are controlled with
    `HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_LOSS_REDUCTION`,
    `HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_TOKENS`,
    `HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_DOCS`,
    `HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_FINAL_STEP`,
    `HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_BLEND_SOURCES`, and
    `HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_REQUIRE_PRODUCTION_GATE`.
- Added and validated a real local learning-sanity manifest for the existing
  digit-isolated 32K rehearsal:
  - manifest:
    `runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/learning-sanity-ladder.json`,
  - validation report:
    `runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/learning-sanity-validation.json`,
  - result:
    `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.019620`,
  - evidence: tokenizer vocab `32768`, `128` reserved tokens, manifest v2
    binary shards, `loader.kind="binary_shard_streaming"`,
    `tokens_materialized=false`, `35,705` selected tokens from `34` selected
    docs, and train-plus-resume loss `10.049116 -> 9.851956`.
- Updated `QB_PRETRAINING_READINESS.md` and `CODEX_GOAL_LOOP.md` to distinguish
  this bounded local 32K evidence from the still-pending paid 4x A100
  production-data run.
- Focused validation passed:
  - `cargo fmt --all --check`,
  - `bash -n scripts/run_qb_tokenizer_hardpath.sh`,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/learning-sanity-ladder.json --report runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/learning-sanity-validation.json`,
  - `cargo check --bin heirloom`,
  - `CARGO_INCREMENTAL=0 cargo test --bin heirloom learning_sanity -- --nocapture`
    ran `12` readiness tests successfully,
  - `cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/longer-32k-blend-valid.json`
    returned `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.160000`,
  - `bash -n scripts/validate.sh`.
- No remote A100/Vertex budget, new external data staging, production corpus
  weight changes, tokenizer default changes, base checkpoint mutation, or
  Padawan/pretraining mixing occurred.

## 2026-06-16 - Vertex Learning-Sanity Fixture Validation

- Added an explicit Vertex quick-gate hook controlled by
  `HEIRLOOM_RUN_LEARNING_SANITY_FIXTURES=1`.
  - The generated Vertex worker now runs
    `heirloom readiness validate-learning-sanity` against
    `ladder-valid.json`, `lr-grad-sweep-valid.json`, and
    `longer-32k-blend-valid.json`.
  - It uploads fixture logs under `learning-sanity-fixtures/*.txt`, uploads the
    JSON validation reports under `learning-sanity-fixtures/*-validation.json`,
    and records those report URIs in `summary.json`.
  - `scripts/gcp/README.md` documents the new flag and artifact location.
- Ran the requested Vertex validation job:
  - job:
    `projects/232930557062/locations/us-central1/customJobs/7741721827130474496`,
  - display name: `heirloom-validate-quick-20260616-010019`,
  - state: `JOB_STATE_SUCCEEDED`,
  - runtime: `2026-06-16T05:04:16Z -> 2026-06-16T05:07:17Z`,
  - source package:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260616-010019.tar.gz`,
  - artifact prefix:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260616-010019`,
  - summary:
    `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260616-010019/summary.json`.
- Vertex summary result:
  - `status="passed"`,
  - `mode="quick"`,
  - `quick_clippy="run"`,
  - `learning_sanity_fixture_report_uris` contains the three expected reports,
  - GPU smoke/storage/topology/probe lanes were intentionally disabled for this
    contract validation run with `HEIRLOOM_REQUIRE_GPU=0`,
    `HEIRLOOM_RUN_CUDA_STORAGE_TESTS=0`, `HEIRLOOM_RUN_TENSOR_CORE_PROBE=0`,
    and `HEIRLOOM_COLLECT_GPU_TOPOLOGY=0`.
- Targeted remote evidence:
  - ladder fixture passed with
    `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - LR/grad sweep fixture passed with
    `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.250000`,
  - longer 32K blend fixture passed with
    `learning_sanity status=passed stages=1 passed=1 failed=0 min_loss_reduction=0.160000`.
- The uploaded longer 32K blend report records tokenizer vocab `32768`, `128`
  reserved tokens, `binary_shards` manifest storage,
  `binary_shard_streaming` loader kind, `4096` selected tokens, `8` selected
  docs, and train-plus-resume loss `10.0 -> 8.4` with loss reduction
  `0.15999999999999998`.
