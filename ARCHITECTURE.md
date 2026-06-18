# Heirloom Tensor Runtime Architecture

This crate is a serious small prototype, not a PyTorch replacement. The goal is to make the difficult parts visible: tensor metadata, view semantics, gradient propagation, module composition, device/dtype boundaries, and the places where a toy design hits real framework pressure.

## Core Tensor Model

`Tensor` is an `Rc<RefCell<TensorInner>>` handle. Clones share identity and storage, which lets parameters move through modules and optimizers without copying. Each tensor tracks:

- `shape: Vec<usize>`
- `strides: Vec<usize>`
- `storage_offset: usize`
- `dtype: DType`
- `device: Device`
- `storage: Rc<Storage>`
- `requires_grad`, `grad`, and optional `grad_fn`

Implemented tensor dtypes are `DType::F32`, `DType::BFloat16`, `DType::F64`, `DType::I64`, and `DType::Bool`. CPU tensors remain the default full training path. `Device::Cuda(id)` now exists as an explicit storage/copy surface backed by `heirloom-kernels::cuda::CudaBuffer`. First Tensor-level CUDA kernels exist only for f32<->bf16 casts, contiguous full-storage f32 elementwise arithmetic, ReLU, GELU, rank-2 matmul with dense or strided rank-2 operands, `[rows, cols] + [cols]` f32 bias-add broadcasting, f32 embedding lookup, all-element sum/mean reductions, last-dimension f32 layer norm, rank-2 f32 cross-entropy loss, strict f32 fused causal attention, SGD updates, and AdamW updates with CUDA-resident moment buffers, with CUDA-resident backward for that same narrow differentiable surface plus cast/view/2D-transpose graph nodes. Module-level `.to_device`, `train-lm --device cuda:0`, and `train-lm --precision bf16` can now drive a tiny transformer LM-shaped CUDA smoke path with BF16 activation rounding.

A first NCCL DDP seam also exists: one process per CUDA device, rank-local backward, NCCL f32 gradient all-reduce, and local AdamW. BF16 `mma.sync` Tensor Core probe/counters exist for sm80+ validation; `Linear` projections can use BF16 Tensor Core RHS-transposed GEMM with device-side pad/crop for ragged dimensions in forward and matching input/weight backward matmuls, now launched by a four-warp shared-memory-staged CTA kernel by default, with an opt-in eight-warp XOR-swizzled CTA kernel available for validation runs. Guarded GEMM-only throughput tiers now also include A100 microbench-validated `ldmatrix` shared-memory MMA and double-buffered `cp.async` staging kernels, but they are not default train-path production kernels. Strict causal attention can use BF16 Tensor Core QK/AV forward plus score-grad/dQ/dK/dV backward matmuls with boundary-predicated ragged `time`/`head_dim` edges. The launcher has now completed both a bounded 4x A100 TinyStories DDP reference tier and the full 500-step 4x TinyStories-valid DDP gate with train/eval/generation plus checkpoint-resume reports. Real production AMP parity, train-path production routing for the new instruction-path GEMM tiers, tuned/general/batched Tensor Core GEMM, fused/streaming attention, broader CUDA operator dispatch, peer copies, multi-node distributed training, and production GPU model training are still deliberately absent.

The crate is now split by concern:

- `storage`: dtype, device, CPU/CUDA storage, explicit host/device copies, and shared storage version counters
- `shape`: shape, stride, indexing, broadcasting, and permutation helpers
- `dispatch`: builtin operator schemas, dtype/device resolution, and CPU scalar kernel entry points
- `extension`: Rust-side custom unary op forward/backward callback API
- `grad_mode`: thread-local `no_grad` behavior
- `npy`: pure Rust `.npy` tensor read/write for the supported CPU dtypes
- `rng`: deterministic RNG and tensor initializers
- `tensor`: tensor API, view construction, graph recording, and forward op methods
- `tensor/autograd`: autograd node definitions, graph traversal helpers, and backward formulas
- `nn`: modules, losses, and optimizer surface

## Storage And Views

Storage is a shared `StorageData` enum behind `RefCell`, with concrete CPU `Vec<f32>`, `Vec<u16>` BF16 bits, `Vec<f64>`, `Vec<i64>`, and `Vec<bool>` variants. CUDA storage is represented by a safe `CudaBuffer` handle owned by `heirloom-kernels`, which keeps all Driver API FFI and `unsafe` code outside the main `#![forbid(unsafe_code)]` crate. Logical tensor data is produced by walking `shape`, `strides`, and `storage_offset`; CUDA data can be explicitly materialized through `Tensor::cpu()` or typed data accessors for inspection. Storage also tracks a shared version counter. In-place mutation through a tensor or view bumps that counter.

`Tensor::to_device`, `Tensor::cuda`, and `Tensor::cpu` provide explicit device copies. CPU-to-CUDA and CUDA-to-CPU copies currently materialize contiguous logical tensors. `Tensor::to_dtype` supports CPU conversions and CUDA f32<->bf16 conversions, recording differentiable cast nodes when gradients are enabled. CUDA-to-CUDA copies, peer access, broader device-side dtype casts, and view-aware device copy kernels are not implemented. Unsupported CUDA math resolves to a clear device error instead of falling back to CPU execution.

`transpose`, `permute`, `narrow`, and `expand` are real strided views. `expand` uses zero strides, which exposes internal-overlap hazards. In-place writes through internally overlapping tensors are rejected. `view` is a metadata-only reshape for contiguous tensors only. `reshape` uses `view` when possible and otherwise materializes a contiguous copy. `as_strided` exists as a checked non-differentiable escape hatch because differentiable overlapping alias semantics are one of the hard parts.

Gradients are represented internally as f64 gradient tensors rather than bare `Vec<f64>` buffers. A gradient tensor carries independent storage, shape, strides, storage offset, dtype, and device metadata. The public `grad()` accessor preserves the original f32-oriented API by downcasting logical gradient values, `grad_f64()` exposes logical f64 values, and `grad_tensor()` exposes the storage-aware gradient tensor itself. First accumulation preserves non-overlapping target layouts, such as transposed and narrowed leaf views, while internally overlapping layouts fall back to dense contiguous gradient storage so arbitrary seed gradients remain representable. This still punts on full alias graphs, exact overlap analysis, and shared-storage gradient accumulation across aliased tensor handles.

## Primitive Ops

Implemented ops:

- `add`, `sub`, `mul`, `div`, including NumPy/PyTorch-style trailing-dimension broadcasting
- `matmul`, rank-2 only
- `matmul`, rank-2 and batched with batch-dimension broadcasting
- `transpose`, `permute`, `narrow`, `view`, `reshape`, `contiguous`, and checked non-differentiable `as_strided`
- `expand` as a zero-stride broadcast view
- `relu`
- `gelu`
- `embedding`
- `layer_norm_last_dim`
- `masked_fill`
- fused causal self-attention
- `argmax_last_dim`
- `softmax_dim`
- `sum`, all elements
- `mean`, all elements
- `sum_dim`, `mean_dim`, and signed-dim variants with `keepdim`

`mse_loss` is now composed from primitive `sub`, `mul`, and `mean`, so shared-subgraph gradient accumulation is exercised by a real loss.

`cross_entropy_for_logits` is intentionally fused and rank-2: stable log-sum-exp classification loss is usually fused in serious runtimes, and it avoids forcing an unstable `log(softmax(x))` path. `cross_entropy_for_logits_tensor` accepts i64 tensor targets, which lets CUDA language-model loss save targets on device rather than rehydrating a host target vector. The tiny language-model path flattens `[batch, time, vocab]` logits and `[batch, time]` targets into this rank-2 loss.

Binary arithmetic has explicit prototype promotion rules: f64 dominates f32, f32 dominates bf16, floats dominate i64/bool, integer-only add/sub/mul produce i64, and integer-only div produces f64. Bool subtraction/division are rejected. `matmul`, `relu`, `gelu`, `softmax_dim`, `cross_entropy_for_logits`, `layer_norm_last_dim`, and fused causal self-attention are floating-only. `embedding` requires i64 indices plus floating weights, `masked_fill` requires a bool mask while preserving input dtype, and `argmax_last_dim` returns non-differentiable i64 indices. `sum` maps bool/i64 to i64, while `mean` maps bool/i64 to f64.

## Operator Registry And Dispatch

`dispatch` now contains a builtin operator catalog rather than only loose kernel functions. Each `OperatorInfo` declares:

- the operator identity and display name
- operator kind, such as binary, unary, matrix multiply, loss, reduction, normalization, indexing, masking, selection, or attention
- dtype rule, such as arithmetic promotion, floating promotion, preserve-floating, reduction output rules, embedding lookup, mask preservation, argmax index output, or same-floating-inputs
- alias policy
- autograd policy

The public `operator_catalog()` API exposes that catalog for inspection, and `OPERATORS.md` summarizes the currently registered surface. Tensor methods create lightweight `TensorMeta { dtype, device }` values and ask the builtin `KernelRegistry` to resolve each operation, including transformer-critical ops such as `embedding`, `gelu`, `layer_norm_last_dim`, `masked_fill`, `argmax_last_dim`, and fused causal self-attention. Resolution performs dtype/device validation, computes output dtype, and returns a CPU kernel function pointer when the kernel lives in dispatch. Tensor methods consult the schema's fresh-output and autograd policies, but still own shape-specific logic, graph recording, and saved tensor metadata.

This is a real architectural step, but it is still far from PyTorch's dispatcher. There is no dynamic registration, dispatch-key set, backend fallback, generated operator schema, boxed calling convention, custom backend hook, or extension ABI.

`DISPATCH.md` outlines the production path from the current builtin registry to schema objects, dynamic registration, dispatch keys, boxed calls, and autograd kernel registration.

## Extension API

`extension` exposes `CustomUnaryOp`, `CustomUnaryRegistry`, `CustomUnaryForwardContext`, and `CustomUnaryBackwardContext`. A custom unary op is a named, floating-only, shape-preserving operator with a Rust forward callback and a Rust backward callback. `Tensor::apply_custom_unary` runs the forward callback, validates output length, records a `GradFn::CustomUnary` node when gradients are enabled, and invokes the custom backward callback during reverse-mode traversal.

The custom backward path receives saved input data, saved output data, upstream gradient, shape, dtype, and device. It validates returned gradient length and uses the same saved tensor version checks as builtin ops. `CustomUnaryRegistry` provides explicit local registration, duplicate-name checks, lookup, and listing. It is intentionally separate from the builtin dispatch catalog until dynamic registration is designed. `EXTENDING.md` documents the supported API and its limits.

`ALIASING.md` documents the current aliasing invariants, the completed first gradient-tensor phase, and the production plan for view metadata, overlap analysis, in-place rules, and alias-aware accumulation.

## Autograd

Each differentiable result stores a `GradFn` enum variant with parent tensors and any metadata needed for backward. Calling `backward` builds a topological ordering from the output to leaves, seeds the output gradient, and walks the graph in reverse order. Gradients accumulate by tensor identity, so shared subgraphs and repeated parents such as `x.mul(&x)` work.

Ops that need forward values during backward use `SavedTensor`, which records the storage version at graph construction. If the saved storage is mutated before backward, backward errors instead of silently computing with stale assumptions. The runtime also has thread-local `no_grad`, `detach`, explicit non-scalar `backward_with_grad`, and version bumping for in-place updates.

By default, successful backward releases non-leaf graph nodes. A second backward through the same graph errors instead of silently stopping at a stale node. `backward_retain_graph`, `backward_with_grad_retain_graph`, and `backward_with_grad_f64_retain_graph` keep the graph alive for explicit repeated backward. Non-leaf gradients are transient engine buffers and are cleared after use unless `retain_grad()` is called. Leaf gradients remain accumulated.

Gradient hooks can be registered on tensors with `register_grad_hook`, removed by id, and can inspect or replace incoming logical f64 gradients before accumulation. Hook return shapes are validated, and hooks on retained non-leaves affect both the retained gradient and the upstream gradient used by parent backward formulas.

This is closer to a small dynamic autograd tape than to PyTorch's production engine. It still lacks thread-safe graph execution, higher-order gradients, arbitrary custom autograd functions, saved tensor hooks, activation checkpointing, alias-graph-aware accumulation, and dispatch-key integration.

CUDA autograd is currently a narrow side path rather than a general engine. For f32 CUDA tensors, elementwise `add`/`sub`/`mul`/`div`, `[rows, cols] + [cols]` bias add, `relu`, `gelu`, rank-2 `matmul` over dense or strided rank-2 layouts, `embedding`, all-element `sum`/`mean`, `layer_norm_last_dim`, rank-2 `cross_entropy_for_logits`, and strict fused `causal_self_attention` can propagate CUDA-resident gradients, and repeated-parent accumulation uses a CUDA add kernel. CUDA f32<->bf16 casts propagate FP32 gradients on device, which lets the BF16 activation policy round activations without CPU staging. CUDA view backward is a dense logical passthrough, and rank-2 transpose backward uses a device-side transpose kernel so `Linear`-style `input.reshape(...).matmul(weight.transpose()).add(bias).reshape(...)` can keep gradients on device. CUDA embedding backward uses a device-side scatter-add kernel so repeated rows accumulate on the GPU. CUDA GELU backward uses the same tanh-approximation derivative as the CPU runtime. CUDA layer norm backward computes input plus affine weight/bias gradients on device; CUDA cross-entropy can save its target tensor on device and compute logits gradients without CPU fallback; CUDA causal attention saves its softmax weights as a CUDA tensor and computes Q/K/V gradients on device. The AMP causal-attention Tensor Core path accelerates QK/AV forward matmuls plus backward score-grad/dQ/dK/dV matmuls with boundary-predicated ragged edges, while f32 CUDA kernels still handle softmax-backward and transpose glue. `Tensor::apply_sgd` can update contiguous full-storage f32 CUDA parameters with CUDA-resident gradients through a device-side SGD kernel. `AdamW::step_mut` can update contiguous f32 CUDA parameters with CUDA-resident first/second moment buffers; gradient clipping uses a device-side sum-of-squares reduction and reads back only a scalar norm contribution. `AdamW::all_reduce_cuda_gradients_nccl` can all-reduce CUDA f32 gradients in place through NCCL and scale them by `1/world_size` before local AdamW. CUDA gradient hooks, general CUDA broadcasting, dimension reductions, general CUDA batched matmul, production AMP, and fully validated production GPU training remain unimplemented.

## NN Layer

The `nn` module exposes:

- `Module` trait
- `Linear`
- `Embedding`
- `LayerNorm`
- `Gelu`
- `CausalSelfAttention`
- `FeedForward`
- `TransformerBlock`
- `TinyTransformerLm`
- `ReLU`
- `Sequential`
- `mse_loss`
- `sgd_step`
- `Optimizer` trait
- `Sgd`
- `AdamW`
- state_dict save/load helpers

`Linear` stores weight as `[out_features, in_features]` and computes `input.matmul(weight.transpose()).add(bias)`. Bias uses broadcasting, so its gradient exercises unbroadcasting.

`Linear::new` uses `Linear::DEFAULT_SEED` through `Linear::new_with_seed`, and `Linear::new_with_rng` accepts a caller-owned Heirloom RNG for reproducible initialization. `Linear` now applies over the last dimension, so rank-3 transformer activations can pass through projections without manual flattening. `save_state_dict` writes a directory containing a `state.tsv` manifest and one `.npy` file per named parameter. `load_state_dict` loads those tensors back into matching module parameters by name and shape.

`TinyTransformerLm` is a one-block decoder-only language model surface: token embedding, position embedding, pre-norm causal self-attention, feed-forward block, final layer norm, and LM head. It trains through the same autograd path as the rest of the runtime. Concrete module types expose `.to_device(Device)` so parameters can move as a tree; the CUDA path avoids position-embedding `expand` by gathering repeated position ids directly into `[batch, time, d_model]`. `forward_bf16_activations` and `loss_bf16_activations` insert differentiable f32->bf16->f32 activation round-trips while leaving parameters, gradients, optimizer state, logits, and losses in f32. `forward_amp_bf16` and `loss_amp_bf16` use the same FP32-parameter/BF16-activation policy but route CUDA `Linear` projections through BF16 Tensor Core RHS-transposed GEMM with explicit device-side pad/crop for ragged dimensions in forward and through matching BF16 Tensor Core matmuls for input/weight gradients. AMP attention routes QK and AV forward matmuls through BF16 Tensor Core kernels and routes the first backward slice through Tensor Core matmuls for score gradients, dQ, dK, and dV, with predicated loads/stores for ragged `time` and `head_dim` edges and f32 CUDA kernels still handling softmax-backward and transposition glue. Unsupported projection devices or dtypes fall back unless `HEIRLOOM_REQUIRE_TENSOR_CORES=1` is set; unsupported attention Tensor Core devices or invalid attention shapes fall back unless `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1` is set. Gradients remain FP32 CUDA tensors.

Generation supports greedy decoding and seeded sampling through temperature, top-k, top-p, repetition penalty, frequency penalty, and presence penalty. The generation API returns generated token ids, per-step probabilities, a finish reason, and final RNG state so CLI reports can be deterministic and inspectable.

Evaluation uses deterministic non-overlapping token windows over a provided token stream. It computes a token-weighted mean cross-entropy loss under `no_grad` and reports perplexity as `exp(loss)`. This is a simple language-model validation loop, not a streaming/distributed evaluator.

## Native Runtime Support

`npy` implements a narrow pure Rust NumPy `.npy` reader/writer for CPU f32, f64, i64, and bool tensors. It supports C-order arrays for those descriptors and preserves logical tensor layout when writing views. It intentionally rejects unsupported dtypes, Fortran-order arrays, and broader `.npz` packaging.

`heirloom-kernels::cuda` is the first real GPU boundary. It dynamically loads the CUDA Driver API and NCCL, uses CUDA primary contexts for owning tensor buffers, provides safe typed `CudaBuffer` host/device copies, exposes compute-capability and peer-access discovery, owns stream-backed `NcclCommunicator` handles, JIT-loads embedded PTX with error-log capture, and launches f32<->bf16 casts, f32 elementwise, bias-add, GELU, rank-2 dense/strided matmul, 2D transpose, embedding gather/scatter-add, layer norm, cross-entropy, causal attention forward/backward, reduction, sum-of-squares, fill, ReLU, ReLU-backward, matmul-backward, SGD-update, and AdamW-update kernels.

The CUDA runtime layer now has a per-thread primary-context allocation cache with stream/event deferred frees, explicit cached compute streams, per-context PTX module caching, and profiler-style counters for launches, syncs, copies, allocations, module loads, Tensor Core padded/remainder tiles, attention padded/remainder/fallback tiles, CTA GEMM calls, CTA tiles, launched CTA warps, active MMA warp tiles, staged CTA GEMM calls, shared-stage tiles/bytes, wide-swizzled CTA GEMM calls, swizzled-stage tiles/bytes, guarded ldmatrix/cp.async GEMM calls and instruction counters, global-fed CTA debug calls, legacy one-warp GEMM calls, and NCCL bytes. It also has a separate sm80 PTX module for `heirloom_bf16_mma_probe`, legacy `heirloom_matmul_bf16_mma_rhs_t_f32`, debug global-fed CTA `heirloom_matmul_bf16_mma_rhs_t_f32_cta`, default shared-memory staged CTA `heirloom_matmul_bf16_mma_rhs_t_f32_cta_staged`, opt-in eight-warp XOR-swizzled CTA `heirloom_matmul_bf16_mma_rhs_t_f32_cta_wide_swizzled`, guarded `ldmatrix` and double-buffered `cp.async` GEMM throughput kernels, and BF16 Tensor Core causal-attention QK/AV forward plus backward matmul tiles with predicated ragged-edge loads/stores.

The Linear GEMM wrapper pads BF16 matrices on device to `M%16=0`, `K%16=0`, `N%8=0`, runs the default four-warp staged CTA kernel over 32x16 output regions, then crops the f32 output back to the logical shape while reporting padded/remainder, CTA tile, and shared-stage counters. `HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM=1` selects the opt-in 32x32 output-region kernel with eight warps per CTA and XOR-swizzled shared-memory staging; `HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM=1` and `HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1` select A100 microbench-validated guarded instruction-path GEMM tiers; `HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM=1` selects the previous global-memory-fed CTA kernel; and `HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM=1` selects the older one-warp-per-16x8-tile kernel for debugging. These guarded instruction-path GEMM tiers are not yet default train-path production kernels.

The attention kernels use ceil launch grids and boundary predicates for ragged `time` and `head_dim`; the Tensor Core tiled flash forward candidate has a separate A100 attention-only forward gate, but remains an isolated no-grad throughput path until flash backward exists. The same GEMM wrapper is used for AMP `Linear` forward and for the matching input/weight backward matmuls, with BF16 layout-materialization, transpose, pad, and crop kernels preparing logical operands. Attention backward decomposes into Tensor Core matmuls for `dAttention = dO * V^T`, `dQ = dScore * K`, `dV = attention^T * dO`, and `dK = dScore^T * Q`, while f32 CUDA kernels handle the causal softmax backward and square transposes. This keeps unsafe CUDA/NCCL FFI outside the main `#![forbid(unsafe_code)]` crate. Tensor CUDA storage now exists, and Tensor f32 arithmetic/ReLU/GELU/rank-2-matmul/Linear-shaped projection pieces/embedding/layer-norm/cross-entropy/causal-attention/sum/mean plus f32<->bf16 casts can dispatch to device-resident forward/backward kernels for the supported layouts; `TinyTransformerLm` can execute a tiny CUDA forward/backward/AdamW/checkpoint smoke path; `apply_sgd` and `AdamW::step_mut` can mutate strict f32 CUDA parameters in place and bump storage versions.

The 4x launcher has now proven NCCL init plus f32 all-reduce on A100s, a bounded 4x DDP tiny fixture has trained/resumed with NCCL gradient all-reduce, Tensor Core Linear matmuls, and zero final/per-step parameter-checksum drift, and the full 4x TinyStories-valid DDP gate has trained/resumed/evaluated/generated on public text with synchronized parameters and positive Linear plus attention Tensor Core counters. Train-path production routing for the new instruction-path GEMM tiers, bank-conflict/occupancy tuning, general batched Tensor Core matmul, production flash attention forward/backward, production allocator tuning, and production kernel scheduling remain open. All other CUDA math is intentionally guarded until each CUDA kernel/autograd formula is registered deliberately.

`rng` implements a deterministic SplitMix64-based generator, uniform and normal scalar sampling, and uniform/normal tensor initializers. This is not a cryptographic RNG and not a PyTorch RNG parity implementation; it exists to make Heirloom initialization reproducible without depending on Python.

`tokenizer` implements a small byte-level BPE-like tokenizer with `<pad>`, `<bos>`, and `<eos>` IDs. Saved tokenizer JSON now carries a format string, version, special-id metadata, training byte count, and a stable training hash; loading validates the byte table and merge graph before use. Legacy pre-metadata tokenizer JSON is still accepted and upgraded in memory.

`data` provides deterministic token batching, dataset RNG/batch state, a TinyStories validation-split downloader, and a prepared language-model dataset path. `heirloom data prepare` encodes a source text file once, writes train/valid token files, records hashes and source/tokenizer provenance in `manifest.json`, and lets `train-lm` consume token files directly through `--dataset-manifest`.

`checkpoint` bundles model state, optimizer state, tokenizer, config, step, dataset RNG state, dataset batch count, and optional prepared-dataset manifest path into a directory. This is enough for exact next-batch continuation of the Heirloom token dataset, but it is still not a production checkpoint archive.

## CLI Demo

`cargo run --bin train` trains a one-layer linear regression model on deterministic synthetic data generated from:

```text
y = 3*x0 - 2*x1 + 0.5
```

The demo prints periodic loss values and fails if final loss is not less than 5% of initial loss.

`cargo run --bin classify` trains a linear three-class classifier on deterministic synthetic clusters with cross-entropy loss. It prints loss and accuracy and fails if loss does not decrease substantially or accuracy does not exceed 95%.

`cargo run --bin custom` runs a chained custom unary autograd demo and checks both forward values and gradients.

`cargo run --bin heirloom` is the small real training CLI. It can download the public TinyStories validation split, train a tokenizer, prepare a manifest-backed token dataset, train/resume a tiny decoder LM, evaluate train/valid loss and perplexity, save checkpoints, and generate text from a prompt. `train-lm`, `eval-lm`, and `generate` accept `--precision f32|bf16|amp-bf16`; `bf16` selects the activation-rounding path, while `amp-bf16` is the CUDA mixed-precision policy surface with Tensor Core Linear forward/backward using explicit pad/crop for ragged dimensions and Tensor Core attention forward/backward matmuls with predicated ragged-edge handling. `AmpBf16Policy` records the current policy: FP32 master parameters, FP32 gradients, FP32 AdamW state, BF16 Tensor Core operands for eligible matmuls, FP32 accumulation/loss/reduction/normalization/softmax, finite checks, and no BF16 loss scaling. Strict AMP scopes reject unexpected CUDA-to-CPU tensor materialization; scalar logging, checkpoint serialization, eval loss reads, generation logits inspection, and explicit inspections are allowed through scoped host-staging allowances and reported. `heirloom gpu tensor-core-probe --device 0` validates that the CUDA driver can JIT and execute BF16 MMA on sm80+. `heirloom gpu nccl-probe --devices cuda:0,cuda:1,... --probe-kind spawn|cuda-context|nccl-init|all-reduce` runs a staged rank launcher before any model training: every rank receives a config file and writes separate stdout, stderr, stage, and report artifacts, while the parent writes `launcher-report.json` with PIDs, exit states, timeout status, and last known stages. For CUDA `Linear` projections, `amp-bf16` casts operands to BF16, pads to MMA tile boundaries on device when needed, calls the default shared-memory staged CTA Tensor Core GEMM for forward plus matching backward matmuls, crops logical outputs, and reports padded/remainder plus CTA and shared-stage counters; setting `HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM=1` switches Linear GEMM to the opt-in wide/swizzled CTA family and reports wide-swizzled calls plus swizzled-stage tiles/bytes. Unsupported devices/dtypes fall back unless `HEIRLOOM_REQUIRE_TENSOR_CORES=1` is set. For attention, `amp-bf16` routes QK/AV forward plus score-grad/dQ/dK/dV backward matmuls through Tensor Core kernels with attention padded/remainder/fallback counters; unsupported attention devices or invalid shapes fall back unless `HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1` is set. Train, rank, eval, and generation reports include AMP policy, finite checks, per-op AMP decisions, host-staging events, and an `amp_bf16_validation` status; train and rank reports also include raw Tensor Core counters, CUDA runtime counters, `tensor_core_coverage`, and a named `tensor_core_pad_crop` section with used/passed/status, logical and padded Linear shapes, padded/remainder tile counts, and scalar/Linear fallback counts. Runtime counters now distinguish staged CTA GEMM, wide-swizzled CTA GEMM, global-fed CTA debug GEMM, legacy one-warp GEMM calls, and attention edge tile handling. `train-lm`, `eval-lm`, and `generate` also accept `--device cpu` by default and a narrow opt-in `--device cuda:<id>` path that moves the model and token batches to CUDA, saves CUDA checkpoints through explicit CPU snapshots, and reloads checkpoints back onto the requested device. `train-lm --devices cuda:0,cuda:1,... --distributed nccl` uses the same bounded launcher lifecycle, gives each rank deterministic sharded batches, all-reduces gradients with NCCL, and writes rank plus aggregate reports including Tensor Core counters, CUDA runtime counters, all-reduce counts/bytes, final scalar parameter-checksum drift, and per-step checksum drift controlled by `--ddp-checksum-every`. Generation still samples on the host after explicitly materializing next-token logits. The validation script exercises tokenizer training, data preparation, manifest-backed train/resume, a tiny BF16 train/eval/generate pass, checkpoint load, heldout eval, sampled generation, and JSON report writing on a tiny local fixture. `scripts/run_tinystories_cuda_reference.sh` provides smoke/reference/full CUDA public-data tiers and writes a summary with model config, precision, distributed metadata, data hashes/token counts, gate threshold, artifacts, timings, optional checkpoint-resume reports, DDP rank-launcher artifacts, Tensor Core coverage, and Tensor Core pad/crop status. The full uncapped TinyStories CUDA run remains opt-in because the current CUDA kernels are correctness-first, but recorded single-A100 and full 4x A100 DDP gates now exist.

`cargo run --example microgpt_heirloom` is a compact Karpathy-inspired GPT training artifact. It intentionally uses Heirloom's tensor runtime, autograd, tokenizer, AdamW, and tiny transformer modules rather than reimplementing scalar micrograd-style autograd.

## Property And Parity Testing

The crate now uses `proptest` for randomized shape/gradient properties over broadcasting, expand, reductions, and matmul. These tests are not a replacement for proof or exhaustive coverage, but they catch classes of shape bugs that fixed fixtures miss.

PyTorch parity has two layers:

- `tests/fixtures/pytorch_parity.json` is consumed by Rust tests and checked during `cargo test`.
- `tools/generate_pytorch_fixtures.py` can refresh that fixture from live PyTorch when `torch` is installed.

The local workspace now has a project-local `.venv` with PyTorch, NumPy, pytest, Hypothesis, and maturin installed. The checked-in fixture is stamped `generated_by: torch 2.12.0` and includes core tensor/autograd cases plus transformer-runtime cases for batched matmul, tanh-GELU, layer norm, embedding scatter-add, and fused causal attention.

`heirloom-python` is a thin PyO3 `cdylib` crate that depends on the safe `heirloom` runtime and builds the `heirloom_py` module through maturin. It exposes enough Tensor construction, metadata, materialization, autograd, view, transformer-op, and loss surface to write direct Python parity tests against live `torch`. The boundary is deliberately a test harness: it does not expose modules, optimizers, dataloaders, checkpointing, Python custom autograd, or a PyTorch-like ergonomic API.

`scripts/python_parity.sh` builds that extension into `.venv` and runs `python/tests/test_heirloom_py_parity.py`. Those tests compare Heirloom directly against live PyTorch for matmul/ReLU/mean backward, broadcast plus transpose gradients, layer norm, embedding repeated-row accumulation, cross-entropy, fused causal attention, NumPy round-trips, and a skipped local-CUDA smoke when CUDA is unavailable.

`Dockerfile` and `scripts/docker_validate.sh` provide a reproducible CPU validation image. The image is not a CUDA/GPU runtime.

## Comparison Against PyTorch Concepts

### Dispatcher

PyTorch routes ops through a dispatcher keyed by dtype, layout, device, autograd state, and extension registrations. Heirloom now has storage variants, promotion rules, a builtin operator catalog, and registry-mediated CPU kernel resolution, but it is not a real dispatcher: there is no dispatch key set, backend fallback, generated bindings, boxed calling convention, or external registration.

### Autograd Tape

PyTorch dynamically records operation nodes and uses a robust engine to execute backwards across devices and threads. Heirloom records a `GradFn` enum per result and runs a single-threaded DFS/topological pass.

### Tensor Views

PyTorch has deep alias tracking, version counters, view replay, and in-place mutation validation. Heirloom has shape/stride/offset metadata, real views, saved-tensor version checks, and storage-aware gradient tensors, but it does not handle overlapping differentiable views, complex in-place rules, exact overlap classification, or alias-graph-aware gradient accumulation.

### Device Backends

PyTorch separates tensor metadata from backend kernels and supports many backends. Heirloom has explicit `Device::Cpu` and `Device::Cuda(id)`, real CUDA storage copies, f32<->bf16 CUDA casts, first narrow Tensor CUDA forward/backward kernels including GELU, rank-2 dense/strided matmul, bias-add broadcasting, embedding scatter-add, layer norm, rank-2 cross-entropy, fused causal attention, and Linear-shaped view propagation, plus narrow CUDA SGD/AdamW updates, BF16 activation rounding, BF16 Tensor Core Linear forward/backward GEMM with explicit pad/crop edge handling, boundary-predicated BF16 Tensor Core attention forward plus backward matmul decomposition, and a full single-node 4x NCCL DDP public-data gate. It still has no backend registry, dispatch-key set, general CUDA autograd engine, broad stream model, general batched Tensor Core kernel table, fused/streaming attention kernel, production AMP, multi-node distributed backend, or full production model-training GPU backend yet.

### Extension Points

PyTorch supports C++/Python extensions, custom operators, custom autograd, dispatch keys, and private-use backends. Heirloom now has a Rust-side custom unary autograd API, a separate custom unary registry, and a thin PyO3 parity-test wrapper, but extension points are still static, narrow, CPU-only from the custom-op side, and not integrated with a dynamic dispatcher or backend registry.
