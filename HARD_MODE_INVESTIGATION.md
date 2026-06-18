# Hard Mode Investigation

This report investigates every area called out in `HARD_MODE.md` and sketches hard-path implementation approaches. It intentionally avoids the happy-easy path of "just add another op" or "just special-case the tiny LM." The current codebase already has meaningful hard-mode foundations: storage-aware gradients, saved tensor version checks, a builtin operator catalog, opt-in CUDA storage/kernels, and live PyTorch parity harnesses. The remaining work is mostly about contracts, not surface area.

## Evidence Read

- Runtime docs: `ARCHITECTURE.md`, `ALIASING.md`, `DISPATCH.md`, `OPERATORS.md`, `PROGRESS.md`, `README.md`, `HARD_MODE.md`.
- Core implementation anchors: `src/storage.rs`, `src/shape.rs`, `src/dispatch.rs`, `src/tensor.rs`, `src/tensor/autograd.rs`, `src/nn.rs`, `src/npy.rs`, `src/data.rs`, `src/checkpoint.rs`, `src/tokenizer.rs`, `src/extension.rs`, `heirloom-kernels/src/cuda.rs`, `heirloom-python/src/lib.rs`.
- Test and gate anchors: `tests/*`, `python/tests/test_heirloom_py_parity.py`, `scripts/validate.sh`, CUDA/TinyStories scripts under `scripts/`.

## Top-Level Recommendation

The next serious implementation should not be another isolated kernel unless it is explicitly in service of one of these contracts:

1. Alias/view/indexing contract.
   Build replayable view metadata, exact overlap classification, PyTorch-like in-place validation, alias-aware gradient routing, and then tensor indexing on top of that model.

2. Dispatcher/backend contract.
   Move from a static operator catalog plus Tensor-owned branches to schema objects, dispatch keys, per-backend kernel tables, and registered autograd kernels.

3. Kernel iteration/memory contract.
   Replace `Vec<f64>` materialization with a stride-aware TensorIterator-style planner for CPU and with registered device kernels for CUDA. Add allocator and stream semantics before broadening CUDA.

4. Parity infrastructure contract.
   Broaden live PyTorch comparison, fuzzing, dtype/device/layout matrices, tolerance tracking, and sanitizer/stress gates before expanding risky semantics.

Everything else should hang off these contracts, or it will deepen the prototype-specific maze.

## Tensor Semantics

### DTypes, Scalar Tensors, Mixed Precision

Current evidence: `DType` is a small enum (`src/storage.rs:7`), `StorageData` has CPU variants plus CUDA handles (`src/storage.rs:22`), CPU casts materialize via f64 (`src/tensor.rs:605`), and CUDA casts only support f32<->bf16 (`src/tensor.rs:625`). `.npy` rejects BF16 (`src/npy.rs:8`).

Hard approach:

- Introduce an explicit dtype promotion table generated or snapshot-tested against PyTorch. Treat it as data, not a chain of local `match` choices.
- Add a scalar representation distinct from rank-0 Tensor construction: scalar literals need dtype/device inference, wrapping behavior, and promotion semantics.
- Add dtype families before individual dtypes: floating, integral, bool, complex, quantized. This keeps kernel registration honest.
- For float16/BF16, define policy layers: storage dtype, compute dtype, accumulation dtype, optimizer master dtype, autocast policy, and serialization policy.
- Complex and quantized tensors should not be "just another StorageData variant"; they need op semantics, conjugate/view behavior, promotion rules, and IO format choices.

Avoid: adding `F16` to `DType` and letting unsupported ops fall through CPU f64 materialization.

### CUDA Backend Scope

Current evidence: `Device::Cuda` exists (`src/storage.rs:235`), CUDA storage is backed by `CudaBuffer` (`src/storage.rs:267`, `src/storage.rs:310`), Tensor branches directly into CUDA kernels for specific ops (`src/tensor.rs:1786`, `src/tensor.rs:1884`, `src/tensor.rs:2738`), and CUDA backward has a separate f32 path (`src/tensor/autograd.rs:285`). The kernel wrapper uses synchronous launches and primary contexts (`heirloom-kernels/src/cuda.rs:134`, `heirloom-kernels/src/cuda.rs:2270`).

Hard approach:

- Define backend execution objects: allocator, stream, event, device guard, synchronization policy, and peer-copy policy.
- Move CUDA op routing into dispatch registration, not `Tensor` conditionals. A missing CUDA kernel should be a dispatch miss with schema context.
- Add device-to-device copy and peer access as first-class copy kernels before distributed or multi-GPU work.
- Use cuBLAS/cuDNN/NCCL only after the dispatcher and stream model can express them.
- Treat the existing A100 gates as narrow training evidence, not backend completion.

Avoid: continuing to add direct `if device != Cpu { return cuda_x(...) }` branches for every new op.

### Layouts

Current evidence: Tensor metadata is dense strided shape/stride/offset (`src/tensor.rs:32`), and overlap detection is zero-stride only (`src/shape.rs:22`). There is no layout field separate from strides.

Hard approach:

- Add a `Layout` abstraction, even if the only supported value remains dense strided at first.
- Add a meta layout early because it improves shape-inference testing without kernels.
- Treat channels-last as a dense-strided format with memory-format metadata; do not fake it as ordinary arbitrary strides.
- Sparse/nested/ragged layouts need separate storage and operator domains, so gate them behind schema layout constraints.

Avoid: accepting arbitrary stride metadata and assuming every op is dense-strided safe.

### Views, Indexing, Overlap

Current evidence: view ops are real (`src/tensor.rs:1484`, `src/tensor.rs:1525`, `src/tensor.rs:1641`, `src/tensor.rs:1687`), differentiable `as_strided` is rejected (`src/tensor.rs:1595`), CUDA view backward is narrow (`src/tensor/autograd.rs:569`), and `ALIASING.md` already lays out phases.

Hard approach:

- Promote view creation into structured `ViewOp` metadata with replay and inverse/reduce-backward behavior.
- Implement exact dense-strided overlap analysis: no overlap, full overlap, partial overlap, unknown. Zero stride is only one case.
- Build tensor indexing only after view metadata exists: basic slicing should be a view when possible; advanced indexing should be a copy/scatter family with its own backward.
- Negative strides require either storage offset normalization plus signed stride support, or an explicit decision to reject them with tests.
- CUDA view support should use view-aware kernels or materializing copy kernels registered per op, not broad CPU fallback.

Avoid: adding `tensor[index]` as ad hoc data extraction that discards aliasing.

### Gradient Tensors And Aliased Accumulation

Current evidence: `GradTensor` carries storage/layout/device metadata (`src/tensor.rs:58`) and CPU accumulation preserves non-overlapping layouts while CUDA accumulation requires dense f32 buffers (`src/tensor.rs:1057`, `src/tensor.rs:1134`). Gradients attach to tensor identity, not alias graph (`src/tensor/autograd.rs:1704`).

Hard approach:

- Keep independent gradient storage as the default invariant.
- Add alias graph IDs: base storage identity, view chain, differentiable/non-differentiable boundary, detached aliases.
- Route view gradients through replay/inverse transforms back to the base when PyTorch would accumulate on the base.
- Make ambiguous overlapping accumulation fail loudly at backward/accumulation time with view provenance.

Avoid: trying to share gradient storage with primal storage to "look alias-aware."

### In-Place And Versioning

Current evidence: storage mutation bumps a shared version (`src/storage.rs:415`), `add_` exists (`src/tensor.rs:1442`), and saved tensors check versions (`src/tensor/autograd.rs:10`).

Hard approach:

- Build a mutation validator with explicit cases: leaf requiring grad, view of leaf, view saved for backward, detached alias, mutation under no_grad, mutation after graph release.
- Add schema-level alias/mutation annotations before adding more in-place ops.
- Preserve version checks, but do not use them as the whole in-place model; PyTorch rejects many mutations before backward.

Avoid: relying on saved-tensor version errors as the only guardrail.

### Broadcasting And TensorIterator

Current evidence: binary CPU ops use shape broadcasting (`src/dispatch.rs:488`) and CUDA only supports equal shapes or Linear bias add (`src/tensor.rs:1786`). Reductions and transformer ops have custom loops.

Hard approach:

- Build a TensorIterator-style plan object: output shape, per-input strides, dtype result, overlap/write checks, parallel chunking, and kernel callback.
- Use the same planner for binary, unary, masked, reductions where possible.
- For CUDA, the same plan should lower to either a generic strided kernel or a registered specialized kernel.

Avoid: adding one broadcast rule per op.

## Autograd

### Engine Shape

Current evidence: graph discovery is recursive (`src/tensor/autograd.rs:1704`), backward walks a topo list in one thread (`src/tensor.rs:1228`), and CUDA has a side path for f32 buffer gradients (`src/tensor/autograd.rs:285`).

Hard approach:

- Replace recursive DFS with an explicit stack and graph task object, even before adding parallelism.
- Separate graph discovery, gradient queueing, accumulation, and node execution.
- Make device execution pluggable so CPU and CUDA backward kernels are registered node handlers rather than special traversal branches.

Avoid: adding threads around the current `Rc<RefCell<_>>` graph.

### Lifecycle, Hooks, Saved Tensors

Current evidence: detach/no_grad/version checks/retain APIs exist in Tensor and autograd (`src/tensor.rs:1182`, `src/tensor/autograd.rs:136`), and tensor grad hooks are CPU-only before accumulation (`src/tensor.rs:1057`, `src/tensor.rs:1134`).

Hard approach:

- Add saved tensor hook policy before activation checkpointing. Checkpointing needs controlled recomputation and saved tensor packing/unpacking.
- Add anomaly detection by recording op name, source location if available, tensor metadata, and saved-tensor provenance during forward.
- Treat hooks as graph-node contracts: tensor hooks, saved tensor hooks, and future custom autograd hooks should have ordering and dtype/device rules.
- For CUDA hooks, either materialize explicitly with a documented sync or require device-side hook support; no hidden CPU hop.

Avoid: implementing activation checkpointing as "drop saved data and rerun closure" without RNG/device/alias semantics.

### Higher-Order Gradients And Custom Autograd

Current evidence: `GradFn` is an enum of backward formulas (`src/tensor/autograd.rs:17`), custom ops are shape-preserving unary Rust callbacks (`src/extension.rs:5`), and backward outputs are plain gradient buffers.

Hard approach:

- Represent backward formulas as differentiable ops if higher-order gradients are desired. Returning `Vec<f64>` gradients cannot support grad-of-grad.
- Define a custom autograd function API with saved tensor policy, multiple inputs/outputs, non-differentiable outputs, dirty/alias annotations, and backward arity validation.
- Integrate custom autograd into dispatcher schema registration after alias metadata exists.

Avoid: adding `CustomBinaryOp` as another detached registry before the schema and alias model are ready.

### Non-Scalar Backward

Current evidence: scalar `backward()` errors on non-scalar outputs and explicit seed APIs exist (`src/tensor.rs:1182`, `src/tensor.rs:1212`).

Hard approach:

- Keep the scalar-only implicit seed behavior because that matches PyTorch's core rule.
- Improve ergonomics with `ones_like`, `sum().backward()`, and Python-style error messages rather than silently creating ones for non-scalars.
- Add parity tests around scalar rank-0 versus shape `[1]` semantics.

Avoid: auto-seeding non-scalar tensors with ones in `backward()`.

### Aliased Gradient Accumulation

Current evidence: gradients are attached to tensor identities; view gradients can preserve layout but are independent (`src/tensor.rs:58`, `src/tensor.rs:1057`).

Hard approach:

- Tie this to alias graph work. There is no safe shortcut.
- Define whether `.grad` appears on base, view, or both under `retain_grad`, and match PyTorch first for leaf views.
- Add fixtures for base/view/backward/mutation combinations before changing accumulation.

Avoid: merging gradients by storage pointer without view replay.

## Ops And Numerics

### Matmul And Linear Algebra

Current evidence: CPU `matmul_cpu` supports rank >= 2 with broadcast batch dims (`src/dispatch.rs:571`); CUDA matmul is rank-2 only (`src/tensor.rs:1884`).

Hard approach:

- Implement PyTorch's vector/matrix cases: 1D x 1D, 1D x 2D, 2D x 1D, batched broadcasting, empty dims.
- Split generic matmul semantics from backend kernels. Use schema lowering to GEMM, batched GEMM, vector dot, or composite reshape paths.
- Add tolerance studies and large randomized parity tests before optimizing.

Avoid: expanding only the shape used by the transformer.

### Reductions

Current evidence: all-element and single-dim reductions exist (`src/tensor.rs:2738`, `src/tensor.rs:2787`); empty mean is rejected (`src/dispatch.rs:641`).

Hard approach:

- Introduce a `ReductionSpec`: dims list, keepdim, dtype override, initial value policy, empty behavior, named-dim placeholder.
- Match PyTorch empty/NaN behavior where possible, including `mse_loss`.
- Add numerically stable reductions for large inputs and dtype-specific accumulation.

Avoid: adding `sum_dims` by looping over `sum_dim` and hoping strides/autograd line up.

### Nonsmooth And Missing Primitive Ops

Current evidence: ReLU derivative is fixed by the backward formula (`src/tensor/autograd.rs:1210`), and many primitives are absent.

Hard approach:

- Document derivative conventions for nonsmooth ops as part of operator schemas.
- Prioritize primitives that unlock composites: `neg`, `pow`, `exp`, `log`, `sqrt`, `maximum/minimum`, comparisons, `where`, `clone`, `copy`, `cat`, `stack`, `slice/index`, `dropout`.
- Let cross-entropy keep its fused implementation, but add reduction modes and a schema-level loss contract.

Avoid: treating fused ops as substitutes for primitive coverage.

### CPU Kernel Strategy

Current evidence: Tensor methods routinely materialize `data_f64()` and allocate output Vecs (`src/tensor.rs:1748`, `src/tensor.rs:2054`, `src/tensor.rs:2886`).

Hard approach:

- Introduce typed storage iterators and TensorIterator CPU loops with f32/f64/i64/bool specialization.
- Use `matrixmultiply` or BLAS for dense GEMM, but only after dtype/layout dispatch can choose the correct kernel.
- Add rayon/chunking where iteration order is deterministic or explicitly documented.
- Track numerical tolerances per op, dtype, backend, and shape.

Avoid: hiding f64 materialization behind helper functions and calling it performance work.

## Dispatcher And Backends

### Builtin Catalog To Real Dispatcher

Current evidence: static `Operator` and `OperatorInfo` exist (`src/dispatch.rs:6`, `src/dispatch.rs:67`), with resolution helpers (`src/dispatch.rs:488`) and CPU kernel checks (`src/dispatch.rs:680`). Tensor still owns shape logic and graph recording.

Hard approach:

- Convert `OperatorInfo` into schema records: namespace, name, overload, args, returns, alias annotations, dtype/layout/device constraints, mutability, and autograd policy.
- Add dynamic registration with duplicate checks and compatibility validation.
- Add dispatch keys: CPU, CUDA, Autograd, CompositeExplicit, layout keys, dtype family keys, and fallback keys.
- Move autograd formula registration out of `Tensor` once schemas can express saved tensors and backward output arity.

Avoid: expanding the static enum and saying the dispatcher grew.

### CUDA Backend And Runtime Model

Current evidence: CUDA wrappers validate buffers but synchronously launch kernels (`heirloom-kernels/src/cuda.rs:2557` and following launch helpers), no reusable allocator/stream is exposed, and Tensor errors on unsupported CUDA ops rather than falling back.

Hard approach:

- Add a reusable allocator abstraction over `CudaBuffer`, with ownership, caching, lifetime, and memory pressure behavior.
- Add stream/event API and make every kernel launch take execution context.
- Add explicit sync points: host materialization, Python `.numpy()`, checkpoint export, validation reports.
- Add peer-copy, all-reduce, and NCCL only after stream/device guards exist.
- Register cuBLAS/cuDNN kernels as backend kernels under schemas, not direct calls from Tensor.

Avoid: using hidden `cuCtxSynchronize` as the permanent correctness model.

### Custom Ops

Current evidence: `CustomUnaryOp` and `CustomUnaryRegistry` are separate and narrow (`src/extension.rs:5`, `src/extension.rs:82`).

Hard approach:

- Delay broad custom ops until schema/alias/autograd registration exists.
- Then add boxed calling convention, type-erased tensor values, custom kernel registration, and optional backend-specific kernels.
- Require custom ops to declare aliasing, mutability, dtype promotion, saved tensor use, and differentiability.

Avoid: plugin loading that bypasses alias and dtype checks.

## Modules And Training

### Module System

Current evidence: `Module` only has `forward`, `parameters`, and named parameters (`src/nn.rs:9`); concrete `.to_device` methods are hand-written (`src/nn.rs:72`, `src/nn.rs:595`).

Hard approach:

- Add `Parameter` and `Buffer` wrappers with registration.
- Add recursive module traversal, train/eval state, named children, and device/dtype movement through trait objects.
- Define serialization keys as part of module registration, not a best-effort `named_parameters` convention.

Avoid: writing `.to_device` by hand for every new module.

### Optimizers

Current evidence: AdamW has JSON state and CUDA f32 moment buffers (`src/nn.rs:1192`, `src/nn.rs:1401`); SGD is intentionally minimal.

Hard approach:

- Add parameter groups with per-group hyperparameters and deterministic state ordering.
- Define optimizer state dtype/device policy, especially for AMP/master weights.
- Add schedulers as composable stateful objects, not CLI-only LR changes.
- Add optimizer hooks only after parameter identity and module traversal are stable.

Avoid: adding momentum to SGD without parameter groups and state serialization.

### Checkpoints

Current evidence: LM checkpoints write a directory with model state, optimizer JSON, tokenizer, and metadata (`src/checkpoint.rs:45`); state_dict writes `.npy` files (`src/nn.rs:1515`).

Hard approach:

- Define a versioned checkpoint archive manifest with tensor records, dtype/device/layout, optimizer/module state, RNG states, data state, and provenance.
- Use atomic write/rename, partial-write detection, and checksums.
- Prefer safetensors-like tensor payloads over ad hoc `.npy` directories for production checkpoint work.
- Keep CPU snapshot behavior explicit for CUDA until device-native checkpointing is designed.

Avoid: zipping the current directory format and calling it production serialization.

### RNG, Generation, Evaluation

Current evidence: RNG is a deterministic SplitMix64 helper, generation has sampling controls, eval computes heldout cross-entropy (`src/nn.rs:618`, `src/nn.rs:841`, `src/data.rs:17`).

Hard approach:

- Add generator objects with CPU/CUDA device scopes and algorithm/version metadata.
- Make reproducibility contracts explicit: stable across versions, or versioned and allowed to change.
- Generation needs KV cache, batched serving, streaming, stop strings, beam/constrained decoding, and extension logit processors.
- Evaluation needs document-boundary masking, stride/overlap options, multiple metrics, benchmark harnesses, calibration, and distributed aggregation.

Avoid: improving text quality by tweaking sampling without adding serving/eval architecture.

## IO And Serialization

### NPY And Tensor Formats

Current evidence: `.npy` handles CPU f32/f64/i64/bool C-order and rejects BF16 (`src/npy.rs:6`).

Hard approach:

- Decide BF16 serialization policy before writing bytes: NumPy descriptor compatibility, safetensors dtype, or Heirloom-specific metadata.
- Add `.npz` only if archive semantics, compression, and atomicity are clear.
- For production model tensors, prefer safetensors-style metadata-rich storage before pickle or `.pt`.
- Add mmap/streaming readers with shape/dtype validation and partial-read errors.

Avoid: encoding BF16 as raw void `|V2` without metadata and expecting interop.

### NPY Parser Completeness

Current evidence: parser is intentionally narrow string parsing (`src/npy.rs:86`).

Hard approach:

- Either adopt a tested parser crate or implement a small Python-literal header parser with exact type handling.
- Add fixtures for endian variants, Fortran order rejection, shape edge cases, v1/v2/v3 headers, malformed headers, and trailing data.

Avoid: expanding ad hoc string searches.

### Tokenizer And Data

Current evidence: tokenizer is versioned byte-BPE-like (`src/tokenizer.rs:1`), prepared data writes hashed token files and manifests (`src/data.rs:17`).

Hard approach:

- Decide whether production tokenizer parity means GPT-2 BPE, `tokenizers`, SentencePiece, or native Heirloom tokenizer. Each implies different normalization/pretokenization contracts.
- Move prepared token arrays from JSON to a binary sharded format with dtype/endian metadata.
- Add mmap, streaming, corruption recovery, document-boundary masks, packed sequence metadata, distributed samplers, worker prefetch, and pinned-memory handoff.

Avoid: optimizing current JSON token arrays while leaving data semantics undefined.

## Python Interop

### Current Binding

Current evidence: PyO3 exposes a narrow unsendable `Tensor` wrapper and parity methods (`heirloom-python/src/lib.rs:7`, `heirloom-python/src/lib.rs:82`).

Hard approach:

- Decide if Python is a parity harness or a product API. Those are different contracts.
- If product API: add modules, optimizers, dataloaders, checkpointing, typed package metadata, `.pyi` stubs, Python exceptions, docs, and stable import namespace.
- Add `__array__`, buffer protocol, DLPack, and optional Torch interop with explicit ownership and sync behavior.
- Python custom autograd should map into the same custom autograd schema model as Rust custom ops.

Avoid: making the parity harness look PyTorch-like without supporting PyTorch-like semantics.

## Testing And Parity

### Current Test Matrix

Current evidence: default validation runs format/test/clippy/demos/LM smoke (`scripts/validate.sh:1`), Rust consumes PyTorch fixtures, Python parity compares against live torch, and CUDA tests are opt-in via `HEIRLOOM_CUDA_TESTS`.

Hard approach:

- Generate randomized live PyTorch parity cases across shape, dtype, layout, scalar, view, reduction, and broadcasting dimensions.
- Promote expected divergences into explicit tests with reason strings.
- Add tolerance tracking by op/backend/dtype instead of one-off epsilon choices.
- Add sanitizer, miri where possible, stress, leak, and long-run graph lifecycle tests.
- Add a reproducible CUDA image and CI lane before treating CUDA gates as routine.

Avoid: increasing fixed fixtures without randomized coverage.

### CUDA And TinyStories Evidence

Current evidence: A100 runs and TinyStories CUDA gates prove a narrow single-GPU path, not broad parity.

Hard approach:

- Keep smoke/reference/full tiers, but add failure classification: kernel compile, device availability, numerical mismatch, performance regression, checkpoint/resume mismatch.
- Add CUDA randomized op parity against CPU/PyTorch for supported kernels.
- Add performance baselines only after stream/allocator and production kernels exist.

Avoid: using a successful TinyStories loss decrease as evidence for distributed or production GPU training.

## Safety And Ownership

### Ownership And Thread Safety

Current evidence: Tensor is `Rc<RefCell<TensorInner>>` (`src/tensor.rs:23`) and Python wrapper is unsendable (`heirloom-python/src/lib.rs:7`).

Hard approach:

- Decide whether Heirloom remains single-threaded by design or moves to `Arc` plus carefully scoped interior mutability.
- If moving to threads, introduce execution ownership first: graph task, parameter state locks, storage mutation policy, and stream/device guards.
- Keep `heirloom-kernels` as the unsafe boundary and audit any new unsafe with narrow wrappers.

Avoid: replacing `Rc<RefCell>` with `Arc<Mutex>` globally and calling it thread-safe.

### Error Diagnostics

Current evidence: `TensorError` is coarse string-bearing variants (`src/error.rs:5`).

Hard approach:

- Add structured error payloads: op name, schema, input metadata, expected/actual dtype/device/layout, source location if available, and graph node id.
- Include saved tensor provenance and mutation history in autograd errors.
- Keep user-facing messages concise, but preserve machine-readable diagnostics for tests and Python exceptions.

Avoid: only polishing message strings.

## Suggested Work Packets

1. Alias + indexing foundation.
   Implement `ViewOp`, overlap classifier, mutation validator, basic indexing API, and alias-aware tests. This unlocks several Tensor Semantics and Autograd items at once.

2. TensorIterator CPU foundation.
   Add a typed iteration planner and port binary/unary/reduction CPU ops away from `Vec<f64>` materialization. This also prepares dtype expansion.

3. Dispatcher phase 1.
   Replace enum-only operator identity with schema records and kernel registration for current ops. Do this before adding in-place schemas and custom op expansion.

4. CUDA runtime contract.
   Add allocator, stream/event, explicit sync, D2D copy, and registered CUDA kernel table. Then consider CUDA batched matmul and cuBLAS.

5. Parity matrix expansion.
   Add live PyTorch randomized tests for dtype/promotion, scalar tensors, views, reductions, and empty semantics. Add CUDA randomized parity for current kernels under the opt-in gate.

6. Production serialization/data path.
   Pick safetensors-like tensor storage plus binary sharded token data with mmap and manifest checksums. Then update checkpoint format around that.

## Bottom Line

The highest-leverage hard-mode path is not feature breadth. It is choosing the next invariant and forcing the code to honor it everywhere: alias graph, dispatcher schema, TensorIterator, CUDA execution context, or parity matrix. If the project wants to become more PyTorch-honest, the best next move is the alias/indexing foundation or dispatcher phase 1. If it wants more GPU capability, the CUDA runtime contract should come before more one-off kernels.
