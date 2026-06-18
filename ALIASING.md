# Aliasing And Storage-Aware Gradient Plan

Heirloom supports real strided views, shared storage version counters, and gradient tensors with explicit storage/layout metadata. That is a meaningful step beyond logical gradient vectors, but it is still not production-grade tensor aliasing.

## Current State

- `Tensor` tracks shape, strides, storage offset, dtype, and device.
- Views share `Storage`.
- In-place writes bump a shared storage version counter.
- Saved tensors record the version and fail backward if storage changed.
- `expand` uses zero strides and rejects in-place writes through internally overlapping views.
- Gradients are stored as f64 gradient tensors with shape, strides, storage offset, device, and independent gradient storage.
- `grad()` and `grad_f64()` keep returning logical values for compatibility, while `grad_tensor()` exposes the storage-aware gradient tensor.
- First gradient accumulation preserves non-overlapping target layouts, such as transposed and narrowed leaf views.
- Internally overlapping target layouts, currently detected through zero strides, fall back to dense contiguous gradient storage so arbitrary seed gradients remain representable.

## Current Invariants

- Non-overlapping views can mutate shared storage and bump the shared version.
- Expanded zero-stride views report internal overlap and reject in-place writes.
- Saved tensor version checks catch mutation through aliases.
- Non-contiguous `reshape` and `contiguous` copy into independent storage.
- Differentiable `as_strided` is intentionally rejected.
- Transposed and narrowed leaf-view gradients preserve gradient strides/storage offsets without aliasing primal storage.
- Expanded leaf-view gradients use dense gradient storage rather than zero-stride gradient storage.

These invariants are covered by tests in `tests/aliasing_invariants.rs`.

## Production Target

A production aliasing model needs:

- view metadata that can replay or invert view operations
- alias graph or view chain tracking
- exact overlap analysis beyond zero strides
- in-place operation validation against autograd history
- leaf/view mutation rules
- gradient accumulation into aliased bases
- explicit behavior for detach, no_grad, copy, clone, and view-of-view operations

## Proposed Phases

### Phase 1: Gradient Tensor Object - Done

`Option<Vec<f64>>` has been replaced with a gradient tensor representation:

- dtype
- device
- shape
- strides
- storage
- accumulation policy

The current accumulation policy preserves non-overlapping layouts and falls back to dense contiguous storage for internally overlapping layouts. Gradient tensors own independent f64 storage rather than sharing primal storage.

### Phase 2: View Metadata

Represent view creation as structured metadata:

- source tensor identity
- view kind
- shape/stride/offset transform
- backward replay transform
- alias policy

Current `GradFn::Permute`, `Narrow`, `Expand`, and `View` variants are the seed for this.

### Phase 3: Overlap Analysis

Implement exact internal overlap detection for dense strided tensors. Zero-stride detection is not enough. The runtime should classify:

- no overlap
- partial overlap
- full overlap
- too hard / unknown

### Phase 4: In-Place Rules

Move from simple version checks to explicit mutation policy:

- leaf tensor requiring grad
- view of leaf requiring grad
- saved tensor needed for backward
- mutation under `no_grad`
- mutation of detached aliases

### Phase 5: Storage-Aware Accumulation

Gradient accumulation should understand aliases across tensor handles and primal storage, not only the layout of each handle's `.grad`. For views, gradients should be routed through replayable view transforms back to the base storage where appropriate, while still detecting ambiguous overlapping accumulation.

## Review Questions

- Which view operations should be represented as replayable metadata first?
- Should any gradient tensors ever share storage with primal tensors, or should independent gradient storage remain a hard rule?
- How should overlapping views fail: at construction, mutation, backward, or accumulation?
- Which PyTorch in-place/view rules should Heirloom intentionally match first?
- What is the minimal alias graph that avoids unsound custom autograd behavior?
