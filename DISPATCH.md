# Dispatcher Production Plan

Heirloom currently has a builtin `KernelRegistry` and public `operator_catalog()`. That is a useful stepping stone, not a production dispatcher. This document describes the path from the current static registry toward something closer to PyTorch's dispatcher architecture.

## Current State

- Operators are represented by the `Operator` enum.
- Builtin schemas are static `OperatorInfo` values.
- Resolution checks dtype/device compatibility and returns output dtype plus CPU scalar kernel pointers where available.
- Tensor methods consult declared alias/autograd policies for supported out-of-place ops, but still perform graph recording and some shape-specific kernels.
- Custom unary ops live in `CustomUnaryRegistry`, separate from the builtin dispatch catalog.
- Transformer-training ops added for the tiny LM runtime are now first-class catalog entries with resolver coverage for dtype/device/autograd policy. Their kernels and backward graph construction still live in `Tensor`.

## Production Target

A serious dispatcher should support:

- operator schemas independent of Rust enum variants
- dynamic registration of kernels and extension operators
- dispatch keys for device, layout, dtype family, autograd, and fallback behavior
- backend-specific kernel tables
- boxed and typed calling conventions
- generated or validated operator signatures
- backend fallback and composite kernels
- extension registration without editing core enums
- explicit autograd kernel registration

The current catalog deliberately does not register in-place variants such as `add_`. Those methods borrow dtype resolution from builtin out-of-place schemas, but their aliasing and mutation semantics are still handwritten Tensor logic.

The catalog also deliberately keeps fused Heirloom-specific operators, such as `heirloom.causal_self_attention`, separate from ATen-like names. That makes the prototype honest about where it is matching PyTorch concepts and where it is choosing a smaller runtime-specific primitive.

## Proposed Phases

### Phase 1: Schema Objects

Replace enum-only operator identity with stable schema records:

- namespace
- name
- overload
- input/output arity
- mutability and alias annotations
- dtype/device/layout constraints
- autograd policy

The existing `OperatorInfo` becomes the seed for this.

### Phase 2: Registration API

Add a mutable registry that accepts builtin and user registrations:

- `register_schema`
- `register_cpu_kernel`
- `register_composite_kernel`
- `register_autograd_kernel`
- duplicate and compatibility checks

Custom unary ops should move from a separate registry into this model only after schema and alias metadata can represent them safely.

### Phase 3: Dispatch Keys

Introduce a compact dispatch key set. Initial keys could be:

- `Cpu`
- `Autograd`
- `CompositeExplicit`
- `Float`
- `Integer`
- `Bool`

This is still smaller than PyTorch, but it moves routing out of ad hoc dtype checks.

### Phase 4: Calling Convention

Keep typed direct calls for internal hot paths, but add a boxed call representation for extension registration and debugging. Production systems need a uniform way to call unknown operators.

### Phase 5: Autograd Integration

Autograd should be a dispatch key or registered kernel layer, not a tensor-method side effect. This requires:

- explicit saved tensor policy
- backward schema
- gradient output arity
- view/in-place annotations
- custom backward registration

## Review Questions

- Which operator metadata should be required before dynamic registration is allowed?
- How much of aliasing policy belongs in schema versus tensor implementation?
- Should autograd be represented as dispatch-key routing or as graph wrapper nodes?
- What is the smallest boxed value representation that does not compromise safety?
- How should custom ops express dtype promotion without reimplementing dispatcher logic?
