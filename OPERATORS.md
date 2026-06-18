# Heirloom Operator Registry

This file summarizes the builtin operator catalog exposed by `operator_catalog()`. It is intentionally small and explicit: these are declarations for the current Rust runtime surface, not a claim of PyTorch dispatcher parity.

| Operator | Name | Kind | DType Rule | Kernel Location |
| --- | --- | --- | --- | --- |
| `Add` | `aten.add` | Binary | Arithmetic promotion | dispatch CPU scalar |
| `Sub` | `aten.sub` | Binary | Arithmetic promotion | dispatch CPU scalar |
| `Mul` | `aten.mul` | Binary | Arithmetic promotion | dispatch CPU scalar |
| `Div` | `aten.div` | Binary | Arithmetic promotion | dispatch CPU scalar |
| `Matmul` | `aten.matmul` | Matrix multiply | Floating promotion | dispatch CPU scalar |
| `Relu` | `aten.relu` | Unary | Preserve floating dtype | dispatch CPU scalar |
| `Gelu` | `aten.gelu` | Unary | Preserve floating dtype | tensor shape-specific kernel |
| `SoftmaxDim` | `aten.softmax.dim` | Unary | Preserve floating dtype | tensor shape-specific kernel |
| `CrossEntropyForLogits` | `heirloom.cross_entropy_for_logits` | Loss | Preserve floating dtype | tensor fused kernel |
| `Sum` | `aten.sum` | Reduction all | Sum reduction | dispatch CPU scalar |
| `Mean` | `aten.mean` | Reduction all | Mean reduction | dispatch CPU scalar |
| `SumDim` | `aten.sum.dim` | Reduction dim | Sum reduction | tensor shape-specific kernel |
| `MeanDim` | `aten.mean.dim` | Reduction dim | Mean reduction | tensor shape-specific kernel |
| `LayerNormLastDim` | `aten.layer_norm.last_dim` | Normalization | Same floating inputs | tensor shape-specific kernel |
| `Embedding` | `aten.embedding` | Indexing | Embedding lookup | tensor shape-specific kernel |
| `MaskedFill` | `aten.masked_fill` | Masking | Mask preserve | tensor shape-specific kernel |
| `ArgmaxLastDim` | `aten.argmax.last_dim` | Selection | Argmax index | tensor shape-specific kernel |
| `CausalSelfAttention` | `heirloom.causal_self_attention` | Attention | Same floating inputs | tensor fused kernel |

## Current Resolution Rules

- Device resolution currently accepts only matching `Device::Cpu`.
- Binary arithmetic promotes f64 over f32, floating dtypes over integer/bool, integer-only add/sub/mul to i64, and integer-only div to f64.
- Bool subtraction and division are rejected.
- Matmul, relu, gelu, softmax, cross-entropy, layer norm, and causal attention require floating tensors.
- Layer norm and causal attention require same dtype/device floating inputs.
- Embedding requires i64 indices and floating weights, returning the weight dtype.
- Masked fill requires a bool mask and preserves the input dtype.
- Argmax returns i64 indices and is explicitly non-differentiable.
- Sum maps f32 to f32, f64 to f64, and bool/i64 to i64.
- Mean maps f32 to f32, f64 to f64, and bool/i64 to f64.
- Fresh-output and autograd policies are consulted by `Tensor` methods for supported out-of-place ops, but graph-node construction is still owned by `Tensor`.
- In-place variants such as `add_` are not first-class operator schemas yet; they reuse dtype resolution but enforce mutation and aliasing rules in Tensor code.

## Not Yet Real Dispatcher Parity

The catalog is builtin and static. There are no dispatch keys, backend fallbacks, dynamic operator registration, generated schemas, boxed calling convention, per-layout kernels, custom backends, extension ABI, or autograd kernel registration.

Custom unary ops created through `CustomUnaryOp` are intentionally outside this builtin catalog for now. They are graph-integrated Rust callbacks, not dynamically registered dispatcher operators.

Transformer-training ops such as `embedding`, `gelu`, `layer_norm_last_dim`, `masked_fill`, `argmax_last_dim`, and fused causal self-attention are now first-class entries in this builtin catalog and resolve through `KernelRegistry` before execution. Their shape-specific kernels, saved tensor choices, and backward graph nodes are still owned by `Tensor`, so this remains a static registry milestone rather than dispatcher parity.
