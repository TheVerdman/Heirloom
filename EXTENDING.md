# Extending Heirloom

Heirloom now has a small Rust-side custom autograd API for floating, shape-preserving unary operators. This is not PyTorch extension parity. It is a concrete prototype surface for experimenting with forward and backward callbacks while retaining saved tensor version checks and graph integration.

## Custom Unary Ops

Create a `CustomUnaryOp` with a stable name, a forward callback, and a backward callback:

```rust
use heirloom::{
    CustomUnaryBackwardContext, CustomUnaryForwardContext, CustomUnaryOp, Result, Tensor,
};

fn square_forward(ctx: CustomUnaryForwardContext<'_>) -> Result<Vec<f64>> {
    Ok(ctx.input.iter().map(|value| value * value).collect())
}

fn square_backward(ctx: CustomUnaryBackwardContext<'_>) -> Result<Vec<f64>> {
    Ok(ctx
        .input
        .iter()
        .zip(ctx.grad_output.iter())
        .map(|(value, grad)| 2.0 * value * grad)
        .collect())
}

let square = CustomUnaryOp::new("example.square", square_forward, square_backward)?;
let x = Tensor::from_f64(vec![2.0, -3.0], &[2], true)?;
let y = x.apply_custom_unary(square)?;
y.sum()?.backward()?;
```

For a named collection of user ops, use `CustomUnaryRegistry`:

```rust
let mut registry = heirloom::CustomUnaryRegistry::new();
registry.register(square)?;
let op = registry.get("example.square").unwrap();
```

## Current Guarantees

- Custom unary ops require floating tensors.
- The output shape is preserved and the returned output length is validated.
- Backward receives the saved input, saved output, upstream gradient, shape, dtype, and device.
- Backward output length is validated before gradient accumulation.
- Saved tensor version checks catch in-place mutation between forward and backward.
- `no_grad` disables custom op graph recording.
- `CustomUnaryRegistry` supports explicit local registration, duplicate-name rejection, lookup, and listing.

## Current Limits

- Only shape-preserving unary ops are supported.
- There is no custom binary op, reduction, view op, in-place op, or multi-output op.
- Custom ops live in a separate user registry, not the builtin dispatcher catalog.
- There is no dynamic dispatcher registration or builtin operator catalog insertion for custom ops.
- There is no Python binding, C ABI, plugin loading, boxed calling convention, or backend-specific custom kernel.
- Custom backward is first-order only; higher-order gradients are not implemented.

Run the demo with:

```text
cargo run --bin custom
```
