use heirloom::{
    CustomUnaryBackwardContext, CustomUnaryForwardContext, CustomUnaryOp, Result, Tensor,
    TensorError,
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

fn shifted_cubic_forward(ctx: CustomUnaryForwardContext<'_>) -> Result<Vec<f64>> {
    Ok(ctx
        .input
        .iter()
        .map(|value| {
            let shifted = value + 1.0;
            shifted * shifted * shifted
        })
        .collect())
}

fn shifted_cubic_backward(ctx: CustomUnaryBackwardContext<'_>) -> Result<Vec<f64>> {
    Ok(ctx
        .input
        .iter()
        .zip(ctx.grad_output.iter())
        .map(|(value, grad)| 3.0 * (value + 1.0) * (value + 1.0) * grad)
        .collect())
}

fn assert_close(name: &str, actual: &[f64], expected: &[f64]) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(TensorError::InvalidOperation(format!(
            "{name} length mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if (actual - expected).abs() > 1e-10 {
            return Err(TensorError::InvalidOperation(format!(
                "{name}[{index}] mismatch: actual {actual}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let square = CustomUnaryOp::new("demo.square", square_forward, square_backward)?;
    let shifted_cubic = CustomUnaryOp::new(
        "demo.shifted_cubic",
        shifted_cubic_forward,
        shifted_cubic_backward,
    )?;
    let x = Tensor::from_f64(vec![-2.0, 0.5, 3.0], &[3], true)?;

    let y = x
        .apply_custom_unary(square)?
        .apply_custom_unary(shifted_cubic)?;
    let loss = y.sum()?;
    loss.backward()?;

    let values = y.data_f64_exact()?;
    let gradients = x.grad_f64().ok_or_else(|| {
        TensorError::Autograd("expected custom autograd demo to populate x.grad".to_string())
    })?;

    assert_close("custom forward", &values, &[125.0, 1.953125, 1000.0])?;
    assert_close("custom gradient", &gradients, &[-300.0, 4.6875, 1800.0])?;

    println!("custom_ops=demo.square -> demo.shifted_cubic");
    println!("input={:?}", x.data_f64_exact()?);
    println!("output={values:?}");
    println!("gradient={gradients:?}");
    Ok(())
}
