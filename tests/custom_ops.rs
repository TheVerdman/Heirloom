use heirloom::{
    no_grad, CustomUnaryBackwardContext, CustomUnaryForwardContext, CustomUnaryOp,
    CustomUnaryRegistry, DType, Result, Tensor,
};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-10,
            "index {index}: actual={a}, expected={e}, actual={actual:?}, expected={expected:?}"
        );
    }
}

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

fn bad_forward_length(ctx: CustomUnaryForwardContext<'_>) -> Result<Vec<f64>> {
    Ok(vec![0.0; ctx.input.len() + 1])
}

fn bad_backward_length(_ctx: CustomUnaryBackwardContext<'_>) -> Result<Vec<f64>> {
    Ok(vec![1.0])
}

#[test]
fn custom_unary_forward_backward_and_chaining_work() {
    let square =
        CustomUnaryOp::new("heirloom.test.square", square_forward, square_backward).unwrap();
    let shifted_cubic = CustomUnaryOp::new(
        "heirloom.test.shifted_cubic",
        shifted_cubic_forward,
        shifted_cubic_backward,
    )
    .unwrap();

    let x = Tensor::from_f64(vec![-2.0, 0.5, 3.0], &[3], true).unwrap();
    let y = x
        .apply_custom_unary(square)
        .unwrap()
        .apply_custom_unary(shifted_cubic)
        .unwrap();

    assert_eq!(y.dtype(), DType::F64);
    assert_close(&y.data_f64_exact().unwrap(), &[125.0, 1.953125, 1000.0]);

    y.sum().unwrap().backward().unwrap();
    assert_close(&x.grad_f64().unwrap(), &[-300.0, 4.6875, 1800.0]);
}

#[test]
fn custom_unary_preserves_f32_dtype_but_uses_f64_internal_values() {
    let square =
        CustomUnaryOp::new("heirloom.test.square", square_forward, square_backward).unwrap();
    let x = Tensor::from_vec(vec![1.5, -2.0], &[2], true).unwrap();

    let y = x.apply_custom_unary(square).unwrap();
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(y.data_f32().unwrap(), vec![2.25, 4.0]);

    y.sum().unwrap().backward().unwrap();
    assert_close(&x.grad_f64().unwrap(), &[3.0, -4.0]);
}

#[test]
fn custom_unary_honors_no_grad_and_rejects_non_float_input() {
    let square =
        CustomUnaryOp::new("heirloom.test.square", square_forward, square_backward).unwrap();
    let x = Tensor::from_f64(vec![2.0], &[1], true).unwrap();
    let y = no_grad(|| x.apply_custom_unary(square)).unwrap();

    assert!(!y.requires_grad());
    assert_eq!(y.data_f64_exact().unwrap(), vec![4.0]);

    let ints = Tensor::from_i64(vec![1, 2], &[2], false).unwrap();
    assert!(ints
        .apply_custom_unary(square)
        .unwrap_err()
        .to_string()
        .contains("floating"));
}

#[test]
fn custom_unary_validates_name_forward_shape_and_backward_shape() {
    assert!(
        CustomUnaryOp::new("bad name", square_forward, square_backward)
            .unwrap_err()
            .to_string()
            .contains("unsupported characters")
    );

    let bad_forward = CustomUnaryOp::new(
        "heirloom.test.bad_forward",
        bad_forward_length,
        square_backward,
    )
    .unwrap();
    let x = Tensor::from_f64(vec![2.0, 3.0], &[2], true).unwrap();
    assert!(x
        .apply_custom_unary(bad_forward)
        .unwrap_err()
        .to_string()
        .contains("returned 3 elements"));

    let bad_backward = CustomUnaryOp::new(
        "heirloom.test.bad_backward",
        square_forward,
        bad_backward_length,
    )
    .unwrap();
    let y = x.apply_custom_unary(bad_backward).unwrap();
    assert!(y
        .sum()
        .unwrap()
        .backward()
        .unwrap_err()
        .to_string()
        .contains("returned gradient length"));
}

#[test]
fn custom_unary_saved_tensor_version_check_catches_mutation() {
    let square =
        CustomUnaryOp::new("heirloom.test.square", square_forward, square_backward).unwrap();
    let x = Tensor::from_f64(vec![2.0, 3.0], &[2], true).unwrap();
    let y = x.apply_custom_unary(square).unwrap();

    x.add_(&Tensor::from_f64(vec![1.0, 1.0], &[2], false).unwrap())
        .unwrap();
    assert!(y
        .sum()
        .unwrap()
        .backward()
        .unwrap_err()
        .to_string()
        .contains("modified in-place"));
}

#[test]
fn custom_unary_registry_tracks_user_ops_without_builtin_dispatch_registration() {
    let square =
        CustomUnaryOp::new("heirloom.test.square", square_forward, square_backward).unwrap();
    let mut registry = CustomUnaryRegistry::new();

    assert!(registry.is_empty());
    registry.register(square).unwrap();
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.operators()[0].name(), "heirloom.test.square");
    assert!(registry
        .register(square)
        .unwrap_err()
        .to_string()
        .contains("already registered"));

    let registered = registry.get("heirloom.test.square").unwrap();
    let x = Tensor::from_f64(vec![2.0, -3.0], &[2], true).unwrap();
    let y = x.apply_custom_unary(registered).unwrap();
    assert_close(&y.data_f64_exact().unwrap(), &[4.0, 9.0]);
}
