use heirloom::{
    operator_catalog, CustomUnaryBackwardContext, CustomUnaryForwardContext, CustomUnaryOp, DType,
    Result, Tensor,
};

fn identity_forward(ctx: CustomUnaryForwardContext<'_>) -> Result<Vec<f64>> {
    Ok(ctx.input.to_vec())
}

fn identity_backward(ctx: CustomUnaryBackwardContext<'_>) -> Result<Vec<f64>> {
    Ok(ctx.grad_output.to_vec())
}

#[test]
fn matches_pytorch_dtype_promotion_for_i64_plus_f32() {
    // PyTorch 2.12: torch.int64 + torch.float32 -> torch.float32.
    let ints = Tensor::from_i64(vec![1, 2], &[2], false).unwrap();
    let floats = Tensor::from_vec(vec![0.5, 1.5], &[2], false).unwrap();

    let out = ints.add(&floats).unwrap();
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.data_f32().unwrap(), vec![1.5, 3.5]);
}

#[test]
fn matches_pytorch_sum_dtype_for_i64_and_bool_inputs() {
    // PyTorch 2.12: int64.sum() and bool.sum() both produce int64.
    let ints = Tensor::from_i64(vec![1, 2, 3], &[3], false).unwrap();
    let bools = Tensor::from_bool(vec![true, false, true], &[3], false).unwrap();

    let int_sum = ints.sum().unwrap();
    let bool_sum = bools.sum().unwrap();

    assert_eq!(int_sum.dtype(), DType::I64);
    assert_eq!(int_sum.data_i64().unwrap(), vec![6]);
    assert_eq!(bool_sum.dtype(), DType::I64);
    assert_eq!(bool_sum.data_i64().unwrap(), vec![2]);
}

#[test]
fn documents_current_divergence_from_pytorch_integer_div_and_mean() {
    let ints = Tensor::from_i64(vec![1, 2, 3], &[3], false).unwrap();

    // PyTorch 2.12 integer division returns float32 for int64 / int64.
    // Heirloom currently promotes integer-only div to f64 for precision-first simplicity.
    let quotient = ints
        .div(&Tensor::from_i64(vec![2, 2, 2], &[3], false).unwrap())
        .unwrap();
    assert_eq!(quotient.dtype(), DType::F64);
    assert_eq!(quotient.data_f64_exact().unwrap(), vec![0.5, 1.0, 1.5]);

    // PyTorch 2.12 rejects mean() on int64 and bool tensors.
    // Heirloom currently returns f64 for both so reductions remain total over supported dtypes.
    assert_eq!(ints.mean().unwrap().dtype(), DType::F64);
    let bools = Tensor::from_bool(vec![true, false, true], &[3], false).unwrap();
    assert_eq!(bools.mean().unwrap().dtype(), DType::F64);
}

#[test]
fn matches_pytorch_expand_in_place_write_rejection_shape_of_behavior() {
    // PyTorch rejects writes through expanded zero-stride views because multiple logical
    // elements alias the same storage. Heirloom rejects the same class of operation.
    let base = Tensor::from_vec(vec![1.0, 2.0], &[2, 1], false).unwrap();
    let expanded = base.expand(&[2, 3]).unwrap();
    let err = expanded
        .add_(&Tensor::ones(&[2, 3], false).unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("internal overlap"));
}

#[test]
fn custom_unary_boundary_is_explicitly_outside_builtin_dispatch_catalog() {
    let op = CustomUnaryOp::new(
        "heirloom.test.identity",
        identity_forward,
        identity_backward,
    )
    .unwrap();
    assert!(!operator_catalog()
        .iter()
        .any(|schema| schema.name == op.name()));

    let x = Tensor::from_vec(vec![1.0, 2.0], &[2], true).unwrap();
    let y = x.apply_custom_unary(op).unwrap();
    y.sum().unwrap().backward().unwrap();
    assert_eq!(x.grad().unwrap(), vec![1.0, 1.0]);
}
