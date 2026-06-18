use heirloom::{DType, Tensor};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "index {index}: actual={a}, expected={e}, actual={actual:?}, expected={expected:?}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-9,
            "index {index}: actual={a}, expected={e}, actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn mutation_through_non_overlapping_view_bumps_shared_storage_version() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false).unwrap();
    let view = base.transpose().unwrap();

    assert_eq!(base.storage_version(), 0);
    assert_eq!(view.storage_version(), 0);

    view.add_(&Tensor::ones(&[2, 2], false).unwrap()).unwrap();

    assert_eq!(base.storage_version(), 1);
    assert_eq!(view.storage_version(), 1);
    assert_close(&base.data(), &[2.0, 3.0, 4.0, 5.0]);
    assert_close(&view.data(), &[2.0, 4.0, 3.0, 5.0]);
}

#[test]
fn saved_tensor_version_check_catches_mutation_through_alias_view() {
    let base = Tensor::from_vec(vec![-1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();
    let relu = base.relu().unwrap();
    let alias = base.transpose().unwrap();

    alias.add_(&Tensor::ones(&[2, 2], false).unwrap()).unwrap();

    let err = relu.sum().unwrap().backward().unwrap_err();
    assert!(err.to_string().contains("modified in-place"));
}

#[test]
fn reshape_copy_from_non_contiguous_view_breaks_storage_aliasing() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let copied = base.transpose().unwrap().reshape(&[6]).unwrap();

    assert_eq!(base.storage_version(), 0);
    assert_eq!(copied.storage_version(), 0);
    assert_close(&copied.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    base.add_(&Tensor::ones(&[2, 3], false).unwrap()).unwrap();

    assert_eq!(base.storage_version(), 1);
    assert_eq!(copied.storage_version(), 0);
    assert_close(&copied.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn contiguous_noop_preserves_alias_but_non_contiguous_contiguous_copies() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false).unwrap();
    let contiguous_alias = base.contiguous().unwrap();
    let contiguous_copy = base.transpose().unwrap().contiguous().unwrap();

    base.add_(&Tensor::ones(&[2, 2], false).unwrap()).unwrap();

    assert_eq!(contiguous_alias.storage_version(), base.storage_version());
    assert_close(&contiguous_alias.data(), &[2.0, 3.0, 4.0, 5.0]);
    assert_eq!(contiguous_copy.storage_version(), 0);
    assert_close(&contiguous_copy.data(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn as_strided_rejects_storage_out_of_bounds() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], false).unwrap();
    let err = base.as_strided(&[2, 2], &[2, 2], 1).unwrap_err();

    assert!(err.to_string().contains("exceeds storage length"));
}

#[test]
fn transposed_leaf_view_gradient_preserves_non_overlapping_layout() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let view = base.transpose().unwrap();
    view.set_requires_grad(true);

    view.mul(&view).unwrap().sum().unwrap().backward().unwrap();

    assert_close_f64(&view.grad_f64().unwrap(), &[2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
    let grad = view.grad_tensor().unwrap();
    assert_eq!(grad.dtype(), DType::F64);
    assert_eq!(grad.shape(), view.shape());
    assert_eq!(grad.strides(), view.strides());
    assert_eq!(grad.storage_offset(), view.storage_offset());
    assert_close_f64(
        &grad.data_f64_exact().unwrap(),
        &[2.0, 8.0, 4.0, 10.0, 6.0, 12.0],
    );
}

#[test]
fn narrowed_leaf_view_gradient_preserves_storage_offset_without_primal_aliasing() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let view = base.narrow(1, 1, 2).unwrap();
    view.set_requires_grad(true);

    view.sum().unwrap().backward().unwrap();

    let grad = view.grad_tensor().unwrap();
    assert_eq!(grad.shape(), vec![2, 2]);
    assert_eq!(grad.strides(), vec![3, 1]);
    assert_eq!(grad.storage_offset(), 1);
    assert_close_f64(&grad.data_f64_exact().unwrap(), &[1.0, 1.0, 1.0, 1.0]);

    base.add_(&Tensor::ones(&[2, 3], false).unwrap()).unwrap();
    assert_close_f64(&grad.data_f64_exact().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn expanded_leaf_view_gradient_uses_dense_layout_to_represent_distinct_values() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], false).unwrap();
    let expanded = base.expand(&[2, 3]).unwrap();
    expanded.set_requires_grad(true);
    let seed = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    expanded.backward_with_grad_f64(seed.clone()).unwrap();

    let grad = expanded.grad_tensor().unwrap();
    assert_eq!(grad.shape(), expanded.shape());
    assert_eq!(grad.strides(), vec![3, 1]);
    assert_eq!(grad.storage_offset(), 0);
    assert_close_f64(&expanded.grad_f64().unwrap(), &seed);
    assert_close_f64(&grad.data_f64_exact().unwrap(), &seed);
}
