use heirloom::Tensor;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "index {index}: actual={a}, expected={e}, actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn default_backward_releases_graph_and_rejects_second_backward() {
    let x = Tensor::from_vec(vec![2.0, 3.0], &[2], true).unwrap();
    let y = x.mul(&x).unwrap().sum().unwrap();

    y.backward().unwrap();
    assert_close(&x.grad().unwrap(), &[4.0, 6.0]);
    assert!(!y.is_leaf());

    x.zero_grad();
    let err = y.backward().unwrap_err();
    assert!(err.to_string().contains("already been released"));
    assert!(x.grad().is_none());
}

#[test]
fn retained_backward_allows_one_more_backward_before_default_release() {
    let x = Tensor::from_vec(vec![2.0, 3.0], &[2], true).unwrap();
    let y = x.mul(&x).unwrap().sum().unwrap();

    y.backward_retain_graph().unwrap();
    assert_close(&x.grad().unwrap(), &[4.0, 6.0]);

    x.zero_grad();
    y.backward().unwrap();
    assert_close(&x.grad().unwrap(), &[4.0, 6.0]);

    x.zero_grad();
    let err = y.backward().unwrap_err();
    assert!(err.to_string().contains("already been released"));
}

#[test]
fn retained_backward_with_explicit_seed_supports_non_scalar_outputs() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], true).unwrap();
    let y = x.mul(&x).unwrap();

    y.backward_with_grad_f64_retain_graph(vec![1.0, 1.0, 1.0])
        .unwrap();
    assert_close(&x.grad().unwrap(), &[2.0, 4.0, 6.0]);

    x.zero_grad();
    y.backward_with_grad(vec![1.0, 1.0, 1.0]).unwrap();
    assert_close(&x.grad().unwrap(), &[2.0, 4.0, 6.0]);
}

#[test]
fn non_leaf_gradients_are_transient_unless_the_tensor_is_a_leaf() {
    let x = Tensor::from_vec(vec![2.0, 3.0], &[2], true).unwrap();
    let y = x.mul(&x).unwrap();
    let loss = y.sum().unwrap();

    loss.backward_retain_graph().unwrap();

    assert_close(&x.grad().unwrap(), &[4.0, 6.0]);
    assert!(y.grad().is_none());
    assert!(loss.grad().is_none());
    assert!(!y.is_leaf());
}

#[test]
fn released_intermediate_cannot_be_used_to_backpropagate_into_old_parents() {
    let x = Tensor::from_vec(vec![1.0, 2.0], &[2], true).unwrap();
    let y = x.relu().unwrap();

    y.sum().unwrap().backward().unwrap();
    x.zero_grad();

    let z = y.mul(&y).unwrap().sum().unwrap();
    let err = z.backward().unwrap_err();
    assert!(err.to_string().contains("already been released"));
    assert!(x.grad().is_none());
}

#[test]
fn retain_grad_keeps_non_leaf_gradient_after_backward() {
    let x = Tensor::from_vec(vec![2.0, 3.0], &[2], true).unwrap();
    let y = x.mul(&x).unwrap();
    y.retain_grad().unwrap();

    y.sum().unwrap().backward().unwrap();

    assert!(y.retains_grad());
    assert_close(&y.grad().unwrap(), &[1.0, 1.0]);
    assert_close(&x.grad().unwrap(), &[4.0, 6.0]);
}

#[test]
fn retain_grad_rejects_tensors_that_do_not_require_grad() {
    let x = Tensor::from_vec(vec![1.0, 2.0], &[2], false).unwrap();

    let err = x.retain_grad().unwrap_err();

    assert!(err.to_string().contains("requires gradients"));
    assert!(!x.retains_grad());
}

#[test]
fn grad_hook_can_modify_leaf_gradient_and_be_removed() {
    let x = Tensor::from_vec(vec![2.0, 3.0], &[2], true).unwrap();
    let hook_id = x.register_grad_hook(|grad| Ok(grad.iter().map(|value| value * 0.5).collect()));

    x.mul(&x).unwrap().sum().unwrap().backward().unwrap();
    assert_close(&x.grad().unwrap(), &[2.0, 3.0]);

    assert!(x.remove_grad_hook(hook_id));
    assert!(!x.remove_grad_hook(hook_id));
    x.zero_grad();

    x.mul(&x).unwrap().sum().unwrap().backward().unwrap();
    assert_close(&x.grad().unwrap(), &[4.0, 6.0]);
}

#[test]
fn retained_non_leaf_hook_modifies_retained_and_propagated_gradient() {
    let x = Tensor::from_vec(vec![2.0, 3.0], &[2], true).unwrap();
    let y = x.mul(&x).unwrap();
    y.retain_grad().unwrap();
    y.register_grad_hook(|grad| Ok(grad.iter().map(|value| value * 3.0).collect()));

    y.sum().unwrap().backward().unwrap();

    assert_close(&y.grad().unwrap(), &[3.0, 3.0]);
    assert_close(&x.grad().unwrap(), &[12.0, 18.0]);
}

#[test]
fn grad_hook_shape_mismatch_errors_before_accumulation() {
    let x = Tensor::from_vec(vec![2.0, 3.0], &[2], true).unwrap();
    x.register_grad_hook(|_| Ok(vec![1.0]));

    let err = x.mul(&x).unwrap().sum().unwrap().backward().unwrap_err();

    assert!(err.to_string().contains("gradient hook"));
    assert!(x.grad().is_none());
}
