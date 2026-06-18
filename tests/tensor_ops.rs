use heirloom::Tensor;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "index {index}: actual={a}, expected={e}, full actual={actual:?}"
        );
    }
}

#[test]
fn transpose_is_a_strided_view_and_matches_pytorch_t_fixture() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();
    let xt = x.transpose().unwrap();

    assert_eq!(x.shape(), vec![2, 3]);
    assert_eq!(x.strides(), vec![3, 1]);
    assert_eq!(xt.shape(), vec![3, 2]);
    assert_eq!(xt.strides(), vec![1, 3]);
    assert_eq!(xt.storage_offset(), 0);
    assert_close(&xt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn add_broadcasts_and_unbroadcasts_gradients() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();
    let bias = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3], true).unwrap();

    let y = x.add(&bias).unwrap();
    assert_eq!(y.shape(), vec![2, 3]);
    assert_close(&y.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

    y.sum().unwrap().backward().unwrap();
    assert_close(&x.grad().unwrap(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_close(&bias.grad().unwrap(), &[2.0, 2.0, 2.0]);
}

#[test]
fn matmul_backward_matches_sum_loss_formula() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], true).unwrap();

    let out = a.matmul(&b).unwrap();
    assert_eq!(out.shape(), vec![2, 2]);
    assert_close(&out.data(), &[58.0, 64.0, 139.0, 154.0]);

    out.sum().unwrap().backward().unwrap();
    assert_close(&a.grad().unwrap(), &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
    assert_close(&b.grad().unwrap(), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn relu_mean_backward_uses_zero_derivative_at_zero() {
    let x = Tensor::from_vec(vec![-1.0, 0.0, 2.0, 4.0], &[4], true).unwrap();
    let loss = x.relu().unwrap().mean().unwrap();

    assert_close(&loss.data(), &[1.5]);
    loss.backward().unwrap();
    assert_close(&x.grad().unwrap(), &[0.0, 0.0, 0.25, 0.25]);
}

#[test]
fn view_keeps_storage_metadata_and_backpropagates_shape_only() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], true).unwrap();
    let viewed = x.view(&[2, 2]).unwrap();

    assert_eq!(viewed.shape(), vec![2, 2]);
    assert_eq!(viewed.strides(), vec![2, 1]);
    assert_close(&viewed.data(), &[1.0, 2.0, 3.0, 4.0]);

    viewed.sum().unwrap().backward().unwrap();
    assert_close(&x.grad().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn view_rejects_non_contiguous_transpose() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let xt = x.transpose().unwrap();
    let err = xt.view(&[6]).unwrap_err();
    assert!(err.to_string().contains("contiguous"));
}
