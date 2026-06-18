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
fn pytorch_parity_add_broadcast_gradient_fixture() {
    // Mirrors:
    // x = torch.arange(6.).reshape(2, 3).requires_grad_()
    // b = torch.tensor([10., 20., 30.], requires_grad=True)
    // (x + b).sum().backward()
    let x = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], true).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3], true).unwrap();

    x.add(&b).unwrap().sum().unwrap().backward().unwrap();

    assert_close(&x.grad().unwrap(), &[1.0; 6]);
    assert_close(&b.grad().unwrap(), &[2.0, 2.0, 2.0]);
}

#[test]
fn pytorch_parity_transpose_then_matmul_fixture() {
    // Mirrors torch behavior for a.T @ b and sum-loss gradients.
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();
    let b = Tensor::from_vec(vec![1.0, -1.0, 2.0, 0.5], &[2, 2], true).unwrap();

    let out = a.transpose().unwrap().matmul(&b).unwrap();
    assert_eq!(out.shape(), vec![3, 2]);
    assert_close(&out.data(), &[9.0, 1.0, 12.0, 0.5, 15.0, 0.0]);

    out.sum().unwrap().backward().unwrap();
    assert_close(&a.grad().unwrap(), &[0.0, 0.0, 0.0, 2.5, 2.5, 2.5]);
    assert_close(&b.grad().unwrap(), &[6.0, 6.0, 15.0, 15.0]);
}

#[test]
fn pytorch_parity_mse_loss_fixture() {
    // Mirrors torch.nn.functional.mse_loss(pred, target, reduction="mean").
    let pred = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3], true).unwrap();
    let target = Tensor::from_vec(vec![0.0, 2.0, 1.0], &[3], false).unwrap();

    let loss = pred.mse_loss(&target).unwrap();
    assert_close(&loss.data(), &[10.0 / 3.0]);
    loss.backward().unwrap();
    assert_close(&pred.grad().unwrap(), &[2.0 / 3.0, 0.0, 2.0]);
}
