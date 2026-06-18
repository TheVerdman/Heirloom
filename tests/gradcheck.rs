use heirloom::Tensor;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "index {index}: actual={a}, expected={e}, tolerance={tolerance}, full actual={actual:?}, expected={expected:?}"
        );
    }
}

fn finite_difference(mut values: Vec<f32>, eps: f32, f: impl Fn(&[f32]) -> f32) -> Vec<f32> {
    let mut grad = vec![0.0; values.len()];
    for index in 0..values.len() {
        let original = values[index];
        values[index] = original + eps;
        let plus = f(&values);
        values[index] = original - eps;
        let minus = f(&values);
        values[index] = original;
        grad[index] = (plus - minus) / (2.0 * eps);
    }
    grad
}

fn mlpish_loss_value(x_data: &[f32], w_data: &[f32], b_data: &[f32]) -> f32 {
    let x = Tensor::from_vec(x_data.to_vec(), &[2, 3], false).unwrap();
    let w = Tensor::from_vec(w_data.to_vec(), &[3, 2], false).unwrap();
    let b = Tensor::from_vec(b_data.to_vec(), &[2], false).unwrap();
    x.matmul(&w)
        .unwrap()
        .add(&b)
        .unwrap()
        .relu()
        .unwrap()
        .mean()
        .unwrap()
        .data()[0]
}

#[test]
fn finite_difference_checks_composed_matmul_add_relu_mean_gradients() {
    let x_data = vec![0.25, -0.4, 0.8, 0.6, 0.1, -0.2];
    let w_data = vec![0.7, -0.3, 0.2, 0.5, -0.1, 0.4];
    let b_data = vec![0.9, 0.7];

    let x = Tensor::from_vec(x_data.clone(), &[2, 3], true).unwrap();
    let w = Tensor::from_vec(w_data.clone(), &[3, 2], true).unwrap();
    let b = Tensor::from_vec(b_data.clone(), &[2], true).unwrap();
    let loss = x
        .matmul(&w)
        .unwrap()
        .add(&b)
        .unwrap()
        .relu()
        .unwrap()
        .mean()
        .unwrap();
    loss.backward().unwrap();

    let eps = 1e-3;
    let x_numeric = finite_difference(x_data.clone(), eps, |candidate| {
        mlpish_loss_value(candidate, &w_data, &b_data)
    });
    let w_numeric = finite_difference(w_data.clone(), eps, |candidate| {
        mlpish_loss_value(&x_data, candidate, &b_data)
    });
    let b_numeric = finite_difference(b_data.clone(), eps, |candidate| {
        mlpish_loss_value(&x_data, &w_data, candidate)
    });

    assert_close(&x.grad().unwrap(), &x_numeric, 2e-2);
    assert_close(&w.grad().unwrap(), &w_numeric, 2e-2);
    assert_close(&b.grad().unwrap(), &b_numeric, 2e-2);
}

fn broadcast_loss_value(x_data: &[f32], y_data: &[f32], denom_data: &[f32]) -> f32 {
    let x = Tensor::from_vec(x_data.to_vec(), &[2, 2], false).unwrap();
    let y = Tensor::from_vec(y_data.to_vec(), &[2, 1], false).unwrap();
    let denom = Tensor::from_vec(denom_data.to_vec(), &[2], false).unwrap();
    x.mul(&y)
        .unwrap()
        .div(&denom)
        .unwrap()
        .sum()
        .unwrap()
        .data()[0]
}

#[test]
fn finite_difference_checks_broadcast_mul_div_gradients() {
    let x_data = vec![1.2, -0.7, 0.4, 2.0];
    let y_data = vec![0.8, -1.3];
    let denom_data = vec![1.5, 2.2];

    let x = Tensor::from_vec(x_data.clone(), &[2, 2], true).unwrap();
    let y = Tensor::from_vec(y_data.clone(), &[2, 1], true).unwrap();
    let denom = Tensor::from_vec(denom_data.clone(), &[2], true).unwrap();
    let loss = x.mul(&y).unwrap().div(&denom).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let eps = 1e-3;
    let x_numeric = finite_difference(x_data.clone(), eps, |candidate| {
        broadcast_loss_value(candidate, &y_data, &denom_data)
    });
    let y_numeric = finite_difference(y_data.clone(), eps, |candidate| {
        broadcast_loss_value(&x_data, candidate, &denom_data)
    });
    let denom_numeric = finite_difference(denom_data.clone(), eps, |candidate| {
        broadcast_loss_value(&x_data, &y_data, candidate)
    });

    assert_close(&x.grad().unwrap(), &x_numeric, 2e-2);
    assert_close(&y.grad().unwrap(), &y_numeric, 2e-2);
    assert_close(&denom.grad().unwrap(), &denom_numeric, 2e-2);
}

fn softmax_squared_loss_value(logits_data: &[f32]) -> f32 {
    let logits = Tensor::from_vec(logits_data.to_vec(), &[2, 3], false).unwrap();
    let probs = logits.softmax_dim(1).unwrap();
    probs.mul(&probs).unwrap().sum().unwrap().data()[0]
}

#[test]
fn finite_difference_checks_softmax_backward() {
    let logits_data = vec![1.0, -0.5, 0.25, 0.7, 0.2, -1.0];
    let logits = Tensor::from_vec(logits_data.clone(), &[2, 3], true).unwrap();
    let probs = logits.softmax_dim(1).unwrap();
    let loss = probs.mul(&probs).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let numeric = finite_difference(logits_data, 1e-3, softmax_squared_loss_value);
    assert_close(&logits.grad().unwrap(), &numeric, 2e-2);
}

fn cross_entropy_loss_value(logits_data: &[f32]) -> f32 {
    Tensor::from_vec(logits_data.to_vec(), &[2, 3], false)
        .unwrap()
        .cross_entropy_for_logits(&[2, 0])
        .unwrap()
        .data()[0]
}

#[test]
fn finite_difference_checks_cross_entropy_logits_backward() {
    let logits_data = vec![0.2, -1.0, 1.7, 1.2, 0.4, -0.3];
    let logits = Tensor::from_vec(logits_data.clone(), &[2, 3], true).unwrap();
    let loss = logits.cross_entropy_for_logits(&[2, 0]).unwrap();
    loss.backward().unwrap();

    let numeric = finite_difference(logits_data, 1e-3, cross_entropy_loss_value);
    assert_close(&logits.grad().unwrap(), &numeric, 2e-2);
}
