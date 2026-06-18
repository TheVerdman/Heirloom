use heirloom::nn::{mse_loss, AdamW, Linear, Module, Optimizer, ReLU, Sequential, Sgd};
use heirloom::Tensor;

fn assert_loss_decreases(model: &Sequential, x: &Tensor, y: &Tensor, epochs: usize) -> (f32, f32) {
    let optimizer = Sgd::new(model.parameters(), 0.08).unwrap();
    let initial = mse_loss(&model.forward(x).unwrap(), y).unwrap().data()[0];

    for _ in 0..epochs {
        optimizer.zero_grad();
        let loss = mse_loss(&model.forward(x).unwrap(), y).unwrap();
        loss.backward().unwrap();
        optimizer.step().unwrap();
    }

    let final_loss = mse_loss(&model.forward(x).unwrap(), y).unwrap().data()[0];
    (initial, final_loss)
}

#[test]
fn sequential_linear_model_trains_on_synthetic_data() {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..80 {
        let x0 = (i as f32 / 80.0) * 2.0 - 1.0;
        let x1 = (((i * 29 + 5) % 80) as f32 / 80.0) * 2.0 - 1.0;
        inputs.push(x0);
        inputs.push(x1);
        targets.push(3.0 * x0 - 2.0 * x1 + 0.5);
    }

    let x = Tensor::from_vec(inputs, &[80, 2], false).unwrap();
    let y = Tensor::from_vec(targets, &[80, 1], false).unwrap();
    let model = Sequential::new(vec![Box::new(Linear::new(2, 1).unwrap())]);

    let (initial, final_loss) = assert_loss_decreases(&model, &x, &y, 180);
    assert!(
        final_loss < initial * 0.05,
        "expected large loss reduction, initial={initial}, final={final_loss}"
    );
}

#[test]
fn sequential_composes_relu_and_linear_layers() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3).unwrap()),
        Box::new(ReLU::new()),
        Box::new(Linear::new(3, 1).unwrap()),
    ]);
    let x = Tensor::from_vec(vec![1.0, -2.0, 0.5, 0.25], &[2, 2], false).unwrap();
    let out = model.forward(&x).unwrap();

    assert_eq!(model.len(), 3);
    assert_eq!(out.shape(), vec![2, 1]);
    assert_eq!(model.parameters().len(), 4);
}

#[test]
fn sgd_rejects_invalid_learning_rates() {
    let model = Sequential::new(vec![Box::new(Linear::new(2, 1).unwrap())]);
    match Sgd::new(model.parameters(), 0.0) {
        Ok(_) => panic!("expected invalid learning rate to fail"),
        Err(err) => assert!(err.to_string().contains("learning rate")),
    }
}

#[test]
fn adamw_rejects_non_finite_clipped_gradient_norm() {
    let parameter = Tensor::from_vec(vec![1.0], &[1], true).unwrap();
    let nan_multiplier = Tensor::from_vec(vec![f32::NAN], &[1], false).unwrap();
    let loss = parameter.mul(&nan_multiplier).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let mut optimizer = AdamW::new(vec![parameter], 0.01)
        .unwrap()
        .with_clip_norm(Some(1.0))
        .unwrap();
    let err = optimizer.step_mut().unwrap_err();

    assert!(err.to_string().contains("global gradient"));
}
