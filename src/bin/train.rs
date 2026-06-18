use heirloom::nn::{mse_loss, Linear, Module, Optimizer, Sequential, Sgd};
use heirloom::{Result, Tensor, TensorError};

fn synthetic_linear_regression(samples: usize) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::with_capacity(samples * 2);
    let mut targets = Vec::with_capacity(samples);

    for i in 0..samples {
        let x0 = (i as f32 / samples as f32) * 2.0 - 1.0;
        let scrambled = (i * 37 + 11) % samples;
        let x1 = (scrambled as f32 / samples as f32) * 2.0 - 1.0;
        let y = 3.0 * x0 - 2.0 * x1 + 0.5;
        inputs.push(x0);
        inputs.push(x1);
        targets.push(y);
    }

    Ok((
        Tensor::from_vec(inputs, &[samples, 2], false)?,
        Tensor::from_vec(targets, &[samples, 1], false)?,
    ))
}

fn main() -> Result<()> {
    let (x, y) = synthetic_linear_regression(96)?;
    let model = Sequential::new(vec![Box::new(Linear::new(2, 1)?)]);
    let optimizer = Sgd::new(model.parameters(), 0.08)?;
    let epochs = 220;

    let mut initial_loss = None;
    for epoch in 0..epochs {
        optimizer.zero_grad();

        let prediction = model.forward(&x)?;
        let loss = mse_loss(&prediction, &y)?;
        let loss_value = loss.data()[0];
        initial_loss.get_or_insert(loss_value);

        loss.backward()?;
        optimizer.step()?;

        if epoch % 40 == 0 || epoch == epochs - 1 {
            println!("epoch={epoch:03} loss={loss_value:.6}");
        }
    }

    let final_loss = mse_loss(&model.forward(&x)?, &y)?.data()[0];
    let first_loss = initial_loss.unwrap_or(final_loss);
    println!("initial_loss={first_loss:.6}");
    println!("final_loss={final_loss:.6}");

    if final_loss >= first_loss * 0.05 {
        return Err(TensorError::InvalidOperation(format!(
            "training did not reduce loss enough: initial={first_loss}, final={final_loss}"
        )));
    }

    println!("loss decreased by {:.2}x", first_loss / final_loss);
    Ok(())
}
