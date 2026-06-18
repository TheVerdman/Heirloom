use heirloom::nn::{cross_entropy_for_logits, Linear, Module, Optimizer, Sequential, Sgd};
use heirloom::{Result, Tensor, TensorError};

fn synthetic_three_class_data(samples_per_class: usize) -> Result<(Tensor, Vec<usize>)> {
    let centers = [(2.0_f32, 0.0_f32), (0.0, 2.0), (-2.0, -2.0)];
    let mut inputs = Vec::with_capacity(samples_per_class * centers.len() * 2);
    let mut targets = Vec::with_capacity(samples_per_class * centers.len());

    for (class, (cx, cy)) in centers.iter().copied().enumerate() {
        for i in 0..samples_per_class {
            let jitter_x = ((i % 5) as f32 - 2.0) * 0.08;
            let jitter_y = (((i * 3 + 1) % 5) as f32 - 2.0) * 0.08;
            inputs.push(cx + jitter_x);
            inputs.push(cy + jitter_y);
            targets.push(class);
        }
    }

    Ok((
        Tensor::from_vec(inputs, &[samples_per_class * centers.len(), 2], false)?,
        targets,
    ))
}

fn accuracy(logits: &Tensor, targets: &[usize]) -> Result<f32> {
    let shape = logits.shape();
    if shape.len() != 2 || shape[0] != targets.len() {
        return Err(TensorError::Shape(format!(
            "accuracy expected logits [batch, classes] matching {} targets, got {:?}",
            targets.len(),
            shape
        )));
    }
    let classes = shape[1];
    let data = logits.data();
    let mut correct = 0usize;

    for (row, target) in targets.iter().enumerate().take(shape[0]) {
        let row_start = row * classes;
        let mut best_class = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for class in 0..classes {
            let value = data[row_start + class];
            if value > best_value {
                best_value = value;
                best_class = class;
            }
        }
        if best_class == *target {
            correct += 1;
        }
    }

    Ok(correct as f32 / targets.len() as f32)
}

fn main() -> Result<()> {
    let (x, targets) = synthetic_three_class_data(32)?;
    let model = Sequential::new(vec![Box::new(Linear::new(2, 3)?)]);
    let optimizer = Sgd::new(model.parameters(), 0.12)?;
    let epochs = 180;

    let mut initial_loss = None;
    for epoch in 0..epochs {
        optimizer.zero_grad();
        let logits = model.forward(&x)?;
        let loss = cross_entropy_for_logits(&logits, &targets)?;
        let loss_value = loss.data()[0];
        initial_loss.get_or_insert(loss_value);
        loss.backward()?;
        optimizer.step()?;

        if epoch % 30 == 0 || epoch == epochs - 1 {
            let acc = accuracy(&model.forward(&x)?, &targets)?;
            println!("epoch={epoch:03} loss={loss_value:.6} acc={acc:.3}");
        }
    }

    let final_logits = model.forward(&x)?;
    let final_loss = cross_entropy_for_logits(&final_logits, &targets)?.data()[0];
    let final_acc = accuracy(&final_logits, &targets)?;
    let first_loss = initial_loss.unwrap_or(final_loss);
    println!("initial_loss={first_loss:.6}");
    println!("final_loss={final_loss:.6}");
    println!("final_accuracy={final_acc:.3}");

    if final_loss >= first_loss * 0.35 || final_acc < 0.95 {
        return Err(TensorError::InvalidOperation(format!(
            "classification training underperformed: initial_loss={first_loss}, final_loss={final_loss}, accuracy={final_acc}"
        )));
    }

    Ok(())
}
