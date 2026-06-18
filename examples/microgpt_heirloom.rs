use heirloom::data::TokenDataset;
use heirloom::nn::{AdamW, Module, Optimizer, TinyTransformerConfig, TinyTransformerLm};
use heirloom::rng::HeirloomRng;
use heirloom::tokenizer::{BpeTokenizer, BOS_ID, EOS_ID};
use heirloom::{Result, TensorError};

fn main() -> Result<()> {
    let corpus = "anna likes rust. anna likes tiny models. rust likes anna. ";
    let tokenizer = BpeTokenizer::train(corpus, 280)?;
    let tokens = tokenizer.encode(corpus, true, true);
    let mut dataset = TokenDataset::new(tokens, 8, 2026)?;

    let config = TinyTransformerConfig {
        vocab_size: tokenizer.vocab_size(),
        block_size: 8,
        d_model: 12,
        n_heads: 3,
        ff_hidden: 24,
    };
    let mut rng = HeirloomRng::new(7);
    let model = TinyTransformerLm::new(config, &mut rng)?;
    let mut optimizer = AdamW::new(model.parameters(), 0.01)?
        .with_weight_decay(0.0)?
        .with_clip_norm(Some(1.0))?;

    let (input, target) = dataset.next_batch(4)?;
    let initial = model.loss(&input, &target)?.data()[0];
    for step in 0..40 {
        optimizer.zero_grad();
        let (input, target) = dataset.next_batch(4)?;
        let loss = model.loss(&input, &target)?;
        let value = loss.data()[0];
        loss.backward()?;
        optimizer.step_mut()?;
        if step % 10 == 0 {
            println!("step={step:03} loss={value:.6}");
        }
    }
    let (input, target) = dataset.next_batch(4)?;
    let final_loss = model.loss(&input, &target)?.data()[0];
    if final_loss >= initial {
        return Err(TensorError::InvalidOperation(format!(
            "microgpt example did not reduce loss: initial={initial}, final={final_loss}"
        )));
    }

    let mut prefix = tokenizer.encode("anna", true, false);
    if prefix.is_empty() {
        prefix.push(BOS_ID);
    }
    let generated = model.generate_greedy(&prefix, 24, EOS_ID)?;
    println!("initial_loss={initial:.6}");
    println!("final_loss={final_loss:.6}");
    println!("{}", tokenizer.decode(&generated));
    Ok(())
}
