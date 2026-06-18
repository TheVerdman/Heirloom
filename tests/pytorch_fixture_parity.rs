use heirloom::Tensor;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct FixtureFile {
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    name: String,
    op: String,
    targets: Option<Vec<usize>>,
    n_heads: Option<usize>,
    features: Option<usize>,
    eps: Option<f64>,
    inputs: Vec<FixtureInput>,
    output_shape: Vec<usize>,
    output_data: Vec<f32>,
    grads: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    name: String,
    shape: Vec<usize>,
    data: Vec<serde_json::Value>,
    requires_grad: bool,
    dtype: Option<String>,
}

fn assert_close(case_name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case_name}: length mismatch actual={actual:?}, expected={expected:?}"
    );
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "{case_name}: index {index}: actual={a}, expected={e}, tolerance={tolerance}, full actual={actual:?}, expected={expected:?}"
        );
    }
}

fn tensor(input: &FixtureInput) -> Tensor {
    match input.dtype.as_deref().unwrap_or("f32") {
        "f32" => Tensor::from_vec(
            input
                .data
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect(),
            &input.shape,
            input.requires_grad,
        )
        .unwrap(),
        "i64" => Tensor::from_i64(
            input
                .data
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect(),
            &input.shape,
            input.requires_grad,
        )
        .unwrap(),
        other => panic!("unsupported fixture dtype {other}"),
    }
}

#[test]
fn consumes_pytorch_parity_fixture_json() {
    let fixture: FixtureFile =
        serde_json::from_str(include_str!("fixtures/pytorch_parity.json")).unwrap();

    for case in fixture.cases {
        let tensors = case.inputs.iter().map(tensor).collect::<Vec<_>>();
        let output = match case.op.as_str() {
            "add_sum" => tensors[0].add(&tensors[1]).unwrap().sum().unwrap(),
            "matmul_mean" => tensors[0].matmul(&tensors[1]).unwrap().mean().unwrap(),
            "batched_matmul_mean" => tensors[0].matmul(&tensors[1]).unwrap().mean().unwrap(),
            "relu_mean" => tensors[0].relu().unwrap().mean().unwrap(),
            "gelu_tanh_mean" => tensors[0].gelu().unwrap().mean().unwrap(),
            "layer_norm_square_sum" => {
                assert_eq!(case.features.unwrap(), tensors[1].shape()[0]);
                let normalized = tensors[0]
                    .layer_norm_last_dim(&tensors[1], &tensors[2], case.eps.unwrap())
                    .unwrap();
                normalized.mul(&normalized).unwrap().sum().unwrap()
            }
            "embedding_sum" => tensors[0].embedding(&tensors[1]).unwrap().sum().unwrap(),
            "softmax_square_sum" => {
                let probs = tensors[0].softmax_dim(1).unwrap();
                probs.mul(&probs).unwrap().sum().unwrap()
            }
            "cross_entropy" => tensors[0]
                .cross_entropy_for_logits(case.targets.as_deref().unwrap())
                .unwrap(),
            "causal_attention_square_sum" => {
                let attended = tensors[0]
                    .causal_self_attention(&tensors[1], &tensors[2], case.n_heads.unwrap())
                    .unwrap();
                attended.mul(&attended).unwrap().sum().unwrap()
            }
            other => panic!("unknown fixture op {other} in {}", case.name),
        };

        assert_eq!(output.shape(), case.output_shape, "{}", case.name);
        assert_close(&case.name, &output.data(), &case.output_data, 2e-5);
        output.backward().unwrap();

        for (input, tensor) in case.inputs.iter().zip(tensors.iter()) {
            if input.requires_grad {
                let expected = case
                    .grads
                    .get(&input.name)
                    .unwrap_or_else(|| panic!("missing grad for {} in {}", input.name, case.name));
                assert_close(
                    &format!("{} grad {}", case.name, input.name),
                    &tensor.grad().unwrap(),
                    expected,
                    2e-5,
                );
            }
        }
    }
}
