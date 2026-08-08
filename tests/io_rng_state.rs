use heirloom::nn::{load_state_dict, save_state_dict, Linear, Module, Sequential};
use heirloom::rng::HeirloomRng;
use heirloom::{npy, Tensor};
use std::fs;
use std::path::PathBuf;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "index {index}: actual={a}, expected={e}, actual={actual:?}, expected={expected:?}"
        );
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "heirloom_{name}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn npy_round_trips_contiguous_f32_tensor() {
    let dir = temp_dir("npy_roundtrip");
    let path = dir.join("tensor.npy");
    let tensor = Tensor::from_vec(vec![1.5, -2.0, 3.25, 4.5], &[2, 2], true).unwrap();

    npy::write_npy(&path, &tensor).unwrap();
    let loaded = npy::read_npy(&path, false).unwrap();

    assert_eq!(loaded.shape(), vec![2, 2]);
    assert!(!loaded.requires_grad());
    assert_close(&loaded.data(), &[1.5, -2.0, 3.25, 4.5]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn npy_writes_logical_data_for_non_contiguous_views() {
    let dir = temp_dir("npy_view");
    let path = dir.join("view.npy");
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let view = tensor.transpose().unwrap();

    npy::write_npy(&path, &view).unwrap();
    let loaded = npy::read_npy(&path, false).unwrap();

    assert_eq!(loaded.shape(), vec![3, 2]);
    assert_close(&loaded.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn npy_rejects_overflowing_shape_before_reading_payload() {
    let dir = temp_dir("npy_shape_overflow");
    let path = dir.join("overflow.npy");
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 2), }}",
        usize::MAX
    );
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    fs::write(&path, bytes).unwrap();

    let error = npy::read_npy(&path, false).expect_err("overflowing shape must be rejected");
    assert!(error.to_string().contains("overflows usize element count"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rng_is_deterministic_for_uniform_and_normal_sequences() {
    let mut a = HeirloomRng::new(42);
    let mut b = HeirloomRng::new(42);

    let uniform_a = (0..8).map(|_| a.uniform_f32()).collect::<Vec<_>>();
    let uniform_b = (0..8).map(|_| b.uniform_f32()).collect::<Vec<_>>();
    assert_close(&uniform_a, &uniform_b);
    assert!(uniform_a.iter().all(|value| *value >= 0.0 && *value < 1.0));

    let normal_a = (0..8).map(|_| a.normal_f32()).collect::<Vec<_>>();
    let normal_b = (0..8).map(|_| b.normal_f32()).collect::<Vec<_>>();
    assert_close(&normal_a, &normal_b);
}

#[test]
fn tensor_and_rng_shape_apis_reject_element_count_overflow() {
    assert!(Tensor::zeros(&[usize::MAX, 2], false).is_err());

    let mut rng = HeirloomRng::new(42);
    assert!(rng
        .uniform_tensor(&[usize::MAX, 2], -1.0, 1.0, false)
        .is_err());
    assert!(rng
        .normal_tensor(&[usize::MAX, 2], 0.0, 1.0, false)
        .is_err());
}

#[test]
fn linear_rng_initialization_is_reproducible() {
    let mut rng_a = HeirloomRng::new(7);
    let mut rng_b = HeirloomRng::new(7);

    let a = Linear::new_with_rng(3, 2, &mut rng_a).unwrap();
    let b = Linear::new_with_rng(3, 2, &mut rng_b).unwrap();

    assert_close(&a.weight().data(), &b.weight().data());
    assert_close(&a.bias().data(), &b.bias().data());
}

#[test]
fn linear_default_initialization_uses_seeded_rng_path() {
    let default = Linear::new(3, 2).unwrap();
    let seeded = Linear::new_with_seed(3, 2, Linear::DEFAULT_SEED).unwrap();
    let alternate = Linear::new_with_seed(3, 2, Linear::DEFAULT_SEED + 1).unwrap();

    assert_close(&default.weight().data(), &seeded.weight().data());
    assert_close(&default.bias().data(), &seeded.bias().data());
    assert!(default
        .weight()
        .data()
        .iter()
        .zip(alternate.weight().data().iter())
        .any(|(a, b)| (a - b).abs() > 1e-4));
}

#[test]
fn state_dict_saves_and_loads_named_parameters() {
    let dir = temp_dir("state_dict");
    let mut rng_a = HeirloomRng::new(11);
    let mut rng_b = HeirloomRng::new(99);
    let source = Sequential::new(vec![Box::new(
        Linear::new_with_rng(2, 3, &mut rng_a).unwrap(),
    )]);
    let target = Sequential::new(vec![Box::new(
        Linear::new_with_rng(2, 3, &mut rng_b).unwrap(),
    )]);
    let input = Tensor::from_vec(vec![1.0, -2.0, 0.5, 0.25], &[2, 2], false).unwrap();

    let source_before = source.forward(&input).unwrap().data();
    let target_before = target.forward(&input).unwrap().data();
    assert!(source_before
        .iter()
        .zip(target_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-4));

    save_state_dict(&source, &dir).unwrap();
    load_state_dict(&target, &dir).unwrap();

    let target_after = target.forward(&input).unwrap().data();
    assert_close(&target_after, &source_before);
    assert!(dir.join("state.tsv").exists());
    let manifest = fs::read_to_string(dir.join("state.tsv")).unwrap();
    assert!(manifest.contains("0.weight\t0000_0_weight.npy"));
    assert!(manifest.contains("0.bias\t0001_0_bias.npy"));
    assert!(dir.join("0000_0_weight.npy").exists());
    assert!(dir.join("0001_0_bias.npy").exists());

    fs::remove_dir_all(dir).unwrap();
}
