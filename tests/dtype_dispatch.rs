use heirloom::{npy, DType, Tensor};
use std::fs;
use std::path::PathBuf;

fn assert_close_f64(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tolerance,
            "index {index}: actual={a}, expected={e}, full actual={actual:?}"
        );
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "heirloom_dtype_{name}_{}_{}",
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
fn typed_constructors_accessors_and_views_preserve_dtype() {
    let ints = Tensor::from_i64(vec![1, 2, 3, 4, 5, 6], &[2, 3], false).unwrap();
    assert_eq!(ints.dtype(), DType::I64);
    assert_eq!(ints.data_i64().unwrap(), vec![1, 2, 3, 4, 5, 6]);

    let transposed = ints.transpose().unwrap();
    assert_eq!(transposed.dtype(), DType::I64);
    assert_eq!(transposed.shape(), vec![3, 2]);
    assert_eq!(transposed.data_i64().unwrap(), vec![1, 4, 2, 5, 3, 6]);

    let err = Tensor::from_bool(vec![true, false], &[2], true).unwrap_err();
    assert!(err.to_string().contains("only floating tensors"));
}

#[test]
fn binary_ops_promote_dtypes_and_keep_floating_autograd() {
    let x = Tensor::from_f32(vec![1.0, 2.0], &[2], true).unwrap();
    let y = Tensor::from_f64(vec![0.5, -1.0], &[2], true).unwrap();

    let z = x.add(&y).unwrap();
    assert_eq!(z.dtype(), DType::F64);
    assert_close_f64(&z.data_f64_exact().unwrap(), &[1.5, 1.0], 1e-12);

    z.sum().unwrap().backward().unwrap();
    assert_close_f64(&x.grad_f64().unwrap(), &[1.0, 1.0], 1e-12);
    assert_close_f64(&y.grad_f64().unwrap(), &[1.0, 1.0], 1e-12);

    let ints = Tensor::from_i64(vec![2, 4], &[2], false).unwrap();
    let floats = Tensor::from_f32(vec![0.5, 1.5], &[2], false).unwrap();
    let promoted = ints.mul(&floats).unwrap();
    assert_eq!(promoted.dtype(), DType::F32);
    assert_eq!(promoted.data_f32().unwrap(), vec![1.0, 6.0]);

    let quotient = ints
        .div(&Tensor::from_i64(vec![4, 2], &[2], false).unwrap())
        .unwrap();
    assert_eq!(quotient.dtype(), DType::F64);
    assert_close_f64(&quotient.data_f64_exact().unwrap(), &[0.5, 2.0], 1e-12);
}

#[test]
fn f64_matmul_and_backward_stay_f64() {
    let a = Tensor::from_f64(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();
    let b = Tensor::from_f64(vec![5.0, 6.0], &[2, 1], true).unwrap();

    let out = a.matmul(&b).unwrap();
    assert_eq!(out.dtype(), DType::F64);
    assert_close_f64(&out.data_f64_exact().unwrap(), &[17.0, 39.0], 1e-12);

    out.sum().unwrap().backward().unwrap();
    assert_close_f64(&a.grad_f64().unwrap(), &[5.0, 6.0, 5.0, 6.0], 1e-12);
    assert_close_f64(&b.grad_f64().unwrap(), &[4.0, 6.0], 1e-12);
}

#[test]
fn reductions_have_explicit_non_float_output_rules() {
    let ints = Tensor::from_i64(vec![1, 2, 3], &[3], false).unwrap();
    let int_sum = ints.sum().unwrap();
    assert_eq!(int_sum.dtype(), DType::I64);
    assert_eq!(int_sum.data_i64().unwrap(), vec![6]);

    let int_mean = ints.mean().unwrap();
    assert_eq!(int_mean.dtype(), DType::F64);
    assert_close_f64(&int_mean.data_f64_exact().unwrap(), &[2.0], 1e-12);

    let flags = Tensor::from_bool(vec![true, false, true], &[3], false).unwrap();
    let flag_sum = flags.sum().unwrap();
    assert_eq!(flag_sum.dtype(), DType::I64);
    assert_eq!(flag_sum.data_i64().unwrap(), vec![2]);

    let flag_mean = flags.mean().unwrap();
    assert_eq!(flag_mean.dtype(), DType::F64);
    assert_close_f64(&flag_mean.data_f64_exact().unwrap(), &[2.0 / 3.0], 1e-12);
}

#[test]
fn non_float_tensors_are_rejected_by_floating_only_ops() {
    let ints = Tensor::from_i64(vec![1, 2, 3, 4], &[2, 2], false).unwrap();

    assert!(ints.relu().unwrap_err().to_string().contains("floating"));
    assert!(ints
        .matmul(&ints)
        .unwrap_err()
        .to_string()
        .contains("floating"));
}

#[test]
fn npy_round_trips_f64_i64_and_bool_tensors() {
    let dir = temp_dir("npy_typed");

    let f64_path = dir.join("f64.npy");
    let f64_tensor = Tensor::from_f64(vec![1.25, -2.5, 3.75, 4.5], &[2, 2], true).unwrap();
    npy::write_npy(&f64_path, &f64_tensor).unwrap();
    let f64_loaded = npy::read_npy(&f64_path, false).unwrap();
    assert_eq!(f64_loaded.dtype(), DType::F64);
    assert_close_f64(
        &f64_loaded.data_f64_exact().unwrap(),
        &[1.25, -2.5, 3.75, 4.5],
        1e-12,
    );

    let i64_path = dir.join("i64.npy");
    let i64_tensor = Tensor::from_i64(vec![10, -20, 30, 40], &[2, 2], false).unwrap();
    npy::write_npy(&i64_path, &i64_tensor).unwrap();
    let i64_loaded = npy::read_npy(&i64_path, false).unwrap();
    assert_eq!(i64_loaded.dtype(), DType::I64);
    assert_eq!(i64_loaded.data_i64().unwrap(), vec![10, -20, 30, 40]);

    let bool_path = dir.join("bool.npy");
    let bool_tensor = Tensor::from_bool(vec![true, false, true, true], &[2, 2], false).unwrap();
    npy::write_npy(&bool_path, &bool_tensor).unwrap();
    let bool_loaded = npy::read_npy(&bool_path, false).unwrap();
    assert_eq!(bool_loaded.dtype(), DType::Bool);
    assert_eq!(
        bool_loaded.data_bool().unwrap(),
        vec![true, false, true, true]
    );

    fs::remove_dir_all(dir).unwrap();
}
