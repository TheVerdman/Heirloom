use heirloom::{DType, Tensor};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) -> Result<(), TestCaseError> {
    prop_assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        prop_assert!(
            (a - e).abs() <= tolerance,
            "index {index}: actual={a}, expected={e}, tolerance={tolerance}, actual={actual:?}, expected={expected:?}"
        );
    }
    Ok(())
}

fn bounded_f32() -> impl Strategy<Value = f32> {
    -3.0_f32..3.0_f32
}

proptest! {
    #[test]
    fn add_broadcast_gradient_counts_repeat_axes(
        batch in 1usize..5,
        cols in 1usize..6,
        x_values in prop::collection::vec(bounded_f32(), 1..30),
        bias_values in prop::collection::vec(bounded_f32(), 1..10),
    ) {
        let x_len = batch * cols;
        let x_data = x_values.into_iter().cycle().take(x_len).collect::<Vec<_>>();
        let bias_data = bias_values.into_iter().cycle().take(cols).collect::<Vec<_>>();
        let x = Tensor::from_vec(x_data, &[batch, cols], true).unwrap();
        let bias = Tensor::from_vec(bias_data, &[cols], true).unwrap();

        x.add(&bias).unwrap().sum().unwrap().backward().unwrap();

        assert_close(&x.grad().unwrap(), &vec![1.0; x_len], 1e-6)?;
        assert_close(&bias.grad().unwrap(), &vec![batch as f32; cols], 1e-6)?;
    }

    #[test]
    fn expand_gradient_sums_across_broadcasted_axes(
        rows in 1usize..5,
        cols in 1usize..6,
        values in prop::collection::vec(bounded_f32(), 1..10),
    ) {
        let base_data = values.into_iter().cycle().take(cols).collect::<Vec<_>>();
        let base = Tensor::from_vec(base_data, &[cols], true).unwrap();
        let expanded = base.expand(&[rows, cols]).unwrap();

        expanded.mean().unwrap().backward().unwrap();

        let expected = vec![1.0 / cols as f32; cols];
        assert_close(&base.grad().unwrap(), &expected, 1e-6)?;
    }

    #[test]
    fn sum_dim_backward_places_ones_at_every_input_element(
        rows in 1usize..5,
        cols in 1usize..6,
        keepdim in any::<bool>(),
        values in prop::collection::vec(bounded_f32(), 1..30),
    ) {
        let len = rows * cols;
        let data = values.into_iter().cycle().take(len).collect::<Vec<_>>();
        let x = Tensor::from_vec(data, &[rows, cols], true).unwrap();
        x.sum_dim(1, keepdim).unwrap().sum().unwrap().backward().unwrap();

        assert_close(&x.grad().unwrap(), &vec![1.0; len], 1e-6)?;
    }

    #[test]
    fn mean_dim_backward_scales_by_reduced_axis(
        rows in 1usize..5,
        cols in 1usize..6,
        keepdim in any::<bool>(),
        values in prop::collection::vec(bounded_f32(), 1..30),
    ) {
        let len = rows * cols;
        let data = values.into_iter().cycle().take(len).collect::<Vec<_>>();
        let x = Tensor::from_vec(data, &[rows, cols], true).unwrap();
        x.mean_dim(1, keepdim).unwrap().sum().unwrap().backward().unwrap();

        assert_close(&x.grad().unwrap(), &vec![1.0 / cols as f32; len], 1e-6)?;
    }

    #[test]
    fn matmul_sum_backward_matches_closed_form(
        m in 1usize..4,
        k in 1usize..5,
        n in 1usize..4,
        left_values in prop::collection::vec(bounded_f32(), 1..40),
        right_values in prop::collection::vec(bounded_f32(), 1..40),
    ) {
        let left_data = left_values.into_iter().cycle().take(m * k).collect::<Vec<_>>();
        let right_data = right_values.into_iter().cycle().take(k * n).collect::<Vec<_>>();
        let left = Tensor::from_vec(left_data.clone(), &[m, k], true).unwrap();
        let right = Tensor::from_vec(right_data.clone(), &[k, n], true).unwrap();

        left.matmul(&right).unwrap().sum().unwrap().backward().unwrap();

        let mut expected_left = vec![0.0; m * k];
        for row in 0..m {
            for shared in 0..k {
                let mut acc = 0.0;
                for col in 0..n {
                    acc += right_data[shared * n + col];
                }
                expected_left[row * k + shared] = acc;
            }
        }

        let mut expected_right = vec![0.0; k * n];
        for shared in 0..k {
            for col in 0..n {
                let mut acc = 0.0;
                for row in 0..m {
                    acc += left_data[row * k + shared];
                }
                expected_right[shared * n + col] = acc;
            }
        }

        assert_close(&left.grad().unwrap(), &expected_left, 1e-5)?;
        assert_close(&right.grad().unwrap(), &expected_right, 1e-5)?;
    }

    #[test]
    fn mixed_i64_f32_add_and_mul_always_promote_to_f32(
        len in 1usize..20,
        int_values in prop::collection::vec(-20i64..20i64, 1..40),
        float_values in prop::collection::vec(bounded_f32(), 1..40),
    ) {
        let ints = int_values.into_iter().cycle().take(len).collect::<Vec<_>>();
        let floats = float_values.into_iter().cycle().take(len).collect::<Vec<_>>();
        let int_tensor = Tensor::from_i64(ints.clone(), &[len], false).unwrap();
        let float_tensor = Tensor::from_vec(floats.clone(), &[len], false).unwrap();

        let added = int_tensor.add(&float_tensor).unwrap();
        let multiplied = int_tensor.mul(&float_tensor).unwrap();

        prop_assert_eq!(added.dtype(), DType::F32);
        prop_assert_eq!(multiplied.dtype(), DType::F32);
        for index in 0..len {
            prop_assert!((added.data()[index] - (ints[index] as f32 + floats[index])).abs() <= 1e-5);
            prop_assert!((multiplied.data()[index] - (ints[index] as f32 * floats[index])).abs() <= 1e-5);
        }
    }

    #[test]
    fn bool_sum_counts_true_values(
        values in prop::collection::vec(any::<bool>(), 1..40),
    ) {
        let tensor = Tensor::from_bool(values.clone(), &[values.len()], false).unwrap();
        let expected = values.iter().filter(|value| **value).count() as i64;
        let sum = tensor.sum().unwrap();

        prop_assert_eq!(sum.dtype(), DType::I64);
        prop_assert_eq!(sum.data_i64().unwrap(), vec![expected]);
    }
}
