use heirloom::{no_grad, Tensor};

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
fn no_grad_and_detach_cut_autograd_tracking_but_not_storage_aliasing() {
    let x = Tensor::from_vec(vec![1.0, -2.0, 3.0], &[3], true).unwrap();
    let y = no_grad(|| x.relu()).unwrap();
    assert!(!y.requires_grad());
    assert!(y
        .backward()
        .unwrap_err()
        .to_string()
        .contains("does not require"));

    let detached = x.detach();
    assert!(!detached.requires_grad());
    x.add_(&Tensor::ones(&[3], false).unwrap()).unwrap();
    assert_close(&detached.data(), &[2.0, -1.0, 4.0]);
}

#[test]
fn saved_tensor_version_check_catches_in_place_mutation_before_backward() {
    let x = Tensor::from_vec(vec![-1.0, 2.0, 3.0], &[3], true).unwrap();
    let y = x.relu().unwrap();
    assert_eq!(x.storage_version(), 0);

    x.add_(&Tensor::ones(&[3], false).unwrap()).unwrap();
    assert_eq!(x.storage_version(), 1);

    let err = y.sum().unwrap().backward().unwrap_err();
    assert!(err.to_string().contains("modified in-place"));
}

#[test]
fn sub_mul_div_backward_cover_non_fused_arithmetic() {
    let x = Tensor::from_vec(vec![2.0, 4.0], &[2], true).unwrap();
    let y = Tensor::from_vec(vec![1.0, 2.0], &[2], true).unwrap();

    let loss = x
        .mul(&y)
        .unwrap()
        .add(&x.div(&y).unwrap())
        .unwrap()
        .sub(&y)
        .unwrap()
        .sum()
        .unwrap();
    loss.backward().unwrap();

    assert_close(&x.grad().unwrap(), &[2.0, 2.5]);
    assert_close(&y.grad().unwrap(), &[-1.0, 2.0]);
}

#[test]
fn mse_loss_is_composed_from_primitives_and_supports_target_gradients() {
    let pred = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3], true).unwrap();
    let target = Tensor::from_vec(vec![0.0, 2.0, 1.0], &[3], true).unwrap();

    let loss = pred.mse_loss(&target).unwrap();
    assert_close(&loss.data(), &[10.0 / 3.0]);
    loss.backward().unwrap();

    assert_close(&pred.grad().unwrap(), &[2.0 / 3.0, 0.0, 2.0]);
    assert_close(&target.grad().unwrap(), &[-2.0 / 3.0, 0.0, -2.0]);
}

#[test]
fn reshape_copies_non_contiguous_views_and_backpropagates_through_copy() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();
    let reshaped = x.transpose().unwrap().reshape(&[6]).unwrap();

    assert_eq!(reshaped.shape(), vec![6]);
    assert!(reshaped.is_contiguous());
    assert_close(&reshaped.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    reshaped.sum().unwrap().backward().unwrap();
    assert_close(&x.grad().unwrap(), &[1.0; 6]);
}

#[test]
fn narrow_backward_scatters_into_the_original_shape() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();
    let y = x.narrow(1, 1, 2).unwrap();

    assert_eq!(y.shape(), vec![2, 2]);
    assert_close(&y.data(), &[2.0, 3.0, 5.0, 6.0]);

    y.sum().unwrap().backward().unwrap();
    assert_close(&x.grad().unwrap(), &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn permute_rank3_is_a_strided_view_and_backpropagates() {
    let x = Tensor::from_vec((0..24).map(|v| v as f32).collect(), &[2, 3, 4], true).unwrap();
    let y = x.permute(&[2, 0, 1]).unwrap();

    assert_eq!(y.shape(), vec![4, 2, 3]);
    assert_eq!(y.strides(), vec![1, 12, 4]);
    y.sum().unwrap().backward().unwrap();
    assert_close(&x.grad().unwrap(), &[1.0; 24]);
}

#[test]
fn as_strided_is_available_only_as_a_checked_non_differentiable_escape_hatch() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false).unwrap();
    let view = x.as_strided(&[2, 2], &[2, 1], 1).unwrap();
    assert_eq!(view.shape(), vec![2, 2]);
    assert_close(&view.data(), &[2.0, 3.0, 4.0, 5.0]);

    let grad_x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], true).unwrap();
    let err = grad_x.as_strided(&[2], &[1], 0).unwrap_err();
    assert!(err.to_string().contains("overlapping alias semantics"));
}

#[test]
fn backward_with_explicit_seed_supports_non_scalar_outputs() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], true).unwrap();
    let y = x.mul(&x).unwrap();

    y.backward_with_grad(vec![1.0, 1.0, 1.0]).unwrap();
    assert_close(&x.grad().unwrap(), &[2.0, 4.0, 6.0]);
}

#[test]
fn dim_reductions_match_basic_pytorch_shape_and_gradient_behavior() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();

    let col_sum = x.sum_dim(0, false).unwrap();
    assert_eq!(col_sum.shape(), vec![3]);
    assert_close(&col_sum.data(), &[5.0, 7.0, 9.0]);

    let row_mean = x.mean_dim(1, true).unwrap();
    assert_eq!(row_mean.shape(), vec![2, 1]);
    assert_close(&row_mean.data(), &[2.0, 5.0]);

    row_mean.sum().unwrap().backward().unwrap();
    assert_close(
        &x.grad().unwrap(),
        &[
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
    );
}

#[test]
fn dim_reduction_to_rank_zero_and_empty_dim_edges_are_explicit() {
    let x = Tensor::from_vec(vec![2.0, 4.0, 6.0], &[3], true).unwrap();
    let total = x.sum_dim(0, false).unwrap();
    assert_eq!(total.shape(), Vec::<usize>::new());
    assert_close(&total.data(), &[12.0]);

    let empty = Tensor::from_vec(Vec::new(), &[0, 3], false).unwrap();
    let empty_sum = empty.sum_dim(0, false).unwrap();
    assert_eq!(empty_sum.shape(), vec![3]);
    assert_close(&empty_sum.data(), &[0.0, 0.0, 0.0]);
    assert!(empty
        .mean_dim(0, false)
        .unwrap_err()
        .to_string()
        .contains("empty"));
}

#[test]
fn expand_is_a_zero_stride_view_and_reduces_gradients_back_to_base_shape() {
    let bias = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3], true).unwrap();
    let expanded = bias.expand(&[2, 3]).unwrap();

    assert_eq!(expanded.shape(), vec![2, 3]);
    assert_eq!(expanded.strides(), vec![0, 1]);
    assert!(expanded.has_internal_overlap());
    assert_close(&expanded.data(), &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);

    expanded.sum().unwrap().backward().unwrap();
    assert_close(&bias.grad().unwrap(), &[2.0, 2.0, 2.0]);
}

#[test]
fn expand_supports_middle_axis_broadcasting_and_blocks_in_place_writes() {
    let x = Tensor::from_vec(vec![1.0, 2.0], &[2, 1], true).unwrap();
    let expanded = x.expand(&[2, 3]).unwrap();

    assert_eq!(expanded.strides(), vec![1, 0]);
    assert_close(&expanded.data(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

    let err = expanded
        .add_(&Tensor::ones(&[2, 3], false).unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("internal overlap"));

    expanded.mean().unwrap().backward().unwrap();
    assert_close(&x.grad().unwrap(), &[0.5, 0.5]);
}

#[test]
fn signed_dim_reductions_accept_negative_dims_and_reject_out_of_range_dims() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();

    let last_dim = x.sum_dim_signed(-1, false).unwrap();
    assert_eq!(last_dim.shape(), vec![2]);
    assert_close(&last_dim.data(), &[6.0, 15.0]);

    let first_dim_mean = x.mean_dim_signed(-2, false).unwrap();
    assert_eq!(first_dim_mean.shape(), vec![3]);
    assert_close(&first_dim_mean.data(), &[2.5, 3.5, 4.5]);

    assert!(x
        .sum_dim_signed(-3, false)
        .unwrap_err()
        .to_string()
        .contains("out of range"));
}

#[test]
fn softmax_dim_is_stable_and_rows_sum_to_one() {
    let logits =
        Tensor::from_vec(vec![1000.0, 1001.0, 1002.0, 1.0, 2.0, 3.0], &[2, 3], true).unwrap();
    let probs = logits.softmax_dim_signed(-1).unwrap();
    let data = probs.data();

    assert_eq!(probs.shape(), vec![2, 3]);
    assert_close(&data[0..3], &[0.09003057, 0.24472848, 0.66524094]);
    assert_close(&data[3..6], &[0.09003057, 0.24472848, 0.66524094]);
    assert_close(&probs.sum_dim(1, false).unwrap().data(), &[1.0, 1.0]);
}

#[test]
fn cross_entropy_for_logits_matches_stable_manual_fixture_and_gradients() {
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.1, 0.5, 1.5, -1.0], &[2, 3], true).unwrap();
    let loss = logits.cross_entropy_for_logits(&[0, 1]).unwrap();

    assert_close(&loss.data(), &[0.39428452]);
    loss.backward().unwrap();
    assert_close(
        &logits.grad().unwrap(),
        &[
            -0.17049943,
            0.12121649,
            0.04928295,
            0.12685809,
            -0.15516396,
            0.02830587,
        ],
    );
}

#[test]
fn cross_entropy_for_logits_tensor_matches_slice_targets() {
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.1, 0.5, 1.5, -1.0], &[2, 3], true).unwrap();
    let targets = Tensor::from_i64(vec![0, 1], &[2], false).unwrap();
    let loss = logits.cross_entropy_for_logits_tensor(&targets).unwrap();

    assert_close(&loss.data(), &[0.39428452]);
    loss.backward().unwrap();
    assert_close(
        &logits.grad().unwrap(),
        &[
            -0.17049943,
            0.12121649,
            0.04928295,
            0.12685809,
            -0.15516396,
            0.02830587,
        ],
    );
}
