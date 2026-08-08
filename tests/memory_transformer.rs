use heirloom::checkpoint::{
    load_lm_checkpoint_on_device, load_memory_lm_checkpoint_on_device,
    save_memory_lm_checkpoint_with_dataset_state,
};
use heirloom::data::TokenDatasetState;
use heirloom::memory_transformer::{
    MemoryAccessCounts, MemoryLookupKind, MemoryTransformerConfig, MemoryTransformerLm,
    MemoryUpdatePolicy, SmftMode, SmftRowMask,
};
use heirloom::nn::{
    load_state_dict, save_state_dict, AdamW, Module, Optimizer, SparseAdamWCompactRowsUpdate,
    SparseAdamWRowsUpdate,
};
use heirloom::rng::HeirloomRng;
use heirloom::tokenizer::BpeTokenizer;
use heirloom::{Device, Result, Tensor};
use std::fs;
use std::path::PathBuf;

fn require_cuda_hardware() {
    assert_eq!(
        std::env::var("HEIRLOOM_CUDA_TESTS").ok().as_deref(),
        Some("1"),
        "ignored CUDA tests must be run through scripts/test_gpu.sh cuda"
    );
    assert!(
        heirloom_kernels::cuda::is_available(),
        "HEIRLOOM_CUDA_TESTS=1 but the CUDA Driver API is unavailable"
    );
}
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn heirloom_bin() -> &'static str {
    env!("CARGO_BIN_EXE_heirloom")
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "heirloom-memory-transformer-{name}-{}-{nanos}",
        std::process::id()
    ))
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

fn test_dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn expected_product_key_indices(
    query: &[f32],
    keys: &[f32],
    tokens: usize,
    slots: usize,
    key_dim: usize,
    top_k: usize,
    beam: usize,
) -> Vec<i64> {
    let side = (slots as f64).sqrt() as usize;
    let half_dim = key_dim / 2;
    let mut out = Vec::with_capacity(tokens * top_k);
    for token in 0..tokens {
        let query_row = &query[token * key_dim..(token + 1) * key_dim];
        let mut left_scores = Vec::with_capacity(side);
        let mut right_scores = Vec::with_capacity(side);
        for left in 0..side {
            let mut best = f32::NEG_INFINITY;
            for right in 0..side {
                let slot = left * side + right;
                let key_row = &keys[slot * key_dim..(slot + 1) * key_dim];
                best = best.max(test_dot(&query_row[..half_dim], &key_row[..half_dim]));
            }
            left_scores.push((left, best));
        }
        for right in 0..side {
            let mut best = f32::NEG_INFINITY;
            for left in 0..side {
                let slot = left * side + right;
                let key_row = &keys[slot * key_dim..(slot + 1) * key_dim];
                best = best.max(test_dot(&query_row[half_dim..], &key_row[half_dim..]));
            }
            right_scores.push((right, best));
        }
        left_scores.sort_by(|(li, ls), (ri, rs)| {
            rs.partial_cmp(ls)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| li.cmp(ri))
        });
        right_scores.sort_by(|(li, ls), (ri, rs)| {
            rs.partial_cmp(ls)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| li.cmp(ri))
        });
        let mut candidates = Vec::with_capacity(beam * beam);
        for &(left, _) in left_scores.iter().take(beam) {
            for &(right, _) in right_scores.iter().take(beam) {
                let slot = left * side + right;
                let key_row = &keys[slot * key_dim..(slot + 1) * key_dim];
                candidates.push((slot, test_dot(query_row, key_row)));
            }
        }
        candidates.sort_by(|(li, ls), (ri, rs)| {
            rs.partial_cmp(ls)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| li.cmp(ri))
        });
        candidates.dedup_by_key(|(slot, _)| *slot);
        out.extend(candidates.iter().take(top_k).map(|(slot, _)| *slot as i64));
    }
    out
}

fn expected_product_key_indices_from_side_scores(
    left_scores: &[f32],
    right_scores: &[f32],
    tokens: usize,
    side: usize,
    top_k: usize,
    beam: usize,
) -> Vec<i64> {
    let mut out = Vec::with_capacity(tokens * top_k);
    for token in 0..tokens {
        let mut left = (0..side)
            .map(|index| (index, left_scores[token * side + index]))
            .collect::<Vec<_>>();
        let mut right = (0..side)
            .map(|index| (index, right_scores[token * side + index]))
            .collect::<Vec<_>>();
        left.sort_by(|(li, ls), (ri, rs)| {
            rs.partial_cmp(ls)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| li.cmp(ri))
        });
        right.sort_by(|(li, ls), (ri, rs)| {
            rs.partial_cmp(ls)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| li.cmp(ri))
        });
        let mut candidates = Vec::with_capacity(beam * beam);
        for &(left_index, left_score) in left.iter().take(beam) {
            for &(right_index, right_score) in right.iter().take(beam) {
                candidates.push((left_index * side + right_index, left_score + right_score));
            }
        }
        candidates.sort_by(|(li, ls), (ri, rs)| {
            rs.partial_cmp(ls)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| li.cmp(ri))
        });
        out.extend(candidates.iter().take(top_k).map(|(slot, _)| *slot as i64));
    }
    out
}

fn small_config() -> MemoryTransformerConfig {
    MemoryTransformerConfig {
        vocab_size: 31,
        block_size: 8,
        n_layers: 2,
        d_model: 8,
        n_heads: 2,
        ff_hidden: 16,
        memory_layer_indices: vec![1],
        memory_slots: 12,
        memory_key_dim: 4,
        memory_value_dim: 8,
        memory_top_k: 3,
        memory_heads: 1,
        memory_lookup: MemoryLookupKind::Exact,
        shared_memory: true,
        memory_plus: true,
        memory_update_policy: MemoryUpdatePolicy::Full,
        smft_mode: SmftMode::Disabled,
    }
}

#[test]
fn constructs_32_layer_memory_transformer_with_stable_memory_names() -> Result<()> {
    let mut config = MemoryTransformerConfig::tiny(64);
    config.d_model = 8;
    config.n_heads = 2;
    config.ff_hidden = 16;
    config.memory_slots = 16;
    config.memory_key_dim = 4;
    config.memory_value_dim = 8;
    config.memory_top_k = 2;
    config.memory_layer_indices = vec![8, 16, 24];
    let mut rng = HeirloomRng::new(7);
    let model = MemoryTransformerLm::new(config, &mut rng)?;

    assert_eq!(model.blocks_len(), 32);
    let names = model
        .named_parameters("")
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"shared_memory.key".to_string()));
    assert!(names.contains(&"shared_memory.value".to_string()));
    assert!(names.contains(&"block.16.feed_forward.memory.query_proj.weight".to_string()));
    assert!(names.contains(&"block.0.feed_forward.dense.fc1.weight".to_string()));
    assert_eq!(
        names
            .iter()
            .filter(|name| *name == "shared_memory.key")
            .count(),
        1
    );
    Ok(())
}

#[test]
fn thirty_two_layer_memory_transformer_forward_shape_and_loss_are_finite() -> Result<()> {
    let mut config = MemoryTransformerConfig::tiny(32);
    config.block_size = 4;
    config.d_model = 4;
    config.n_heads = 2;
    config.ff_hidden = 8;
    config.memory_slots = 8;
    config.memory_key_dim = 2;
    config.memory_value_dim = 4;
    config.memory_top_k = 2;
    config.memory_heads = 1;
    config.memory_layer_indices = vec![8, 16, 24];

    let mut rng = HeirloomRng::new(23);
    let model = MemoryTransformerLm::new(config, &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4], &[1, 4], false)?;
    let targets = Tensor::from_i64(vec![2, 3, 4, 5], &[1, 4], false)?;

    let logits = model.forward(&input)?;
    assert_eq!(logits.shape(), vec![1, 4, 32]);
    let loss = model.loss(&input, &targets)?;
    assert!(loss.data()[0].is_finite());
    Ok(())
}

#[test]
fn memory_transformer_forward_shape_and_loss_are_finite() -> Result<()> {
    let mut rng = HeirloomRng::new(11);
    let model = MemoryTransformerLm::new(small_config(), &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 2, 3, 4, 5], &[2, 4], false)?;
    let targets = Tensor::from_i64(vec![2, 3, 4, 5, 3, 4, 5, 6], &[2, 4], false)?;

    let logits = model.forward(&input)?;
    assert_eq!(logits.shape(), vec![2, 4, 31]);
    let loss = model.loss(&input, &targets)?;
    assert!(loss.data()[0].is_finite());
    Ok(())
}

#[test]
fn product_key_lookup_validates_shape_requirements() {
    let mut non_square = small_config();
    non_square.memory_lookup = MemoryLookupKind::ProductKey;
    non_square.memory_slots = 12;
    let err = match MemoryTransformerLm::new(non_square, &mut HeirloomRng::new(19)) {
        Ok(_) => panic!("expected non-square product-key config to fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("product-key memory lookup requires square memory_slots"),
        "unexpected error: {err}"
    );

    let mut odd_key_dim = small_config();
    odd_key_dim.memory_lookup = MemoryLookupKind::ProductKey;
    odd_key_dim.memory_slots = 16;
    odd_key_dim.memory_key_dim = 3;
    let err = match MemoryTransformerLm::new(odd_key_dim, &mut HeirloomRng::new(19)) {
        Ok(_) => panic!("expected odd-key-dim product-key config to fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("product-key memory lookup requires even memory_key_dim"),
        "unexpected error: {err}"
    );
}

#[test]
fn product_key_cpu_lookup_runs_through_memory_forward_and_reports_rows() -> Result<()> {
    let mut config = small_config();
    config.memory_lookup = MemoryLookupKind::ProductKey;
    config.memory_slots = 16;
    config.memory_top_k = 2;
    let mut rng = HeirloomRng::new(23);
    let model = MemoryTransformerLm::new(config, &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 2, 3, 4, 5], &[2, 4], false)?;

    let logits = model.forward(&input)?;
    assert_eq!(logits.shape(), vec![2, 4, 31]);
    let report = model.memory_selection_report()?;
    assert_eq!(report.memory_lookup, MemoryLookupKind::ProductKey);
    assert_eq!(report.configured_memory_layers, 1);
    assert_eq!(report.captured_memory_layers, 1);
    assert_eq!(report.selected_row_events, 16);
    assert!(report.unique_selected_rows > 0);
    assert_eq!(report.selected_rows_device.as_deref(), Some("cpu"));
    Ok(())
}

#[test]
fn smft_access_counts_rank_foreground_rows_against_background() -> Result<()> {
    let foreground = MemoryAccessCounts::from_rows(6, vec![1, 1, 2, 4, 4, 4])?;
    let background = MemoryAccessCounts::from_rows(6, vec![1, 2, 2, 2, 3, 4])?;

    assert_eq!(foreground.total_events, 6);
    assert_eq!(foreground.unique_rows, 3);
    assert_eq!(foreground.row_counts, vec![0, 2, 1, 0, 3, 0]);
    foreground.validate()?;
    let from_counts = MemoryAccessCounts::from_row_counts(6, vec![0, 2, 1, 0, 3, 0])?;
    assert_eq!(from_counts, foreground);
    let mut merged = MemoryAccessCounts::empty(6);
    merged.merge_in(&foreground)?;
    merged.merge_in(&background)?;
    assert_eq!(merged.total_events, 12);
    assert_eq!(merged.row_counts, vec![0, 3, 4, 1, 4, 0]);
    let mut invalid = foreground.clone();
    invalid.total_events += 1;
    let err = invalid.validate().unwrap_err();
    assert!(
        err.to_string().contains("total_events"),
        "unexpected error: {err}"
    );

    let mask = foreground.smft_mask_against(&background, 0.5, 1)?;
    assert_eq!(mask.trainable_rows, vec![4, 1]);
    assert_eq!(mask.frozen_rows, 4);
    assert_eq!(mask.scores[0].row, 4);
    assert_eq!(mask.scores[0].foreground_count, 3);
    assert_eq!(mask.scores[0].background_count, 1);
    assert!(mask.scores[0].score > mask.scores[1].score);

    let mismatched_background = MemoryAccessCounts::empty(7);
    let err = foreground
        .smft_mask_against(&mismatched_background, 0.5, 1)
        .unwrap_err();
    assert!(
        err.to_string().contains("memory_slots mismatch"),
        "unexpected error: {err}"
    );
    Ok(())
}

#[test]
fn smft_access_report_uses_last_memory_forward_rows() -> Result<()> {
    let mut rng = HeirloomRng::new(27);
    let model = MemoryTransformerLm::new(small_config(), &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?;
    let _ = model.forward(&input)?;

    let report = model.smft_access_report(4)?;
    assert_eq!(report.counts.memory_slots, 12);
    assert_eq!(report.counts.total_events, 24);
    assert!(report.counts.unique_rows > 0);
    assert!(!report.top_rows.is_empty());
    assert!(report.top_rows.len() <= 4);
    assert!(report.top_rows[0].foreground_count > 0);
    Ok(())
}

#[test]
fn smft_row_mask_materializes_bool_tensor_and_masks_sparse_updates() -> Result<()> {
    let mask = SmftRowMask {
        memory_slots: 4,
        trainable_rows: vec![1, 3],
        frozen_rows: 2,
        trainable_fraction: 0.5,
        scores: Vec::new(),
    };
    let tensor = mask.to_tensor(Device::Cpu)?;
    assert_eq!(tensor.data_bool()?, vec![false, true, false, true]);

    let invalid = SmftRowMask {
        trainable_rows: vec![4],
        ..mask.clone()
    };
    let err = match invalid.to_tensor(Device::Cpu) {
        Ok(_) => panic!("expected invalid SMFT mask to fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("out of range"),
        "unexpected error: {err}"
    );
    let duplicate = SmftRowMask {
        trainable_rows: vec![1, 1],
        ..mask.clone()
    };
    let err = match duplicate.to_tensor(Device::Cpu) {
        Ok(_) => panic!("expected duplicate SMFT mask row to fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("appears more than once"),
        "unexpected error: {err}"
    );
    let stale_frozen_count = SmftRowMask {
        frozen_rows: 99,
        ..mask.clone()
    };
    let err = match stale_frozen_count.to_tensor(Device::Cpu) {
        Ok(_) => panic!("expected stale frozen-row count to fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("does not match"),
        "unexpected error: {err}"
    );

    let mut rng = HeirloomRng::new(31);
    let model = MemoryTransformerLm::new(small_config(), &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?;
    let _ = model.forward(&input)?;
    let smft_mask = model.smft_row_mask_against(&MemoryAccessCounts::empty(12), 0.5, 1)?;
    let updates = model.memory_sparse_adamw_updates_with_mask(&smft_mask)?;
    assert_eq!(updates.len(), 2);
    for update in updates {
        let row_mask = update.row_mask.expect("expected SMFT row mask");
        assert_eq!(row_mask.shape(), vec![12]);
        assert!(row_mask.data_bool()?.iter().any(|selected| *selected));
    }
    Ok(())
}

#[test]
fn product_key_smft_mask_projects_conservative_half_key_masks() -> Result<()> {
    let mut config = small_config();
    config.memory_lookup = MemoryLookupKind::ProductKey;
    config.memory_slots = 4;
    config.memory_top_k = 2;
    let mut rng = HeirloomRng::new(89);
    let model = MemoryTransformerLm::new(config, &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?;
    let _ = model.forward(&input)?;
    let mask = SmftRowMask {
        memory_slots: 4,
        trainable_rows: vec![0, 1],
        frozen_rows: 2,
        trainable_fraction: 0.5,
        scores: Vec::new(),
    };
    let projection = mask.product_key_projection()?;
    assert_eq!(projection.side, 2);
    assert_eq!(projection.left_trainable_rows, vec![0]);
    assert!(projection.right_trainable_rows.is_empty());

    let updates = model.memory_sparse_adamw_updates_with_mask(&mask)?;
    let names = model
        .named_parameters("")
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    assert_eq!(updates.len(), 3);
    assert_eq!(
        names[updates[0].parameter_index],
        "shared_memory.product_key_left"
    );
    assert_eq!(
        updates[0]
            .row_mask
            .as_ref()
            .expect("left row mask")
            .data_bool()?,
        vec![true, false]
    );
    assert_eq!(
        names[updates[1].parameter_index],
        "shared_memory.product_key_right"
    );
    assert_eq!(
        updates[1]
            .row_mask
            .as_ref()
            .expect("right row mask")
            .data_bool()?,
        vec![false, false]
    );
    assert_eq!(names[updates[2].parameter_index], "shared_memory.value");
    assert_eq!(
        updates[2]
            .row_mask
            .as_ref()
            .expect("value row mask")
            .data_bool()?,
        vec![true, true, false, false]
    );
    Ok(())
}

#[test]
fn memory_transformer_state_dict_round_trips() -> Result<()> {
    let dir = temp_dir("state-dict");
    let mut source_rng = HeirloomRng::new(13);
    let mut target_rng = HeirloomRng::new(29);
    let config = small_config();
    let source = MemoryTransformerLm::new(config.clone(), &mut source_rng)?;
    let target = MemoryTransformerLm::new(config, &mut target_rng)?;

    save_state_dict(&source, &dir)?;
    load_state_dict(&target, &dir)?;

    for ((source_name, source_param), (target_name, target_param)) in source
        .named_parameters("")
        .into_iter()
        .zip(target.named_parameters(""))
    {
        assert_eq!(source_name, target_name);
        assert_eq!(source_param.shape(), target_param.shape());
        assert_eq!(source_param.data(), target_param.data());
    }
    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn memory_lm_checkpoint_records_family_and_loads_memory_model() -> Result<()> {
    let dir = temp_dir("family-checkpoint");
    let tokenizer = BpeTokenizer::train("the cat sat. the dog ran.", 280)?;
    let mut rng = HeirloomRng::new(41);
    let model = MemoryTransformerLm::new(small_config(), &mut rng)?;
    let optimizer = AdamW::new(model.parameters(), 0.01)?;

    save_memory_lm_checkpoint_with_dataset_state(
        &dir,
        &model,
        &optimizer,
        &tokenizer,
        TokenDatasetState::from_seed(9),
        Some("fixture-manifest.json".to_string()),
    )?;

    let metadata_json = fs::read_to_string(dir.join("metadata.json")).unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&metadata_json).unwrap();
    assert_eq!(metadata["model_family"], "memory_transformer");
    assert_eq!(metadata["memory_config"]["memory_slots"], 12);
    assert_eq!(metadata["dataset_manifest_path"], "fixture-manifest.json");

    let loaded = load_memory_lm_checkpoint_on_device(&dir, Device::Cpu)?;
    assert_eq!(
        loaded.metadata.model_family,
        heirloom::checkpoint::LmModelFamily::MemoryTransformer
    );
    assert_eq!(loaded.model.config, model.config);
    let tiny_loader_error = match load_lm_checkpoint_on_device(&dir, Device::Cpu) {
        Ok(_) => panic!("tiny loader should reject memory-transformer checkpoints"),
        Err(err) => err,
    };
    assert!(
        tiny_loader_error
            .to_string()
            .contains("checkpoint model_family is MemoryTransformer"),
        "unexpected error: {tiny_loader_error}"
    );

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn memory_cuda_kernel_contract_validates_dims_and_counters() {
    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let dims = heirloom_kernels::cuda::MemoryLookupDims {
        tokens: 6,
        slots: 16,
        key_dim: 4,
        value_dim: 8,
        top_k: 3,
    };
    heirloom_kernels::cuda::validate_memory_lookup_dims(dims, "test_memory_lookup").unwrap();
    heirloom_kernels::cuda::record_memory_cuda_lookup_rejected(dims).unwrap();
    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.lookup_rejected_calls, 1);
    assert_eq!(counters.selected_tokens, 6);
    assert_eq!(counters.selected_rows, 18);
    assert_eq!(counters.query_key_score_calls, 0);
    assert_eq!(counters.weighted_value_forward_calls, 0);

    let invalid = heirloom_kernels::cuda::MemoryLookupDims { top_k: 17, ..dims };
    let err = heirloom_kernels::cuda::validate_memory_lookup_dims(invalid, "test_memory_lookup")
        .unwrap_err();
    assert!(
        err.to_string().contains("top_k <= slots"),
        "unexpected error: {err}"
    );
    let product_key_dims = heirloom_kernels::cuda::MemoryProductKeyDims {
        tokens: 2,
        slots: 16,
        key_dim: 4,
        top_k: 3,
        beam: 2,
    };
    heirloom_kernels::cuda::validate_memory_product_key_dims(product_key_dims, "test_product_key")
        .unwrap();
    let product_key_invalid = heirloom_kernels::cuda::MemoryProductKeyDims {
        slots: 12,
        ..product_key_dims
    };
    let err = heirloom_kernels::cuda::validate_memory_product_key_dims(
        product_key_invalid,
        "test_product_key",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("square slots"),
        "unexpected error: {err}"
    );

    let topk_dims = heirloom_kernels::cuda::MemoryTopkDims {
        rows: 2,
        cols: 4,
        top_k: 2,
    };
    heirloom_kernels::cuda::validate_memory_topk_dims(topk_dims, "test_memory_topk").unwrap();
    let topk_invalid = heirloom_kernels::cuda::MemoryTopkDims {
        top_k: 5,
        ..topk_dims
    };
    let err = heirloom_kernels::cuda::validate_memory_topk_dims(topk_invalid, "test_memory_topk")
        .unwrap_err();
    assert!(
        err.to_string().contains("top_k <= cols"),
        "unexpected error: {err}"
    );

    let weighted_dims = heirloom_kernels::cuda::MemoryWeightedValueDims {
        tokens: 2,
        top_k: 2,
        slots: 3,
        value_dim: 4,
    };
    heirloom_kernels::cuda::validate_memory_weighted_value_dims(
        weighted_dims,
        "test_memory_weighted_value",
    )
    .unwrap();
    let weighted_invalid = heirloom_kernels::cuda::MemoryWeightedValueDims {
        top_k: 4,
        ..weighted_dims
    };
    let err = heirloom_kernels::cuda::validate_memory_weighted_value_dims(
        weighted_invalid,
        "test_memory_weighted_value",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("top_k <= slots"),
        "unexpected error: {err}"
    );

    let selected_score_dims = heirloom_kernels::cuda::MemorySelectedScoreDims {
        tokens: 2,
        top_k: 2,
        slots: 3,
        key_dim: 4,
    };
    heirloom_kernels::cuda::validate_memory_selected_score_dims(
        selected_score_dims,
        "test_memory_selected_scores",
    )
    .unwrap();
    let selected_score_invalid = heirloom_kernels::cuda::MemorySelectedScoreDims {
        top_k: 4,
        ..selected_score_dims
    };
    let err = heirloom_kernels::cuda::validate_memory_selected_score_dims(
        selected_score_invalid,
        "test_memory_selected_scores",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("top_k <= slots"),
        "unexpected error: {err}"
    );

    let sparse_adamw_dims = heirloom_kernels::cuda::SparseAdamWRowsDims {
        selected_rows: 3,
        rows: 4,
        row_dim: 2,
    };
    heirloom_kernels::cuda::validate_sparse_adamw_rows_dims(
        sparse_adamw_dims,
        "test_sparse_adamw_rows",
    )
    .unwrap();
    let sparse_adamw_invalid = heirloom_kernels::cuda::SparseAdamWRowsDims {
        rows: 0,
        ..sparse_adamw_dims
    };
    let err = heirloom_kernels::cuda::validate_sparse_adamw_rows_dims(
        sparse_adamw_invalid,
        "test_sparse_adamw_rows",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("non-zero rows/row_dim"),
        "unexpected error: {err}"
    );
}

#[test]
fn topk_indices_dim1_is_deterministic_on_cpu() -> Result<()> {
    let scores = Tensor::from_f32(
        vec![1.0, 3.0, 3.0, -1.0, 0.5, 0.25, 0.75, 0.75],
        &[2, 4],
        false,
    )?;
    let indices = scores.topk_indices_dim1(2)?;
    assert_eq!(indices.data_i64()?, vec![1, 2, 2, 3]);
    Ok(())
}

#[test]
fn memory_weighted_value_cpu_forward_backward_accumulates_repeated_rows() -> Result<()> {
    let indices = Tensor::from_i64(vec![0, 2, 2, 1], &[2, 2], false)?;
    let weights = Tensor::from_f32(vec![0.25, 0.75, 1.0, -0.5], &[2, 2], true)?;
    let values = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true)?;

    let output = indices.memory_weighted_value(&weights, &values)?;
    assert_eq!(output.shape(), vec![2, 2]);
    assert_close_f64(&output.data_f64(), &[4.0, 5.0, 3.5, 4.0], 1e-6);

    output.sum()?.backward()?;
    assert_close_f32(&weights.grad().unwrap(), &[3.0, 11.0, 11.0, 7.0], 1e-6);
    assert_close_f32(
        &values.grad().unwrap(),
        &[0.25, 0.25, -0.5, -0.5, 1.75, 1.75],
        1e-6,
    );
    Ok(())
}

#[test]
fn memory_selected_scores_cpu_forward_backward_accumulates_repeated_rows() -> Result<()> {
    let indices = Tensor::from_i64(vec![0, 2, 2, 1], &[2, 2], false)?;
    let query = Tensor::from_f32(vec![1.0, 2.0, 3.0, -1.0], &[2, 2], true)?;
    let keys = Tensor::from_f32(vec![1.0, 0.0, 0.5, 2.0, -1.0, 1.0], &[3, 2], true)?;

    let scores = indices.memory_selected_scores(&query, &keys)?;
    assert_eq!(scores.shape(), vec![2, 2]);
    assert_close_f64(&scores.data_f64(), &[1.0, 1.0, -4.0, -0.5], 1e-6);

    scores.backward_with_grad(vec![1.0, 2.0, -1.0, 0.5])?;
    assert_close_f32(&query.grad().unwrap(), &[-1.0, 2.0, 1.25, 0.0], 1e-6);
    assert_close_f32(
        &keys.grad().unwrap(),
        &[1.0, 2.0, 1.5, -0.5, -1.0, 5.0],
        1e-6,
    );
    Ok(())
}

#[test]
fn product_key_memory_uses_trainable_half_key_tables() -> Result<()> {
    let mut config = small_config();
    config.memory_lookup = MemoryLookupKind::ProductKey;
    config.memory_slots = 16;
    config.memory_top_k = 2;
    let mut rng = HeirloomRng::new(73);
    let model = MemoryTransformerLm::new(config, &mut rng)?;
    let named = model.named_parameters("");
    let names = named
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    assert!(names.contains(&"shared_memory.product_key_left".to_string()));
    assert!(names.contains(&"shared_memory.product_key_right".to_string()));
    assert!(names.contains(&"shared_memory.value".to_string()));
    assert!(!names.contains(&"shared_memory.key".to_string()));

    let left = named
        .iter()
        .find(|(name, _)| name == "shared_memory.product_key_left")
        .expect("product-key left table")
        .1
        .clone();
    let right = named
        .iter()
        .find(|(name, _)| name == "shared_memory.product_key_right")
        .expect("product-key right table")
        .1
        .clone();
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?;
    let targets = Tensor::from_i64(vec![2, 3, 4, 5, 2, 3, 4, 5], &[2, 4], false)?;
    let loss = model.loss(&input, &targets)?;
    assert!(loss.data()[0].is_finite());
    loss.backward()?;
    assert!(
        left.grad().is_some(),
        "left half-key table should receive gradients"
    );
    assert!(
        right.grad().is_some(),
        "right half-key table should receive gradients"
    );

    let updates = model.memory_sparse_adamw_updates()?;
    assert_eq!(updates.len(), 3);
    assert_eq!(
        names[updates[0].parameter_index],
        "shared_memory.product_key_left"
    );
    assert_eq!(
        names[updates[1].parameter_index],
        "shared_memory.product_key_right"
    );
    assert_eq!(names[updates[2].parameter_index], "shared_memory.value");
    assert_eq!(updates[0].rows, 4);
    assert_eq!(updates[0].row_dim, 2);
    assert_eq!(updates[1].rows, 4);
    assert_eq!(updates[1].row_dim, 2);
    assert_eq!(updates[2].rows, 16);
    assert_eq!(updates[2].row_dim, 8);
    assert_eq!(updates[0].selected_rows.shape(), vec![16]);
    assert_eq!(updates[1].selected_rows.shape(), vec![16]);
    assert_eq!(updates[2].selected_rows.shape(), vec![16]);
    Ok(())
}

#[test]
fn memory_sparse_adamw_updates_map_shared_memory_parameters() -> Result<()> {
    let mut rng = HeirloomRng::new(31);
    let model = MemoryTransformerLm::new(small_config(), &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?;
    let _ = model.forward(&input)?;

    let updates = model.memory_sparse_adamw_updates()?;
    assert_eq!(updates.len(), 2);
    let names = model
        .named_parameters("")
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    assert_eq!(names[updates[0].parameter_index], "shared_memory.key");
    assert_eq!(names[updates[1].parameter_index], "shared_memory.value");
    assert_eq!(updates[0].selected_rows.shape(), vec![24]);
    assert_eq!(updates[1].selected_rows.shape(), vec![24]);
    assert_eq!(updates[0].selected_rows.data_i64()?.len(), 24);
    assert_eq!(updates[0].rows, 12);
    assert_eq!(updates[0].row_dim, 4);
    assert_eq!(updates[1].rows, 12);
    assert_eq!(updates[1].row_dim, 8);
    Ok(())
}

#[test]
fn memory_only_adamw_updates_only_memory_tables() -> Result<()> {
    let mut config = small_config();
    config.memory_update_policy = MemoryUpdatePolicy::MemoryOnly;
    let mut rng = HeirloomRng::new(37);
    let model = MemoryTransformerLm::new(config, &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?;
    let targets = Tensor::from_i64(vec![2, 3, 4, 5, 2, 3, 4, 5], &[2, 4], false)?;
    let named = model.named_parameters("");
    let token_embedding = named
        .iter()
        .find(|(name, _)| name == "token_embedding.weight")
        .expect("token embedding")
        .1
        .clone();
    let shared_memory_key = named
        .iter()
        .find(|(name, _)| name == "shared_memory.key")
        .expect("shared memory key")
        .1
        .clone();
    let token_before = token_embedding.data_f32()?;
    let key_before = shared_memory_key.data_f32()?;

    let mut optimizer = AdamW::new(model.parameters(), 0.01)?.with_weight_decay(0.0)?;
    let loss = model.loss(&input, &targets)?;
    loss.backward()?;
    assert!(
        token_embedding.grad().is_some(),
        "fixture should produce dense parameter gradients"
    );
    assert!(
        shared_memory_key.grad().is_some(),
        "fixture should produce memory table gradients"
    );
    optimizer.step_parameter_indices_mut(&model.memory_table_parameter_indices()?)?;

    assert_eq!(token_embedding.data_f32()?, token_before);
    assert_ne!(shared_memory_key.data_f32()?, key_before);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_softmax_dim1_backward_is_device_resident() -> Result<()> {
    require_cuda_hardware();

    let input = Tensor::from_f32(vec![1.0, 2.0, -1.0, 0.5, -0.5, 1.5], &[2, 3], true)?.cuda(0)?;
    let output = input.softmax_dim(1)?;
    assert_eq!(output.device(), Device::Cuda(0));
    output.backward_with_grad(vec![0.25, -0.5, 1.0, 1.5, -0.25, 0.75])?;
    let grad = input.grad_tensor().expect("softmax input gradient");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.shape(), vec![2, 3]);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_memory_access_count_rows_uses_device_kernel() -> Result<()> {
    require_cuda_hardware();

    let selected_rows =
        heirloom_kernels::cuda::CudaBuffer::from_i64(0, &[2, 0, 2, 1, 2]).expect("selected rows");
    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let counts = heirloom_kernels::cuda::memory_access_count_rows_i64_u64(&selected_rows, 4)
        .expect("count rows kernel")
        .to_u64()
        .expect("count rows copy");

    assert_eq!(counts, vec![1, 1, 3, 0]);
    assert_eq!(
        heirloom_kernels::cuda::memory_kernel_counters().access_count_calls,
        1
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_product_key_candidate_lookup_matches_cpu_reference() -> Result<()> {
    require_cuda_hardware();

    let tokens = 2;
    let slots = 9;
    let key_dim = 4;
    let top_k = 2;
    let beam = 2;
    let query = vec![1.0, 0.5, -0.25, 0.75, -0.5, 1.25, 0.5, -1.0];
    let keys = vec![
        0.2, 0.1, 0.4, -0.1, //
        1.0, 0.0, -0.2, 0.5, //
        -0.4, 0.6, 0.9, 0.1, //
        0.3, 1.0, -0.6, 0.8, //
        -0.7, 0.2, 0.5, 1.2, //
        0.9, -0.5, -0.1, -0.8, //
        -0.2, 0.4, 0.3, 0.7, //
        0.6, 0.6, -0.9, 0.2, //
        -1.0, 0.3, 1.1, -0.4,
    ];
    let expected = expected_product_key_indices(&query, &keys, tokens, slots, key_dim, top_k, beam);
    let query = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &query).expect("query buffer");
    let keys = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &keys).expect("keys buffer");

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let actual = heirloom_kernels::cuda::memory_product_key_topk_indices_f32(
        &query,
        &keys,
        heirloom_kernels::cuda::MemoryProductKeyDims {
            tokens,
            slots,
            key_dim,
            top_k,
            beam,
        },
    )
    .expect("product-key CUDA lookup")
    .to_i64()
    .expect("copy product-key indices");

    assert_eq!(actual, expected);
    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.product_key_calls, 1);
    assert_eq!(counters.selected_tokens, tokens);
    assert_eq!(counters.selected_rows, tokens * top_k);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_product_key_half_table_primitives_match_cpu_reference() -> Result<()> {
    require_cuda_hardware();

    let tokens = 2;
    let side = 3;
    let slots = side * side;
    let key_dim = 4;
    let top_k = 2;
    let beam = 2;
    let left_scores = vec![0.2, 1.0, -0.5, 0.3, -0.2, 0.8];
    let right_scores = vec![0.4, -0.1, 0.7, 1.2, 0.1, -0.4];
    let expected = expected_product_key_indices_from_side_scores(
        &left_scores,
        &right_scores,
        tokens,
        side,
        top_k,
        beam,
    );
    let left_scores_cuda =
        heirloom_kernels::cuda::CudaBuffer::from_f32(0, &left_scores).expect("left scores");
    let right_scores_cuda =
        heirloom_kernels::cuda::CudaBuffer::from_f32(0, &right_scores).expect("right scores");
    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let actual = heirloom_kernels::cuda::memory_product_key_topk_indices_from_side_scores_f32(
        &left_scores_cuda,
        &right_scores_cuda,
        heirloom_kernels::cuda::MemoryProductKeyDims {
            tokens,
            slots,
            key_dim,
            top_k,
            beam,
        },
    )
    .expect("product-key side-score CUDA lookup")
    .to_i64()
    .expect("copy product-key side-score indices");
    assert_eq!(actual, expected);

    let selected =
        heirloom_kernels::cuda::CudaBuffer::from_i64(0, &[0, 5, 8]).expect("selected rows");
    let (left_rows, right_rows) =
        heirloom_kernels::cuda::memory_product_key_split_rows_i64(&selected, side)
            .expect("split rows");
    assert_eq!(left_rows.to_i64().expect("left rows"), vec![0, 1, 2]);
    assert_eq!(right_rows.to_i64().expect("right rows"), vec![0, 2, 2]);

    let indices = heirloom_kernels::cuda::CudaBuffer::from_i64(0, &[0, 5, 8, 2]).expect("indices");
    let query = vec![
        1.0, 0.5, -0.25, 0.75, //
        -0.5, 1.25, 0.5, -1.0,
    ];
    let left_keys = vec![0.2, 0.1, 1.0, -0.5, -0.4, 0.6];
    let right_keys = vec![0.4, -0.1, -0.2, 0.5, 0.9, 0.1];
    let query_cuda = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &query).expect("query");
    let left_cuda = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &left_keys).expect("left keys");
    let right_cuda =
        heirloom_kernels::cuda::CudaBuffer::from_f32(0, &right_keys).expect("right keys");
    let dims = heirloom_kernels::cuda::MemoryProductKeySelectedScoreDims {
        tokens,
        top_k,
        side,
        key_dim,
    };
    let scores =
        heirloom_kernels::cuda::memory_product_key_selected_scores_forward_f32_i64_buffers(
            &indices,
            &query_cuda,
            &left_cuda,
            &right_cuda,
            dims,
        )
        .expect("product-key selected score forward")
        .to_f32()
        .expect("copy scores");
    assert_close_f32(&scores, &[0.075, 0.6, 1.3, 0.375], 1e-6);

    let grad_scores = heirloom_kernels::cuda::CudaBuffer::from_f32(0, &[1.0, 2.0, -1.0, 0.5])
        .expect("grad scores");
    let grad_query =
        heirloom_kernels::cuda::memory_product_key_selected_scores_backward_query_f32_i64_buffers(
            &indices,
            &left_cuda,
            &right_cuda,
            &grad_scores,
            dims,
        )
        .expect("product-key selected score backward query")
        .to_f32()
        .expect("copy grad query");
    assert_close_f32(
        &grad_query,
        &[2.2, -0.9, 2.2, 0.1, 0.5, -0.55, -0.45, -0.05],
        1e-6,
    );
    let grad_left =
        heirloom_kernels::cuda::memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers(
            &indices,
            &query_cuda,
            &grad_scores,
            dims,
            true,
        )
        .expect("product-key selected score backward left")
        .to_f32()
        .expect("copy grad left");
    assert_close_f32(&grad_left, &[0.75, 1.125, 2.0, 1.0, 0.5, -1.25], 1e-6);
    let grad_right =
        heirloom_kernels::cuda::memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers(
            &indices,
            &query_cuda,
            &grad_scores,
            dims,
            false,
        )
        .expect("product-key selected score backward right")
        .to_f32()
        .expect("copy grad right");
    assert_close_f32(&grad_right, &[-0.25, 0.75, 0.0, 0.0, -0.75, 2.0], 1e-6);
    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.product_key_selected_score_forward_calls, 1);
    assert_eq!(counters.product_key_backward_query_calls, 1);
    assert_eq!(counters.product_key_backward_half_key_calls, 2);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_memory_selected_scores_forward_backward_uses_fused_kernels() -> Result<()> {
    require_cuda_hardware();

    let indices = Tensor::from_i64(vec![0, 2, 2, 1], &[2, 2], false)?.cuda(0)?;
    let query = Tensor::from_f32(vec![1.0, 2.0, 3.0, -1.0], &[2, 2], true)?.cuda(0)?;
    let keys = Tensor::from_f32(vec![1.0, 0.0, 0.5, 2.0, -1.0, 1.0], &[3, 2], true)?.cuda(0)?;

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let scores = indices.memory_selected_scores(&query, &keys)?;
    assert_eq!(scores.device(), Device::Cuda(0));
    assert_eq!(scores.shape(), vec![2, 2]);
    assert_close_f64(&scores.cpu()?.data_f64(), &[1.0, 1.0, -4.0, -0.5], 1e-4);
    scores.backward_with_grad(vec![1.0, 2.0, -1.0, 0.5])?;

    let query_grad = query.grad_tensor().expect("memory query gradient");
    let key_grad = keys.grad_tensor().expect("memory key gradient");
    assert_eq!(query_grad.device(), Device::Cuda(0));
    assert_eq!(key_grad.device(), Device::Cuda(0));
    assert_close_f64(&query_grad.cpu()?.data_f64(), &[-1.0, 2.0, 1.25, 0.0], 1e-4);
    assert_close_f64(
        &key_grad.cpu()?.data_f64(),
        &[1.0, 2.0, 1.5, -0.5, -1.0, 5.0],
        1e-4,
    );

    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.query_key_score_calls, 1);
    assert_eq!(counters.selected_key_backward_calls, 1);
    assert_eq!(counters.scatter_add_rows_calls, 1);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_memory_weighted_value_forward_backward_uses_fused_kernels() -> Result<()> {
    require_cuda_hardware();

    let indices = Tensor::from_i64(vec![0, 2, 2, 1], &[2, 2], false)?.cuda(0)?;
    let weights = Tensor::from_f32(vec![0.25, 0.75, 1.0, -0.5], &[2, 2], true)?.cuda(0)?;
    let values = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true)?.cuda(0)?;

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let output = indices.memory_weighted_value(&weights, &values)?;
    assert_eq!(output.device(), Device::Cuda(0));
    assert_eq!(output.shape(), vec![2, 2]);
    output.backward_with_grad(vec![1.0, 1.0, 1.0, 1.0])?;

    let weight_grad = weights.grad_tensor().expect("memory weights gradient");
    let value_grad = values.grad_tensor().expect("memory values gradient");
    assert_eq!(weight_grad.device(), Device::Cuda(0));
    assert_eq!(value_grad.device(), Device::Cuda(0));
    assert_close_f64(
        &weight_grad.cpu()?.data_f64(),
        &[3.0, 11.0, 11.0, 7.0],
        1e-4,
    );
    assert_close_f64(
        &value_grad.cpu()?.data_f64(),
        &[0.25, 0.25, -0.5, -0.5, 1.75, 1.75],
        1e-4,
    );

    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.weighted_value_forward_calls, 1);
    assert_eq!(counters.weighted_value_backward_calls, 1);
    assert_eq!(counters.scatter_add_rows_calls, 1);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_memory_gather_selected_rows_and_gradients_stays_device_resident() -> Result<()> {
    require_cuda_hardware();

    let table = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true)?.cuda(0)?;
    let selected_rows = Tensor::from_i64(vec![2, 0, 2], &[3], false)?.cuda(0)?;

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let gathered = table.cuda_memory_gather_selected_rows_f32_i64(&selected_rows)?;
    assert_eq!(gathered.device(), Device::Cuda(0));
    assert_eq!(gathered.shape(), vec![3, 2]);
    assert_close_f64(
        &gathered.cpu()?.data_f64(),
        &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0],
        1e-6,
    );

    table.backward_with_grad(vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.25])?;
    let gathered_grad = table
        .cuda_memory_gather_selected_grad_rows_f32_i64(&selected_rows)?
        .expect("selected gradient rows");
    assert_eq!(gathered_grad.device(), Device::Cuda(0));
    assert_eq!(gathered_grad.shape(), vec![3, 2]);
    assert_close_f64(
        &gathered_grad.cpu()?.data_f64(),
        &[0.5, -0.25, 1.0, -2.0, 0.5, -0.25],
        1e-6,
    );
    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.gather_selected_rows_calls, 2);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_bool_mask_to_i64_indices_compacts_rows_on_device() -> Result<()> {
    require_cuda_hardware();

    let mask = Tensor::from_bool(vec![false, true, true, false, true], &[5], false)?.cuda(0)?;
    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let rows = mask.cuda_bool_mask_to_i64_indices()?;
    assert_eq!(rows.device(), Device::Cuda(0));
    assert_eq!(rows.shape(), vec![3]);
    assert_eq!(rows.cpu()?.data_i64()?, vec![1, 2, 4]);
    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.bool_mask_to_indices_calls, 1);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_sparse_adamw_rows_updates_only_selected_unique_rows() -> Result<()> {
    require_cuda_hardware();

    let param = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true)?.cuda(0)?;
    param.backward_with_grad(vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.25])?;
    let selected_rows = Tensor::from_i64(vec![2, 0, 2], &[3], false)?.cuda(0)?;
    let mut optimizer = AdamW::new(vec![param.clone()], 0.1)?.with_weight_decay(0.0)?;

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    optimizer.step_cuda_sparse_rows_mut(&[SparseAdamWRowsUpdate {
        parameter_index: 0,
        selected_rows,
        row_mask: None,
        rows: 3,
        row_dim: 2,
    }])?;

    assert_close_f64(
        &param.cpu()?.data_f64(),
        &[0.9, 2.1, 3.0, 4.0, 4.9, 6.1],
        1e-4,
    );
    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.sparse_adamw_rows_calls, 1);
    assert_eq!(counters.sparse_adamw_compact_rows_calls, 1);
    assert_eq!(counters.gather_selected_rows_calls, 1);
    Ok(())
}

#[test]
fn cpu_sparse_adamw_rows_updates_only_selected_unique_rows() -> Result<()> {
    let param = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true)?;
    param.backward_with_grad(vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.25])?;
    let selected_rows = Tensor::from_i64(vec![2, 0, 2], &[3], false)?;
    let mut optimizer = AdamW::new(vec![param.clone()], 0.1)?.with_weight_decay(0.0)?;

    optimizer.step_sparse_rows_mut(&[SparseAdamWRowsUpdate {
        parameter_index: 0,
        selected_rows,
        row_mask: None,
        rows: 3,
        row_dim: 2,
    }])?;

    assert_close_f64(&param.data_f64(), &[0.9, 2.1, 3.0, 4.0, 4.9, 6.1], 1e-4);
    Ok(())
}

#[test]
fn compact_sparse_adamw_rejects_mismatched_compact_gradient_shape() -> Result<()> {
    let param = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true)?;
    let selected_rows = Tensor::from_i64(vec![0, 2], &[2], false)?;
    let compact_grad_rows = Tensor::from_f32(vec![1.0, -1.0], &[1, 2], false)?;
    let mut optimizer = AdamW::new(vec![param], 0.1)?.with_weight_decay(0.0)?;

    let err = optimizer
        .step_cuda_sparse_compact_rows_mut(&[SparseAdamWCompactRowsUpdate {
            parameter_index: 0,
            selected_rows,
            compact_grad_rows,
            rows: 3,
            row_dim: 2,
        }])
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("compact_grad_rows must have shape"),
        "{err}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_sparse_adamw_rows_respects_smft_row_mask() -> Result<()> {
    require_cuda_hardware();

    let param = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true)?.cuda(0)?;
    param.backward_with_grad(vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.25])?;
    let selected_rows = Tensor::from_i64(vec![2, 0, 1, 2], &[4], false)?.cuda(0)?;
    let row_mask = Tensor::from_bool(vec![false, true, true], &[3], false)?.cuda(0)?;
    let mut optimizer = AdamW::new(vec![param.clone()], 0.1)?.with_weight_decay(0.0)?;

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    optimizer.step_cuda_sparse_rows_mut(&[SparseAdamWRowsUpdate {
        parameter_index: 0,
        selected_rows,
        row_mask: Some(row_mask),
        rows: 3,
        row_dim: 2,
    }])?;

    assert_close_f64(
        &param.cpu()?.data_f64(),
        &[1.0, 2.0, 2.9, 4.1, 4.9, 6.1],
        1e-4,
    );
    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.sparse_adamw_rows_calls, 1);
    assert_eq!(counters.sparse_adamw_compact_rows_calls, 1);
    assert_eq!(counters.gather_selected_rows_calls, 1);
    Ok(())
}

#[test]
fn cpu_sparse_adamw_rows_respects_smft_row_mask() -> Result<()> {
    let param = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true)?;
    param.backward_with_grad(vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.25])?;
    let selected_rows = Tensor::from_i64(vec![2, 0, 1, 2], &[4], false)?;
    let row_mask = Tensor::from_bool(vec![false, true, true], &[3], false)?;
    let mut optimizer = AdamW::new(vec![param.clone()], 0.1)?.with_weight_decay(0.0)?;

    optimizer.step_sparse_rows_mut(&[SparseAdamWRowsUpdate {
        parameter_index: 0,
        selected_rows,
        row_mask: Some(row_mask),
        rows: 3,
        row_dim: 2,
    }])?;

    assert_close_f64(&param.data_f64(), &[1.0, 2.0, 2.9, 4.1, 4.9, 6.1], 1e-4);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_memory_transformer_sparse_row_adamw_updates_memory_tables() -> Result<()> {
    require_cuda_hardware();

    let mut config = small_config();
    config.memory_update_policy = MemoryUpdatePolicy::SparseRows;
    let mut rng = HeirloomRng::new(53);
    let model = MemoryTransformerLm::new(config, &mut rng)?.to_device(Device::Cuda(0))?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?.cuda(0)?;
    let targets = Tensor::from_i64(vec![2, 3, 4, 5, 2, 3, 4, 5], &[2, 4], false)?.cuda(0)?;
    let mut optimizer = AdamW::new(model.parameters(), 0.01)?;

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let loss = model.loss(&input, &targets)?;
    loss.backward()?;
    let updates = model.memory_sparse_adamw_updates()?;
    assert_eq!(updates.len(), 2);
    assert_eq!(updates[0].selected_rows.device(), Device::Cuda(0));
    optimizer.step_cuda_sparse_rows_mut(&updates)?;

    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.sparse_adamw_rows_calls, 2);
    assert_eq!(counters.sparse_adamw_compact_rows_calls, 2);
    assert_eq!(counters.gather_selected_rows_calls, 2);
    assert!(counters.selected_key_backward_calls >= 1);
    assert!(counters.scatter_add_rows_calls >= 2);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_memory_transformer_step_uses_device_topk_and_memory_grads() -> Result<()> {
    require_cuda_hardware();

    let mut rng = HeirloomRng::new(47);
    let model = MemoryTransformerLm::new(small_config(), &mut rng)?.to_device(Device::Cuda(0))?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?.cuda(0)?;
    let targets = Tensor::from_i64(vec![2, 3, 4, 5, 2, 3, 4, 5], &[2, 4], false)?.cuda(0)?;
    let memory_value = model
        .named_parameters("")
        .into_iter()
        .find(|(name, _)| name == "shared_memory.value")
        .expect("shared memory value parameter")
        .1;
    let memory_key = model
        .named_parameters("")
        .into_iter()
        .find(|(name, _)| name == "shared_memory.key")
        .expect("shared memory key parameter")
        .1;
    let mut optimizer = AdamW::new(model.parameters(), 0.01)?;

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let logits = model.forward(&input)?;
    assert_eq!(logits.device(), Device::Cuda(0));
    assert_eq!(logits.shape(), vec![2, 4, 31]);
    let loss = model.loss(&input, &targets)?;
    assert!(loss.data_f32()?[0].is_finite());
    loss.backward()?;
    let memory_grad = memory_value
        .grad_tensor()
        .expect("memory value should receive selected-row gradients");
    assert_eq!(memory_grad.device(), Device::Cuda(0));
    let memory_key_grad = memory_key
        .grad_tensor()
        .expect("memory key should receive selected-row gradients");
    assert_eq!(memory_key_grad.device(), Device::Cuda(0));
    optimizer.step_mut()?;

    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert_eq!(counters.lookup_rejected_calls, 0);
    assert!(counters.query_key_score_calls >= 2, "counters={counters:?}");
    assert!(counters.topk_calls >= 2, "counters={counters:?}");
    assert!(
        counters.weighted_value_forward_calls >= 2,
        "counters={counters:?}"
    );
    assert!(
        counters.weighted_value_backward_calls >= 1,
        "counters={counters:?}"
    );
    assert!(
        counters.scatter_add_rows_calls >= 1,
        "counters={counters:?}"
    );
    assert!(
        counters.selected_key_backward_calls >= 1,
        "counters={counters:?}"
    );
    assert!(counters.selected_tokens >= 16, "counters={counters:?}");
    assert!(counters.selected_rows >= 48, "counters={counters:?}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_memory_transformer_product_key_lookup_runs_on_device() -> Result<()> {
    require_cuda_hardware();

    let mut config = small_config();
    config.memory_lookup = MemoryLookupKind::ProductKey;
    config.memory_slots = 16;
    config.memory_top_k = 2;
    let mut rng = HeirloomRng::new(71);
    let model = MemoryTransformerLm::new(config, &mut rng)?.to_device(Device::Cuda(0))?;
    let named = model.named_parameters("");
    let left = named
        .iter()
        .find(|(name, _)| name == "shared_memory.product_key_left")
        .expect("product-key left table")
        .1
        .clone();
    let right = named
        .iter()
        .find(|(name, _)| name == "shared_memory.product_key_right")
        .expect("product-key right table")
        .1
        .clone();
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?.cuda(0)?;
    let targets = Tensor::from_i64(vec![2, 3, 4, 5, 2, 3, 4, 5], &[2, 4], false)?.cuda(0)?;

    heirloom_kernels::cuda::reset_memory_kernel_counters();
    let logits = model.forward(&input)?;
    assert_eq!(logits.device(), Device::Cuda(0));
    assert_eq!(logits.shape(), vec![2, 4, 31]);
    let loss = model.loss(&input, &targets)?;
    assert!(loss.cpu()?.data_f32()?[0].is_finite());
    loss.backward()?;
    let report = model.memory_selection_report()?;
    assert_eq!(report.memory_lookup, MemoryLookupKind::ProductKey);
    assert_eq!(report.selected_rows_device.as_deref(), Some("cuda:0"));
    assert_eq!(report.selected_row_events, 16);
    assert_eq!(
        left.grad_tensor().expect("left half-key grad").device(),
        Device::Cuda(0)
    );
    assert_eq!(
        right.grad_tensor().expect("right half-key grad").device(),
        Device::Cuda(0)
    );
    let counters = heirloom_kernels::cuda::memory_kernel_counters();
    assert!(counters.product_key_calls >= 2, "counters={counters:?}");
    assert!(
        counters.weighted_value_forward_calls >= 2,
        "counters={counters:?}"
    );
    assert!(
        counters.product_key_selected_score_forward_calls >= 2,
        "counters={counters:?}"
    );
    assert!(
        counters.product_key_backward_query_calls >= 1,
        "counters={counters:?}"
    );
    assert!(
        counters.product_key_backward_half_key_calls >= 2,
        "counters={counters:?}"
    );
    Ok(())
}

#[test]
fn memory_transformer_fixed_batch_training_decreases_loss() -> Result<()> {
    let mut rng = HeirloomRng::new(17);
    let model = MemoryTransformerLm::new(small_config(), &mut rng)?;
    let input = Tensor::from_i64(vec![1, 2, 3, 4, 1, 2, 3, 4], &[2, 4], false)?;
    let targets = Tensor::from_i64(vec![2, 3, 4, 5, 2, 3, 4, 5], &[2, 4], false)?;
    let mut optimizer = AdamW::new(model.parameters(), 0.02)?.with_weight_decay(0.0)?;

    let initial = model.loss(&input, &targets)?.data()[0];
    for _ in 0..30 {
        optimizer.zero_grad();
        let loss = model.loss(&input, &targets)?;
        loss.backward()?;
        optimizer.step_mut()?;
    }
    let final_loss = model.loss(&input, &targets)?.data()[0];
    assert!(
        final_loss < initial,
        "expected training to reduce fixed-batch loss, initial={initial} final={final_loss}"
    );
    Ok(())
}

#[test]
fn train_memory_lm_cli_writes_family_checkpoint_and_report() -> Result<()> {
    let dir = temp_dir("cli");
    fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("tiny.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let checkpoint = dir.join("checkpoint");
    let report = dir.join("report.json");
    let text = "the cat sat. the dog ran. the cat ran. the dog sat. ";
    fs::write(&data_path, text.repeat(8)).unwrap();
    let tokenizer = BpeTokenizer::train(&fs::read_to_string(&data_path).unwrap(), 280)?;
    tokenizer.save(&tokenizer_path)?;

    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--data",
            data_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "4",
            "--batch-size",
            "2",
            "--block-size",
            "4",
            "--n-layers",
            "2",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--memory-layer-indices",
            "1",
            "--memory-slots",
            "8",
            "--memory-key-dim",
            "4",
            "--memory-value-dim",
            "8",
            "--memory-top-k",
            "2",
            "--lr",
            "0.01",
            "--log-every",
            "2",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom train-memory-lm");

    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(report_json["command"], "train-memory-lm");
    assert_eq!(report_json["model_family"], "memory_transformer");
    assert_eq!(report_json["memory_config"]["n_layers"], 2);
    assert_eq!(report_json["memory_config"]["memory_layer_indices"][0], 1);
    assert_eq!(
        report_json["memory_selection"]["configured_memory_layers"],
        1
    );
    assert_eq!(report_json["memory_selection"]["captured_memory_layers"], 1);
    assert!(
        report_json["memory_selection"]["selected_row_events"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        report_json["memory_optimizer"]["applied_path"],
        "dense_all_parameters"
    );
    assert_eq!(
        report_json["memory_optimizer"]["memory_table_parameter_count"],
        2
    );
    assert!(report_json["memory_table_parameter_checksum_sum"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(
        report_json["smft_access"]["counts"]["total_events"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(report_json["smft_access"]["top_rows"]
        .as_array()
        .unwrap()
        .iter()
        .all(|row| row["foreground_count"].as_u64().unwrap() > 0));
    assert_eq!(report_json["cuda_memory_kernels"]["implemented"], true);
    assert_eq!(
        report_json["cuda_memory_kernels"]["counters"]["lookup_rejected_calls"],
        0
    );
    assert_eq!(
        report_json["cuda_memory_kernels"]["counters"]["weighted_value_forward_calls"],
        0
    );
    assert!(report_json["final_loss"].as_f64().unwrap().is_finite());

    let metadata_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(checkpoint.join("metadata.json")).unwrap())
            .unwrap();
    assert_eq!(metadata_json["model_family"], "memory_transformer");
    assert_eq!(metadata_json["memory_config"]["memory_top_k"], 2);

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn train_memory_lm_cli_uses_binary_shard_manifest_streaming_loader() -> Result<()> {
    let dir = temp_dir("cli-v2-manifest");
    fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("tiny.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let prepared_dir = dir.join("prepared");
    let checkpoint = dir.join("checkpoint");
    let report = dir.join("report.json");
    let eval_report = dir.join("eval.json");
    let generation_report = dir.join("generation.json");
    let text = "TRACE qb STATE route memory DELTA sparse rows ACTION read EVIDENCE manifest. ";
    fs::write(&data_path, text.repeat(12)).unwrap();
    let tokenizer = BpeTokenizer::train(&fs::read_to_string(&data_path).unwrap(), 280)?;
    tokenizer.save(&tokenizer_path)?;

    let prepare = Command::new(heirloom_bin())
        .args([
            "data",
            "prepare",
            "--input",
            data_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--out-dir",
            prepared_dir.to_str().unwrap(),
            "--format",
            "binary-shard",
            "--valid-fraction",
            "0.1",
            "--shard-tokens",
            "32",
        ])
        .output()
        .expect("failed to run heirloom data prepare");
    assert!(
        prepare.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&prepare.stdout),
        String::from_utf8_lossy(&prepare.stderr)
    );

    let manifest = prepared_dir.join("manifest.json");
    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--dataset-manifest",
            manifest.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--block-size",
            "4",
            "--n-layers",
            "2",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--memory-layer-indices",
            "1",
            "--memory-slots",
            "8",
            "--memory-key-dim",
            "4",
            "--memory-value-dim",
            "8",
            "--memory-top-k",
            "2",
            "--lr",
            "0.01",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom train-memory-lm");

    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(report_json["command"], "train-memory-lm");
    assert_eq!(report_json["model_family"], "memory_transformer");
    assert_eq!(report_json["loader"]["kind"], "binary_shard_streaming");
    assert_eq!(report_json["loader"]["tokens_materialized"], false);
    assert!(report_json["performance"]["tokens_seen"].as_u64().unwrap() > 0);
    assert!(
        report_json["memory_selection"]["selected_row_events"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        report_json["smft_access"]["counts"]["total_events"]
            .as_u64()
            .unwrap()
            > 0
    );

    let eval = Command::new(heirloom_bin())
        .args([
            "eval-memory-lm",
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--dataset-manifest",
            manifest.to_str().unwrap(),
            "--split",
            "valid",
            "--batch-size",
            "1",
            "--max-batches",
            "1",
            "--report",
            eval_report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom eval-memory-lm");
    assert!(
        eval.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&eval.stdout),
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&eval_report).unwrap()).unwrap();
    assert_eq!(eval_json["command"], "eval-memory-lm");
    assert_eq!(eval_json["model_family"], "memory_transformer");
    assert_eq!(eval_json["loader"]["kind"], "binary_shard_streaming");
    assert_eq!(eval_json["loader"]["tokens_materialized"], false);
    assert!(eval_json["metrics"]["tokens"].as_u64().unwrap() > 0);

    let generation = Command::new(heirloom_bin())
        .args([
            "generate-memory-lm",
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--prompt",
            "TRACE qb",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0.0",
            "--report",
            generation_report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom generate-memory-lm");
    assert!(
        generation.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&generation.stdout),
        String::from_utf8_lossy(&generation.stderr)
    );
    let generation_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&generation_report).unwrap()).unwrap();
    assert_eq!(generation_json["command"], "generate-memory-lm");
    assert_eq!(generation_json["model_family"], "memory_transformer");
    assert!(generation_json["generated_tokens"].as_u64().unwrap() <= 2);
    assert!(generation_json["total_tokens"].as_u64().unwrap() >= 1);

    let metadata_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(checkpoint.join("metadata.json")).unwrap())
            .unwrap();
    assert_eq!(metadata_json["model_family"], "memory_transformer");
    assert_eq!(
        metadata_json["dataset_manifest_path"],
        manifest.display().to_string()
    );

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn train_memory_lm_cli_rejects_smft_mask_without_sparse_mode() -> Result<()> {
    let dir = temp_dir("cli-smft-mask-reject");
    fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("tiny.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let mask_path = dir.join("mask.json");
    let checkpoint = dir.join("checkpoint");
    let text = "the cat sat. the dog ran. ";
    fs::write(&data_path, text.repeat(4)).unwrap();
    let tokenizer = BpeTokenizer::train(&fs::read_to_string(&data_path).unwrap(), 280)?;
    tokenizer.save(&tokenizer_path)?;
    let mask = SmftRowMask {
        memory_slots: 8,
        trainable_rows: vec![1, 3],
        frozen_rows: 6,
        trainable_fraction: 0.25,
        scores: Vec::new(),
    };
    fs::write(&mask_path, serde_json::to_string_pretty(&mask).unwrap()).unwrap();

    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--data",
            data_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "4",
            "--n-layers",
            "2",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--memory-layer-indices",
            "1",
            "--memory-slots",
            "8",
            "--memory-key-dim",
            "4",
            "--memory-value-dim",
            "8",
            "--memory-top-k",
            "2",
            "--smft-row-mask",
            mask_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom train-memory-lm");

    assert!(!output.status.success(), "expected command to reject mask");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--smft-row-mask was supplied"),
        "unexpected stderr: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn train_memory_lm_cli_rejects_masked_smft_without_initial_mask() -> Result<()> {
    let dir = temp_dir("cli-masked-smft-no-mask-reject");
    fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("tiny.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let checkpoint = dir.join("checkpoint");
    let text = "the cat sat. the dog ran. ";
    fs::write(&data_path, text.repeat(4)).unwrap();
    let tokenizer = BpeTokenizer::train(&fs::read_to_string(&data_path).unwrap(), 280)?;
    tokenizer.save(&tokenizer_path)?;

    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--data",
            data_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "4",
            "--n-layers",
            "2",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--memory-layer-indices",
            "1",
            "--memory-slots",
            "8",
            "--memory-key-dim",
            "4",
            "--memory-value-dim",
            "8",
            "--memory-top-k",
            "2",
            "--smft-mode",
            "masked-memory-rows",
        ])
        .output()
        .expect("failed to run heirloom train-memory-lm");

    assert!(
        !output.status.success(),
        "expected command to reject missing initial SMFT mask"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("single-rank masked-memory-rows SMFT requires --smft-row-mask"),
        "unexpected stderr: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn train_memory_lm_frozen_policy_reports_training_step_not_optimizer_step() -> Result<()> {
    let dir = temp_dir("cli-memory-frozen-step");
    fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("tiny.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let checkpoint = dir.join("checkpoint");
    let report = dir.join("report.json");
    let text = "the cat sat. the dog ran. ";
    fs::write(&data_path, text.repeat(6)).unwrap();
    let tokenizer = BpeTokenizer::train(&fs::read_to_string(&data_path).unwrap(), 280)?;
    tokenizer.save(&tokenizer_path)?;

    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--data",
            data_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--block-size",
            "4",
            "--n-layers",
            "2",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--memory-layer-indices",
            "1",
            "--memory-slots",
            "8",
            "--memory-key-dim",
            "4",
            "--memory-value-dim",
            "8",
            "--memory-top-k",
            "2",
            "--memory-update-policy",
            "frozen",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom train-memory-lm");

    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(report_json["start_step"], 0);
    assert_eq!(report_json["final_step"], 2);
    assert_eq!(
        report_json["memory_optimizer"]["applied_path"],
        "frozen_no_optimizer_step"
    );

    let metadata_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(checkpoint.join("metadata.json")).unwrap())
            .unwrap();
    assert_eq!(metadata_json["step"], 2);

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn train_memory_lm_cli_rejects_smft_refresh_without_sparse_mode() -> Result<()> {
    let dir = temp_dir("cli-smft-refresh-reject");
    fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("tiny.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let checkpoint = dir.join("checkpoint");
    let text = "the cat sat. the dog ran. ";
    fs::write(&data_path, text.repeat(4)).unwrap();
    let tokenizer = BpeTokenizer::train(&fs::read_to_string(&data_path).unwrap(), 280)?;
    tokenizer.save(&tokenizer_path)?;

    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--data",
            data_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "4",
            "--n-layers",
            "2",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--memory-layer-indices",
            "1",
            "--memory-slots",
            "8",
            "--memory-key-dim",
            "4",
            "--memory-value-dim",
            "8",
            "--memory-top-k",
            "2",
            "--smft-refresh-every",
            "1",
        ])
        .output()
        .expect("failed to run heirloom train-memory-lm");

    assert!(
        !output.status.success(),
        "expected command to reject refresh"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--smft-refresh-every was supplied"),
        "unexpected stderr: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn train_memory_lm_cli_writes_smft_counts_and_mask_artifacts() -> Result<()> {
    let dir = temp_dir("cli-smft-artifacts");
    fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("tiny.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let background_path = dir.join("background-counts.json");
    let counts_path = dir.join("access-counts.json");
    let mask_path = dir.join("derived-mask.json");
    let checkpoint = dir.join("checkpoint");
    let report = dir.join("report.json");
    let text = "the cat sat. the dog ran. the cat ran. the dog sat. ";
    fs::write(&data_path, text.repeat(8)).unwrap();
    let tokenizer = BpeTokenizer::train(&fs::read_to_string(&data_path).unwrap(), 280)?;
    tokenizer.save(&tokenizer_path)?;
    let background = MemoryAccessCounts::empty(8);
    fs::write(
        &background_path,
        serde_json::to_string_pretty(&background).unwrap(),
    )
    .unwrap();

    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--data",
            data_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "3",
            "--batch-size",
            "2",
            "--block-size",
            "4",
            "--n-layers",
            "2",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--memory-layer-indices",
            "1",
            "--memory-slots",
            "8",
            "--memory-key-dim",
            "4",
            "--memory-value-dim",
            "8",
            "--memory-top-k",
            "2",
            "--lr",
            "0.01",
            "--smft-background-counts",
            background_path.to_str().unwrap(),
            "--smft-access-counts-out",
            counts_path.to_str().unwrap(),
            "--smft-mask-out",
            mask_path.to_str().unwrap(),
            "--smft-trainable-fraction",
            "0.5",
            "--smft-min-rows",
            "1",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom train-memory-lm");

    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let counts: MemoryAccessCounts =
        serde_json::from_str(&fs::read_to_string(&counts_path).unwrap()).unwrap();
    counts.validate()?;
    assert_eq!(counts.memory_slots, 8);
    assert!(counts.total_events > 0);
    assert!(counts.unique_rows > 0);

    let mask: SmftRowMask = serde_json::from_str(&fs::read_to_string(&mask_path).unwrap()).unwrap();
    mask.validate()?;
    assert_eq!(mask.memory_slots, 8);
    assert!(!mask.trainable_rows.is_empty());
    assert!(mask.trainable_rows.len() <= counts.unique_rows);

    let report_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(
        report_json["smft_artifacts"]["access_counts_out"],
        counts_path.display().to_string()
    );
    assert_eq!(
        report_json["smft_artifacts"]["mask_out"],
        mask_path.display().to_string()
    );
    assert_eq!(
        report_json["smft_artifacts"]["accumulated_counts"]["total_events"],
        counts.total_events
    );
    assert_eq!(
        report_json["smft_artifacts"]["generated_mask"]["trainable_rows"],
        mask.trainable_rows.len()
    );

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run scripts/test_gpu.sh cuda"]
fn cuda_train_memory_lm_cli_refreshes_smft_mask_online() -> Result<()> {
    require_cuda_hardware();

    let dir = temp_dir("cli-smft-refresh-cuda");
    fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("tiny.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let background_path = dir.join("background-counts.json");
    let checkpoint = dir.join("checkpoint");
    let report = dir.join("report.json");
    let text = "the cat sat. the dog ran. the cat ran. the dog sat. ";
    fs::write(&data_path, text.repeat(8)).unwrap();
    let tokenizer = BpeTokenizer::train(&fs::read_to_string(&data_path).unwrap(), 280)?;
    tokenizer.save(&tokenizer_path)?;
    let background = MemoryAccessCounts::empty(8);
    fs::write(
        &background_path,
        serde_json::to_string_pretty(&background).unwrap(),
    )
    .unwrap();

    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--data",
            data_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--block-size",
            "4",
            "--n-layers",
            "2",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--memory-layer-indices",
            "1",
            "--memory-slots",
            "8",
            "--memory-key-dim",
            "4",
            "--memory-value-dim",
            "8",
            "--memory-top-k",
            "2",
            "--lr",
            "0.01",
            "--device",
            "cuda:0",
            "--memory-update-policy",
            "sparse-rows",
            "--smft-background-counts",
            background_path.to_str().unwrap(),
            "--smft-trainable-fraction",
            "0.5",
            "--smft-min-rows",
            "1",
            "--smft-refresh-every",
            "1",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom train-memory-lm");

    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    let online_refresh = &report_json["smft_artifacts"]["online_refresh"];
    assert_eq!(online_refresh["enabled"], true);
    assert_eq!(online_refresh["refresh_every"], 1);
    assert_eq!(online_refresh["refresh_count"], 2);
    assert_eq!(online_refresh["last_refresh_step"], 2);
    assert_eq!(
        online_refresh["active_mask_source"],
        "online_refresh_step_2"
    );
    assert!(
        online_refresh["active_mask"]["trainable_rows"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        report_json["memory_optimizer"]["smft_row_mask"]["source"],
        "online_refresh_step_2"
    );
    assert!(
        report_json["memory_optimizer"]["row_mask_attached_sparse_update_count"]
            .as_u64()
            .unwrap()
            > 0
    );

    fs::remove_dir_all(&dir).ok();
    Ok(())
}
