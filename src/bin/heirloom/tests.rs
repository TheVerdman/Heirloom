#[cfg(test)]
mod tests {
    use super::*;

    fn valid_hex(byte: u8) -> String {
        (0..128).map(|_| format!("{byte:02x}")).collect()
    }

    fn env_value(env: &[LauncherEnvVar], name: &str) -> Option<String> {
        env.iter()
            .find(|var| var.name == name)
            .map(|var| var.value.clone())
    }

    fn padawan_tmp_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("heirloom-padawan-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_test_json(path: &Path, value: serde_json::Value) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, serde_json::to_string_pretty(&value).unwrap() + "\n").unwrap();
    }

    fn write_learning_sanity_sweep_fixture(root: &Path, world_size: usize, include_lr: bool) {
        let configs = [(0.001, 1usize), (0.001, 2), (0.0005, 1), (0.0005, 2)];
        let mut runs = Vec::new();
        for (index, (learning_rate, grad_accumulation)) in configs.iter().enumerate() {
            let learning_rate = *learning_rate;
            let grad_accumulation = *grad_accumulation;
            let report_name = format!("sweep-run-{index}.json");
            let global_effective_batch_size = 2 * grad_accumulation * world_size;
            let mut report = serde_json::json!({
                "command": "train-lm",
                "model_family": "tiny_transformer",
                "distributed": "nccl",
                "precision": "amp-bf16",
                "world_size": world_size,
                "devices": (0..world_size).map(|device| format!("cuda:{device}")).collect::<Vec<_>>(),
                "per_rank_batch_size": 2,
                "global_batch_size": 2 * world_size,
                "grad_accumulation_steps": grad_accumulation,
                "per_rank_effective_batch_size": 2 * grad_accumulation,
                "global_micro_batch_size": 2 * world_size,
                "global_effective_batch_size": global_effective_batch_size,
                "initial_loss": 4.0,
                "final_loss": 3.0,
                "loss_reduction": 0.25,
                "rank_loss_reductions": vec![0.25; world_size],
                "all_reduce_calls": 8,
                "all_reduce_bytes": 4096,
                "loader": {
                    "kind": "binary_shard_streaming"
                },
                "performance": {
                    "tokens_seen": 1024,
                    "micro_batch_size": 2,
                    "grad_accumulation_steps": grad_accumulation,
                    "data_parallel_world_size": world_size,
                    "global_effective_batch_size": global_effective_batch_size
                }
            });
            if include_lr {
                report["learning_rate"] = serde_json::json!(learning_rate);
            }
            write_test_json(&root.join(&report_name), report);
            runs.push(serde_json::json!({
                "label": format!("lr-{learning_rate}-ga-{grad_accumulation}"),
                "report": report_name,
                "expected_learning_rate": learning_rate,
                "expected_grad_accumulation_steps": grad_accumulation
            }));
        }
        write_test_json(
            &root.join("lr-grad-sweep.json"),
            serde_json::json!({
                "format": LEARNING_SANITY_LR_GRAD_SWEEP_FORMAT,
                "version": 0,
                "runs": runs
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": LEARNING_SANITY_MANIFEST_FORMAT,
                "version": 0,
                "stages": [
                    {
                        "stage_id": "lr_grad_accumulation_sweep",
                        "report": "lr-grad-sweep.json",
                        "expected_command": "train-lm",
                        "expected_model_family": "tiny_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "min_loss_reduction": 0.2
                    }
                ]
            }),
        );
    }

    #[test]
    fn learning_sanity_manifest_validates_loss_improvement() {
        let root = padawan_tmp_root("learning-sanity-pass");
        write_test_json(
            &root.join("dense-report.json"),
            serde_json::json!({
                "command": "train-lm",
                "model_family": "tiny_transformer",
                "initial_loss": 4.0,
                "final_loss": 3.0,
                "loss_reduction": 0.25,
                "loader": {
                    "kind": "binary_shard_streaming"
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "dense_fixed_shard",
                        "report": "dense-report.json",
                        "expected_command": "train-lm",
                        "expected_model_family": "tiny_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "min_loss_reduction": 0.2
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
        assert_eq!(report["min_loss_reduction_observed"], 0.25);
    }

    #[test]
    fn learning_sanity_manifest_rejects_flat_loss() {
        let root = padawan_tmp_root("learning-sanity-flat");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 5.0,
                "loss_reduction": 0.0,
                "memory_config": {
                    "memory_lookup": "exact",
                    "smft_mode": "disabled"
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "memory_exact_smft_disabled",
                        "report": "memory-report.json",
                        "expected_command": "train-memory-lm",
                        "expected_model_family": "memory_transformer",
                        "expected_memory_lookup": "exact",
                        "expected_smft_mode": "disabled",
                        "min_loss_reduction": 0.001
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("did not improve loss"));
    }

    #[test]
    fn learning_sanity_memory_layers_disabled_rejects_active_indices() {
        let root = padawan_tmp_root("learning-sanity-memory-disabled");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.0,
                "loss_reduction": 0.2,
                "loader": {
                    "kind": "binary_shard_streaming"
                },
                "memory_config": {
                    "n_layers": 2,
                    "memory_layer_indices": [1],
                    "memory_lookup": "exact",
                    "smft_mode": "disabled"
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "memory_layers_disabled",
                        "report": "memory-report.json",
                        "expected_command": "train-memory-lm",
                        "expected_model_family": "memory_transformer",
                        "expected_loader_kind": "binary_shard_streaming"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("expected memory layers disabled"));
    }

    #[test]
    fn learning_sanity_smft_enabled_validates_artifact_evidence() {
        let root = padawan_tmp_root("learning-sanity-smft-enabled");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.9,
                "loss_reduction": 0.02,
                "loader": {
                    "kind": "binary_shard_streaming"
                },
                "memory_config": {
                    "memory_lookup": "exact",
                    "memory_update_policy": "sparse_rows",
                    "smft_mode": "masked_memory_rows"
                },
                "memory_optimizer": {
                    "sparse_optimizer_updates_selected_rows": true,
                    "sparse_update_parameter_count": 2,
                    "row_mask_attached_sparse_update_count": 2,
                    "sparse_update_selected_row_events": 64,
                    "smft_row_mask": {
                        "applied": true,
                        "trainable_rows": 8
                    }
                },
                "smft_artifacts": {
                    "accumulated_counts": {
                        "total_events": 512,
                        "unique_rows": 16
                    },
                    "generated_mask": {
                        "trainable_rows": 8
                    },
                    "online_refresh": {
                        "enabled": true,
                        "refresh_count": 4,
                        "active_mask": {
                            "trainable_rows": 8
                        }
                    }
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "memory_exact_smft_enabled",
                        "report": "memory-report.json",
                        "expected_command": "train-memory-lm",
                        "expected_model_family": "memory_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "expected_memory_lookup": "exact",
                        "expected_memory_update_policy": "sparse_rows",
                        "expected_smft_mode": "masked_memory_rows"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
    }

    #[test]
    fn learning_sanity_smft_enabled_rejects_missing_artifacts() {
        let root = padawan_tmp_root("learning-sanity-smft-missing");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.9,
                "loss_reduction": 0.02,
                "memory_config": {
                    "memory_lookup": "exact",
                    "memory_update_policy": "sparse_rows",
                    "smft_mode": "masked_memory_rows"
                },
                "memory_optimizer": {
                    "sparse_optimizer_updates_selected_rows": true,
                    "sparse_update_parameter_count": 2,
                    "row_mask_attached_sparse_update_count": 2,
                    "sparse_update_selected_row_events": 64,
                    "smft_row_mask": {
                        "applied": true,
                        "trainable_rows": 8
                    }
                }
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "memory_exact_smft_enabled",
                        "report": "memory-report.json"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("requires smft_artifacts evidence"));
    }

    #[test]
    fn learning_sanity_product_key_parity_validates_lookup_evidence() {
        let root = padawan_tmp_root("learning-sanity-product-key");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.0,
                "loss_reduction": 0.2,
                "loader": {
                    "kind": "binary_shard_streaming"
                },
                "memory_config": {
                    "memory_key_dim": 8,
                    "memory_lookup": "product_key",
                    "memory_slots": 16,
                    "memory_update_policy": "full",
                    "smft_mode": "disabled"
                },
                "memory_optimizer": {
                    "memory_table_parameter_count": 3
                },
                "memory_selection": {
                    "captured_memory_layers": 1,
                    "configured_memory_layers": 1,
                    "memory_lookup": "product_key",
                    "selected_row_events": 64,
                    "unique_selected_rows": 7
                },
                "amp_bf16_op_decisions": [
                    {
                        "kernel_path": "cpu_product_key_candidate_topk_memory_lookup"
                    }
                ]
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "product_key_parity",
                        "report": "memory-report.json",
                        "expected_command": "train-memory-lm",
                        "expected_model_family": "memory_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "expected_memory_lookup": "product_key",
                        "expected_memory_update_policy": "full",
                        "expected_smft_mode": "disabled"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
    }

    #[test]
    fn learning_sanity_product_key_parity_rejects_non_square_slots() {
        let root = padawan_tmp_root("learning-sanity-product-key-shape");
        write_test_json(
            &root.join("memory-report.json"),
            serde_json::json!({
                "command": "train-memory-lm",
                "model_family": "memory_transformer",
                "initial_loss": 5.0,
                "final_loss": 4.0,
                "loss_reduction": 0.2,
                "memory_config": {
                    "memory_key_dim": 8,
                    "memory_lookup": "product_key",
                    "memory_slots": 18,
                    "smft_mode": "disabled"
                },
                "memory_optimizer": {
                    "memory_table_parameter_count": 3
                },
                "memory_selection": {
                    "captured_memory_layers": 1,
                    "configured_memory_layers": 1,
                    "memory_lookup": "product_key",
                    "selected_row_events": 64,
                    "unique_selected_rows": 7
                },
                "amp_bf16_op_decisions": [
                    {
                        "kernel_path": "cpu_product_key_candidate_topk_memory_lookup"
                    }
                ]
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "product_key_parity",
                        "report": "memory-report.json"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("memory_slots must be square"));
    }

    #[test]
    fn learning_sanity_lr_grad_sweep_validates_ddp_matrix() {
        let root = padawan_tmp_root("learning-sanity-lr-grad-sweep");
        write_learning_sanity_sweep_fixture(&root, 4, true);
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
        assert_eq!(report["stages"][0]["loss_reduction"], 0.25);
        assert_eq!(report["stages"][0]["best_loss_reduction"], 0.25);
        assert_eq!(report["stages"][0]["recommended_learning_rate"], 0.001);
        assert_eq!(
            report["stages"][0]["recommended_grad_accumulation_steps"],
            1
        );
        assert_eq!(report["stages"][0]["best_run"]["label"], "lr-0.001-ga-1");
        assert_eq!(report["stages"][0]["runs"].as_array().unwrap().len(), 4);
        assert_eq!(
            report["stages"][0]["min_observed_data_parallel_world_size"],
            4
        );
    }

    #[test]
    fn learning_sanity_lr_grad_sweep_rejects_weak_best_run() {
        let root = padawan_tmp_root("learning-sanity-lr-grad-weak-best");
        write_learning_sanity_sweep_fixture(&root, 4, true);
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": LEARNING_SANITY_MANIFEST_FORMAT,
                "version": 0,
                "stages": [
                    {
                        "stage_id": "lr_grad_accumulation_sweep",
                        "report": "lr-grad-sweep.json",
                        "expected_command": "train-lm",
                        "expected_model_family": "tiny_transformer",
                        "expected_loader_kind": "binary_shard_streaming",
                        "min_loss_reduction": 0.2,
                        "min_best_loss_reduction": 0.3
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("best loss_reduction 0.25 < required 0.3"));
    }

    #[test]
    fn learning_sanity_lr_grad_sweep_rejects_missing_learning_rate() {
        let root = padawan_tmp_root("learning-sanity-lr-grad-missing-lr");
        write_learning_sanity_sweep_fixture(&root, 4, false);
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("learning_rate must be numeric"));
    }

    #[test]
    fn learning_sanity_lr_grad_sweep_rejects_underpowered_world_size() {
        let root = padawan_tmp_root("learning-sanity-lr-grad-small-world");
        write_learning_sanity_sweep_fixture(&root, 2, true);
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("data_parallel_world_size 2 < required 4"));
    }

    #[test]
    fn learning_sanity_longer_32k_blend_validates_hardpath_bundle() {
        let report = validate_learning_sanity_manifest(Path::new(
            "tests/fixtures/learning_sanity/longer-32k-blend-valid.json",
        ))
        .unwrap();
        assert_eq!(report["status"], "passed");
        assert_eq!(report["passed"], 1);
        assert_eq!(report["failed"], 0);
        assert_eq!(report["stages"][0]["tokenizer_vocab_size"], 32768);
        assert!((report["stages"][0]["loss_reduction"].as_f64().unwrap() - 0.16).abs() < 1.0e-12);
    }

    #[test]
    fn learning_sanity_longer_32k_blend_rejects_small_tokenizer() {
        let root = padawan_tmp_root("learning-sanity-32k-small-tokenizer");
        write_test_json(
            &root.join("summary.json"),
            serde_json::json!({
                "status": "passed",
                "tokenizer_version": 2,
                "tokenizer_vocab_size": 512,
                "reserved_tokens": 128,
                "manifest_version": 2,
                "manifest_storage": "binary_shards",
                "loader_kind": "binary_shard_streaming",
                "tokens_materialized": false,
                "target_tokens": 4096,
                "selected_tokens": 4096,
                "selected_docs": 8,
                "artifacts": {}
            }),
        );
        write_test_json(
            &root.join("ladder.json"),
            serde_json::json!({
                "format": "heirloom.learning_sanity_ladder",
                "version": 0,
                "stages": [
                    {
                        "stage_id": "longer_32k_blend",
                        "report": "summary.json"
                    }
                ]
            }),
        );
        let report = validate_learning_sanity_manifest(&root.join("ladder.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["passed"], 0);
        assert_eq!(report["failed"], 1);
        assert!(report["stages"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("tokenizer_vocab_size expected 32768"));
    }

    #[test]
    fn padawan_sha256_matches_known_vector() {
        assert_eq!(
            sha256_prefixed_hex(b"abc"),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn padawan_fixture_validates_and_verifies_in_rust() {
        let episodes = vec![PathBuf::from("padawan/fixtures/episode_valid.jsonl")];
        let root = Path::new("padawan/fixtures");
        let validation = padawan_validate_report(&episodes, Some(root)).unwrap();
        assert_eq!(validation["status"], "passed");
        assert_eq!(validation["episodes"], 1);
        assert_eq!(validation["sft_eligible"], 1);
        assert_eq!(validation["smft_eligible"], 1);

        let verification = padawan_verify_report(&episodes, Some(root), &[], &[]).unwrap();
        assert_eq!(verification["status"], "passed");
        assert_eq!(verification["family_counts"]["passed"], 4);
        assert_eq!(verification["family_counts"]["failed"], 0);
    }

    #[test]
    fn padawan_code_patch_verifier_rejects_non_sidecar_paths() {
        let root = padawan_tmp_root("bad-patch");
        std::fs::create_dir_all(root.join("artifacts")).unwrap();
        std::fs::write(
            root.join("artifacts/bad.diff"),
            "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@\n+bad\n",
        )
        .unwrap();
        let episode = serde_json::json!({
            "padawan": {
                "artifact_ref": "artifact://padawan/artifacts/bad.diff"
            },
            "verifier": {
                "unrelated_changes": 0
            }
        });
        let err = padawan_verify_code_patch(&episode, &root, &["padawan/".to_string()])
            .expect_err("patch verifier should reject non-sidecar paths");
        assert!(err
            .to_string()
            .contains("outside Padawan sidecar allowlist"));
    }

    #[test]
    fn padawan_json_tool_call_verifier_rejects_inline_results() {
        let root = padawan_tmp_root("bad-tool-call");
        write_test_json(
            &root.join("traces/trace.json"),
            serde_json::json!({
                "tool_calls": [
                    {
                        "tool": "shell",
                        "arguments": {},
                        "result": "inline result should live in observations"
                    }
                ]
            }),
        );
        write_test_json(
            &root.join("artifacts/bundle.json"),
            serde_json::json!({
                "artifacts": [
                    {
                        "name": "trace",
                        "ref": "artifact://padawan/traces/trace.json",
                        "media_type": "application/json"
                    }
                ]
            }),
        );
        let episode = serde_json::json!({
            "padawan": {
                "trace_ref": "artifact://padawan/traces/trace.json",
                "artifact_bundle_ref": "artifact://padawan/artifacts/bundle.json"
            }
        });
        let err = padawan_verify_json_tool_call(&episode, &root)
            .expect_err("tool-call verifier should reject inline results");
        assert!(err.to_string().contains("should not inline tool results"));
    }

    #[test]
    fn padawan_evidence_verifier_rejects_missing_observations() {
        let root = padawan_tmp_root("bad-evidence");
        write_test_json(
            &root.join("traces/trace.json"),
            serde_json::json!({
                "observations": [],
                "final_validation": {
                    "validator": "heirloom padawan verify",
                    "expected": "passes"
                }
            }),
        );
        std::fs::create_dir_all(root.join("finals")).unwrap();
        std::fs::write(root.join("finals/final.txt"), "done\n").unwrap();
        let episode = serde_json::json!({
            "padawan": {
                "trace_ref": "artifact://padawan/traces/trace.json",
                "final_response_ref": "artifact://padawan/finals/final.txt"
            },
            "verifier": {
                "failure_class": null
            }
        });
        let err = padawan_verify_evidence_citation(&episode, &root)
            .expect_err("evidence verifier should require observations");
        assert!(err.to_string().contains("at least one observation"));
    }

    #[test]
    fn padawan_memory_smft_verifier_rejects_row_budget_overflow() {
        let root = padawan_tmp_root("bad-smft");
        let rows = (0..33).collect::<Vec<usize>>();
        write_test_json(
            &root.join("memory/selection.json"),
            serde_json::json!({
                "format": "heirloom.padawan.memory_selection",
                "memory_slots": 1024,
                "layers": {
                    "layer_8": {
                        "selected_rows": rows,
                        "selected_row_events": 33
                    }
                }
            }),
        );
        write_test_json(
            &root.join("memory/counts.json"),
            serde_json::json!({
                "format": "heirloom.padawan.smft_access_counts",
                "memory_slots": 1024,
                "foreground_rows": rows,
                "background_rows": [99, 100],
                "foreground_background_lift": 4.5,
                "replay_jaccard_overlap": 0.34,
                "per_episode_row_cap": 32,
                "task_family_row_cap": 128
            }),
        );
        let episode = serde_json::json!({
            "selection": {
                "eligible_for_smft": true
            },
            "memory": {
                "selection_report_ref": "artifact://padawan/memory/selection.json",
                "smft_access_counts_ref": "artifact://padawan/memory/counts.json"
            }
        });
        let err = padawan_verify_memory_smft(&episode, &root)
            .expect_err("SMFT verifier should reject row budget overflow");
        assert!(err
            .to_string()
            .contains("selected rows exceed per-episode cap"));
    }

    #[test]
    fn tensor_core_flash_microbench_pass_requires_ok_and_passed_when_requested() {
        let error_report = serde_json::json!({
            "status": "error",
            "passed": false,
        });
        assert!(tensor_core_flash_microbench_passed(&error_report, false));
        assert!(!tensor_core_flash_microbench_passed(&error_report, true));

        let failed_report = serde_json::json!({
            "status": "ok",
            "passed": false,
        });
        assert!(!tensor_core_flash_microbench_passed(&failed_report, true));

        let passed_report = serde_json::json!({
            "status": "ok",
            "passed": true,
        });
        assert!(tensor_core_flash_microbench_passed(&passed_report, true));
    }

    #[test]
    fn flash_tensor_core_forward_mma_flops_count_head_dim_chunks() {
        let head_dim_16 = flash_tensor_core_forward_mma_flops(1, 1, 16, 16);
        let head_dim_64 = flash_tensor_core_forward_mma_flops(1, 1, 16, 64);

        assert_eq!(head_dim_16, 24_576.0);
        assert_eq!(head_dim_64, 294_912.0);
    }

    #[test]
    fn grad_accumulation_helpers_preserve_optimizer_step_semantics() {
        assert!(validate_grad_accumulation_steps(0).is_err());
        validate_grad_accumulation_steps(1).unwrap();
        assert_eq!(accumulated_training_tokens_seen(3, 2, 5, 4).unwrap(), 120);
        assert_eq!(ddp_sample_step(10, 2, 3, 4).unwrap(), 51);
    }

    #[test]
    fn aggregate_rank_performance_preserves_timing_buckets() {
        let rank_reports = vec![
            serde_json::json!({
                "performance": {
                    "tokens_seen": 100,
                    "train_elapsed_ms": 10,
                    "dataloader_elapsed_ms": 1,
                    "host_to_device_elapsed_ms": 2,
                    "forward_backward_elapsed_ms": 3,
                    "all_reduce_elapsed_ms": 4,
                    "optimizer_elapsed_ms": 5,
                    "host_to_device_cuda_elapsed_ms": 0.2,
                    "forward_backward_cuda_elapsed_ms": 1.5,
                    "all_reduce_cuda_elapsed_ms": 0.7,
                    "optimizer_cuda_elapsed_ms": 0.3,
                    "active_dense_flops_per_token_estimate": 1000.0
                }
            }),
            serde_json::json!({
                "performance": {
                    "tokens_seen": 200,
                    "train_elapsed_ms": 12,
                    "dataloader_elapsed_ms": 2,
                    "host_to_device_elapsed_ms": 1,
                    "forward_backward_elapsed_ms": 4,
                    "all_reduce_elapsed_ms": 6,
                    "optimizer_elapsed_ms": 3,
                    "host_to_device_cuda_elapsed_ms": 0.1,
                    "forward_backward_cuda_elapsed_ms": 2.5,
                    "all_reduce_cuda_elapsed_ms": 0.5,
                    "optimizer_cuda_elapsed_ms": 0.4,
                    "active_dense_flops_per_token_estimate": 1000.0
                }
            }),
        ];

        let report = aggregate_rank_performance(&rank_reports);

        assert_eq!(report["tokens_seen"], 300);
        assert_eq!(report["train_elapsed_ms"], 12);
        assert_eq!(report["dataloader_elapsed_ms"], 2);
        assert_eq!(report["host_to_device_elapsed_ms"], 2);
        assert_eq!(report["forward_backward_elapsed_ms"], 4);
        assert_eq!(report["all_reduce_elapsed_ms"], 6);
        assert_eq!(report["optimizer_elapsed_ms"], 5);
        assert_eq!(report["host_to_device_host_elapsed_ms"], 2);
        assert_eq!(report["forward_backward_host_elapsed_ms"], 4);
        assert_eq!(report["host_to_device_cuda_elapsed_ms"], 0.2);
        assert_eq!(report["forward_backward_cuda_elapsed_ms"], 2.5);
        assert_eq!(report["all_reduce_cuda_elapsed_ms"], 0.7);
        assert_eq!(report["optimizer_cuda_elapsed_ms"], 0.4);
        assert_eq!(report["cuda_event_timing_available"], true);
        assert_eq!(
            report["mfu_timing_source"]["dense_core_mfu_estimate"],
            "cuda_event_forward_backward_elapsed_ms"
        );
        assert!(report["tokens_per_second"].as_f64().unwrap() > 0.0);
        assert!(report["dense_core_mfu_estimate"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn aggregate_rank_cuda_runtime_sums_flash_backward_fields() {
        let rank_reports = vec![
            serde_json::json!({
                "cuda_runtime": {
                    "kernel_launch_family_elapsed_us": 800,
                    "kernel_launch_families": {
                        "rank2_elementwise": {"calls": 3, "elements": 30, "elapsed_us": 300},
                        "tensor_core_gemm_cp_async": {"calls": 5, "elements": 50, "elapsed_us": 500}
                    },
                    "flash_bf16_tensor_core_backward_requested_calls": 1,
                    "flash_bf16_tensor_core_backward_executed_calls": 2,
                    "flash_bf16_tensor_core_backward_fallback_calls": 3,
                    "flash_bf16_tensor_core_backward_row_dot_calls": 4,
                    "flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls": 5,
                    "flash_bf16_tensor_core_backward_dp_mma_tile_calls": 6,
                    "flash_bf16_tensor_core_backward_dq_mma_tile_calls": 7,
                    "flash_bf16_tensor_core_backward_dk_mma_tile_calls": 8,
                    "flash_bf16_tensor_core_backward_dv_mma_tile_calls": 9,
                    "flash_bf16_tensor_core_backward_scalar_tile_calls": 10,
                    "flash_bf16_tensor_core_backward_ragged_tile_count": 11,
                    "flash_bf16_tensor_core_backward_causal_masked_tile_count": 12,
                    "flash_bf16_tensor_core_backward_elapsed_us": 13
                }
            }),
            serde_json::json!({
                "cuda_runtime": {
                    "kernel_launch_family_elapsed_us": 1800,
                    "kernel_launch_families": {
                        "rank2_elementwise": {"calls": 7, "elements": 70, "elapsed_us": 700},
                        "flash_attention_bf16_tensor_core_backward": {"calls": 11, "elements": 110, "elapsed_us": 1100}
                    },
                    "flash_bf16_tensor_core_backward_requested_calls": 10,
                    "flash_bf16_tensor_core_backward_executed_calls": 20,
                    "flash_bf16_tensor_core_backward_fallback_calls": 30,
                    "flash_bf16_tensor_core_backward_row_dot_calls": 40,
                    "flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls": 50,
                    "flash_bf16_tensor_core_backward_dp_mma_tile_calls": 60,
                    "flash_bf16_tensor_core_backward_dq_mma_tile_calls": 70,
                    "flash_bf16_tensor_core_backward_dk_mma_tile_calls": 80,
                    "flash_bf16_tensor_core_backward_dv_mma_tile_calls": 90,
                    "flash_bf16_tensor_core_backward_scalar_tile_calls": 100,
                    "flash_bf16_tensor_core_backward_ragged_tile_count": 110,
                    "flash_bf16_tensor_core_backward_causal_masked_tile_count": 120,
                    "flash_bf16_tensor_core_backward_elapsed_us": 130
                }
            }),
        ];

        let report = aggregate_rank_cuda_runtime(&rank_reports);

        assert_eq!(
            report["kernel_launch_families"]["rank2_elementwise"]["calls"],
            10
        );
        assert_eq!(
            report["kernel_launch_families"]["rank2_elementwise"]["elements"],
            100
        );
        assert_eq!(
            report["kernel_launch_families"]["rank2_elementwise"]["elapsed_us"],
            1000
        );
        assert_eq!(
            report["kernel_launch_families"]["tensor_core_gemm_cp_async"]["calls"],
            5
        );
        assert_eq!(
            report["kernel_launch_families"]["flash_attention_bf16_tensor_core_backward"]["calls"],
            11
        );
        assert_eq!(report["kernel_launch_family_elapsed_us"], 2600);
        assert_eq!(
            report["flash_bf16_tensor_core_backward_requested_calls"],
            11
        );
        assert_eq!(report["flash_bf16_tensor_core_backward_executed_calls"], 22);
        assert_eq!(report["flash_bf16_tensor_core_backward_fallback_calls"], 33);
        assert_eq!(report["flash_bf16_tensor_core_backward_row_dot_calls"], 44);
        assert_eq!(
            report["flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls"],
            55
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_dp_mma_tile_calls"],
            66
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_dq_mma_tile_calls"],
            77
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_dk_mma_tile_calls"],
            88
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_dv_mma_tile_calls"],
            99
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_scalar_tile_calls"],
            110
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_ragged_tile_count"],
            121
        );
        assert_eq!(
            report["flash_bf16_tensor_core_backward_causal_masked_tile_count"],
            132
        );
        assert_eq!(report["flash_bf16_tensor_core_backward_elapsed_us"], 143);
    }

    #[test]
    fn tensor_core_pad_crop_report_names_ragged_linear_evidence() {
        let cuda_runtime = serde_json::json!({
            "tensor_core_padded_tiles": 17,
            "tensor_core_remainder_tiles": 5,
        });
        let tensor_core = serde_json::json!({
            "bf16_scalar_matmul_fallback_calls": 0,
        });
        let coverage = serde_json::json!({
            "linear_totals": {
                "tensor_core_calls": 4,
                "fallback_calls": 0,
            },
            "linear_modules": [
                {
                    "module": "lm_head",
                    "calls": 4,
                    "tensor_core_calls": 4,
                    "fallback_calls": 0,
                    "last_path": "tensor_core",
                    "last_m": 15,
                    "last_k": 18,
                    "last_n": 281,
                }
            ],
        });

        let report =
            tensor_core_pad_crop_report_from_json(&cuda_runtime, &tensor_core, &coverage, "test");

        assert_eq!(report["used"], true);
        assert_eq!(report["passed"], true);
        assert_eq!(report["status"], "passed");
        assert_eq!(report["padded_tiles"], 17);
        assert_eq!(report["remainder_tiles"], 5);
        assert_eq!(report["scalar_fallbacks"], 0);
        assert_eq!(report["linear_fallbacks"], 0);
        assert_eq!(report["linear_modules"][0]["logical_shape"]["m"], 15);
        assert_eq!(report["linear_modules"][0]["padded_shape"]["m"], 16);
        assert_eq!(report["linear_modules"][0]["padded_shape"]["k"], 32);
        assert_eq!(report["linear_modules"][0]["padded_shape"]["n"], 288);
        assert_eq!(report["linear_modules"][0]["needs_padding"], true);
    }

    #[test]
    fn tensor_core_pad_crop_report_distinguishes_tile_aligned_shapes() {
        let cuda_runtime = serde_json::json!({
            "tensor_core_padded_tiles": 0,
            "tensor_core_remainder_tiles": 0,
        });
        let tensor_core = serde_json::json!({
            "bf16_scalar_matmul_fallback_calls": 0,
        });
        let coverage = serde_json::json!({
            "linear_totals": {
                "tensor_core_calls": 1,
                "fallback_calls": 0,
            },
            "linear_modules": [
                {
                    "module": "linear",
                    "calls": 1,
                    "tensor_core_calls": 1,
                    "fallback_calls": 0,
                    "last_path": "tensor_core",
                    "last_m": 16,
                    "last_k": 16,
                    "last_n": 8,
                }
            ],
        });

        let report =
            tensor_core_pad_crop_report_from_json(&cuda_runtime, &tensor_core, &coverage, "test");

        assert_eq!(report["used"], false);
        assert_eq!(report["passed"], false);
        assert_eq!(report["status"], "not_used");
        assert_eq!(report["linear_modules"][0]["needs_padding"], false);
    }

    #[test]
    fn cuda_runtime_counters_json_includes_flash_attention_fields() {
        let json = cuda_runtime_counters_json(cuda::CudaRuntimeCounters::default());

        assert_eq!(json["tensor_core_ldmatrix_gemm_executed_calls"], 0);
        assert_eq!(json["tensor_core_ldmatrix_gemm_hard_require_failures"], 0);
        assert_eq!(json["tensor_core_ldmatrix_gemm_instructions"], 0);
        assert_eq!(json["tensor_core_ldmatrix_gemm_elapsed_us"], 0);
        assert_eq!(json["tensor_core_cp_async_gemm_executed_calls"], 0);
        assert_eq!(json["tensor_core_cp_async_gemm_hard_require_failures"], 0);
        assert_eq!(json["tensor_core_cp_async_gemm_instructions"], 0);
        assert_eq!(json["tensor_core_cp_async_gemm_elapsed_us"], 0);
        assert_eq!(json["kernel_launch_family_elapsed_us"], 0);
        assert!(json["kernel_launch_families"]
            .as_object()
            .expect("kernel launch families is an object")
            .is_empty());
        assert_eq!(json["tensor_core_staged_cta_gemm_elapsed_us"], 0);
        assert_eq!(json["tensor_core_wide_swizzled_cta_gemm_elapsed_us"], 0);
        assert_eq!(json["tensor_core_global_cta_gemm_elapsed_us"], 0);
        assert_eq!(json["tensor_core_legacy_warp_gemm_elapsed_us"], 0);
        assert_eq!(json["bf16_attention_materialized_reference_calls"], 0);
        assert_eq!(json["flash_bf16_attention_requested_calls"], 0);
        assert_eq!(json["flash_bf16_attention_executed_calls"], 0);
        assert_eq!(json["flash_bf16_attention_fallback_calls"], 0);
        assert_eq!(json["flash_bf16_attention_scalar_fallback_calls"], 0);
        assert_eq!(json["flash_bf16_attention_qk_tile_calls"], 0);
        assert_eq!(json["flash_bf16_attention_av_tile_calls"], 0);
        assert_eq!(json["flash_bf16_attention_ragged_tile_count"], 0);
        assert_eq!(json["flash_bf16_attention_causal_masked_tile_count"], 0);
        assert_eq!(json["flash_bf16_attention_elapsed_us"], 0);
        assert_eq!(json["flash_bf16_attention_hard_require_failures"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_requested_calls"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_executed_calls"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_qk_tile_calls"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_av_tile_calls"], 0);
        assert_eq!(json["flash_bf16_scalar_streaming_elapsed_us"], 0);
        assert_eq!(json["flash_bf16_tensor_core_requested_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_executed_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_fallback_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_qk_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_av_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_ragged_tile_count"], 0);
        assert_eq!(json["flash_bf16_tensor_core_causal_masked_tile_count"], 0);
        assert_eq!(json["flash_bf16_tensor_core_elapsed_us"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_requested_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_executed_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_fallback_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_row_dot_calls"], 0);
        assert_eq!(
            json["flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls"],
            0
        );
        assert_eq!(json["flash_bf16_tensor_core_backward_dp_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_dq_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_dk_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_dv_mma_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_scalar_tile_calls"], 0);
        assert_eq!(json["flash_bf16_tensor_core_backward_ragged_tile_count"], 0);
        assert_eq!(
            json["flash_bf16_tensor_core_backward_causal_masked_tile_count"],
            0
        );
        assert_eq!(json["flash_bf16_tensor_core_backward_elapsed_us"], 0);
    }

    #[test]
    fn memory_optimizer_report_records_supplied_smft_row_mask() {
        let mut config = MemoryTransformerConfig::tiny(32);
        config.block_size = 4;
        config.n_layers = 2;
        config.d_model = 8;
        config.n_heads = 2;
        config.ff_hidden = 16;
        config.memory_layer_indices = vec![1];
        config.memory_slots = 8;
        config.memory_key_dim = 4;
        config.memory_value_dim = 8;
        config.memory_top_k = 2;
        config.memory_update_policy = MemoryUpdatePolicy::SparseRows;
        config.smft_mode = SmftMode::MaskedMemoryRows;
        let mut rng = HeirloomRng::new(515);
        let model = MemoryTransformerLm::new(config, &mut rng).unwrap();
        let input = Tensor::from_i64(vec![1, 2, 3, 4], &[1, 4], false).unwrap();
        let _ = model.forward(&input).unwrap();
        let mask = SmftRowMask {
            memory_slots: 8,
            trainable_rows: vec![1, 3],
            frozen_rows: 6,
            trainable_fraction: 0.25,
            scores: Vec::new(),
        };

        let report = memory_optimizer_report(&model, Some(&mask), Some("mask.json")).unwrap();

        assert_eq!(
            report["applied_path"],
            "cpu_sparse_rows_selected_memory_tables"
        );
        assert_eq!(report["sparse_update_parameter_count"], 2);
        assert_eq!(report["row_mask_attached_sparse_update_count"], 2);
        assert_eq!(
            report["sparse_updates_accumulate_dense_gradient_buffers"],
            true
        );
        assert_eq!(report["sparse_optimizer_updates_selected_rows"], true);
        assert_eq!(
            report["sparse_optimizer_gathers_compact_gradient_rows"],
            false
        );
        assert_eq!(report["compressed_sparse_gradient_transport"], false);
        assert_eq!(report["smft_row_mask"]["source"], "mask.json");
        assert_eq!(report["smft_row_mask"]["memory_slots"], 8);
        assert_eq!(report["smft_row_mask"]["trainable_rows"], 2);
        assert_eq!(report["smft_row_mask"]["frozen_rows"], 6);
    }

    #[test]
    fn memory_optimizer_report_records_product_key_smft_projection() {
        let mut config = MemoryTransformerConfig::tiny(32);
        config.block_size = 4;
        config.n_layers = 2;
        config.d_model = 8;
        config.n_heads = 2;
        config.ff_hidden = 16;
        config.memory_layer_indices = vec![1];
        config.memory_slots = 4;
        config.memory_key_dim = 4;
        config.memory_value_dim = 8;
        config.memory_top_k = 2;
        config.memory_lookup = MemoryLookupKind::ProductKey;
        config.memory_update_policy = MemoryUpdatePolicy::SparseRows;
        config.smft_mode = SmftMode::MaskedMemoryRows;
        let mut rng = HeirloomRng::new(616);
        let model = MemoryTransformerLm::new(config, &mut rng).unwrap();
        let input = Tensor::from_i64(vec![1, 2, 3, 4], &[1, 4], false).unwrap();
        let _ = model.forward(&input).unwrap();
        let mask = SmftRowMask {
            memory_slots: 4,
            trainable_rows: vec![0, 1],
            frozen_rows: 2,
            trainable_fraction: 0.5,
            scores: Vec::new(),
        };

        let report = memory_optimizer_report(&model, Some(&mask), Some("mask.json")).unwrap();

        assert_eq!(report["sparse_update_parameter_count"], 3);
        assert_eq!(report["row_mask_attached_sparse_update_count"], 3);
        let projection = &report["smft_row_mask"]["product_key_projection"];
        assert_eq!(projection["policy"], "conservative_all_slots");
        assert_eq!(projection["side"], 2);
        assert_eq!(projection["value_trainable_rows"], 2);
        assert_eq!(projection["left_trainable_rows"], 1);
        assert_eq!(projection["right_trainable_rows"], 0);
        assert_eq!(projection["half_key_rows_are_conservative"], true);
    }

    #[test]
    fn distributed_memory_evidence_requires_dense_counters_for_full_updates() {
        let err =
            validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
                memory_update_policy: &MemoryUpdatePolicy::Full,
                all_reduce_calls: 0,
                all_reduce_bytes: 0,
                row_union_all_reduce_calls: 0,
                row_union_all_reduce_bytes: 0,
                row_union_candidate_rows: 0,
                compact_gradient_all_reduce_calls: 0,
                compact_gradient_all_reduce_bytes: 0,
            })
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("without recorded gradient all-reduces"),
            "{err}"
        );

        validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
            memory_update_policy: &MemoryUpdatePolicy::Full,
            all_reduce_calls: 2,
            all_reduce_bytes: 128,
            row_union_all_reduce_calls: 0,
            row_union_all_reduce_bytes: 0,
            row_union_candidate_rows: 0,
            compact_gradient_all_reduce_calls: 0,
            compact_gradient_all_reduce_bytes: 0,
        })
        .unwrap();
    }

    #[test]
    fn distributed_memory_sparse_evidence_requires_compact_gradient_transport() {
        let err =
            validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
                memory_update_policy: &MemoryUpdatePolicy::SparseRows,
                all_reduce_calls: 4,
                all_reduce_bytes: 256,
                row_union_all_reduce_calls: 2,
                row_union_all_reduce_bytes: 64,
                row_union_candidate_rows: 8,
                compact_gradient_all_reduce_calls: 0,
                compact_gradient_all_reduce_bytes: 0,
            })
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("without recorded compact sparse all-reduces"),
            "{err}"
        );

        validate_distributed_memory_all_reduce_evidence(DistributedMemoryAllReduceEvidence {
            memory_update_policy: &MemoryUpdatePolicy::SparseRows,
            all_reduce_calls: 4,
            all_reduce_bytes: 256,
            row_union_all_reduce_calls: 2,
            row_union_all_reduce_bytes: 64,
            row_union_candidate_rows: 8,
            compact_gradient_all_reduce_calls: 2,
            compact_gradient_all_reduce_bytes: 512,
        })
        .unwrap();
    }

    #[test]
    fn parses_marked_nccl_unique_id_from_noisy_stdout() {
        let hex = valid_hex(0xab);
        let stdout =
            format!("NCCL INFO graph line\n{NCCL_UNIQUE_ID_HELPER_MARKER}{hex}\nNCCL INFO done\n");

        assert_eq!(parse_nccl_unique_id_helper_stdout(&stdout).unwrap(), hex);
    }

    #[test]
    fn parses_exact_hex_nccl_unique_id_for_backwards_compatibility() {
        let hex = valid_hex(0x17);

        assert_eq!(parse_nccl_unique_id_helper_stdout(&hex).unwrap(), hex);
    }

    #[test]
    fn rejects_missing_nccl_unique_id_marker_or_exact_hex_line() {
        let err = parse_nccl_unique_id_helper_stdout("NCCL INFO only\n").unwrap_err();

        assert!(format!("{err}").contains("did not contain"));
    }

    #[test]
    fn rejects_invalid_marked_nccl_unique_id_payload() {
        let stdout = format!("{NCCL_UNIQUE_ID_HELPER_MARKER}abc\n");
        let err = parse_nccl_unique_id_helper_stdout(&stdout).unwrap_err();

        assert!(format!("{err}").contains("invalid id"));
    }

    #[test]
    fn rejects_multiple_nccl_unique_id_candidates() {
        let first = valid_hex(0x01);
        let second = valid_hex(0x02);
        let stdout = format!(
            "{NCCL_UNIQUE_ID_HELPER_MARKER}{first}\n{NCCL_UNIQUE_ID_HELPER_MARKER}{second}"
        );
        let err = parse_nccl_unique_id_helper_stdout(&stdout).unwrap_err();

        assert!(format!("{err}").contains("multiple candidate ids"));
    }

    #[test]
    fn nccl_launcher_env_defaults_to_single_node_loopback_bootstrap() {
        let env = launcher_env_from_lookup(|_| None);

        assert_eq!(env_value(&env, "NCCL_DEBUG").as_deref(), Some("INFO"));
        assert_eq!(
            env_value(&env, "NCCL_DEBUG_SUBSYS").as_deref(),
            Some("INIT,COLL,GRAPH")
        );
        assert_eq!(env_value(&env, "NCCL_NET_PLUGIN").as_deref(), Some("none"));
        assert_eq!(env_value(&env, "NCCL_SOCKET_IFNAME").as_deref(), Some("lo"));
        assert_eq!(env_value(&env, "NCCL_IB_DISABLE").as_deref(), Some("1"));
        assert!(env_value(&env, "NCCL_CUMEM_ENABLE").is_none());
        assert!(env_value(&env, "NCCL_CUMEM_HOST_ENABLE").is_none());
        assert!(env_value(&env, "NCCL_P2P_DISABLE").is_none());
        assert!(env_value(&env, "NCCL_P2P_LEVEL").is_none());
        assert!(env_value(&env, "HEIRLOOM_NCCL_TRACE").is_none());
    }

    #[test]
    fn nccl_launcher_env_ignores_inherited_vertex_bootstrap_overrides() {
        let env = launcher_env_from_lookup(|name| match name {
            "NCCL_SOCKET_IFNAME" => Some("^cbr,veth,docker,lo,cali,gke,node,cilium".to_string()),
            "NCCL_IB_DISABLE" => Some("0".to_string()),
            "NCCL_NET_PLUGIN" => Some("FastSocket".to_string()),
            "HEIRLOOM_NCCL_TRACE" => Some("1".to_string()),
            _ => None,
        });

        assert_eq!(env_value(&env, "NCCL_SOCKET_IFNAME").as_deref(), Some("lo"));
        assert_eq!(env_value(&env, "NCCL_IB_DISABLE").as_deref(), Some("1"));
        assert_eq!(env_value(&env, "NCCL_NET_PLUGIN").as_deref(), Some("none"));
        assert_eq!(env_value(&env, "HEIRLOOM_NCCL_TRACE").as_deref(), Some("1"));
    }

    #[test]
    fn nccl_launcher_env_preserves_heirloom_specific_overrides() {
        let env = launcher_env_from_lookup(|name| match name {
            "HEIRLOOM_NCCL_SOCKET_IFNAME" => Some("eth0".to_string()),
            "HEIRLOOM_NCCL_IB_DISABLE" => Some("0".to_string()),
            "HEIRLOOM_NCCL_NET_PLUGIN" => Some("none".to_string()),
            "HEIRLOOM_NCCL_CUMEM_ENABLE" => Some("0".to_string()),
            "HEIRLOOM_NCCL_CUMEM_HOST_ENABLE" => Some("0".to_string()),
            "HEIRLOOM_NCCL_P2P_DISABLE" => Some("1".to_string()),
            "HEIRLOOM_NCCL_P2P_LEVEL" => Some("NVL".to_string()),
            "HEIRLOOM_NCCL_TRACE" => Some("1".to_string()),
            _ => None,
        });

        assert_eq!(
            env_value(&env, "NCCL_SOCKET_IFNAME").as_deref(),
            Some("eth0")
        );
        assert_eq!(env_value(&env, "NCCL_IB_DISABLE").as_deref(), Some("0"));
        assert_eq!(env_value(&env, "NCCL_NET_PLUGIN").as_deref(), Some("none"));
        assert_eq!(env_value(&env, "NCCL_CUMEM_ENABLE").as_deref(), Some("0"));
        assert_eq!(
            env_value(&env, "NCCL_CUMEM_HOST_ENABLE").as_deref(),
            Some("0")
        );
        assert_eq!(env_value(&env, "NCCL_P2P_DISABLE").as_deref(), Some("1"));
        assert_eq!(env_value(&env, "NCCL_P2P_LEVEL").as_deref(), Some("NVL"));
        assert_eq!(env_value(&env, "HEIRLOOM_NCCL_TRACE").as_deref(), Some("1"));
    }

    #[test]
    fn ddp_step_checksum_drifts_reports_synced_ranks() {
        let reports = vec![
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0},
                    {"step": 2, "parameter_checksum_sum": 11.0, "parameter_checksum_sumsq": 31.0}
                ]
            }),
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0},
                    {"step": 2, "parameter_checksum_sum": 11.0, "parameter_checksum_sumsq": 31.0}
                ]
            }),
        ];

        let drifts = ddp_step_checksum_drifts(&reports).unwrap();

        assert_eq!(drifts.len(), 2);
        assert_eq!(
            max_step_checksum_error(&drifts, "parameter_checksum_sum_max_error"),
            0.0
        );
        assert_eq!(
            max_step_checksum_error(&drifts, "parameter_checksum_sumsq_max_error"),
            0.0
        );
    }

    #[test]
    fn ddp_step_checksum_drifts_detects_rank_drift() {
        let reports = vec![
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0}
                ]
            }),
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.25, "parameter_checksum_sumsq": 30.5}
                ]
            }),
        ];

        let drifts = ddp_step_checksum_drifts(&reports).unwrap();

        assert_eq!(
            max_step_checksum_error(&drifts, "parameter_checksum_sum_max_error"),
            0.25
        );
        assert_eq!(
            max_step_checksum_error(&drifts, "parameter_checksum_sumsq_max_error"),
            0.5
        );
    }

    #[test]
    fn ddp_step_checksum_drifts_rejects_misaligned_steps() {
        let reports = vec![
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0}
                ]
            }),
            serde_json::json!({
                "step_checksums": [
                    {"step": 2, "parameter_checksum_sum": 10.0, "parameter_checksum_sumsq": 30.0}
                ]
            }),
        ];

        let err = ddp_step_checksum_drifts(&reports).unwrap_err();

        assert!(format!("{err}").contains("does not match rank 0 step"));
    }

    #[test]
    fn ddp_memory_step_checksum_drifts_report_memory_table_sync() {
        let reports = vec![
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "memory_table_checksum_sum": 4.0, "memory_table_checksum_sumsq": 16.0},
                    {"step": 2, "memory_table_checksum_sum": 5.0, "memory_table_checksum_sumsq": 25.0}
                ]
            }),
            serde_json::json!({
                "step_checksums": [
                    {"step": 1, "memory_table_checksum_sum": 4.0, "memory_table_checksum_sumsq": 16.0},
                    {"step": 2, "memory_table_checksum_sum": 5.125, "memory_table_checksum_sumsq": 25.5}
                ]
            }),
        ];

        let drifts = ddp_memory_step_checksum_drifts(&reports).unwrap();

        assert_eq!(drifts.len(), 2);
        assert_eq!(
            max_step_checksum_error(&drifts, "memory_table_checksum_sum_max_error"),
            0.125
        );
        assert_eq!(
            max_step_checksum_error(&drifts, "memory_table_checksum_sumsq_max_error"),
            0.5
        );
        assert_eq!(drifts[0]["memory_table_checksum_sum_max_error"], 0.0);
    }
}
