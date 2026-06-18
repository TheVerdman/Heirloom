use heirloom::checkpoint::{load_lm_checkpoint, save_lm_checkpoint_with_dataset_state};
use heirloom::data::{
    build_qb_native_corpus_blend_manifest, prepare_lm_data, prepare_lm_data_binary_shards,
    read_token_shard, write_corpus_blend_manifest, write_token_shard, PreparedDataOptions,
    PreparedTokenData, TokenDataSplit, TokenDataset, TokenDatasetState, TokenShardDType,
};
use heirloom::nn::{
    AdamW, GenerationOptions, Module, Optimizer, TinyTransformerConfig, TinyTransformerLm,
};
use heirloom::rng::HeirloomRng;
use heirloom::tokenizer::{
    default_reserved_tokens, validate_reserved_tokens, BpeTokenizer, BpeTokenizerV2Options,
    BpeTrainingSample, BYTE_OFFSET, TOKENIZER_VERSION_V2,
};
use heirloom::{DType, Tensor};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, full actual={actual:?}"
        );
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "heirloom_transformer_{name}_{}_{}",
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
fn tokenizer_round_trips_and_dataset_batches_are_deterministic() {
    let tokenizer = BpeTokenizer::train("hello hello hello", 270).unwrap();
    let encoded = tokenizer.encode("hello", true, true);
    assert_eq!(tokenizer.decode(&encoded), "hello");
    assert!(tokenizer.vocab_size() > 259);
    assert_eq!(tokenizer.metadata().format.as_str(), "heirloom.byte_bpe");
    assert_eq!(tokenizer.metadata().vocab_size, tokenizer.vocab_size());

    let tokens = tokenizer.encode("hello hello hello hello", true, true);
    let mut a = TokenDataset::new(tokens.clone(), 4, 123).unwrap();
    let mut b = TokenDataset::new(tokens, 4, 123).unwrap();
    let (ax, ay) = a.next_batch(3).unwrap();
    let (bx, by) = b.next_batch(3).unwrap();
    assert_eq!(ax.data_i64().unwrap(), bx.data_i64().unwrap());
    assert_eq!(ay.data_i64().unwrap(), by.data_i64().unwrap());

    let sharded_tokens = (0..100).collect::<Vec<_>>();
    let (rank0_x, rank0_y) =
        TokenDataset::deterministic_sharded_batch(&sharded_tokens, 4, 2, 777, 5, 0, 2).unwrap();
    let (rank0_x_again, rank0_y_again) =
        TokenDataset::deterministic_sharded_batch(&sharded_tokens, 4, 2, 777, 5, 0, 2).unwrap();
    let (rank1_x, _) =
        TokenDataset::deterministic_sharded_batch(&sharded_tokens, 4, 2, 777, 5, 1, 2).unwrap();
    assert_eq!(
        rank0_x.data_i64().unwrap(),
        rank0_x_again.data_i64().unwrap()
    );
    assert_eq!(
        rank0_y.data_i64().unwrap(),
        rank0_y_again.data_i64().unwrap()
    );
    assert_ne!(rank0_x.data_i64().unwrap(), rank1_x.data_i64().unwrap());
}

#[test]
fn tokenizer_load_accepts_legacy_json_and_validates_metadata_files() {
    let dir = temp_dir("tokenizer_legacy");
    let path = dir.join("tokenizer.json");
    let legacy_path = dir.join("legacy-tokenizer.json");
    let tokenizer = BpeTokenizer::train("legacy legacy tokenizer", 270).unwrap();
    tokenizer.save(&path).unwrap();

    let mut value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("metadata");
    fs::write(
        &legacy_path,
        serde_json::to_string_pretty(&value).unwrap() + "\n",
    )
    .unwrap();

    let loaded = BpeTokenizer::load(&legacy_path).unwrap();
    assert_eq!(
        loaded.decode(&loaded.encode("legacy", true, true)),
        "legacy"
    );
    assert_eq!(loaded.metadata().training_hash.as_str(), "legacy");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tokenizer_v2_reserved_tokens_are_atomic_and_bytes_round_trip() {
    let reserved = default_reserved_tokens();
    validate_reserved_tokens(&reserved).unwrap();
    let bytes = b"ab ac <|user|> hello\x00\xff <|assistant|> answer".to_vec();
    let samples = vec![BpeTrainingSample {
        source_id: "fixture".to_string(),
        bytes: bytes.clone(),
        weight: 1,
    }];
    let tokenizer = BpeTokenizer::train_v2(
        &samples,
        reserved.clone(),
        BpeTokenizerV2Options {
            tokenizer_id: "fixture-tokenizer-v2".to_string(),
            vocab_size: 512,
            sample_bytes: bytes.len() as u64,
            seed: 7,
            memory_limit_bytes: Some(1024 * 1024),
            source_blend_hash: "blend".to_string(),
            sample_manifest_hash: "sample".to_string(),
            digit_isolation: true,
            require_exact_vocab: false,
        },
    )
    .unwrap();
    assert_eq!(tokenizer.metadata().version, TOKENIZER_VERSION_V2);
    assert_eq!(tokenizer.reserved_tokens().len(), reserved.len());
    let user_id = reserved
        .iter()
        .find(|token| token.token == "<|user|>")
        .unwrap()
        .id;
    let encoded = tokenizer.encode_bytes(&bytes, false, false);
    assert!(encoded.contains(&user_id));
    assert_eq!(tokenizer.decode_bytes(&encoded), bytes);

    let tokenizer_again = BpeTokenizer::train_v2(
        &samples,
        reserved,
        BpeTokenizerV2Options {
            tokenizer_id: "fixture-tokenizer-v2".to_string(),
            vocab_size: 512,
            sample_bytes: bytes.len() as u64,
            seed: 7,
            memory_limit_bytes: Some(1024 * 1024),
            source_blend_hash: "blend".to_string(),
            sample_manifest_hash: "sample".to_string(),
            digit_isolation: true,
            require_exact_vocab: false,
        },
    )
    .unwrap();
    assert_eq!(
        tokenizer.fingerprint().unwrap(),
        tokenizer_again.fingerprint().unwrap()
    );

    let tie_tokenizer = BpeTokenizer::train_v2(
        &[BpeTrainingSample {
            source_id: "tie".to_string(),
            bytes: b"ab ac".to_vec(),
            weight: 1,
        }],
        default_reserved_tokens(),
        BpeTokenizerV2Options {
            tokenizer_id: "tie-tokenizer-v2".to_string(),
            vocab_size: 512,
            require_exact_vocab: false,
            ..BpeTokenizerV2Options::default()
        },
    )
    .unwrap();
    let first_merge = tie_tokenizer.merges().first().unwrap();
    assert_eq!(first_merge.left, BYTE_OFFSET + b'a' as usize);
    assert_eq!(first_merge.right, BYTE_OFFSET + b'b' as usize);
}

#[test]
fn tokenizer_v2_digit_isolation_blocks_digit_merges() {
    let samples = vec![BpeTrainingSample {
        source_id: "math-fixture".to_string(),
        bytes: b"20 <= 31 <= 32. 3) and 2) should segment consistently. x^2 + 10.5 = 42".repeat(32),
        weight: 1,
    }];
    let tokenizer = BpeTokenizer::train_v2(
        &samples,
        default_reserved_tokens(),
        BpeTokenizerV2Options {
            tokenizer_id: "digit-isolated-tokenizer-v2".to_string(),
            vocab_size: 768,
            sample_bytes: samples[0].bytes.len() as u64,
            seed: 11,
            memory_limit_bytes: Some(1024 * 1024),
            source_blend_hash: "blend".to_string(),
            sample_manifest_hash: "sample".to_string(),
            digit_isolation: true,
            require_exact_vocab: false,
        },
    )
    .unwrap();
    assert!(tokenizer.digit_isolation_enabled());
    assert_eq!(tokenizer.reserved_tokens().len(), 128);
    for merge in tokenizer.merges() {
        let piece = tokenizer.decode_bytes(&[merge.id]);
        assert!(
            !piece.iter().any(|byte| byte.is_ascii_digit()),
            "merge {} unexpectedly contains digit payload {:?}",
            merge.id,
            String::from_utf8_lossy(&piece)
        );
    }
    let encoded = tokenizer.encode("20 <= 31 <= 32. 3) 2)", false, false);
    let digit_ids = (b'0'..=b'9')
        .map(|byte| BYTE_OFFSET + byte as usize)
        .collect::<std::collections::BTreeSet<_>>();
    for token in encoded {
        let piece = tokenizer.decode_bytes(&[token]);
        if piece.iter().any(|byte| byte.is_ascii_digit()) {
            assert!(
                digit_ids.contains(&token),
                "non-byte token {} contains digit payload {:?}",
                token,
                String::from_utf8_lossy(&piece)
            );
        }
    }
}

#[test]
fn tokenizer_v2_rejects_duplicate_reserved_tokens() {
    let bytes = b"hello tokenizer".to_vec();
    let mut reserved = default_reserved_tokens();
    reserved[1].token = reserved[0].token.clone();
    let err = BpeTokenizer::train_v2(
        &[BpeTrainingSample {
            source_id: "fixture".to_string(),
            bytes,
            weight: 1,
        }],
        reserved,
        BpeTokenizerV2Options {
            vocab_size: 512,
            ..BpeTokenizerV2Options::default()
        },
    )
    .unwrap_err();
    assert!(format!("{err}").contains("duplicate reserved token string"));
}

#[test]
fn prepared_data_manifest_round_trips_and_dataset_state_resumes() {
    let dir = temp_dir("prepared_data");
    let input_path = dir.join("corpus.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let prepared_dir = dir.join("prepared");
    let text = "the cat sat. the cat ran. the dog sat. the dog ran. ".repeat(4);
    fs::write(&input_path, &text).unwrap();

    let tokenizer = BpeTokenizer::train(&text, 280).unwrap();
    tokenizer.save(&tokenizer_path).unwrap();
    let prepared = prepare_lm_data(
        &input_path,
        &tokenizer,
        &tokenizer_path,
        &prepared_dir,
        PreparedDataOptions {
            valid_fraction: 0.25,
            max_bytes: None,
            shard_tokens: None,
        },
    )
    .unwrap();

    let loaded = PreparedTokenData::load(&prepared.manifest_path).unwrap();
    assert_eq!(loaded.manifest.version, 1);
    assert!(loaded.manifest.train_tokens > loaded.manifest.valid_tokens);
    assert_eq!(
        loaded.load_tokenizer().unwrap().fingerprint().unwrap(),
        tokenizer.fingerprint().unwrap()
    );

    let train_tokens = loaded.train_tokens().unwrap();
    assert_eq!(
        loaded.valid_tokens().unwrap().len(),
        loaded.manifest.valid_tokens
    );

    let mut uninterrupted = TokenDataset::new(train_tokens.clone(), 4, 123).unwrap();
    let _ = uninterrupted.next_batch(2).unwrap();
    let state = uninterrupted.state();
    assert_eq!(state.batches_seen, 1);
    let expected = uninterrupted.next_batch(2).unwrap();

    let mut resumed = TokenDataset::with_state(train_tokens, 4, state).unwrap();
    let actual = resumed.next_batch(2).unwrap();
    assert_eq!(expected.0.data_i64().unwrap(), actual.0.data_i64().unwrap());
    assert_eq!(expected.1.data_i64().unwrap(), actual.1.data_i64().unwrap());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn binary_token_shards_round_trip_and_validate_payloads() {
    let dir = temp_dir("token_shard");
    let payload_path = dir.join("train.tokens.bin");
    let metadata_path = dir.join("train.tokens.json");
    let tokens = vec![1, 2, 3, 65535, 42, 0];

    let metadata = write_token_shard(&payload_path, &metadata_path, &tokens).unwrap();
    assert_eq!(metadata.dtype, TokenShardDType::U16);
    assert_eq!(metadata.token_count, tokens.len());
    assert_eq!(metadata.payload_bytes, tokens.len() * 2);
    assert_eq!(read_token_shard(&metadata_path).unwrap(), tokens);

    let wide_payload_path = dir.join("wide.tokens.bin");
    let wide_metadata_path = dir.join("wide.tokens.json");
    let wide_tokens = vec![0, 65536, 1_000_000];
    let wide_metadata =
        write_token_shard(&wide_payload_path, &wide_metadata_path, &wide_tokens).unwrap();
    assert_eq!(wide_metadata.dtype, TokenShardDType::U32);
    assert_eq!(wide_metadata.payload_bytes, wide_tokens.len() * 4);
    assert_eq!(read_token_shard(&wide_metadata_path).unwrap(), wide_tokens);

    fs::write(&payload_path, [0u8, 1u8]).unwrap();
    assert!(read_token_shard(&metadata_path).is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn prepared_binary_shard_manifest_streams_without_materializing_tokens() {
    let dir = temp_dir("prepared_binary_shards");
    let input_a = dir.join("TinyStories-valid.txt");
    let input_b = dir.join("qb-traces.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let prepared_dir = dir.join("prepared-v2");
    let text_a = "the cat sat. the cat ran. the dog sat. the dog ran. ".repeat(6);
    let text_b = [
        "STATE known local report DELTA new time ACTION read_file EVIDENCE report ANSWER passed.",
        "STATE known user asks current time DELTA elapsed days ACTION direct ANSWER use envelope.",
    ]
    .join("\n");
    fs::write(&input_a, &text_a).unwrap();
    fs::write(&input_b, &text_b).unwrap();

    let tokenizer = BpeTokenizer::train(&(text_a.clone() + &text_b), 300).unwrap();
    tokenizer.save(&tokenizer_path).unwrap();
    let prepared = prepare_lm_data_binary_shards(
        &[input_a.clone(), input_b.clone()],
        &tokenizer,
        &tokenizer_path,
        &prepared_dir,
        PreparedDataOptions {
            valid_fraction: 0.2,
            max_bytes: None,
            shard_tokens: Some(32),
        },
    )
    .unwrap();

    let loaded = PreparedTokenData::load(&prepared.manifest_path).unwrap();
    assert_eq!(loaded.manifest.version, 2);
    assert_eq!(loaded.manifest.storage, "binary_shards");
    assert_eq!(loaded.manifest.sources.len(), 2);
    assert!(loaded.manifest.train_shards.len() > 1);
    assert!(!loaded.manifest.valid_shards.is_empty());
    assert_eq!(
        loaded
            .manifest
            .train_shards
            .iter()
            .map(|shard| shard.tokens)
            .sum::<usize>(),
        loaded.manifest.train_tokens
    );

    let materialized = loaded.train_tokens().unwrap();
    let state = TokenDatasetState::from_seed(99);
    let mut in_memory = TokenDataset::with_state(materialized, 8, state).unwrap();
    let mut streaming = loaded
        .streaming_dataset(TokenDataSplit::Train, 8, state)
        .unwrap();
    let expected = in_memory.next_batch(3).unwrap();
    let actual = streaming.next_batch(3).unwrap();
    assert_eq!(expected.0.data_i64().unwrap(), actual.0.data_i64().unwrap());
    assert_eq!(expected.1.data_i64().unwrap(), actual.1.data_i64().unwrap());
    assert!(!streaming.loader_stats().bytes_read.eq(&0));

    let mut stream_a = loaded
        .streaming_dataset(TokenDataSplit::Train, 8, TokenDatasetState::from_seed(0))
        .unwrap();
    let mut stream_b = loaded
        .streaming_dataset(TokenDataSplit::Train, 8, TokenDatasetState::from_seed(0))
        .unwrap();
    let rank0 = stream_a
        .deterministic_sharded_batch(2, 123, 5, 0, 2)
        .unwrap();
    let rank0_again = stream_b
        .deterministic_sharded_batch(2, 123, 5, 0, 2)
        .unwrap();
    let rank1 = stream_b
        .deterministic_sharded_batch(2, 123, 5, 1, 2)
        .unwrap();
    assert_eq!(
        rank0.0.data_i64().unwrap(),
        rank0_again.0.data_i64().unwrap()
    );
    assert_ne!(rank0.0.data_i64().unwrap(), rank1.0.data_i64().unwrap());

    let mut uninterrupted = loaded
        .streaming_dataset(TokenDataSplit::Train, 8, TokenDatasetState::from_seed(2026))
        .unwrap();
    let _ = uninterrupted.next_batch(2).unwrap();
    let resume_state = uninterrupted.state();
    let expected = uninterrupted.next_batch(2).unwrap();
    let mut resumed = loaded
        .streaming_dataset(TokenDataSplit::Train, 8, resume_state)
        .unwrap();
    let actual = resumed.next_batch(2).unwrap();
    assert_eq!(expected.0.data_i64().unwrap(), actual.0.data_i64().unwrap());
    assert_eq!(expected.1.data_i64().unwrap(), actual.1.data_i64().unwrap());

    let rejected = prepare_lm_data_binary_shards(
        &[input_a, input_b],
        &tokenizer,
        &tokenizer_path,
        dir.join("rejected"),
        PreparedDataOptions {
            valid_fraction: 0.2,
            max_bytes: Some(16),
            shard_tokens: None,
        },
    );
    assert!(rejected.is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn corpus_blend_manifest_summarizes_qb_trace_sources() {
    let dir = temp_dir("corpus_blend");
    let qb_root = dir.join("VECL-QB-data");
    let source_dir = qb_root.join("synthetic").join("v1-hard");
    let out = dir.join("blend.json");
    fs::create_dir_all(&source_dir).unwrap();
    let corpus = [
        r#"{"domain":"sympy","split":"train","target_text":"{}"}"#,
        r#"{"domain":"timesfm","split":"heldout","target_text":"answer"}"#,
    ]
    .join("\n")
        + "\n";
    fs::write(source_dir.join("corpus.jsonl"), corpus).unwrap();
    fs::write(
        source_dir.join("metadata.json"),
        r#"{
  "actual_count": 2,
  "dataset_hash": "fixture-hash",
  "validation": {
    "split_counts": {"train": 1, "heldout": 1},
    "domain_counts": {"sympy": 1, "timesfm": 1},
    "category_counts": {"sympy_tool_call": 1, "timesfm_inventory_final_answer": 1},
    "task_kind_counts": {"tool_call_json": 1, "final_answer": 1},
    "executable_records": 2,
    "supervised_records": 2
  }
}"#,
    )
    .unwrap();

    let manifest =
        build_qb_native_corpus_blend_manifest("fixture-blend", 32768, Some(&qb_root)).unwrap();
    write_corpus_blend_manifest(&manifest, &out).unwrap();
    let written: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(written["format"], "heirloom.corpus_blend");
    assert_eq!(written["version"], 1);
    assert_eq!(written["tokenizer_target_vocab_size"], 32768);
    assert_eq!(written["planned_source_count"], 4);
    assert_eq!(written["local_source_count"], 1);
    assert_eq!(written["local_record_count"], 2);

    let qb_source = written["sources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|source| source["source_id"] == "vecl_qb.synthetic.v1-hard")
        .unwrap();
    assert_eq!(qb_source["status"], "available");
    assert_eq!(qb_source["license_status"], "internal_synthetic");
    assert_eq!(qb_source["local_records"], 2);
    assert_eq!(qb_source["split_counts"]["train"], 1);
    assert_eq!(qb_source["domain_counts"]["sympy"], 1);
    assert!(qb_source["content_hash"].as_str().unwrap().len() > 10);

    let dolma = written["sources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|source| source["source_id"] == "allenai.dolma.v1_7")
        .unwrap();
    assert_eq!(dolma["license_status"], "odc_by_internal_attribution");
    assert_eq!(dolma["include_in_tokenizer_training"], true);
    assert_eq!(dolma["sampling_weight"], 0.35);

    let nemotron = written["sources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|source| source["source_id"] == "nvidia.nemotron_cc.high_actual")
        .unwrap();
    assert_eq!(
        nemotron["license_status"],
        "nvidia_data_agreement_internal_training"
    );

    let olmo = written["sources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|source| source["source_id"] == "allenai.dolma3_dolmino_mix-100B-1125")
        .unwrap();
    assert_eq!(olmo["status"], "planned");
    assert_eq!(olmo["license_status"], "odc_by_internal_attribution");
    assert_eq!(olmo["sampling_weight"], 0.20);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tokenizer_train_corpus_cli_emits_v2_artifacts_and_fertility_report() {
    let exe = option_env!("CARGO_BIN_EXE_heirloom").unwrap_or("target/debug/heirloom");
    let dir = temp_dir("tokenizer_train_corpus_cli");
    let corpus_path = dir.join("qb-v1-hard.jsonl");
    let blend_path = dir.join("corpus-blend.json");
    let work_dir = dir.join("work");
    let tokenizer_path = dir.join("tokenizer-v2.json");
    let train_report = dir.join("tokenizer-train-report.json");
    let fertility_report = dir.join("tokenizer-fertility-report.json");
    let bench_report = dir.join("tokenizer-bench-report.json");
    let corpus = [
        serde_json::json!({
            "source_id": "fixture-tool",
            "prompt": "Use the local fixture tool for alpha.",
            "target_text": "{\"specialist_id\":\"fixture\",\"confidence\":0.9}",
            "task_kind": "tool_call_json"
        })
        .to_string(),
        serde_json::json!({
            "source_id": "fixture-memory",
            "prompt": "Record memory trace beta.",
            "target_text": "The trace is supported by local evidence.",
            "task_kind": "final_answer"
        })
        .to_string(),
    ]
    .join("\n")
        + "\n";
    fs::write(&corpus_path, corpus).unwrap();
    let blend = serde_json::json!({
        "format": "heirloom.corpus_blend",
        "version": 1,
        "blend_id": "fixture-tokenizer-blend",
        "tokenizer_target_vocab_size": 512,
        "tokenizer_family": "heirloom.byte_bpe",
        "sources": [{
            "source_id": "vecl_qb.synthetic.v1-hard",
            "display_name": "fixture QB v1 hard",
            "kind": "local_qb_trace_corpus",
            "status": "available",
            "role": "tool_routing_memory_trace_supervision",
            "source_url": null,
            "path": corpus_path.to_str().unwrap(),
            "metadata_path": null,
            "data_format": "jsonl",
            "license": "project-internal synthetic corpus; no production user data",
            "license_url": null,
            "license_status": "internal_synthetic",
            "provenance": ["VECL-QB", "synthetic_tool_use", "v1-hard"],
            "sampling_weight": 1.0,
            "include_in_tokenizer_training": true,
            "include_in_pretraining": true,
            "include_in_memory_trace_training": true,
            "synthetic": true,
            "local_bytes": fs::metadata(&corpus_path).unwrap().len() as usize,
            "local_records": 2,
            "content_hash": null,
            "metadata_hash": null,
            "split_counts": {"train": 2},
            "domain_counts": {"fixture": 2},
            "category_counts": {"fixture": 2},
            "task_kind_counts": {"tool_call_json": 1, "final_answer": 1},
            "extra": {}
        }],
        "local_source_count": 1,
        "planned_source_count": 0,
        "local_record_count": 2,
        "local_bytes": fs::metadata(&corpus_path).unwrap().len() as usize,
        "notes": []
    });
    fs::write(&blend_path, serde_json::to_string_pretty(&blend).unwrap()).unwrap();

    let status = Command::new(exe)
        .args([
            "tokenizer",
            "train-corpus",
            "--corpus-blend",
            blend_path.to_str().unwrap(),
            "--out",
            tokenizer_path.to_str().unwrap(),
            "--work-dir",
            work_dir.to_str().unwrap(),
            "--vocab-size",
            "512",
            "--sample-bytes",
            "512",
            "--seed",
            "42",
            "--report",
            train_report.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let tokenizer = BpeTokenizer::load(&tokenizer_path).unwrap();
    assert_eq!(tokenizer.metadata().version, TOKENIZER_VERSION_V2);
    assert!(!tokenizer.reserved_tokens().is_empty());
    assert_eq!(
        tokenizer.decode(&tokenizer.encode("<|user|>hello", false, false)),
        "<|user|>hello"
    );
    let train: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&train_report).unwrap()).unwrap();
    assert_eq!(train["tokenizer"]["version"], TOKENIZER_VERSION_V2);
    assert_eq!(train["hard_path"]["external_tokenizer_dependency"], false);
    assert!(work_dir.join("tokenizer-sample-manifest.json").exists());

    let status = Command::new(exe)
        .args(["tokenizer", "validate", tokenizer_path.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());

    let status = Command::new(exe)
        .args([
            "tokenizer",
            "fertility",
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--corpus-blend",
            blend_path.to_str().unwrap(),
            "--report",
            fertility_report.to_str().unwrap(),
            "--sample-bytes",
            "512",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let fertility: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&fertility_report).unwrap()).unwrap();
    assert_eq!(fertility["status"], "passed");
    assert_eq!(fertility["tokenizer"]["version"], TOKENIZER_VERSION_V2);
    assert!(fertility["sources"][0]["tokens_per_byte"].as_f64().unwrap() > 0.0);

    let status = Command::new(exe)
        .args([
            "tokenizer",
            "bench-encode",
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--input",
            corpus_path.to_str().unwrap(),
            "--report",
            bench_report.to_str().unwrap(),
            "--max-bytes",
            "512",
            "--iterations",
            "2",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let bench: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&bench_report).unwrap()).unwrap();
    assert_eq!(bench["status"], "passed");
    assert_eq!(bench["tokenizer"]["version"], TOKENIZER_VERSION_V2);
    assert_eq!(bench["settings"]["iterations"], 2);
    assert!(bench["throughput"]["tokens_per_second"].as_f64().unwrap() > 0.0);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tokenizer_train_corpus_preserves_general_jsonl_text_samples() {
    let exe = option_env!("CARGO_BIN_EXE_heirloom").unwrap_or("target/debug/heirloom");
    let dir = temp_dir("tokenizer_train_corpus_general_jsonl");
    let corpus_path = dir.join("general.jsonl");
    let blend_path = dir.join("corpus-blend.json");
    let work_dir = dir.join("work");
    let tokenizer_path = dir.join("tokenizer-v2.json");
    let train_report = dir.join("tokenizer-train-report.json");
    let long_text = [
        "This general corpus record must be preserved as document text for tokenizer sampling.",
        "It contains ordinary prose, numbers 12345, punctuation, code-like tokens fn main,",
        "and enough repeated but varied language to exceed the tiny sample quota.",
    ]
    .join(" ")
    .repeat(16);
    let corpus = serde_json::json!({
        "text": long_text,
        "source_file": "fixture/general.jsonl",
        "source_line": 1
    })
    .to_string()
        + "\n";
    fs::write(&corpus_path, corpus).unwrap();
    let blend = serde_json::json!({
        "format": "heirloom.corpus_blend",
        "version": 1,
        "blend_id": "fixture-general-tokenizer-blend",
        "tokenizer_target_vocab_size": 512,
        "tokenizer_family": "heirloom.byte_bpe",
        "sources": [{
            "source_id": "allenai.dolma.v1_7",
            "display_name": "fixture general jsonl",
            "kind": "materialized_pretraining_corpus",
            "status": "available",
            "role": "general_language_backbone",
            "source_url": null,
            "path": corpus_path.to_str().unwrap(),
            "metadata_path": null,
            "data_format": "jsonl",
            "license": "fixture",
            "license_url": null,
            "license_status": "odc_by_internal_attribution",
            "provenance": ["fixture"],
            "sampling_weight": 1.0,
            "include_in_tokenizer_training": true,
            "include_in_pretraining": true,
            "include_in_memory_trace_training": false,
            "synthetic": false,
            "local_bytes": fs::metadata(&corpus_path).unwrap().len() as usize,
            "local_records": 1,
            "content_hash": null,
            "metadata_hash": null,
            "split_counts": {},
            "domain_counts": {},
            "category_counts": {},
            "task_kind_counts": {},
            "extra": {}
        }],
        "local_source_count": 1,
        "planned_source_count": 0,
        "local_record_count": 1,
        "local_bytes": fs::metadata(&corpus_path).unwrap().len() as usize,
        "notes": []
    });
    fs::write(&blend_path, serde_json::to_string_pretty(&blend).unwrap()).unwrap();

    let status = Command::new(exe)
        .args([
            "tokenizer",
            "train-corpus",
            "--corpus-blend",
            blend_path.to_str().unwrap(),
            "--out",
            tokenizer_path.to_str().unwrap(),
            "--work-dir",
            work_dir.to_str().unwrap(),
            "--vocab-size",
            "512",
            "--sample-bytes",
            "1024",
            "--seed",
            "42",
            "--report",
            train_report.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let sample_manifest_path = work_dir.join("tokenizer-sample-manifest.json");
    let sample_manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(sample_manifest_path).unwrap()).unwrap();
    let source = &sample_manifest["sources"][0];
    assert!(source["sampled_bytes"].as_u64().unwrap() >= 1024);
    assert_eq!(source["exhausted"], false);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn data_materialize_blend_cli_emits_prepared_manifest_v2() {
    let exe = option_env!("CARGO_BIN_EXE_heirloom").unwrap_or("target/debug/heirloom");
    let dir = temp_dir("data_materialize_blend_cli");
    let general_path = dir.join("general.jsonl");
    let qb_path = dir.join("qb-hard.jsonl");
    let tokenizer_path = dir.join("tokenizer.json");
    let blend_path = dir.join("corpus-blend.json");
    let out_dir = dir.join("materialized");
    let checkpoint_dir = out_dir.join("checkpoints");
    let general_docs = [
        "A careful derivation explains matrix multiplication, gradients, and optimizer states.",
        "A short instruction answer compares source evidence and final claims for review.",
        "A technical note describes sharded token datasets and deterministic replay metadata.",
        "A math passage proves a small identity and checks each algebraic transformation.",
    ];
    let qb_docs = [
        serde_json::json!({
            "source_id": "fixture-tool",
            "prompt": "Read artifact claim qb-17 and cite the supporting evidence.",
            "target_text": "{\"tool\":\"memory_read\",\"claim_ref\":\"qb-17\",\"confidence\":0.92}",
            "task_kind": "tool_call_json"
        })
        .to_string(),
        serde_json::json!({
            "source_id": "fixture-memory",
            "prompt": "Write a compact memory update for a verifier trace.",
            "target_text": "The memory write links artifact qb-18 to the cited local trace.",
            "task_kind": "final_answer"
        })
        .to_string(),
    ];
    let general_jsonl = general_docs
        .iter()
        .map(|text| serde_json::json!({"text": text}).to_string())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&general_path, &general_jsonl).unwrap();
    fs::write(&qb_path, qb_docs.join("\n") + "\n").unwrap();
    let tokenizer_training_text = format!(
        "{}\n{}",
        general_jsonl,
        fs::read_to_string(&qb_path).unwrap()
    );
    let tokenizer = BpeTokenizer::train(&tokenizer_training_text, 320).unwrap();
    tokenizer.save(&tokenizer_path).unwrap();
    let blend = serde_json::json!({
        "format": "heirloom.corpus_blend",
        "version": 1,
        "blend_id": "fixture-materializer-blend",
        "tokenizer_target_vocab_size": 320,
        "tokenizer_family": "heirloom.byte_bpe",
        "sources": [
            {
                "source_id": "fixture.general.approved",
                "display_name": "Fixture approved general source",
                "kind": "local_text_jsonl",
                "status": "available",
                "role": "general_instruction_reasoning",
                "source_url": null,
                "path": general_path.to_str().unwrap(),
                "metadata_path": null,
                "data_format": "jsonl",
                "license": "fixture approved",
                "license_url": null,
                "license_status": "source_terms_verified",
                "provenance": ["fixture", "general"],
                "sampling_weight": 0.7,
                "include_in_tokenizer_training": true,
                "include_in_pretraining": true,
                "include_in_memory_trace_training": false,
                "synthetic": true,
                "local_bytes": fs::metadata(&general_path).unwrap().len() as usize,
                "local_records": general_docs.len(),
                "content_hash": null,
                "metadata_hash": null,
                "split_counts": {"train": general_docs.len()},
                "domain_counts": {"general": general_docs.len()},
                "category_counts": {"fixture": general_docs.len()},
                "task_kind_counts": {},
                "extra": {}
            },
            {
                "source_id": "vecl_qb.fixture.v1-hard",
                "display_name": "Fixture QB hard source",
                "kind": "local_qb_trace_corpus",
                "status": "available",
                "role": "tool_routing_memory_trace_supervision",
                "source_url": null,
                "path": qb_path.to_str().unwrap(),
                "metadata_path": null,
                "data_format": "jsonl",
                "license": "project-internal synthetic corpus",
                "license_url": null,
                "license_status": "internal_synthetic",
                "provenance": ["VECL-QB", "fixture"],
                "sampling_weight": 0.3,
                "include_in_tokenizer_training": true,
                "include_in_pretraining": true,
                "include_in_memory_trace_training": true,
                "synthetic": true,
                "local_bytes": fs::metadata(&qb_path).unwrap().len() as usize,
                "local_records": qb_docs.len(),
                "content_hash": null,
                "metadata_hash": null,
                "split_counts": {"train": qb_docs.len()},
                "domain_counts": {"qb": qb_docs.len()},
                "category_counts": {"trace": qb_docs.len()},
                "task_kind_counts": {"tool_call_json": 1, "final_answer": 1},
                "extra": {}
            }
        ],
        "local_source_count": 2,
        "planned_source_count": 0,
        "local_record_count": general_docs.len() + qb_docs.len(),
        "local_bytes": (fs::metadata(&general_path).unwrap().len() + fs::metadata(&qb_path).unwrap().len()) as usize,
        "notes": []
    });
    fs::write(&blend_path, serde_json::to_string_pretty(&blend).unwrap()).unwrap();

    let status = Command::new(exe)
        .args([
            "data",
            "materialize-blend",
            "--corpus-blend",
            blend_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--target-tokens",
            "96",
            "--mode",
            "sample",
            "--valid-fraction",
            "0.5",
            "--shard-tokens",
            "24",
            "--text-shard-bytes",
            "256",
            "--candidate-text-mode",
            "rescan",
            "--candidate-retention-token-multiplier",
            "1.0",
            "--candidate-retention-min-docs",
            "1",
            "--candidate-prune-every",
            "1",
            "--progress-every-records",
            "1",
            "--progress-every-bytes",
            "64",
            "--checkpoint-dir",
            checkpoint_dir.to_str().unwrap(),
            "--checkpoint-every-records",
            "1",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let prepared_manifest_path = out_dir.join("prepared").join("manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&prepared_manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["format"], "heirloom.token_dataset");
    assert_eq!(manifest["version"], 2);
    assert_eq!(manifest["storage"], "binary_shards");
    assert!(manifest["train_tokens"].as_u64().unwrap() > 0);
    assert!(manifest["valid_tokens"].as_u64().unwrap() > 0);
    assert!(!manifest["train_shards"].as_array().unwrap().is_empty());
    assert!(!manifest["valid_shards"].as_array().unwrap().is_empty());

    let report: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out_dir.join("curation-report.json")).unwrap())
            .unwrap();
    assert_eq!(report["status"], "passed");
    assert_eq!(report["mode"], "sample");
    assert!(report["selected_docs"].as_u64().unwrap() >= 2);
    assert!(report["prepared_manifest"]
        .as_str()
        .unwrap()
        .ends_with("prepared/manifest.json"));
    assert!(report["timing"]["load_elapsed_ms"].is_number());
    assert!(report["timing"]["scan_elapsed_ms"].is_number());
    assert!(report["timing"]["selection_elapsed_ms"].is_number());
    assert!(report["timing"]["write_outputs_elapsed_ms"].is_number());
    assert!(report["timing"]["score_elapsed_ms"].is_number());
    assert!(report["timing"]["tokenizer_encode_elapsed_ms"].is_number());
    assert!(report["timing"]["hash_elapsed_ms"].is_number());
    assert!(report["timing"]["total_elapsed_ms"].is_number());
    assert!(report["throughput"]["scan_bytes_per_second"].is_number());
    assert!(report["throughput"]["scan_docs_per_second"].is_number());
    assert!(report["throughput"]["tokenizer_encode_tokens_per_second"].is_number());
    assert!(report["throughput"]["selected_tokens_per_second_end_to_end"].is_number());
    assert_eq!(report["candidate_retention"]["enabled"], true);
    assert_eq!(
        report["candidate_retention"]["exact_for_quota_when_multiplier_at_least_one"],
        true
    );
    assert_eq!(report["candidate_text"]["mode"], "rescan");
    assert_eq!(report["candidate_text"]["retained_in_scan"], false);
    assert_eq!(report["checkpoint"]["enabled"], true);
    assert_eq!(report["checkpoint"]["resume_checkpoint"], false);
    assert_eq!(
        report["checkpoint"]["source_checkpoint_count"]
            .as_u64()
            .unwrap(),
        2
    );
    assert_eq!(report["sizing"]["estimated_token_dtype"], "u16");
    assert_eq!(
        report["sizing"]["selected_tokens"].as_u64().unwrap(),
        report["selected_tokens"].as_u64().unwrap()
    );
    assert!(
        report["sizing"]["estimated_token_payload_bytes"]
            .as_u64()
            .unwrap()
            >= report["selected_tokens"].as_u64().unwrap() * 2
    );
    assert_eq!(
        report["output"]["written_docs"].as_u64().unwrap(),
        report["selected_docs"].as_u64().unwrap()
    );
    assert_eq!(
        report["output"]["written_tokens"].as_u64().unwrap(),
        manifest["train_tokens"].as_u64().unwrap() + manifest["valid_tokens"].as_u64().unwrap()
    );
    let sources = report["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 2);
    for source in sources {
        assert!(source["source_file_count"].as_u64().unwrap() >= 1);
        assert!(source["scan_elapsed_ms"].is_number());
        assert!(source["scan_bytes_per_second"].is_number());
        assert!(source["scan_docs_per_second"].is_number());
        assert!(source["score_elapsed_ms"].is_number());
        assert!(source["tokenizer_encode_elapsed_ms"].is_number());
        assert!(source["tokenizer_encode_tokens_per_second"].is_number());
        assert!(source["candidate_tokens"].as_u64().unwrap() > 0);
        assert!(
            source["retained_candidate_docs"].as_u64().unwrap()
                >= source["selected_docs"].as_u64().unwrap()
        );
        assert!(
            source["retained_candidate_tokens"].as_u64().unwrap()
                >= source["selected_tokens"].as_u64().unwrap()
        );
        assert!(source["candidate_retention_enabled"].as_bool().unwrap());
        assert!(source["candidate_retention_exact_for_quota"]
            .as_bool()
            .unwrap());
        assert_eq!(source["candidate_text_mode"], "rescan");
        assert_eq!(source["candidate_text_retained"], false);
        assert_eq!(source["retained_candidate_text_bytes"], 0);
        assert!(source["checkpoint_path"]
            .as_str()
            .unwrap()
            .contains("checkpoint"));
        assert!(source["checkpoint_completed"].as_bool().unwrap());
        assert!(source["checkpoint_saved_count"].as_u64().unwrap() >= 1);
        assert_eq!(
            source["written_docs"].as_u64().unwrap(),
            source["selected_docs"].as_u64().unwrap()
        );
        assert_eq!(
            source["written_tokens"].as_u64().unwrap(),
            source["selected_tokens"].as_u64().unwrap()
        );
    }
    assert!(out_dir.join("source-index.json").exists());
    assert!(out_dir.join("selected-docs.jsonl").exists());
    assert!(out_dir.join("tokenizer-sample-manifest.json").exists());
    assert!(checkpoint_dir.join("manifest.json").exists());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn data_materialize_blend_resumes_from_scan_checkpoint() {
    let exe = option_env!("CARGO_BIN_EXE_heirloom").unwrap_or("target/debug/heirloom");
    let dir = temp_dir("data_materialize_blend_resume_checkpoint");
    let corpus_path = dir.join("source.jsonl");
    let tokenizer_path = dir.join("tokenizer.json");
    let blend_path = dir.join("corpus-blend.json");
    let out_dir = dir.join("materialized");
    let checkpoint_dir = out_dir.join("checkpoints");
    let docs = [
        "{\"text\":\"Checkpoint rehearsal document one covers sparse memory routing and trace evidence.\"}",
        "{\"text\":\"Checkpoint rehearsal document two covers tokenizer curation and source governance.\"}",
        "{\"text\":\"Checkpoint rehearsal document three covers deterministic shard materialization.\"}",
        "{\"text\":\"Checkpoint rehearsal document four covers resumed scanning and selected outputs.\"}",
    ];
    fs::write(&corpus_path, docs.join("\n") + "\n").unwrap();
    let tokenizer = BpeTokenizer::train(&docs.join("\n"), 320).unwrap();
    tokenizer.save(&tokenizer_path).unwrap();
    let blend = serde_json::json!({
        "format": "heirloom.corpus_blend",
        "version": 1,
        "blend_id": "fixture-resume-materializer-blend",
        "tokenizer_target_vocab_size": 320,
        "tokenizer_family": "heirloom.byte_bpe",
        "sources": [
            {
                "source_id": "fixture.resume",
                "display_name": "Resume fixture",
                "kind": "local_fixture",
                "status": "available",
                "role": "resume_checkpoint_test",
                "source_url": null,
                "path": corpus_path.to_str().unwrap(),
                "metadata_path": null,
                "data_format": "jsonl",
                "license": "fixture",
                "license_url": null,
                "license_status": "approved",
                "provenance": ["fixture"],
                "sampling_weight": 1.0,
                "include_in_tokenizer_training": true,
                "include_in_pretraining": true,
                "include_in_memory_trace_training": false,
                "synthetic": true,
                "local_bytes": fs::metadata(&corpus_path).unwrap().len() as usize,
                "local_records": docs.len(),
                "content_hash": null,
                "metadata_hash": null,
                "split_counts": {"train": docs.len()},
                "domain_counts": {"fixture": docs.len()},
                "category_counts": {"resume": docs.len()},
                "task_kind_counts": {},
                "extra": {}
            }
        ],
        "local_source_count": 1,
        "planned_source_count": 0,
        "local_record_count": docs.len(),
        "local_bytes": fs::metadata(&corpus_path).unwrap().len() as usize,
        "notes": []
    });
    fs::write(&blend_path, serde_json::to_string_pretty(&blend).unwrap()).unwrap();

    let first = Command::new(exe)
        .args([
            "data",
            "materialize-blend",
            "--corpus-blend",
            blend_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--target-tokens",
            "200",
            "--mode",
            "sample",
            "--valid-fraction",
            "0.5",
            "--shard-tokens",
            "16",
            "--text-shard-bytes",
            "512",
            "--candidate-text-mode",
            "rescan",
            "--candidate-retention-token-multiplier",
            "1.0",
            "--candidate-retention-min-docs",
            "1",
            "--candidate-prune-every",
            "1",
            "--checkpoint-dir",
            checkpoint_dir.to_str().unwrap(),
            "--checkpoint-every-records",
            "1",
            "--checkpoint-stop-after-records",
            "2",
        ])
        .status()
        .unwrap();
    assert!(!first.success());
    assert!(checkpoint_dir.join("manifest.json").exists());
    let checkpoint_files = fs::read_dir(&checkpoint_dir)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".scan-checkpoint.json"))
        })
        .collect::<Vec<_>>();
    assert_eq!(checkpoint_files.len(), 1);

    let second = Command::new(exe)
        .args([
            "data",
            "materialize-blend",
            "--corpus-blend",
            blend_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--target-tokens",
            "200",
            "--mode",
            "sample",
            "--valid-fraction",
            "0.5",
            "--shard-tokens",
            "16",
            "--text-shard-bytes",
            "512",
            "--candidate-text-mode",
            "rescan",
            "--candidate-retention-token-multiplier",
            "1.0",
            "--candidate-retention-min-docs",
            "1",
            "--candidate-prune-every",
            "1",
            "--checkpoint-dir",
            checkpoint_dir.to_str().unwrap(),
            "--checkpoint-every-records",
            "1",
            "--resume-checkpoint",
        ])
        .status()
        .unwrap();
    assert!(second.success());

    let report: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out_dir.join("curation-report.json")).unwrap())
            .unwrap();
    assert_eq!(report["status"], "passed");
    assert_eq!(report["checkpoint"]["enabled"], true);
    assert_eq!(report["checkpoint"]["resume_checkpoint"], true);
    assert_eq!(report["sizing"]["estimated_token_dtype"], "u16");
    assert_eq!(
        report["output"]["written_tokens"].as_u64().unwrap(),
        report["sizing"]["selected_tokens"].as_u64().unwrap()
    );
    let source = &report["sources"].as_array().unwrap()[0];
    assert_eq!(source["checkpoint_loaded"], true);
    assert_eq!(source["checkpoint_completed"], true);
    assert_eq!(source["scanned_docs"].as_u64().unwrap(), docs.len() as u64);
    assert!(source["checkpoint_saved_count"].as_u64().unwrap() >= 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn data_materialize_blend_rejects_unapproved_license_status() {
    let exe = option_env!("CARGO_BIN_EXE_heirloom").unwrap_or("target/debug/heirloom");
    let dir = temp_dir("data_materialize_blend_license");
    let corpus_path = dir.join("requires-review.jsonl");
    let tokenizer_path = dir.join("tokenizer.json");
    let blend_path = dir.join("corpus-blend.json");
    let out_dir = dir.join("materialized");
    let corpus = "{\"text\":\"This source is intentionally marked as requiring review.\"}\n";
    fs::write(&corpus_path, corpus).unwrap();
    let tokenizer = BpeTokenizer::train(corpus, 270).unwrap();
    tokenizer.save(&tokenizer_path).unwrap();
    let blend = serde_json::json!({
        "format": "heirloom.corpus_blend",
        "version": 1,
        "blend_id": "fixture-unapproved-license",
        "tokenizer_target_vocab_size": 270,
        "tokenizer_family": "heirloom.byte_bpe",
        "sources": [{
            "source_id": "fixture.needs_review",
            "display_name": "Fixture requires review",
            "kind": "local_text_jsonl",
            "status": "available",
            "role": "general_text",
            "source_url": null,
            "path": corpus_path.to_str().unwrap(),
            "metadata_path": null,
            "data_format": "jsonl",
            "license": "fixture requires review",
            "license_url": null,
            "license_status": "source_terms_required",
            "provenance": ["fixture"],
            "sampling_weight": 1.0,
            "include_in_tokenizer_training": true,
            "include_in_pretraining": true,
            "include_in_memory_trace_training": false,
            "synthetic": true,
            "local_bytes": fs::metadata(&corpus_path).unwrap().len() as usize,
            "local_records": 1,
            "content_hash": null,
            "metadata_hash": null,
            "split_counts": {"train": 1},
            "domain_counts": {"fixture": 1},
            "category_counts": {"fixture": 1},
            "task_kind_counts": {},
            "extra": {}
        }],
        "local_source_count": 1,
        "planned_source_count": 0,
        "local_record_count": 1,
        "local_bytes": fs::metadata(&corpus_path).unwrap().len() as usize,
        "notes": []
    });
    fs::write(&blend_path, serde_json::to_string_pretty(&blend).unwrap()).unwrap();

    let status = Command::new(exe)
        .args([
            "data",
            "materialize-blend",
            "--corpus-blend",
            blend_path.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--target-tokens",
            "16",
        ])
        .status()
        .unwrap();
    assert!(!status.success());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn binary_shard_cli_train_and_eval_reports_loader_and_performance() {
    let exe = option_env!("CARGO_BIN_EXE_heirloom").unwrap_or("target/debug/heirloom");
    let dir = temp_dir("binary_shard_cli");
    let input_a = dir.join("TinyStories-valid.txt");
    let input_b = dir.join("qb-traces.txt");
    let tokenizer_path = dir.join("tokenizer.json");
    let prepared_dir = dir.join("prepared");
    let checkpoint = dir.join("checkpoint");
    let train_report = dir.join("train-report.json");
    let eval_report = dir.join("eval-report.json");
    let text_a = "alpha beta gamma delta epsilon zeta eta theta. ".repeat(12);
    let text_b = "STATE known ACTION read_file EVIDENCE local ANSWER concise.\n".repeat(4);
    fs::write(&input_a, &text_a).unwrap();
    fs::write(&input_b, &text_b).unwrap();
    let tokenizer = BpeTokenizer::train(&(text_a + &text_b), 300).unwrap();
    tokenizer.save(&tokenizer_path).unwrap();

    let status = Command::new(exe)
        .args([
            "data",
            "prepare",
            "--input",
            input_a.to_str().unwrap(),
            "--input",
            input_b.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--out-dir",
            prepared_dir.to_str().unwrap(),
            "--format",
            "binary-shard",
            "--valid-fraction",
            "0.2",
            "--shard-tokens",
            "24",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let rejected_dir = dir.join("rejected");
    let rejected = Command::new(exe)
        .args([
            "data",
            "prepare",
            "--input",
            input_a.to_str().unwrap(),
            "--input",
            input_b.to_str().unwrap(),
            "--tokenizer",
            tokenizer_path.to_str().unwrap(),
            "--out-dir",
            rejected_dir.to_str().unwrap(),
            "--format",
            "binary-shard",
            "--max-bytes",
            "16",
        ])
        .status()
        .unwrap();
    assert!(!rejected.success());

    let manifest_path = prepared_dir.join("manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["version"], 2);
    assert_eq!(manifest["storage"], "binary_shards");
    assert!(manifest["train_shards"].as_array().unwrap().len() > 1);

    let status = Command::new(exe)
        .args([
            "train-lm",
            "--dataset-manifest",
            manifest_path.to_str().unwrap(),
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--block-size",
            "8",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--ff-hidden",
            "16",
            "--lr",
            "0.01",
            "--log-every",
            "1",
            "--report",
            train_report.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let train: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&train_report).unwrap()).unwrap();
    assert_eq!(train["loader"]["kind"], "binary_shard_streaming");
    assert_eq!(train["loader"]["tokens_materialized"], false);
    assert!(train["performance"]["tokens_seen"].as_u64().unwrap() > 0);
    assert!(train["performance"]["tokens_per_second"].as_f64().unwrap() >= 0.0);
    assert!(train["performance"]["forward_backward_host_elapsed_ms"].is_number());
    assert!(train["performance"]["forward_backward_cuda_elapsed_ms"].is_number());
    assert!(train["performance"]["cuda_event_timing_available"].is_boolean());
    assert!(train["performance"]["mfu_timing_source"].is_object());
    assert!(train["tokenizer"]["tokenizer_hash"].is_string());
    assert!(train["tokenizer"]["vocab_size"].as_u64().unwrap() >= 259);

    let status = Command::new(exe)
        .args([
            "eval-lm",
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--dataset-manifest",
            manifest_path.to_str().unwrap(),
            "--split",
            "valid",
            "--batch-size",
            "2",
            "--max-batches",
            "1",
            "--report",
            eval_report.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let eval: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&eval_report).unwrap()).unwrap();
    assert_eq!(eval["loader"]["kind"], "binary_shard_streaming");
    assert_eq!(eval["loader"]["tokens_materialized"], false);
    assert!(eval["tokenizer"]["tokenizer_hash"].is_string());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn embedding_backward_scatter_adds_repeated_rows() {
    let indices = Tensor::from_i64(vec![0, 1, 0], &[3], false).unwrap();
    let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();

    indices
        .embedding(&weight)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    assert_close(&weight.grad().unwrap(), &[2.0, 2.0, 1.0, 1.0], 1e-6);
}

#[test]
fn masked_fill_blocks_gradients_at_masked_positions() {
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], true).unwrap();
    let mask = Tensor::from_bool(vec![false, true, false], &[3], false).unwrap();

    let output = input.masked_fill(&mask, -100.0).unwrap();
    assert_close(&output.data(), &[1.0, -100.0, 3.0], 1e-6);
    output.sum().unwrap().backward().unwrap();
    assert_close(&input.grad().unwrap(), &[1.0, 0.0, 1.0], 1e-6);
}

#[test]
fn batched_matmul_broadcasts_and_backpropagates() {
    let left = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, 2.0, 1.0, 0.0,
        ],
        &[2, 2, 3],
        true,
    )
    .unwrap();
    let right = Tensor::from_vec(vec![1.0, 0.5, -1.0, 2.0, 0.25, -0.5], &[3, 2], true).unwrap();

    let output = left.matmul(&right).unwrap();
    assert_eq!(output.shape(), vec![2, 2, 2]);
    output.sum().unwrap().backward().unwrap();

    assert_eq!(left.grad().unwrap().len(), left.numel());
    assert_eq!(right.grad().unwrap().len(), right.numel());
}

#[test]
fn layernorm_gelu_and_causal_attention_have_trainable_gradients() {
    let x = Tensor::from_vec(
        vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8],
        &[1, 2, 4],
        true,
    )
    .unwrap();
    let weight = Tensor::ones(&[4], true).unwrap();
    let bias = Tensor::zeros(&[4], true).unwrap();

    let normalized = x.layer_norm_last_dim(&weight, &bias, 1e-5).unwrap();
    let activated = normalized.gelu().unwrap();
    let attended = activated
        .causal_self_attention(&activated, &activated, 2)
        .unwrap();
    attended.sum().unwrap().backward().unwrap();

    assert_eq!(x.grad().unwrap().len(), x.numel());
    assert_eq!(weight.grad().unwrap().len(), 4);
    assert_eq!(bias.grad().unwrap().len(), 4);
}

#[test]
fn generation_sampling_controls_and_eval_metrics_are_deterministic() {
    let text = "the cat sat. the cat ran. the dog sat. the dog ran. ".repeat(3);
    let tokenizer = BpeTokenizer::train(&text, 280).unwrap();
    let tokens = tokenizer.encode(&text, true, true);
    let config = TinyTransformerConfig {
        vocab_size: tokenizer.vocab_size(),
        block_size: 6,
        d_model: 8,
        n_heads: 2,
        ff_hidden: 16,
    };
    let mut model_rng = HeirloomRng::new(7);
    let model = TinyTransformerLm::new(config, &mut model_rng).unwrap();
    let prefix = tokenizer.encode("the", true, false);

    let greedy = model.generate_greedy(&prefix, 4, 2).unwrap();
    let mut sample_rng = HeirloomRng::new(123);
    let sampled = model
        .generate(
            &prefix,
            &GenerationOptions {
                max_new_tokens: 4,
                eos_id: 2,
                temperature: 1.0,
                top_k: Some(1),
                top_p: Some(1.0),
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
            },
            &mut sample_rng,
        )
        .unwrap();
    assert_eq!(sampled.tokens, greedy);
    assert_eq!(sampled.steps.len(), sampled.new_token_count);
    assert!(sampled.steps.iter().all(|step| step.probability > 0.0));

    let metrics = model.evaluate_token_loss(&tokens, 2, Some(1)).unwrap();
    assert_eq!(metrics.batches, 1);
    assert_eq!(metrics.examples, 2);
    assert_eq!(metrics.tokens, 12);
    assert!(metrics.loss.is_finite());
    assert!(metrics.perplexity.is_finite());
    assert!(metrics.perplexity > 0.0);
}

#[test]
fn tiny_transformer_bf16_activation_policy_trains_fixed_batch() {
    let text = "the cat sat. the cat ran. the dog sat. the dog ran. ";
    let tokenizer = BpeTokenizer::train(text, 280).unwrap();
    let tokens = tokenizer.encode(text, true, true);
    let mut dataset = TokenDataset::new(tokens.clone(), 6, 123).unwrap();
    let config = TinyTransformerConfig {
        vocab_size: tokenizer.vocab_size(),
        block_size: 6,
        d_model: 8,
        n_heads: 2,
        ff_hidden: 16,
    };
    let mut rng = HeirloomRng::new(19);
    let model = TinyTransformerLm::new(config, &mut rng).unwrap();
    assert!(model
        .parameters()
        .iter()
        .all(|parameter| parameter.dtype() == DType::F32));
    let mut optimizer = AdamW::new(model.parameters(), 0.01)
        .unwrap()
        .with_clip_norm(Some(1.0))
        .unwrap();

    let (input, target) = dataset.next_batch(4).unwrap();
    let logits = model.forward_bf16_activations(&input).unwrap();
    assert_eq!(logits.dtype(), DType::F32);
    let initial = model.loss_bf16_activations(&input, &target).unwrap().data()[0];

    for _ in 0..30 {
        optimizer.zero_grad();
        let loss = model.loss_bf16_activations(&input, &target).unwrap();
        loss.backward().unwrap();
        optimizer.step_mut().unwrap();
    }

    let final_loss = model.loss_bf16_activations(&input, &target).unwrap().data()[0];
    assert!(
        final_loss < initial,
        "expected BF16 activation policy to train a fixed batch, initial={initial}, final={final_loss}"
    );

    let metrics = model
        .evaluate_token_loss_bf16_activations(&tokens, 2, Some(1))
        .unwrap();
    assert_eq!(metrics.batches, 1);
    assert!(metrics.loss.is_finite());
    assert!(metrics.perplexity.is_finite());

    let mut generation_rng = HeirloomRng::new(23);
    let generated = model
        .generate_bf16_activations(
            &tokenizer.encode("the", true, false),
            &GenerationOptions::greedy(2, 2),
            &mut generation_rng,
        )
        .unwrap();
    assert!(generated.tokens.len() >= 2);
}

#[test]
fn tiny_transformer_loss_decreases_and_checkpoint_loads() {
    let text = "the cat sat. the cat ran. the dog sat. the dog ran. ";
    let tokenizer = BpeTokenizer::train(text, 280).unwrap();
    let tokens = tokenizer.encode(text, true, true);
    let mut dataset = TokenDataset::new(tokens, 6, 99).unwrap();
    let config = TinyTransformerConfig {
        vocab_size: tokenizer.vocab_size(),
        block_size: 6,
        d_model: 8,
        n_heads: 2,
        ff_hidden: 16,
    };
    let mut rng = HeirloomRng::new(7);
    let model = TinyTransformerLm::new(config, &mut rng).unwrap();
    let mut optimizer = AdamW::new(model.parameters(), 0.01)
        .unwrap()
        .with_clip_norm(Some(1.0))
        .unwrap();

    let (input, target) = dataset.next_batch(4).unwrap();
    let initial = model.loss(&input, &target).unwrap().data()[0];
    for _ in 0..25 {
        optimizer.zero_grad();
        let (input, target) = dataset.next_batch(4).unwrap();
        let loss = model.loss(&input, &target).unwrap();
        loss.backward().unwrap();
        optimizer.step_mut().unwrap();
    }
    let (input, target) = dataset.next_batch(4).unwrap();
    let final_loss = model.loss(&input, &target).unwrap().data()[0];
    assert!(
        final_loss < initial,
        "expected loss to decrease, initial={initial}, final={final_loss}"
    );

    let dir = temp_dir("checkpoint");
    let dataset_state = dataset.state();
    save_lm_checkpoint_with_dataset_state(
        &dir,
        &model,
        &optimizer,
        &tokenizer,
        dataset_state,
        Some("manifest.json".to_string()),
    )
    .unwrap();
    let loaded = load_lm_checkpoint(&dir).unwrap();
    assert_eq!(loaded.metadata.dataset_state(), dataset_state);
    assert_eq!(
        loaded.metadata.dataset_manifest_path.as_deref(),
        Some("manifest.json")
    );
    let generated = loaded
        .model
        .generate_greedy(&tokenizer.encode("the", true, false), 4, 2)
        .unwrap();
    assert!(generated.len() > 1);
    fs::remove_dir_all(dir).unwrap();
}
