use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn heirloom_bin() -> &'static str {
    env!("CARGO_BIN_EXE_heirloom")
}

fn temp_checkpoint(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("heirloom-{name}-{}", std::process::id()))
        .display()
        .to_string()
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("heirloom-{name}-{}-{nanos}", std::process::id()))
}

fn read_json(path: &Path) -> Value {
    let json = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    serde_json::from_str(&json)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()))
}

#[test]
fn amp_bf16_rejects_cpu_training() {
    let output = Command::new(heirloom_bin())
        .args([
            "train-lm",
            "--checkpoint",
            &temp_checkpoint("amp-bf16-cpu"),
            "--precision",
            "amp-bf16",
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("amp-bf16 precision requires a CUDA device"),
        "stderr={stderr}"
    );
}

#[test]
fn distributed_training_rejects_non_cuda_devices_before_nccl() {
    let output = Command::new(heirloom_bin())
        .args([
            "train-lm",
            "--checkpoint",
            &temp_checkpoint("ddp-non-cuda"),
            "--devices",
            "cpu,cuda:0",
            "--distributed",
            "nccl",
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("distributed --devices accepts only cuda:<id> entries"),
        "stderr={stderr}"
    );
}

#[test]
fn distributed_training_rejects_duplicate_devices_before_nccl() {
    let output = Command::new(heirloom_bin())
        .args([
            "train-lm",
            "--checkpoint",
            &temp_checkpoint("ddp-duplicate"),
            "--devices",
            "cuda:0,cuda:0",
            "--distributed",
            "nccl",
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("duplicate CUDA device cuda:0"),
        "stderr={stderr}"
    );
}

#[test]
fn distributed_memory_training_rejects_non_cuda_devices_before_nccl() {
    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--checkpoint",
            &temp_checkpoint("memory-ddp-non-cuda"),
            "--devices",
            "cpu,cuda:0",
            "--distributed",
            "nccl",
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("distributed --devices accepts only cuda:<id> entries"),
        "stderr={stderr}"
    );
}

#[test]
fn distributed_memory_training_rejects_duplicate_devices_before_nccl() {
    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--checkpoint",
            &temp_checkpoint("memory-ddp-duplicate"),
            "--devices",
            "cuda:0,cuda:0",
            "--distributed",
            "nccl",
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("duplicate CUDA device cuda:0"),
        "stderr={stderr}"
    );
}

#[test]
fn distributed_memory_training_rejects_smft_before_nccl() {
    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--checkpoint",
            &temp_checkpoint("memory-ddp-smft"),
            "--devices",
            "cuda:0,cuda:1",
            "--distributed",
            "nccl",
            "--smft-mode",
            "masked-memory-rows",
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("distributed masked-memory-rows SMFT requires --smft-row-mask"),
        "stderr={stderr}"
    );
}

#[test]
fn distributed_memory_training_rejects_smft_mask_without_sparse_rows_before_nccl() {
    let output = Command::new(heirloom_bin())
        .args([
            "train-memory-lm",
            "--checkpoint",
            &temp_checkpoint("memory-ddp-smft-mask-full"),
            "--devices",
            "cuda:0,cuda:1",
            "--distributed",
            "nccl",
            "--memory-update-policy",
            "full",
            "--smft-row-mask",
            "/tmp/heirloom-nonexistent-smft-mask.json",
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("distributed --smft-row-mask requires --memory-update-policy sparse-rows"),
        "stderr={stderr}"
    );
}

#[test]
fn nccl_probe_rejects_non_cuda_devices_before_nccl() {
    let output = Command::new(heirloom_bin())
        .args(["gpu", "nccl-probe", "--devices", "cpu,cuda:0"])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("distributed --devices accepts only cuda:<id> entries"),
        "stderr={stderr}"
    );
}

#[test]
fn nccl_probe_rejects_duplicate_devices_before_nccl() {
    let output = Command::new(heirloom_bin())
        .args(["gpu", "nccl-probe", "--devices", "cuda:0,cuda:0"])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("duplicate CUDA device cuda:0"),
        "stderr={stderr}"
    );
}

#[test]
fn launcher_test_records_rank_artifacts_on_success() {
    let dir = temp_dir("launcher-success");
    let report = dir.join("report.json");
    let output = Command::new(heirloom_bin())
        .args([
            "launcher-test",
            "--ranks",
            "2",
            "--rank-start-timeout-secs",
            "5",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = read_json(&report);
    assert_eq!(summary["status"], "passed");
    let launcher_report_path = PathBuf::from(summary["launcher_report"].as_str().unwrap());
    let launcher_report = read_json(&launcher_report_path);
    assert_eq!(launcher_report["status"], "passed");
    let ranks = launcher_report["ranks"].as_array().unwrap();
    assert_eq!(ranks.len(), 2);
    for rank in ranks {
        assert!(rank["pid"].as_u64().unwrap() > 0);
        assert_eq!(rank["completed"], true);
        assert_eq!(rank["rank_report_exists"], true);
        assert_eq!(rank["stdout_exists"], true);
        assert_eq!(rank["stderr_exists"], true);
        assert_eq!(rank["last_stage"]["stage"], "report_written");
    }
}

#[test]
fn launcher_test_kills_sibling_on_one_rank_failure() {
    let dir = temp_dir("launcher-failure");
    let report = dir.join("report.json");
    let output = Command::new(heirloom_bin())
        .args([
            "launcher-test",
            "--ranks",
            "2",
            "--fail-rank",
            "0",
            "--hang-rank",
            "1",
            "--timeout-secs",
            "5",
            "--rank-start-timeout-secs",
            "5",
            "--kill-grace-secs",
            "1",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let summary = read_json(&report);
    assert_eq!(summary["status"], "failed");
    let launcher_report_path = PathBuf::from(summary["launcher_report"].as_str().unwrap());
    let launcher_report = read_json(&launcher_report_path);
    assert_eq!(launcher_report["status"], "failed");
    assert!(
        launcher_report["reason"]
            .as_str()
            .unwrap()
            .contains("rank 0 exited"),
        "report={launcher_report}"
    );
    let ranks = launcher_report["ranks"].as_array().unwrap();
    assert_eq!(ranks.len(), 2);
    assert_eq!(ranks[0]["last_stage"]["stage"], "failed");
    assert_eq!(ranks[1]["completed"], true);
    let sibling_stage = ranks[1]["last_stage"]["stage"].as_str().unwrap();
    assert!(
        matches!(sibling_stage, "spawned" | "config_loaded"),
        "unexpected sibling stage: {sibling_stage}; report={launcher_report}"
    );
}

#[test]
fn launcher_test_times_out_hung_rank_and_writes_report() {
    let dir = temp_dir("launcher-timeout");
    let report = dir.join("report.json");
    let output = Command::new(heirloom_bin())
        .args([
            "launcher-test",
            "--ranks",
            "2",
            "--hang-rank",
            "0",
            "--timeout-secs",
            "1",
            "--rank-start-timeout-secs",
            "5",
            "--kill-grace-secs",
            "1",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(!output.status.success());
    let summary = read_json(&report);
    assert_eq!(summary["status"], "failed");
    let launcher_report_path = PathBuf::from(summary["launcher_report"].as_str().unwrap());
    let launcher_report = read_json(&launcher_report_path);
    assert_eq!(launcher_report["status"], "timeout");
    assert!(
        launcher_report["reason"]
            .as_str()
            .unwrap()
            .contains("timeout after 1s"),
        "report={launcher_report}"
    );
    let ranks = launcher_report["ranks"].as_array().unwrap();
    assert_eq!(ranks.len(), 2);
    assert_eq!(ranks[0]["last_stage"]["stage"], "config_loaded");
    assert_eq!(ranks[0]["completed"], true);
}

#[test]
fn nccl_probe_spawn_kind_runs_without_cuda_or_nccl() {
    let dir = temp_dir("nccl-probe-spawn");
    let report = dir.join("nccl-probe.json");
    let output = Command::new(heirloom_bin())
        .args([
            "gpu",
            "nccl-probe",
            "--devices",
            "cuda:0,cuda:1",
            "--probe-kind",
            "spawn",
            "--timeout-secs",
            "5",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run heirloom binary");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = read_json(&report);
    assert_eq!(summary["status"], "passed");
    assert_eq!(summary["probe_kind"], "spawn");
    assert_eq!(summary["world_size"], 2);
    let launcher_report_path = PathBuf::from(summary["launcher_report"].as_str().unwrap());
    let launcher_report = read_json(&launcher_report_path);
    assert_eq!(launcher_report["status"], "passed");
    let ranks = launcher_report["ranks"].as_array().unwrap();
    assert_eq!(ranks.len(), 2);
    for rank in ranks {
        assert_eq!(rank["last_stage"]["stage"], "report_written");
        assert_eq!(rank["rank_report_exists"], true);
    }
}
