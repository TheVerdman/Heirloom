#!/usr/bin/env zsh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-49b1b523-d248-434f-bd4}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-gs://${PROJECT_ID}-vecl-qb-artifacts}"
JOB_TS="$(date +%Y%m%d-%H%M%S)"
STREAM_LOGS="${STREAM_LOGS:-true}"
VALIDATE_MODE="${HEIRLOOM_VERTEX_VALIDATE_MODE:-quick}"
ACCELERATOR_TYPE="${HEIRLOOM_VERTEX_ACCELERATOR_TYPE:-NVIDIA_A100_80GB}"
ACCELERATOR_COUNT="${HEIRLOOM_VERTEX_ACCELERATOR_COUNT:-1}"

case "$VALIDATE_MODE" in
  quick|full) ;;
  *)
    print -u2 "HEIRLOOM_VERTEX_VALIDATE_MODE must be quick or full, got ${VALIDATE_MODE}"
    exit 2
    ;;
esac

case "$ACCELERATOR_COUNT" in
  1) DEFAULT_MACHINE_TYPE="a2-ultragpu-1g" ;;
  2) DEFAULT_MACHINE_TYPE="a2-ultragpu-2g" ;;
  4) DEFAULT_MACHINE_TYPE="a2-ultragpu-4g" ;;
  8) DEFAULT_MACHINE_TYPE="a2-ultragpu-8g" ;;
  *)
    print -u2 "Unsupported HEIRLOOM_VERTEX_ACCELERATOR_COUNT=${ACCELERATOR_COUNT}; expected 1, 2, 4, or 8."
    exit 2
    ;;
esac
MACHINE_TYPE="${HEIRLOOM_VERTEX_MACHINE_TYPE:-$DEFAULT_MACHINE_TYPE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PKG_DIR="/tmp/heirloom-vertex-validate-src-${JOB_TS}"
JOB_PKG_DIR="/tmp/heirloom-vertex-validate-job-${JOB_TS}"
SOURCE_TGZ="/tmp/heirloom-source-${JOB_TS}.tar.gz"
PACKAGE_TGZ="/tmp/heirloom-vertex-validate-${JOB_TS}.tar.gz"
CONFIG_YAML="/tmp/heirloom-vertex-validate-${JOB_TS}.yaml"
DISPLAY_NAME="heirloom-validate-${VALIDATE_MODE}-${JOB_TS}"
SOURCE_URI="${BUCKET}/heirloom/packages/heirloom-source-${JOB_TS}.tar.gz"
PACKAGE_URI="${BUCKET}/heirloom/packages/heirloom-vertex-validate-${JOB_TS}.tar.gz"
ARTIFACT_PREFIX="${BUCKET}/heirloom/reference-runs/${DISPLAY_NAME}"

cleanup() {
  rm -rf "$PKG_DIR" "$JOB_PKG_DIR"
  rm -f "$SOURCE_TGZ" "$PACKAGE_TGZ" "$CONFIG_YAML"
}
trap cleanup EXIT

retry_cloud_command() {
  local label="$1"
  shift
  local attempts="${HEIRLOOM_GCS_RETRY_ATTEMPTS:-5}"
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt >= attempts )); then
      return 1
    fi
    local sleep_seconds=$(( 1 << (attempt - 1) ))
    if (( sleep_seconds > 30 )); then
      sleep_seconds=30
    fi
    print -u2 "cloud I/O retry label=${label} attempt=${attempt}/${attempts} sleep_secs=${sleep_seconds}"
    sleep "$sleep_seconds"
    attempt=$((attempt + 1))
  done
}

print "Packaging Heirloom source from ${REPO_ROOT}"
mkdir -p "$PKG_DIR"
tar -C "$REPO_ROOT" \
  --exclude "./target" \
  --exclude "./.git" \
  --exclude "./.venv" \
  --exclude "./runs" \
  --exclude "./tmp" \
  --exclude "./.DS_Store" \
  -czf "$SOURCE_TGZ" .

mkdir -p "$JOB_PKG_DIR/heirloom_vertex_validate_job"
touch "$JOB_PKG_DIR/heirloom_vertex_validate_job/__init__.py"

cat > "$JOB_PKG_DIR/heirloom_vertex_validate_job/__main__.py" <<'PY'
from __future__ import annotations

import os
import json
import pathlib
import signal
import shutil
import subprocess
import tarfile
import time

from google.cloud import storage

UNSET_SENTINEL = "__HEIRLOOM_UNSET__"


def child_env(source: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in source.items() if value != UNSET_SENTINEL}


def run(command: list[str], *, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> None:
    print("\n==> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_to_file(
    command: list[str],
    output_path: pathlib.Path,
    *,
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> None:
    print("\n==> " + " ".join(command) + f" | tee {output_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if timeout_seconds is not None:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            output_path.write_text(output or "", encoding="utf-8")
            if output:
                print(output, end="", flush=True)
            print(
                f"command timed out after {timeout_seconds}s: {' '.join(command)}",
                flush=True,
            )
            raise subprocess.CalledProcessError(124, command) from exc
        output_path.write_text(output or "", encoding="utf-8")
        if output:
            print(output, end="", flush=True)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        return

    with output_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        exit_code = process.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {uri}")
    without_scheme = uri[5:]
    bucket, _, blob = without_scheme.partition("/")
    if not bucket or not blob:
        raise ValueError(f"expected gs://bucket/object URI, got {uri}")
    return bucket, blob


def cloud_io_retry(label: str, operation):
    attempts = max(1, int(os.environ.get("HEIRLOOM_GCS_RETRY_ATTEMPTS", "5")))
    base_sleep = max(
        0.1, float(os.environ.get("HEIRLOOM_GCS_RETRY_BASE_SLEEP_SECS", "1.0"))
    )
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as err:
            if attempt == attempts:
                raise
            sleep_seconds = min(30.0, base_sleep * (2 ** (attempt - 1)))
            print(
                "cloud I/O retry "
                + f"label={label} attempt={attempt}/{attempts} "
                + f"error_type={type(err).__name__} sleep_secs={sleep_seconds:.1f}",
                flush=True,
            )
            time.sleep(sleep_seconds)


def download_gcs(uri: str, destination: pathlib.Path) -> None:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client(project=os.environ.get("PROJECT_ID"))
    bucket = client.bucket(bucket_name)
    cloud_io_retry(
        f"download:{bucket_name}/{blob_name}",
        lambda: bucket.blob(blob_name).download_to_filename(destination),
    )


def upload_gcs(source: pathlib.Path, uri: str) -> None:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client(project=os.environ.get("PROJECT_ID"))
    bucket = client.bucket(bucket_name)
    cloud_io_retry(
        f"upload:{bucket_name}/{blob_name}",
        lambda: bucket.blob(blob_name).upload_from_filename(source),
    )


def upload_directory_gcs(source_dir: pathlib.Path, uri_prefix: str) -> list[str]:
    uploaded: list[str] = []
    if not source_dir.exists():
        return uploaded
    prefix = uri_prefix.rstrip("/")
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir).as_posix()
        uri = f"{prefix}/{relative}"
        upload_gcs(path, uri)
        uploaded.append(uri)
    return uploaded


def ensure_rust_toolchain(env: dict[str, str]) -> None:
    cargo = shutil.which("cargo", path=env["PATH"])
    if cargo is None:
        run(
            [
                "bash",
                "-lc",
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
                "sh -s -- -y --profile minimal",
            ],
            env=env,
        )
    cargo_home = pathlib.Path.home() / ".cargo" / "bin"
    env["PATH"] = f"{cargo_home}:{env['PATH']}"
    if shutil.which("rustup", path=env["PATH"]):
        run(["rustup", "component", "add", "rustfmt", "clippy"], env=env)
    run(["cargo", "--version"], env=env)
    run(["rustc", "--version"], env=env)


def main() -> int:
    source_uri = os.environ["HEIRLOOM_SOURCE_URI"]
    validate_mode = os.environ.get("HEIRLOOM_VALIDATE_MODE", "quick")
    require_gpu = os.environ.get("HEIRLOOM_REQUIRE_GPU", "1") == "1"
    artifact_prefix = os.environ["HEIRLOOM_ARTIFACT_PREFIX"]
    gpu_smoke_len = os.environ.get("HEIRLOOM_GPU_SMOKE_LEN", "4096")
    gpu_smoke_devices = os.environ.get("HEIRLOOM_GPU_SMOKE_DEVICES", "0")
    collect_gpu_topology = os.environ.get("HEIRLOOM_COLLECT_GPU_TOPOLOGY", "1") == "1"
    run_quick_clippy = os.environ.get("HEIRLOOM_RUN_QUICK_CLIPPY", "1") == "1"
    quick_clippy_status = "run" if run_quick_clippy else "skipped"
    quick_clippy_reason = None if run_quick_clippy else "HEIRLOOM_RUN_QUICK_CLIPPY=0"
    run_tensor_core_probe = os.environ.get("HEIRLOOM_RUN_TENSOR_CORE_PROBE", "1") == "1"
    run_tensor_core_microbench = (
        os.environ.get("HEIRLOOM_RUN_TENSOR_CORE_MICROBENCH", "0") == "1"
    )
    tensor_core_microbench_device = os.environ.get("HEIRLOOM_TENSOR_CORE_MICROBENCH_DEVICE", "0")
    tensor_core_microbench_iterations = os.environ.get(
        "HEIRLOOM_TENSOR_CORE_MICROBENCH_ITERATIONS", "32"
    )
    tensor_core_microbench_warmup = os.environ.get("HEIRLOOM_TENSOR_CORE_MICROBENCH_WARMUP", "4")
    tensor_core_microbench_sections = [
        section.strip().lower()
        for section in os.environ.get(
            "HEIRLOOM_TENSOR_CORE_MICROBENCH_SECTIONS", "gemm,attention"
        ).split(",")
        if section.strip()
    ]
    if not tensor_core_microbench_sections:
        tensor_core_microbench_sections = ["gemm", "attention"]
    invalid_microbench_sections = [
        section
        for section in tensor_core_microbench_sections
        if section not in {"all", "gemm", "attention"}
    ]
    if invalid_microbench_sections:
        raise ValueError(
            "HEIRLOOM_TENSOR_CORE_MICROBENCH_SECTIONS entries must be all, gemm, or attention; "
            + f"got {invalid_microbench_sections}"
        )
    tensor_core_microbench_m = os.environ.get("HEIRLOOM_TENSOR_CORE_MICROBENCH_M", "4096")
    tensor_core_microbench_k = os.environ.get("HEIRLOOM_TENSOR_CORE_MICROBENCH_K", "1024")
    tensor_core_microbench_n = os.environ.get("HEIRLOOM_TENSOR_CORE_MICROBENCH_N", "4096")
    tensor_core_microbench_attention_batch = os.environ.get(
        "HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_BATCH", "4"
    )
    tensor_core_microbench_attention_heads = os.environ.get(
        "HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEADS", "16"
    )
    tensor_core_microbench_attention_time = os.environ.get(
        "HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_TIME", "512"
    )
    tensor_core_microbench_attention_head_dim = os.environ.get(
        "HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEAD_DIM", "64"
    )
    enable_nccl_debug = os.environ.get("HEIRLOOM_ENABLE_NCCL_DEBUG", "1") == "1"
    run_nccl_probe = os.environ.get("HEIRLOOM_RUN_NCCL_PROBE", "0") == "1"
    nccl_probe_devices = os.environ.get("HEIRLOOM_NCCL_PROBE_DEVICES", "")
    if nccl_probe_devices == UNSET_SENTINEL:
        nccl_probe_devices = ""
    nccl_probe_len = os.environ.get("HEIRLOOM_NCCL_PROBE_LEN", "1024")
    nccl_probe_kind = os.environ.get("HEIRLOOM_NCCL_PROBE_KIND", "all-reduce")
    nccl_probe_timeout_secs = int(os.environ.get("HEIRLOOM_NCCL_PROBE_TIMEOUT_SECS", "120"))
    nccl_probe_rank_start_timeout_secs = os.environ.get(
        "HEIRLOOM_NCCL_PROBE_RANK_START_TIMEOUT_SECS", "10"
    )
    nccl_probe_kill_grace_secs = os.environ.get("HEIRLOOM_NCCL_PROBE_KILL_GRACE_SECS", "5")
    run_cuda_storage_tests = os.environ.get("HEIRLOOM_RUN_CUDA_STORAGE_TESTS", "1") == "1"
    run_cuda_train_lm_fixture = os.environ.get("HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE", "0") == "1"
    run_cuda_train_memory_lm_fixture = (
        os.environ.get("HEIRLOOM_RUN_CUDA_TRAIN_MEMORY_LM_FIXTURE", "0") == "1"
    )
    run_tinystories_cuda_reference = (
        os.environ.get("HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE", "0") == "1"
    )
    run_qb_data_hardpath = os.environ.get("HEIRLOOM_RUN_QB_DATA_HARDPATH", "0") == "1"
    run_qb_tokenizer_hardpath = (
        os.environ.get("HEIRLOOM_RUN_QB_TOKENIZER_HARDPATH", "0") == "1"
    )
    run_learning_sanity_fixtures = (
        os.environ.get("HEIRLOOM_RUN_LEARNING_SANITY_FIXTURES", "0") == "1"
    )
    work_root = pathlib.Path("/tmp/heirloom-vertex-work")
    source_tgz = work_root / "heirloom-source.tar.gz"
    source_dir = work_root / "src"

    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    print(f"Downloading source package: {source_uri}", flush=True)
    download_gcs(source_uri, source_tgz)
    source_dir.mkdir()
    with tarfile.open(source_tgz, "r:gz") as archive:
        archive.extractall(source_dir)

    env = child_env(os.environ.copy())
    env["PATH"] = f"{pathlib.Path.home() / '.cargo' / 'bin'}:{env.get('PATH', '')}"
    env.setdefault("RUST_BACKTRACE", "1")
    ensure_rust_toolchain(env)

    gpu_report_uris: list[str] = []
    gpu_log_uris: list[str] = []
    topology_report_uris: list[str] = []
    topology_log_uris: list[str] = []
    tensor_core_probe_report_uris: list[str] = []
    tensor_core_probe_log_uris: list[str] = []
    tensor_core_microbench_report_uris: list[str] = []
    tensor_core_microbench_log_uris: list[str] = []
    tensor_core_microbench_sections_requested = (
        tensor_core_microbench_sections if run_tensor_core_microbench else []
    )
    tensor_core_microbench_sections_completed: list[str] = []
    tensor_core_microbench_sections_failed: list[str] = []
    nccl_probe_report_uris: list[str] = []
    nccl_probe_log_uris: list[str] = []
    diagnostic_log_uris: list[str] = []
    cuda_train_lm_fixture_report_uris: list[str] = []
    cuda_train_lm_fixture_pad_crop_report_uris: list[str] = []
    cuda_train_memory_lm_fixture_report_uris: list[str] = []
    cuda_train_memory_lm_fixture_checkpoint_prefix: str | None = None
    tinystories_cuda_reference_report_uris: list[str] = []
    tinystories_cuda_reference_checkpoint_prefix: str | None = None
    qb_data_hardpath_report_uris: list[str] = []
    qb_data_hardpath_checkpoint_prefix: str | None = None
    qb_tokenizer_hardpath_report_uris: list[str] = []
    qb_tokenizer_hardpath_checkpoint_prefix: str | None = None
    learning_sanity_fixture_report_uris: list[str] = []

    def upload_diagnostic(path: pathlib.Path, relative: str) -> str | None:
        if not path.exists():
            return None
        uri = f"{artifact_prefix}/{relative}"
        upload_gcs(path, uri)
        diagnostic_log_uris.append(uri)
        return uri

    def upload_diagnostic_tree(root: pathlib.Path, relative_prefix: str) -> list[str]:
        uploaded: list[str] = []
        if not root.exists():
            return uploaded
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = pathlib.Path(relative_prefix) / path.relative_to(root)
            uri = f"{artifact_prefix}/{relative.as_posix()}"
            upload_gcs(path, uri)
            diagnostic_log_uris.append(uri)
            uploaded.append(uri)
        return uploaded

    if require_gpu:
        if collect_gpu_topology:
            for relative, command in [
                ("nvidia-smi.txt", ["bash", "-lc", "command -v nvidia-smi && nvidia-smi"]),
                ("nvidia-smi-topo.txt", ["bash", "-lc", "nvidia-smi topo -m || true"]),
                ("nvidia-smi-nvlink.txt", ["bash", "-lc", "nvidia-smi nvlink -s || true"]),
                (
                    "nvidia-smi-gpu-bus-ids.csv",
                    [
                        "bash",
                        "-lc",
                        "nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv || true",
                    ],
                ),
            ]:
                topology_log = work_root / relative
                try:
                    run_to_file(command, topology_log, env=env)
                finally:
                    uploaded = upload_diagnostic(topology_log, relative)
                    if uploaded:
                        topology_log_uris.append(uploaded)

        gpu_info_log = work_root / "gpu-info.txt"
        try:
            run_to_file(
                ["cargo", "run", "--bin", "heirloom", "--", "gpu", "info"],
                gpu_info_log,
                cwd=source_dir,
                env=env,
            )
        finally:
            uploaded = upload_diagnostic(gpu_info_log, "gpu-info.txt")
            if uploaded:
                gpu_log_uris.append(uploaded)

        if collect_gpu_topology:
            topology_report_path = work_root / "heirloom-gpu-topology.json"
            topology_log = work_root / "heirloom-gpu-topology.txt"
            try:
                run_to_file(
                    [
                        "cargo",
                        "run",
                        "--bin",
                        "heirloom",
                        "--",
                        "gpu",
                        "topology",
                        "--report",
                        str(topology_report_path),
                    ],
                    topology_log,
                    cwd=source_dir,
                    env=env,
                )
            finally:
                uploaded = upload_diagnostic(topology_log, "heirloom-gpu-topology.txt")
                if uploaded:
                    topology_log_uris.append(uploaded)
                if topology_report_path.exists():
                    topology_report_uri = f"{artifact_prefix}/heirloom-gpu-topology.json"
                    upload_gcs(topology_report_path, topology_report_uri)
                    topology_report_uris.append(topology_report_uri)

        if gpu_smoke_devices == "all":
            device_count = int(os.environ.get("HEIRLOOM_ACCELERATOR_COUNT", "1"))
            devices = list(range(device_count))
        else:
            devices = [int(item.strip()) for item in gpu_smoke_devices.split(",") if item.strip()]
        for device in devices:
            report_path = work_root / f"gpu-smoke-device-{device}.json"
            smoke_log = work_root / f"gpu-smoke-device-{device}.txt"
            try:
                run_to_file(
                    [
                        "cargo",
                        "run",
                        "--bin",
                        "heirloom",
                        "--",
                        "gpu",
                        "smoke",
                        "--device",
                        str(device),
                        "--len",
                        gpu_smoke_len,
                        "--report",
                        str(report_path),
                    ],
                    smoke_log,
                    cwd=source_dir,
                    env=env,
                )
            finally:
                uploaded = upload_diagnostic(smoke_log, f"gpu-smoke-device-{device}.txt")
                if uploaded:
                    gpu_log_uris.append(uploaded)
                if report_path.exists():
                    report_uri = f"{artifact_prefix}/gpu-smoke-device-{device}.json"
                    upload_gcs(report_path, report_uri)
                    gpu_report_uris.append(report_uri)

            if run_tensor_core_probe:
                probe_report_path = work_root / f"tensor-core-probe-device-{device}.json"
                probe_log = work_root / f"tensor-core-probe-device-{device}.txt"
                try:
                    run_to_file(
                        [
                            "cargo",
                            "run",
                            "--bin",
                            "heirloom",
                            "--",
                            "gpu",
                            "tensor-core-probe",
                            "--device",
                            str(device),
                            "--report",
                            str(probe_report_path),
                        ],
                        probe_log,
                        cwd=source_dir,
                        env=env,
                    )
                finally:
                    uploaded = upload_diagnostic(
                        probe_log,
                        f"tensor-core-probe-device-{device}.txt",
                    )
                    if uploaded:
                        tensor_core_probe_log_uris.append(uploaded)
                    if probe_report_path.exists():
                        probe_report_uri = (
                            f"{artifact_prefix}/tensor-core-probe-device-{device}.json"
                        )
                        upload_gcs(probe_report_path, probe_report_uri)
                        tensor_core_probe_report_uris.append(probe_report_uri)

        if run_nccl_probe:
            probe_devices = nccl_probe_devices.strip()
            if not probe_devices:
                probe_devices = ",".join(f"cuda:{device}" for device in devices)
            probe_report_path = work_root / "nccl-probe.json"
            probe_log = work_root / "nccl-probe.txt"
            probe_env = env.copy()
            if enable_nccl_debug:
                probe_env.setdefault("NCCL_DEBUG", "INFO")
                probe_env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL,GRAPH")
                probe_env.setdefault("HEIRLOOM_NCCL_TRACE", "1")
            nccl_net_plugin = os.environ.get("HEIRLOOM_NCCL_NET_PLUGIN")
            if nccl_net_plugin and nccl_net_plugin != "__HEIRLOOM_UNSET__":
                probe_env["NCCL_NET_PLUGIN"] = nccl_net_plugin
            try:
                probe_command = [
                    "cargo",
                    "run",
                    "--bin",
                    "heirloom",
                    "--",
                    "gpu",
                    "nccl-probe",
                    "--devices",
                    probe_devices,
                    "--len",
                    nccl_probe_len,
                    "--probe-kind",
                    nccl_probe_kind,
                    "--timeout-secs",
                    str(nccl_probe_timeout_secs),
                    "--rank-start-timeout-secs",
                    nccl_probe_rank_start_timeout_secs,
                    "--kill-grace-secs",
                    nccl_probe_kill_grace_secs,
                    "--report",
                    str(probe_report_path),
                ]
                run_to_file(
                    probe_command,
                    probe_log,
                    cwd=source_dir,
                    env=probe_env,
                    timeout_seconds=nccl_probe_timeout_secs + 30,
                )
            finally:
                uploaded = upload_diagnostic(probe_log, "nccl-probe.txt")
                if uploaded:
                    nccl_probe_log_uris.append(uploaded)
                uploaded = upload_diagnostic(work_root / "launcher-report.json", "launcher-report.json")
                if uploaded:
                    nccl_probe_report_uris.append(uploaded)
                nccl_probe_log_uris.extend(
                    upload_diagnostic_tree(work_root / "nccl-probe-ranks", "nccl-probe-ranks")
                )
                if probe_report_path.exists():
                    probe_report_uri = f"{artifact_prefix}/nccl-probe.json"
                    upload_gcs(probe_report_path, probe_report_uri)
                    nccl_probe_report_uris.append(probe_report_uri)

        if run_cuda_storage_tests:
            cuda_test_env = env.copy()
            cuda_test_env["HEIRLOOM_CUDA_TESTS"] = "1"
            cuda_test_log = work_root / "cuda-storage-tests.txt"
            try:
                run_to_file(
                    [
                        "cargo",
                        "test",
                        "--workspace",
                        "--test",
                        "cuda_storage",
                        "--",
                        "--nocapture",
                    ],
                    cuda_test_log,
                    cwd=source_dir,
                    env=cuda_test_env,
                )
            finally:
                upload_diagnostic(cuda_test_log, "cuda-storage-tests.txt")

        if run_tensor_core_microbench:
            microbench_env = env.copy()
            microbench_env["HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM"] = "1"
            microbench_env["HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM"] = "1"
            microbench_env["HEIRLOOM_CUDA_FLASH_BF16_ATTENTION"] = "1"
            microbench_env["HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE"] = "1"
            microbench_env["HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING"] = "1"
            microbench_failures: list[dict[str, object]] = []
            microbench_sections: list[dict[str, object]] = []
            for microbench_section in tensor_core_microbench_sections:
                stem = (
                    "tensor-core-microbench"
                    if microbench_section == "all"
                    else f"tensor-core-microbench-{microbench_section}"
                )
                microbench_report_path = work_root / f"{stem}.json"
                microbench_log = work_root / f"{stem}.txt"
                microbench_command = [
                    "cargo",
                    "run",
                    "--release",
                    "--bin",
                    "heirloom",
                    "--",
                    "gpu",
                    "tensor-core-microbench",
                    "--device",
                    tensor_core_microbench_device,
                    "--section",
                    microbench_section,
                    "--iterations",
                    tensor_core_microbench_iterations,
                    "--warmup",
                    tensor_core_microbench_warmup,
                    "--m",
                    tensor_core_microbench_m,
                    "--k",
                    tensor_core_microbench_k,
                    "--n",
                    tensor_core_microbench_n,
                    "--attention-batch",
                    tensor_core_microbench_attention_batch,
                    "--attention-heads",
                    tensor_core_microbench_attention_heads,
                    "--attention-time",
                    tensor_core_microbench_attention_time,
                    "--attention-head-dim",
                    tensor_core_microbench_attention_head_dim,
                    "--report",
                    str(microbench_report_path),
                ]
                section_result: dict[str, object] = {
                    "section": microbench_section,
                    "status": "passed",
                    "command": microbench_command,
                }
                try:
                    run_to_file(
                        microbench_command,
                        microbench_log,
                        cwd=source_dir,
                        env=microbench_env,
                    )
                except subprocess.CalledProcessError as err:
                    section_result["status"] = "failed"
                    section_result["return_code"] = err.returncode
                    microbench_failures.append(section_result.copy())
                finally:
                    uploaded = upload_diagnostic(microbench_log, f"{stem}.txt")
                    if uploaded:
                        tensor_core_microbench_log_uris.append(uploaded)
                        section_result["log_uri"] = uploaded
                    if microbench_report_path.exists():
                        microbench_report_uri = f"{artifact_prefix}/{stem}.json"
                        upload_gcs(microbench_report_path, microbench_report_uri)
                        tensor_core_microbench_report_uris.append(microbench_report_uri)
                        section_result["report_uri"] = microbench_report_uri
                microbench_sections.append(section_result)
                if section_result["status"] == "passed":
                    tensor_core_microbench_sections_completed.append(microbench_section)
                else:
                    tensor_core_microbench_sections_failed.append(microbench_section)

            microbench_summary_path = work_root / "tensor-core-microbench-summary.json"
            microbench_summary_path.write_text(
                json.dumps(
                    {
                        "status": "failed" if microbench_failures else "passed",
                        "sections_requested": tensor_core_microbench_sections_requested,
                        "sections_completed": tensor_core_microbench_sections_completed,
                        "sections_failed": tensor_core_microbench_sections_failed,
                        "sections": microbench_sections,
                        "failures": microbench_failures,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            microbench_summary_uri = f"{artifact_prefix}/tensor-core-microbench-summary.json"
            upload_gcs(microbench_summary_path, microbench_summary_uri)
            tensor_core_microbench_report_uris.append(microbench_summary_uri)
            diagnostic_log_uris.append(microbench_summary_uri)
            if microbench_failures:
                failed_sections = ", ".join(
                    str(failure["section"]) for failure in microbench_failures
                )
                raise RuntimeError(
                    f"tensor-core microbench section(s) failed: {failed_sections}"
                )

        if run_cuda_train_lm_fixture:
            fixture_dir = work_root / "cuda-train-lm-fixture"
            fixture_log = fixture_dir / "run.log"
            fixture_env = env.copy()
            fixture_env["HEIRLOOM_CUDA_FIXTURE_DEVICE"] = os.environ.get(
                "HEIRLOOM_CUDA_FIXTURE_DEVICE", "cuda:0"
            )
            fixture_env["HEIRLOOM_CUDA_FIXTURE_OUT_DIR"] = str(fixture_dir)
            for name in [
                "HEIRLOOM_CUDA_FIXTURE_DEVICES",
                "HEIRLOOM_CUDA_FIXTURE_DISTRIBUTED",
                "HEIRLOOM_CUDA_FIXTURE_DDP_INIT_TIMEOUT_SECS",
                "HEIRLOOM_CUDA_FIXTURE_DDP_CHECKSUM_EVERY",
                "HEIRLOOM_CUDA_FIXTURE_STEPS",
                "HEIRLOOM_CUDA_FIXTURE_RESUME_STEPS",
                "HEIRLOOM_CUDA_FIXTURE_MIN_REDUCTION",
                "HEIRLOOM_CUDA_FIXTURE_BATCH_SIZE",
                "HEIRLOOM_CUDA_FIXTURE_BLOCK_SIZE",
                "HEIRLOOM_CUDA_FIXTURE_D_MODEL",
                "HEIRLOOM_CUDA_FIXTURE_N_HEADS",
                "HEIRLOOM_CUDA_FIXTURE_FF_HIDDEN",
                "HEIRLOOM_CUDA_FIXTURE_VOCAB_SIZE",
                "HEIRLOOM_CUDA_FIXTURE_AUTO_TILE",
                "HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES",
                "HEIRLOOM_CUDA_FIXTURE_RAGGED_ATTENTION_TENSOR_CORES",
                "HEIRLOOM_CUDA_FIXTURE_LR",
                "HEIRLOOM_CUDA_FIXTURE_SEED",
                "HEIRLOOM_CUDA_FIXTURE_PRECISION",
                "HEIRLOOM_REQUIRE_TENSOR_CORES",
                "HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM",
                "HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM",
                "HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM",
                "HEIRLOOM_EXPECT_TENSOR_CORES",
                "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES",
                "HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES",
                "HEIRLOOM_EXPECT_TENSOR_CORE_PADDING",
                "HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORE_PADDING",
                "HEIRLOOM_NCCL_TRACE",
                "HEIRLOOM_NCCL_DEBUG",
                "HEIRLOOM_NCCL_DEBUG_SUBSYS",
                "HEIRLOOM_NCCL_SOCKET_IFNAME",
                "HEIRLOOM_NCCL_IB_DISABLE",
                "HEIRLOOM_NCCL_NET_PLUGIN",
            ]:
                value = os.environ.get(name)
                if value and value != "__HEIRLOOM_UNSET__":
                    fixture_env[name] = value
            try:
                run_to_file(
                    ["bash", "scripts/cuda_train_lm_fixture.sh"],
                    fixture_log,
                    cwd=source_dir,
                    env=fixture_env,
                )
            finally:
                upload_diagnostic(fixture_log, "cuda-train-lm-fixture/run.log")
                for filename in ["train-report.json", "resume-report.json"]:
                    report_path = fixture_dir / filename
                    if report_path.exists():
                        report_uri = f"{artifact_prefix}/cuda-train-lm-fixture/{filename}"
                        upload_gcs(report_path, report_uri)
                        cuda_train_lm_fixture_report_uris.append(report_uri)
                for filename in [
                    "tensor-core-pad-crop-summary.json",
                    "tensor-core-pad-crop-summary.txt",
                ]:
                    report_path = fixture_dir / filename
                    if report_path.exists():
                        report_uri = f"{artifact_prefix}/cuda-train-lm-fixture/{filename}"
                        upload_gcs(report_path, report_uri)
                        cuda_train_lm_fixture_pad_crop_report_uris.append(report_uri)
                for dirname in ["train-ddp-ranks", "resume-ddp-ranks", "ddp-ranks"]:
                    rank_artifacts = fixture_dir / dirname
                    if rank_artifacts.exists():
                        upload_diagnostic_tree(
                            rank_artifacts,
                            f"cuda-train-lm-fixture/{dirname}",
                        )

        if run_cuda_train_memory_lm_fixture:
            memory_fixture_dir = work_root / "cuda-train-memory-lm-fixture"
            memory_fixture_log = memory_fixture_dir / "run.log"
            memory_fixture_env = env.copy()
            memory_fixture_env["HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICE"] = os.environ.get(
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICE", "cuda:0"
            )
            memory_fixture_env["HEIRLOOM_CUDA_MEMORY_FIXTURE_OUT_DIR"] = str(
                memory_fixture_dir
            )
            for name in [
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_INIT_TIMEOUT_SECS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_CHECKSUM_EVERY",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_SOURCE",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_INPUT",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_MAX_BYTES",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_SOURCE_BYTES",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_TRAIN_TOKENS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_BATCH_SIZE",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_BLOCK_SIZE",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_N_LAYERS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_D_MODEL",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_N_HEADS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_FF_HIDDEN",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LAYER_INDICES",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_SLOTS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_KEY_DIM",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_VALUE_DIM",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_TOP_K",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_HEADS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LOOKUP",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_SHARED_MEMORY",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_PLUS",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_VOCAB_SIZE",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_LR",
                "HEIRLOOM_CUDA_MEMORY_FIXTURE_SEED",
                "HEIRLOOM_NCCL_TRACE",
                "HEIRLOOM_NCCL_DEBUG",
                "HEIRLOOM_NCCL_DEBUG_SUBSYS",
                "HEIRLOOM_NCCL_SOCKET_IFNAME",
                "HEIRLOOM_NCCL_IB_DISABLE",
                "HEIRLOOM_NCCL_NET_PLUGIN",
            ]:
                value = os.environ.get(name)
                if value and value != "__HEIRLOOM_UNSET__":
                    memory_fixture_env[name] = value
            if (
                enable_nccl_debug
                and memory_fixture_env.get("HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED")
                == "nccl"
            ):
                memory_fixture_env.setdefault("NCCL_DEBUG", "INFO")
                memory_fixture_env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL,GRAPH")
                memory_fixture_env.setdefault("HEIRLOOM_NCCL_TRACE", "1")
            memory_fixture_succeeded = False
            try:
                run_to_file(
                    ["bash", "scripts/cuda_train_memory_lm_fixture.sh"],
                    memory_fixture_log,
                    cwd=source_dir,
                    env=memory_fixture_env,
                )
                memory_fixture_succeeded = True
            finally:
                upload_diagnostic(
                    memory_fixture_log, "cuda-train-memory-lm-fixture/run.log"
                )
                validator_log = memory_fixture_dir / "artifact-validator.txt"
                validator_command = [
                    "python3",
                    "scripts/validate_memory_fixture_artifacts.py",
                    str(memory_fixture_dir),
                ]
                if (
                    memory_fixture_env.get("HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED")
                    == "nccl"
                ):
                    validator_command.append("--expect-distributed")
                if (
                    memory_fixture_env.get(
                        "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY", ""
                    )
                    .lower()
                    .replace("_", "-")
                    == "sparse-rows"
                ):
                    validator_command.append("--expect-sparse-rows")
                resume_steps = int(
                    memory_fixture_env.get(
                        "HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS", "3"
                    )
                )
                if resume_steps > 0:
                    validator_command.append("--require-resume")
                    validator_command.extend(["--expect-resume-steps", str(resume_steps)])
                for env_name, flag_name in [
                    ("HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION", "--expect-precision"),
                    ("HEIRLOOM_CUDA_MEMORY_FIXTURE_N_LAYERS", "--expect-n-layers"),
                    (
                        "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LAYER_INDICES",
                        "--expect-memory-layer-indices",
                    ),
                    (
                        "HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY",
                        "--expect-memory-update-policy",
                    ),
                    ("HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE", "--expect-smft-mode"),
                    ("HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS", "--expect-steps"),
                    ("HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_SOURCE", "--expect-data-source"),
                    (
                        "HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_SOURCE_BYTES",
                        "--expect-min-source-bytes",
                    ),
                    (
                        "HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_TRAIN_TOKENS",
                        "--expect-min-train-tokens",
                    ),
                ]:
                    value = memory_fixture_env.get(env_name)
                    if value and value != "__HEIRLOOM_UNSET__":
                        validator_command.extend([flag_name, value])
                devices_for_validator = memory_fixture_env.get(
                    "HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES"
                )
                if devices_for_validator and devices_for_validator != "__HEIRLOOM_UNSET__":
                    world_size = len(
                        [
                            device
                            for device in devices_for_validator.split(",")
                            if device.strip()
                        ]
                    )
                    validator_command.extend(["--expect-world-size", str(world_size)])
                validator_command.append("--require-memory-kernel-counters")
                if (
                    memory_fixture_env.get("HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION")
                    == "amp-bf16"
                ):
                    validator_command.append("--require-tensor-core-counters")
                try:
                    run_to_file(
                        validator_command,
                        validator_log,
                        cwd=source_dir,
                        env=memory_fixture_env,
                    )
                except subprocess.CalledProcessError:
                    if memory_fixture_succeeded:
                        raise
                finally:
                    validator_uri = upload_diagnostic(
                        validator_log,
                        "cuda-train-memory-lm-fixture/artifact-validator.txt",
                    )
                    if validator_uri:
                        cuda_train_memory_lm_fixture_report_uris.append(validator_uri)
                for relative in [
                    "summary.json",
                    "train-report.json",
                    "resume-report.json",
                    "tokenizer.json",
                    "prepared/manifest.json",
                    "launcher-report.json",
                ]:
                    report_path = memory_fixture_dir / relative
                    if report_path.exists():
                        report_uri = (
                            f"{artifact_prefix}/cuda-train-memory-lm-fixture/{relative}"
                        )
                        upload_gcs(report_path, report_uri)
                        cuda_train_memory_lm_fixture_report_uris.append(report_uri)
                for dirname in [
                    "train-ddp-memory-ranks",
                    "resume-ddp-memory-ranks",
                    "ddp-memory-ranks",
                ]:
                    rank_artifacts = memory_fixture_dir / dirname
                    if rank_artifacts.exists():
                        upload_diagnostic_tree(
                            rank_artifacts,
                            f"cuda-train-memory-lm-fixture/{dirname}",
                        )
                memory_checkpoint_prefix = (
                    f"{artifact_prefix}/cuda-train-memory-lm-fixture/checkpoint"
                )
                uploaded_memory_checkpoint_uris = upload_directory_gcs(
                    memory_fixture_dir / "checkpoint",
                    memory_checkpoint_prefix,
                )
                if uploaded_memory_checkpoint_uris:
                    cuda_train_memory_lm_fixture_checkpoint_prefix = (
                        memory_checkpoint_prefix
                    )

        if run_tinystories_cuda_reference:
            reference_dir = work_root / "tinystories-cuda-reference"
            reference_log = reference_dir / "run.log"
            reference_env = env.copy()
            reference_env["HEIRLOOM_TINYSTORIES_CUDA_DEVICE"] = os.environ.get(
                "HEIRLOOM_TINYSTORIES_CUDA_DEVICE", "cuda:0"
            )
            optional_reference_names = [
                "HEIRLOOM_TINYSTORIES_CUDA_MODE",
                "HEIRLOOM_TINYSTORIES_CUDA_DEVICES",
                "HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED",
                "HEIRLOOM_REQUIRE_TENSOR_CORES",
                "HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM",
                "HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM",
                "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES",
                "HEIRLOOM_TINYSTORIES_CUDA_STEPS",
                "HEIRLOOM_TINYSTORIES_CUDA_BATCH",
                "HEIRLOOM_TINYSTORIES_CUDA_BLOCK",
                "HEIRLOOM_TINYSTORIES_CUDA_D_MODEL",
                "HEIRLOOM_TINYSTORIES_CUDA_HEADS",
                "HEIRLOOM_TINYSTORIES_CUDA_FF",
                "HEIRLOOM_TINYSTORIES_CUDA_LR",
                "HEIRLOOM_TINYSTORIES_CUDA_MIN_REDUCTION",
                "HEIRLOOM_TINYSTORIES_CUDA_EVAL_BATCHES",
                "HEIRLOOM_TINYSTORIES_CUDA_VOCAB",
                "HEIRLOOM_TINYSTORIES_CUDA_MAX_BYTES",
                "HEIRLOOM_TINYSTORIES_CUDA_GENERATION_TOKENS",
                "HEIRLOOM_TINYSTORIES_CUDA_RESUME_STEPS",
                "HEIRLOOM_TINYSTORIES_CUDA_PRECISION",
            ]
            for name in optional_reference_names:
                value = os.environ.get(name)
                if value and value != "__HEIRLOOM_UNSET__":
                    reference_env[name] = value
                else:
                    reference_env.pop(name, None)
            if (
                enable_nccl_debug
                and reference_env.get("HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED") == "nccl"
            ):
                reference_env.setdefault("NCCL_DEBUG", "INFO")
                reference_env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL,GRAPH")
                reference_env.setdefault("HEIRLOOM_NCCL_TRACE", "1")
            nccl_net_plugin = os.environ.get("HEIRLOOM_NCCL_NET_PLUGIN")
            if nccl_net_plugin and nccl_net_plugin != "__HEIRLOOM_UNSET__":
                reference_env["HEIRLOOM_NCCL_NET_PLUGIN"] = nccl_net_plugin
            try:
                run_to_file(
                    ["bash", "scripts/run_tinystories_cuda_reference.sh", str(reference_dir)],
                    reference_log,
                    cwd=source_dir,
                    env=reference_env,
                )
            finally:
                upload_diagnostic(reference_log, "tinystories-cuda-reference/run.log")
                for relative in [
                    "summary.json",
                    "timings.json",
                    "report.json",
                    "resume-report.json",
                    "eval.json",
                    "generation.json",
                    "generation.txt",
                    "tokenizer.json",
                    "prepared/manifest.json",
                    "launcher-report.json",
                    "train-launcher-report.json",
                    "resume-launcher-report.json",
                ]:
                    report_path = reference_dir / relative
                    if report_path.exists():
                        report_uri = f"{artifact_prefix}/tinystories-cuda-reference/{relative}"
                        upload_gcs(report_path, report_uri)
                        tinystories_cuda_reference_report_uris.append(report_uri)
                for dirname in ["train-ddp-ranks", "resume-ddp-ranks", "ddp-ranks"]:
                    rank_artifacts = reference_dir / dirname
                    if rank_artifacts.exists():
                        upload_diagnostic_tree(
                            rank_artifacts,
                            f"tinystories-cuda-reference/{dirname}",
                        )
                checkpoint_prefix = f"{artifact_prefix}/tinystories-cuda-reference/checkpoint"
                uploaded_checkpoint_uris = upload_directory_gcs(
                    reference_dir / "checkpoint",
                    checkpoint_prefix,
                )
                if uploaded_checkpoint_uris:
                    tinystories_cuda_reference_checkpoint_prefix = checkpoint_prefix
                resume_checkpoint_prefix = (
                    f"{artifact_prefix}/tinystories-cuda-reference/checkpoint-resume"
                )
                upload_directory_gcs(
                    reference_dir / "checkpoint-resume",
                    resume_checkpoint_prefix,
                )

        if run_qb_data_hardpath:
            hardpath_dir = work_root / "qb-data-hardpath"
            hardpath_log = hardpath_dir / "run.log"
            hardpath_env = env.copy()
            optional_hardpath_names = [
                "HEIRLOOM_QB_DATA_HARDPATH_MODEL",
                "HEIRLOOM_QB_DATA_HARDPATH_MODE",
                "HEIRLOOM_QB_DATA_HARDPATH_PROFILE",
                "HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE",
                "HEIRLOOM_QB_DATA_HARDPATH_DEVICE",
                "HEIRLOOM_QB_DATA_HARDPATH_DEVICES",
                "HEIRLOOM_QB_DATA_HARDPATH_DISTRIBUTED",
                "HEIRLOOM_QB_DATA_HARDPATH_PRECISION",
                "HEIRLOOM_QB_DATA_HARDPATH_STEPS",
                "HEIRLOOM_QB_DATA_HARDPATH_BATCH",
                "HEIRLOOM_QB_DATA_HARDPATH_GRAD_ACCUMULATION_STEPS",
                "HEIRLOOM_QB_DATA_HARDPATH_BLOCK",
                "HEIRLOOM_QB_DATA_HARDPATH_D_MODEL",
                "HEIRLOOM_QB_DATA_HARDPATH_HEADS",
                "HEIRLOOM_QB_DATA_HARDPATH_FF",
                "HEIRLOOM_QB_DATA_HARDPATH_LR",
                "HEIRLOOM_QB_DATA_HARDPATH_MIN_REDUCTION",
                "HEIRLOOM_QB_DATA_HARDPATH_EVAL_BATCHES",
                "HEIRLOOM_QB_DATA_HARDPATH_VOCAB",
                "HEIRLOOM_QB_DATA_HARDPATH_MAX_BYTES",
                "HEIRLOOM_QB_DATA_HARDPATH_SHARD_TOKENS",
                "HEIRLOOM_QB_DATA_HARDPATH_GENERATION_TOKENS",
                "HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS",
                "HEIRLOOM_QB_DATA_HARDPATH_LOG_EVERY",
                "HEIRLOOM_QB_DATA_HARDPATH_RESUME_LOG_EVERY",
                "HEIRLOOM_QB_DATA_HARDPATH_DDP_INIT_TIMEOUT_SECS",
                "HEIRLOOM_QB_DATA_HARDPATH_DDP_CHECKSUM_EVERY",
                "HEIRLOOM_QB_DATA_HARDPATH_SKIP_EVAL_GENERATION",
                "HEIRLOOM_QB_DATA_HARDPATH_THROUGHPUT_REPORT",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_N_LAYERS",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LAYER_INDICES",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SLOTS",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_KEY_DIM",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_VALUE_DIM",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_TOP_K",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_HEADS",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LOOKUP",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SHARED_MEMORY",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_PLUS",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_UPDATE_POLICY",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_MODE",
                "HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_ROW_MASK",
                "HEIRLOOM_TINYSTORIES_CUDA_DEVICE",
                "HEIRLOOM_TINYSTORIES_CUDA_DEVICES",
                "HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED",
                "HEIRLOOM_TINYSTORIES_CUDA_STEPS",
                "HEIRLOOM_TINYSTORIES_CUDA_BATCH",
                "HEIRLOOM_TINYSTORIES_CUDA_GRAD_ACCUMULATION_STEPS",
                "HEIRLOOM_TINYSTORIES_CUDA_BLOCK",
                "HEIRLOOM_TINYSTORIES_CUDA_D_MODEL",
                "HEIRLOOM_TINYSTORIES_CUDA_HEADS",
                "HEIRLOOM_TINYSTORIES_CUDA_FF",
                "HEIRLOOM_TINYSTORIES_CUDA_LR",
                "HEIRLOOM_TINYSTORIES_CUDA_MIN_REDUCTION",
                "HEIRLOOM_TINYSTORIES_CUDA_EVAL_BATCHES",
                "HEIRLOOM_TINYSTORIES_CUDA_VOCAB",
                "HEIRLOOM_TINYSTORIES_CUDA_MAX_BYTES",
                "HEIRLOOM_TINYSTORIES_CUDA_GENERATION_TOKENS",
                "HEIRLOOM_TINYSTORIES_CUDA_RESUME_STEPS",
                "HEIRLOOM_TINYSTORIES_CUDA_PRECISION",
                "HEIRLOOM_REQUIRE_TENSOR_CORES",
                "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES",
                "HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM",
                "HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_LDMATRIX_GEMM",
                "HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM",
                "HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM",
                "HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING",
                "HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM",
                "HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM",
                "HEIRLOOM_CUDA_FLASH_BF16_ATTENTION",
                "HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE",
                "HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD",
                "HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING",
                "HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION",
                "HEIRLOOM_EXPECT_FLASH_BF16_ATTENTION",
            ]
            for name in optional_hardpath_names:
                value = os.environ.get(name)
                if value and value != "__HEIRLOOM_UNSET__":
                    hardpath_env[name] = value
                else:
                    hardpath_env.pop(name, None)
            if (
                enable_nccl_debug
                and (
                    hardpath_env.get("HEIRLOOM_QB_DATA_HARDPATH_DISTRIBUTED") == "nccl"
                    or hardpath_env.get("HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED") == "nccl"
                )
            ):
                hardpath_env.setdefault("NCCL_DEBUG", "INFO")
                hardpath_env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL,GRAPH")
                hardpath_env.setdefault("HEIRLOOM_NCCL_TRACE", "1")
            nccl_net_plugin = os.environ.get("HEIRLOOM_NCCL_NET_PLUGIN")
            if nccl_net_plugin and nccl_net_plugin != "__HEIRLOOM_UNSET__":
                hardpath_env["HEIRLOOM_NCCL_NET_PLUGIN"] = nccl_net_plugin
            try:
                run_to_file(
                    ["bash", "scripts/run_qb_data_hardpath.sh", str(hardpath_dir)],
                    hardpath_log,
                    cwd=source_dir,
                    env=hardpath_env,
                )
            finally:
                upload_diagnostic(hardpath_log, "qb-data-hardpath/run.log")
                for relative in [
                    "summary.json",
                    "timings.json",
                    "report.json",
                    "resume-report.json",
                    "eval.json",
                    "generation.json",
                    "generation.txt",
                    "tokenizer.json",
                    "qb-traces.jsonl",
                    "qb-traces.txt",
                    "smft-row-mask.json",
                    "prepared/manifest.json",
                    "launcher-report.json",
                    "train-launcher-report.json",
                    "resume-launcher-report.json",
                ]:
                    report_path = hardpath_dir / relative
                    if report_path.exists():
                        report_uri = f"{artifact_prefix}/qb-data-hardpath/{relative}"
                        upload_gcs(report_path, report_uri)
                        qb_data_hardpath_report_uris.append(report_uri)
                shard_artifacts = hardpath_dir / "prepared" / "shards"
                if shard_artifacts.exists():
                    qb_data_hardpath_report_uris.extend(
                        upload_directory_gcs(
                            shard_artifacts,
                            f"{artifact_prefix}/qb-data-hardpath/prepared/shards",
                        )
                    )
                for dirname in [
                    "train-ddp-ranks",
                    "resume-ddp-ranks",
                    "ddp-ranks",
                    "train-ddp-memory-ranks",
                    "resume-ddp-memory-ranks",
                    "ddp-memory-ranks",
                ]:
                    rank_artifacts = hardpath_dir / dirname
                    if rank_artifacts.exists():
                        upload_diagnostic_tree(
                            rank_artifacts,
                            f"qb-data-hardpath/{dirname}",
                        )
                checkpoint_prefix = f"{artifact_prefix}/qb-data-hardpath/checkpoint"
                uploaded_checkpoint_uris = upload_directory_gcs(
                    hardpath_dir / "checkpoint",
                    checkpoint_prefix,
                )
                if uploaded_checkpoint_uris:
                    qb_data_hardpath_checkpoint_prefix = checkpoint_prefix
                resume_checkpoint_prefix = (
                    f"{artifact_prefix}/qb-data-hardpath/checkpoint-resume"
                )
                upload_directory_gcs(
                    hardpath_dir / "checkpoint-resume",
                    resume_checkpoint_prefix,
                )

        if run_qb_tokenizer_hardpath:
            tokenizer_dir = work_root / "qb-tokenizer-hardpath"
            tokenizer_log = tokenizer_dir / "run.log"
            tokenizer_env = env.copy()
            optional_tokenizer_names = [
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MODE",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_QB_ROOT",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_VOCAB",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_SAMPLE_BYTES",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_TARGET_TOKENS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MATERIALIZE_MODE",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_SEED",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_VALID_FRACTION",
                "HEIRLOOM_QB_TOKENIZER_SOURCE_STAGE_DIR",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_DEVICE",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_DEVICES",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_DISTRIBUTED",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_PRECISION",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_STEPS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_RESUME_STEPS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_BATCH",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_GRAD_ACCUMULATION_STEPS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_BLOCK",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_D_MODEL",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_HEADS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_FF",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_LR",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_EVAL_BATCHES",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_GENERATION_TOKENS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_SHARD_TOKENS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_N_LAYERS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_LAYER_INDICES",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_SLOTS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_KEY_DIM",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_VALUE_DIM",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_TOP_K",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_HEADS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_LOSS_REDUCTION",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_TOKENS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_DOCS",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_FINAL_STEP",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_BLEND_SOURCES",
                "HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_REQUIRE_PRODUCTION_GATE",
                "HEIRLOOM_QB_TOKENIZER_DOLMA_PATH",
                "HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH",
                "HEIRLOOM_QB_TOKENIZER_OLMO3_PATH",
                "HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH",
                "HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH",
                "HEIRLOOM_NCCL_NET_PLUGIN",
                "HEIRLOOM_REQUIRE_TENSOR_CORES",
                "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES",
                "HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM",
                "HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM",
            ]
            for name in optional_tokenizer_names:
                value = os.environ.get(name)
                if value and value != "__HEIRLOOM_UNSET__":
                    tokenizer_env[name] = value
                else:
                    tokenizer_env.pop(name, None)
            if (
                enable_nccl_debug
                and tokenizer_env.get("HEIRLOOM_QB_TOKENIZER_HARDPATH_DISTRIBUTED") == "nccl"
            ):
                tokenizer_env.setdefault("NCCL_DEBUG", "INFO")
                tokenizer_env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL,GRAPH")
                tokenizer_env.setdefault("HEIRLOOM_NCCL_TRACE", "1")
            try:
                run_to_file(
                    ["bash", "scripts/run_qb_tokenizer_hardpath.sh", str(tokenizer_dir)],
                    tokenizer_log,
                    cwd=source_dir,
                    env=tokenizer_env,
                )
            finally:
                upload_diagnostic(tokenizer_log, "qb-tokenizer-hardpath/run.log")
                for relative in [
                    "summary.json",
                    "corpus-blend.json",
                    "tokenizer.json",
                    "tokenizer-train-report.json",
                    "tokenizer-fertility-report.json",
                    "materialized/curation-report.json",
                    "materialized/source-index.json",
                    "materialized/selected-docs.jsonl",
                    "materialized/tokenizer-sample-manifest.json",
                    "materialized/prepared/manifest.json",
                    "train-report.json",
                    "resume-report.json",
                    "eval-report.json",
                    "generation-report.json",
                    "learning-sanity-ladder.json",
                    "learning-sanity-validation.json",
                    "launcher-report.json",
                    "train-launcher-report.json",
                    "resume-launcher-report.json",
                ]:
                    report_path = tokenizer_dir / relative
                    if report_path.exists():
                        report_uri = f"{artifact_prefix}/qb-tokenizer-hardpath/{relative}"
                        upload_gcs(report_path, report_uri)
                        qb_tokenizer_hardpath_report_uris.append(report_uri)
                shard_artifacts = tokenizer_dir / "materialized" / "prepared" / "shards"
                if shard_artifacts.exists():
                    qb_tokenizer_hardpath_report_uris.extend(
                        upload_directory_gcs(
                            shard_artifacts,
                            f"{artifact_prefix}/qb-tokenizer-hardpath/materialized/prepared/shards",
                        )
                    )
                tokenizer_work = tokenizer_dir / "tokenizer-work"
                if tokenizer_work.exists():
                    upload_diagnostic_tree(
                        tokenizer_work,
                        "qb-tokenizer-hardpath/tokenizer-work",
                    )
                for dirname in [
                    "train-ddp-memory-ranks",
                    "resume-ddp-memory-ranks",
                    "ddp-memory-ranks",
                ]:
                    rank_artifacts = tokenizer_dir / dirname
                    if rank_artifacts.exists():
                        upload_diagnostic_tree(
                            rank_artifacts,
                            f"qb-tokenizer-hardpath/{dirname}",
                        )
                checkpoint_prefix = f"{artifact_prefix}/qb-tokenizer-hardpath/checkpoint"
                uploaded_checkpoint_uris = upload_directory_gcs(
                    tokenizer_dir / "checkpoint",
                    checkpoint_prefix,
                )
                if uploaded_checkpoint_uris:
                    qb_tokenizer_hardpath_checkpoint_prefix = checkpoint_prefix

    if run_learning_sanity_fixtures:
        fixture_dir = work_root / "learning-sanity-fixtures"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        for name, manifest in [
            ("ladder-valid", "tests/fixtures/learning_sanity/ladder-valid.json"),
            (
                "lr-grad-sweep-valid",
                "tests/fixtures/learning_sanity/lr-grad-sweep-valid.json",
            ),
            (
                "longer-32k-blend-valid",
                "tests/fixtures/learning_sanity/longer-32k-blend-valid.json",
            ),
        ]:
            report_path = fixture_dir / f"{name}-validation.json"
            log_path = fixture_dir / f"{name}.txt"
            try:
                run_to_file(
                    [
                        "cargo",
                        "run",
                        "--bin",
                        "heirloom",
                        "--",
                        "readiness",
                        "validate-learning-sanity",
                        "--manifest",
                        manifest,
                        "--report",
                        str(report_path),
                    ],
                    log_path,
                    cwd=source_dir,
                    env=env,
                )
            finally:
                upload_diagnostic(log_path, f"learning-sanity-fixtures/{name}.txt")
                if report_path.exists():
                    report_uri = (
                        f"{artifact_prefix}/learning-sanity-fixtures/"
                        f"{name}-validation.json"
                    )
                    upload_gcs(report_path, report_uri)
                    learning_sanity_fixture_report_uris.append(report_uri)

    if validate_mode == "quick":
        run(["cargo", "fmt", "--all", "--check"], cwd=source_dir, env=env)
        run(
            ["cargo", "test", "--workspace", "--exclude", "heirloom-python"],
            cwd=source_dir,
            env=env,
        )
        if run_quick_clippy:
            run(
                [
                    "cargo",
                    "clippy",
                    "--workspace",
                    "--exclude",
                    "heirloom-python",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                ],
                cwd=source_dir,
                env=env,
            )
        else:
            print(
                "Skipping quick clippy because HEIRLOOM_RUN_QUICK_CLIPPY=0",
                flush=True,
            )
    elif validate_mode == "full":
        run(["bash", "scripts/validate.sh"], cwd=source_dir, env=env)
    else:
        raise ValueError(f"unsupported HEIRLOOM_VALIDATE_MODE={validate_mode}")

    summary_path = work_root / "summary.json"
    summary = {
        "mode": validate_mode,
        "status": "passed",
        "source_uri": source_uri,
        "artifact_prefix": artifact_prefix,
        "quick_clippy": quick_clippy_status,
        "gpu_smoke_report_uris": gpu_report_uris,
        "gpu_log_uris": gpu_log_uris,
        "topology_report_uris": topology_report_uris,
        "topology_log_uris": topology_log_uris,
        "tensor_core_probe_report_uris": tensor_core_probe_report_uris,
        "tensor_core_probe_log_uris": tensor_core_probe_log_uris,
        "tensor_core_microbench_report_uris": tensor_core_microbench_report_uris,
        "tensor_core_microbench_log_uris": tensor_core_microbench_log_uris,
        "tensor_core_microbench_sections_requested": tensor_core_microbench_sections_requested,
        "tensor_core_microbench_sections_completed": tensor_core_microbench_sections_completed,
        "tensor_core_microbench_sections_failed": tensor_core_microbench_sections_failed,
        "nccl_probe_report_uris": nccl_probe_report_uris,
        "nccl_probe_log_uris": nccl_probe_log_uris,
        "diagnostic_log_uris": diagnostic_log_uris,
        "cuda_train_lm_fixture_report_uris": cuda_train_lm_fixture_report_uris,
        "cuda_train_lm_fixture_pad_crop_report_uris": cuda_train_lm_fixture_pad_crop_report_uris,
        "cuda_train_memory_lm_fixture_report_uris": cuda_train_memory_lm_fixture_report_uris,
        "cuda_train_memory_lm_fixture_checkpoint_prefix": cuda_train_memory_lm_fixture_checkpoint_prefix,
        "tinystories_cuda_reference_report_uris": tinystories_cuda_reference_report_uris,
        "tinystories_cuda_reference_checkpoint_prefix": tinystories_cuda_reference_checkpoint_prefix,
        "qb_data_hardpath_report_uris": qb_data_hardpath_report_uris,
        "qb_data_hardpath_checkpoint_prefix": qb_data_hardpath_checkpoint_prefix,
        "qb_tokenizer_hardpath_report_uris": qb_tokenizer_hardpath_report_uris,
        "qb_tokenizer_hardpath_checkpoint_prefix": qb_tokenizer_hardpath_checkpoint_prefix,
        "learning_sanity_fixture_report_uris": learning_sanity_fixture_report_uris,
    }
    if quick_clippy_reason is not None:
        summary["quick_clippy_reason"] = quick_clippy_reason
    summary_path.write_text(
        json.dumps(summary, indent=2)
        + "\n",
        encoding="utf-8",
    )
    summary_uri = f"{artifact_prefix}/summary.json"
    upload_gcs(summary_path, summary_uri)
    print(
        "HEIRLOOM_VERTEX_VALIDATE_SUMMARY "
        + f"mode={validate_mode} status=passed artifact_prefix={artifact_prefix} summary={summary_uri}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        artifact_prefix = os.environ.get("HEIRLOOM_ARTIFACT_PREFIX")
        if artifact_prefix:
            failure_path = pathlib.Path("/tmp/heirloom-vertex-failure-summary.json")
            failure_path.write_text(
                json.dumps(
                    {
                        "mode": os.environ.get("HEIRLOOM_VALIDATE_MODE", "quick"),
                        "status": "failed",
                        "source_uri": os.environ.get("HEIRLOOM_SOURCE_URI"),
                        "artifact_prefix": artifact_prefix,
                        "error_type": type(err).__name__,
                        "error": str(err),
                        "note": "Inspect diagnostic *.txt artifacts under artifact_prefix for streamed command output. Full Vertex job specs are intentionally not dumped because they can contain env values.",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            failure_uri = f"{artifact_prefix}/failure-summary.json"
            try:
                upload_gcs(failure_path, failure_uri)
                print(
                    "HEIRLOOM_VERTEX_VALIDATE_FAILURE "
                    + f"summary={failure_uri} artifact_prefix={artifact_prefix}",
                    flush=True,
                )
            except Exception as upload_err:
                print(
                    "HEIRLOOM_VERTEX_VALIDATE_FAILURE "
                    + f"artifact_prefix={artifact_prefix} failure_summary_upload_error={upload_err}",
                    flush=True,
                )
        raise
PY

cat > "$JOB_PKG_DIR/setup.py" <<'PY'
from setuptools import find_packages, setup

setup(
    name="heirloom-vertex-validate",
    version="0.0.0",
    packages=find_packages(),
    install_requires=["google-cloud-storage>=2.18"],
)
PY

tar -C "$JOB_PKG_DIR" -czf "$PACKAGE_TGZ" .

print "Checking artifact bucket ${BUCKET}"
retry_cloud_command "bucket-describe:${BUCKET}" \
  gcloud storage buckets describe "$BUCKET" --project="$PROJECT_ID" >/dev/null
retry_cloud_command "upload-source:${SOURCE_URI}" \
  gcloud storage cp "$SOURCE_TGZ" "$SOURCE_URI" --project="$PROJECT_ID"
retry_cloud_command "upload-package:${PACKAGE_URI}" \
  gcloud storage cp "$PACKAGE_TGZ" "$PACKAGE_URI" --project="$PROJECT_ID"

cat > "$CONFIG_YAML" <<EOF
workerPoolSpecs:
- machineSpec:
    machineType: ${MACHINE_TYPE}
    acceleratorType: ${ACCELERATOR_TYPE}
    acceleratorCount: ${ACCELERATOR_COUNT}
  replicaCount: 1
  diskSpec:
    bootDiskType: pd-ssd
    bootDiskSizeGb: 300
  pythonPackageSpec:
    executorImageUri: us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
    packageUris:
    - ${PACKAGE_URI}
    pythonModule: heirloom_vertex_validate_job
    env:
    - name: PROJECT_ID
      value: "${PROJECT_ID}"
    - name: HEIRLOOM_SOURCE_URI
      value: "${SOURCE_URI}"
    - name: HEIRLOOM_VALIDATE_MODE
      value: "${VALIDATE_MODE}"
    - name: HEIRLOOM_RUN_LEARNING_SANITY_FIXTURES
      value: "${HEIRLOOM_RUN_LEARNING_SANITY_FIXTURES:-0}"
    - name: HEIRLOOM_REQUIRE_GPU
      value: "${HEIRLOOM_REQUIRE_GPU:-1}"
    - name: HEIRLOOM_ACCELERATOR_COUNT
      value: "${ACCELERATOR_COUNT}"
    - name: HEIRLOOM_GPU_SMOKE_LEN
      value: "${HEIRLOOM_GPU_SMOKE_LEN:-4096}"
    - name: HEIRLOOM_GPU_SMOKE_DEVICES
      value: "${HEIRLOOM_GPU_SMOKE_DEVICES:-0}"
    - name: HEIRLOOM_COLLECT_GPU_TOPOLOGY
      value: "${HEIRLOOM_COLLECT_GPU_TOPOLOGY:-1}"
    - name: HEIRLOOM_RUN_QUICK_CLIPPY
      value: "${HEIRLOOM_RUN_QUICK_CLIPPY:-1}"
    - name: HEIRLOOM_RUN_TENSOR_CORE_PROBE
      value: "${HEIRLOOM_RUN_TENSOR_CORE_PROBE:-1}"
    - name: HEIRLOOM_RUN_TENSOR_CORE_MICROBENCH
      value: "${HEIRLOOM_RUN_TENSOR_CORE_MICROBENCH:-0}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_DEVICE
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_DEVICE:-0}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_ITERATIONS
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_ITERATIONS:-32}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_WARMUP
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_WARMUP:-4}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_SECTIONS
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_SECTIONS:-gemm,attention}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_M
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_M:-4096}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_K
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_K:-1024}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_N
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_N:-4096}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_BATCH
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_BATCH:-4}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEADS
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEADS:-16}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_TIME
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_TIME:-512}"
    - name: HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEAD_DIM
      value: "${HEIRLOOM_TENSOR_CORE_MICROBENCH_ATTENTION_HEAD_DIM:-64}"
    - name: HEIRLOOM_ENABLE_NCCL_DEBUG
      value: "${HEIRLOOM_ENABLE_NCCL_DEBUG:-1}"
    - name: HEIRLOOM_RUN_NCCL_PROBE
      value: "${HEIRLOOM_RUN_NCCL_PROBE:-0}"
    - name: HEIRLOOM_NCCL_PROBE_DEVICES
      value: "${HEIRLOOM_NCCL_PROBE_DEVICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_NCCL_PROBE_LEN
      value: "${HEIRLOOM_NCCL_PROBE_LEN:-1024}"
    - name: HEIRLOOM_NCCL_PROBE_KIND
      value: "${HEIRLOOM_NCCL_PROBE_KIND:-all-reduce}"
    - name: HEIRLOOM_NCCL_PROBE_TIMEOUT_SECS
      value: "${HEIRLOOM_NCCL_PROBE_TIMEOUT_SECS:-120}"
    - name: HEIRLOOM_NCCL_PROBE_RANK_START_TIMEOUT_SECS
      value: "${HEIRLOOM_NCCL_PROBE_RANK_START_TIMEOUT_SECS:-10}"
    - name: HEIRLOOM_NCCL_PROBE_KILL_GRACE_SECS
      value: "${HEIRLOOM_NCCL_PROBE_KILL_GRACE_SECS:-5}"
    - name: HEIRLOOM_NCCL_TRACE
      value: "${HEIRLOOM_NCCL_TRACE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_NCCL_DEBUG
      value: "${HEIRLOOM_NCCL_DEBUG:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_NCCL_DEBUG_SUBSYS
      value: "${HEIRLOOM_NCCL_DEBUG_SUBSYS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_NCCL_SOCKET_IFNAME
      value: "${HEIRLOOM_NCCL_SOCKET_IFNAME:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_NCCL_IB_DISABLE
      value: "${HEIRLOOM_NCCL_IB_DISABLE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_RUN_CUDA_STORAGE_TESTS
      value: "${HEIRLOOM_RUN_CUDA_STORAGE_TESTS:-1}"
    - name: HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE
      value: "${HEIRLOOM_RUN_CUDA_TRAIN_LM_FIXTURE:-0}"
    - name: HEIRLOOM_RUN_CUDA_TRAIN_MEMORY_LM_FIXTURE
      value: "${HEIRLOOM_RUN_CUDA_TRAIN_MEMORY_LM_FIXTURE:-0}"
    - name: HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM
      value: "${HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_LDMATRIX_GEMM
      value: "${HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_LDMATRIX_GEMM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM
      value: "${HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM
      value: "${HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING
      value: "${HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FLASH_BF16_ATTENTION
      value: "${HEIRLOOM_CUDA_FLASH_BF16_ATTENTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE
      value: "${HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD
      value: "${HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING
      value: "${HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION
      value: "${HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_EXPECT_FLASH_BF16_ATTENTION
      value: "${HEIRLOOM_EXPECT_FLASH_BF16_ATTENTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_DEVICE
      value: "${HEIRLOOM_CUDA_FIXTURE_DEVICE:-cuda:0}"
    - name: HEIRLOOM_CUDA_FIXTURE_DEVICES
      value: "${HEIRLOOM_CUDA_FIXTURE_DEVICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_DISTRIBUTED
      value: "${HEIRLOOM_CUDA_FIXTURE_DISTRIBUTED:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_DDP_INIT_TIMEOUT_SECS
      value: "${HEIRLOOM_CUDA_FIXTURE_DDP_INIT_TIMEOUT_SECS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_DDP_CHECKSUM_EVERY
      value: "${HEIRLOOM_CUDA_FIXTURE_DDP_CHECKSUM_EVERY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_STEPS
      value: "${HEIRLOOM_CUDA_FIXTURE_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_RESUME_STEPS
      value: "${HEIRLOOM_CUDA_FIXTURE_RESUME_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_MIN_REDUCTION
      value: "${HEIRLOOM_CUDA_FIXTURE_MIN_REDUCTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_BATCH_SIZE
      value: "${HEIRLOOM_CUDA_FIXTURE_BATCH_SIZE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_BLOCK_SIZE
      value: "${HEIRLOOM_CUDA_FIXTURE_BLOCK_SIZE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_D_MODEL
      value: "${HEIRLOOM_CUDA_FIXTURE_D_MODEL:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_N_HEADS
      value: "${HEIRLOOM_CUDA_FIXTURE_N_HEADS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_FF_HIDDEN
      value: "${HEIRLOOM_CUDA_FIXTURE_FF_HIDDEN:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_VOCAB_SIZE
      value: "${HEIRLOOM_CUDA_FIXTURE_VOCAB_SIZE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_AUTO_TILE
      value: "${HEIRLOOM_CUDA_FIXTURE_AUTO_TILE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES
      value: "${HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_RAGGED_ATTENTION_TENSOR_CORES
      value: "${HEIRLOOM_CUDA_FIXTURE_RAGGED_ATTENTION_TENSOR_CORES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_LR
      value: "${HEIRLOOM_CUDA_FIXTURE_LR:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_SEED
      value: "${HEIRLOOM_CUDA_FIXTURE_SEED:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_FIXTURE_PRECISION
      value: "${HEIRLOOM_CUDA_FIXTURE_PRECISION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICE
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICE:-cuda:0}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_INIT_TIMEOUT_SECS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_INIT_TIMEOUT_SECS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_CHECKSUM_EVERY
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_CHECKSUM_EVERY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_SOURCE
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_SOURCE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_INPUT
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_INPUT:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_MAX_BYTES
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_MAX_BYTES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_SOURCE_BYTES
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_SOURCE_BYTES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_TRAIN_TOKENS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_EXPECT_MIN_TRAIN_TOKENS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_BATCH_SIZE
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_BATCH_SIZE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_BLOCK_SIZE
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_BLOCK_SIZE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_N_LAYERS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_N_LAYERS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_D_MODEL
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_D_MODEL:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_N_HEADS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_N_HEADS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_FF_HIDDEN
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_FF_HIDDEN:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LAYER_INDICES
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LAYER_INDICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_SLOTS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_SLOTS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_KEY_DIM
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_KEY_DIM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_VALUE_DIM
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_VALUE_DIM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_TOP_K
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_TOP_K:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_HEADS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_HEADS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LOOKUP
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LOOKUP:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_SHARED_MEMORY
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_SHARED_MEMORY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_PLUS
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_PLUS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_VOCAB_SIZE
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_VOCAB_SIZE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_LR
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_LR:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_MEMORY_FIXTURE_SEED
      value: "${HEIRLOOM_CUDA_MEMORY_FIXTURE_SEED:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_EXPECT_TENSOR_CORES
      value: "${HEIRLOOM_EXPECT_TENSOR_CORES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES
      value: "${HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_EXPECT_TENSOR_CORE_PADDING
      value: "${HEIRLOOM_EXPECT_TENSOR_CORE_PADDING:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORE_PADDING
      value: "${HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORE_PADDING:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE
      value: "${HEIRLOOM_RUN_TINYSTORIES_CUDA_REFERENCE:-0}"
    - name: HEIRLOOM_RUN_QB_DATA_HARDPATH
      value: "${HEIRLOOM_RUN_QB_DATA_HARDPATH:-0}"
    - name: HEIRLOOM_RUN_QB_TOKENIZER_HARDPATH
      value: "${HEIRLOOM_RUN_QB_TOKENIZER_HARDPATH:-0}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MODE
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MODE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_QB_ROOT
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_QB_ROOT:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_VOCAB
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_VOCAB:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_SAMPLE_BYTES
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_SAMPLE_BYTES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_TARGET_TOKENS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_TARGET_TOKENS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MATERIALIZE_MODE
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MATERIALIZE_MODE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_SEED
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_SEED:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_VALID_FRACTION
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_VALID_FRACTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_SOURCE_STAGE_DIR
      value: "${HEIRLOOM_QB_TOKENIZER_SOURCE_STAGE_DIR:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_DEVICE
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_DEVICE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_DEVICES
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_DEVICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_DISTRIBUTED
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_DISTRIBUTED:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_PRECISION
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_PRECISION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_STEPS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_RESUME_STEPS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_RESUME_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_BATCH
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_BATCH:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_GRAD_ACCUMULATION_STEPS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_GRAD_ACCUMULATION_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_BLOCK
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_BLOCK:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_D_MODEL
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_D_MODEL:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_HEADS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_HEADS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_FF
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_FF:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_LR
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_LR:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_EVAL_BATCHES
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_EVAL_BATCHES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_GENERATION_TOKENS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_GENERATION_TOKENS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_SHARD_TOKENS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_SHARD_TOKENS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_N_LAYERS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_N_LAYERS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_LAYER_INDICES
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_LAYER_INDICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_SLOTS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_SLOTS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_KEY_DIM
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_KEY_DIM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_VALUE_DIM
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_VALUE_DIM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_TOP_K
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_TOP_K:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_HEADS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_HEADS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_LOSS_REDUCTION
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_LOSS_REDUCTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_TOKENS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_TOKENS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_DOCS
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_DOCS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_FINAL_STEP
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_FINAL_STEP:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_BLEND_SOURCES
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_BLEND_SOURCES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_REQUIRE_PRODUCTION_GATE
      value: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_REQUIRE_PRODUCTION_GATE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_DOLMA_PATH
      value: "${HEIRLOOM_QB_TOKENIZER_DOLMA_PATH:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH
      value: "${HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_OLMO3_PATH
      value: "${HEIRLOOM_QB_TOKENIZER_OLMO3_PATH:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH
      value: "${HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH
      value: "${HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MODEL
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MODEL:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MODE
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MODE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_PROFILE
      value: "${HEIRLOOM_QB_DATA_HARDPATH_PROFILE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE
      value: "${HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_DEVICE
      value: "${HEIRLOOM_QB_DATA_HARDPATH_DEVICE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_DEVICES
      value: "${HEIRLOOM_QB_DATA_HARDPATH_DEVICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_DISTRIBUTED
      value: "${HEIRLOOM_QB_DATA_HARDPATH_DISTRIBUTED:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_PRECISION
      value: "${HEIRLOOM_QB_DATA_HARDPATH_PRECISION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_STEPS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_BATCH
      value: "${HEIRLOOM_QB_DATA_HARDPATH_BATCH:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_GRAD_ACCUMULATION_STEPS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_GRAD_ACCUMULATION_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_BLOCK
      value: "${HEIRLOOM_QB_DATA_HARDPATH_BLOCK:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_D_MODEL
      value: "${HEIRLOOM_QB_DATA_HARDPATH_D_MODEL:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_HEADS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_HEADS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_FF
      value: "${HEIRLOOM_QB_DATA_HARDPATH_FF:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_LR
      value: "${HEIRLOOM_QB_DATA_HARDPATH_LR:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MIN_REDUCTION
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MIN_REDUCTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_EVAL_BATCHES
      value: "${HEIRLOOM_QB_DATA_HARDPATH_EVAL_BATCHES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_VOCAB
      value: "${HEIRLOOM_QB_DATA_HARDPATH_VOCAB:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MAX_BYTES
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MAX_BYTES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_SHARD_TOKENS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_SHARD_TOKENS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_GENERATION_TOKENS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_GENERATION_TOKENS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_LOG_EVERY
      value: "${HEIRLOOM_QB_DATA_HARDPATH_LOG_EVERY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_RESUME_LOG_EVERY
      value: "${HEIRLOOM_QB_DATA_HARDPATH_RESUME_LOG_EVERY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_DDP_INIT_TIMEOUT_SECS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_DDP_INIT_TIMEOUT_SECS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_DDP_CHECKSUM_EVERY
      value: "${HEIRLOOM_QB_DATA_HARDPATH_DDP_CHECKSUM_EVERY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_SKIP_EVAL_GENERATION
      value: "${HEIRLOOM_QB_DATA_HARDPATH_SKIP_EVAL_GENERATION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_THROUGHPUT_REPORT
      value: "${HEIRLOOM_QB_DATA_HARDPATH_THROUGHPUT_REPORT:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_N_LAYERS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_N_LAYERS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LAYER_INDICES
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LAYER_INDICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SLOTS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SLOTS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_KEY_DIM
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_KEY_DIM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_VALUE_DIM
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_VALUE_DIM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_TOP_K
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_TOP_K:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_HEADS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_HEADS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LOOKUP
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LOOKUP:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SHARED_MEMORY
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SHARED_MEMORY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_PLUS
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_PLUS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_UPDATE_POLICY
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_UPDATE_POLICY:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_MODE
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_MODE:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_ROW_MASK
      value: "${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_ROW_MASK:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_DEVICE
      value: "${HEIRLOOM_TINYSTORIES_CUDA_DEVICE:-cuda:0}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_DEVICES
      value: "${HEIRLOOM_TINYSTORIES_CUDA_DEVICES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED
      value: "${HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_NCCL_NET_PLUGIN
      value: "${HEIRLOOM_NCCL_NET_PLUGIN:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_REQUIRE_TENSOR_CORES
      value: "${HEIRLOOM_REQUIRE_TENSOR_CORES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM
      value: "${HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM
      value: "${HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM
      value: "${HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES
      value: "${HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_MODE
      value: "${HEIRLOOM_TINYSTORIES_CUDA_MODE:-reference}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_STEPS
      value: "${HEIRLOOM_TINYSTORIES_CUDA_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_BATCH
      value: "${HEIRLOOM_TINYSTORIES_CUDA_BATCH:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_GRAD_ACCUMULATION_STEPS
      value: "${HEIRLOOM_TINYSTORIES_CUDA_GRAD_ACCUMULATION_STEPS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_BLOCK
      value: "${HEIRLOOM_TINYSTORIES_CUDA_BLOCK:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_D_MODEL
      value: "${HEIRLOOM_TINYSTORIES_CUDA_D_MODEL:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_HEADS
      value: "${HEIRLOOM_TINYSTORIES_CUDA_HEADS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_FF
      value: "${HEIRLOOM_TINYSTORIES_CUDA_FF:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_LR
      value: "${HEIRLOOM_TINYSTORIES_CUDA_LR:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_MIN_REDUCTION
      value: "${HEIRLOOM_TINYSTORIES_CUDA_MIN_REDUCTION:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_EVAL_BATCHES
      value: "${HEIRLOOM_TINYSTORIES_CUDA_EVAL_BATCHES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_VOCAB
      value: "${HEIRLOOM_TINYSTORIES_CUDA_VOCAB:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_MAX_BYTES
      value: "${HEIRLOOM_TINYSTORIES_CUDA_MAX_BYTES:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_GENERATION_TOKENS
      value: "${HEIRLOOM_TINYSTORIES_CUDA_GENERATION_TOKENS:-__HEIRLOOM_UNSET__}"
    - name: HEIRLOOM_TINYSTORIES_CUDA_PRECISION
      value: "${HEIRLOOM_TINYSTORIES_CUDA_PRECISION:-f32}"
    - name: HEIRLOOM_ARTIFACT_PREFIX
      value: "${ARTIFACT_PREFIX}"
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: PIP_ROOT_USER_ACTION
      value: "ignore"
scheduling:
  timeout: 7200s
  disableRetries: true
baseOutputDirectory:
  outputUriPrefix: ${BUCKET}/vertex-outputs
EOF

python3 - "$CONFIG_YAML" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
filtered: list[str] = []
removed = 0
index = 0
while index < len(lines):
    line = lines[index]
    if (
        line.startswith("    - name: ")
        and index + 1 < len(lines)
        and lines[index + 1].startswith("      value: ")
        and lines[index + 1].split("value: ", 1)[1].strip() == '"__HEIRLOOM_UNSET__"'
    ):
        removed += 1
        index += 2
        continue
    filtered.append(line)
    index += 1

env_count = sum(1 for line in filtered if line.startswith("    - name: "))
if env_count > 100:
    raise SystemExit(
        f"Vertex custom job env var count {env_count} exceeds limit 100 after filtering {removed} unset entries"
    )
path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
print(f"Filtered Vertex env entries: kept={env_count} removed_unset={removed}")
PY

chmod 600 "$CONFIG_YAML"

print "Submitting ${DISPLAY_NAME}"
gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display-name="$DISPLAY_NAME" \
  --config="$CONFIG_YAML" \
  --format='value(name)'

rm -f "$CONFIG_YAML"

JOB_NAME="$(gcloud ai custom-jobs list \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --filter="displayName=${DISPLAY_NAME}" \
  --sort-by=~createTime \
  --limit=1 \
  --format='value(name)')"
JOB_ID="${JOB_NAME##*/}"

print "Vertex custom job id: ${JOB_ID}"
print "Source package: ${SOURCE_URI}"
print "Temporary config deleted after submission."

if [[ "$STREAM_LOGS" == "true" ]]; then
  gcloud ai custom-jobs stream-logs "$JOB_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --polling-interval=15
fi
