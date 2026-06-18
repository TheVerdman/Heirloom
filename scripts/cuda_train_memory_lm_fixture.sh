#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICE:-cuda:0}"
DEVICES="${HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES:-}"
DISTRIBUTED="${HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED:-}"
DDP_INIT_TIMEOUT_SECS="${HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_INIT_TIMEOUT_SECS:-120}"
DDP_CHECKSUM_EVERY="${HEIRLOOM_CUDA_MEMORY_FIXTURE_DDP_CHECKSUM_EVERY:-1}"
PRECISION="${HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION:-amp-bf16}"
STEPS="${HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS:-40}"
RESUME_STEPS="${HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS:-3}"
MIN_REDUCTION="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION:-0.0}"
DATA_SOURCE="${HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_SOURCE:-synthetic}"
DATA_INPUT="${HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_INPUT:-}"
DATA_MAX_BYTES="${HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_MAX_BYTES:-}"
BATCH_SIZE="${HEIRLOOM_CUDA_MEMORY_FIXTURE_BATCH_SIZE:-4}"
BLOCK_SIZE="${HEIRLOOM_CUDA_MEMORY_FIXTURE_BLOCK_SIZE:-4}"
N_LAYERS="${HEIRLOOM_CUDA_MEMORY_FIXTURE_N_LAYERS:-4}"
D_MODEL="${HEIRLOOM_CUDA_MEMORY_FIXTURE_D_MODEL:-16}"
N_HEADS="${HEIRLOOM_CUDA_MEMORY_FIXTURE_N_HEADS:-4}"
FF_HIDDEN="${HEIRLOOM_CUDA_MEMORY_FIXTURE_FF_HIDDEN:-64}"
MEMORY_LAYER_INDICES="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LAYER_INDICES:-1,3}"
MEMORY_SLOTS="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_SLOTS:-16}"
MEMORY_KEY_DIM="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_KEY_DIM:-8}"
MEMORY_VALUE_DIM="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_VALUE_DIM:-16}"
MEMORY_TOP_K="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_TOP_K:-2}"
MEMORY_HEADS="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_HEADS:-1}"
MEMORY_LOOKUP="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LOOKUP:-exact}"
SHARED_MEMORY="${HEIRLOOM_CUDA_MEMORY_FIXTURE_SHARED_MEMORY:-true}"
MEMORY_PLUS="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_PLUS:-true}"
MEMORY_UPDATE_POLICY="${HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY:-sparse-rows}"
SMFT_MODE="${HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE:-disabled}"
SMFT_ROW_MASK="${HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK:-}"
VOCAB_SIZE="${HEIRLOOM_CUDA_MEMORY_FIXTURE_VOCAB_SIZE:-280}"
LR="${HEIRLOOM_CUDA_MEMORY_FIXTURE_LR:-0.02}"
SEED="${HEIRLOOM_CUDA_MEMORY_FIXTURE_SEED:-321}"
OUT_DIR="${HEIRLOOM_CUDA_MEMORY_FIXTURE_OUT_DIR:-}"

DISTRIBUTED_MODE=0
if [[ -n "$DEVICES" || -n "$DISTRIBUTED" ]]; then
  DISTRIBUTED_MODE=1
  DISTRIBUTED="${DISTRIBUTED:-nccl}"
  if [[ -z "$DEVICES" ]]; then
    echo "HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES is required when HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED is set" >&2
    exit 1
  fi
fi

if [[ "$MEMORY_UPDATE_POLICY" != "sparse-rows" && "$DISTRIBUTED_MODE" == "1" ]]; then
  echo "warning: memory fixture DDP row-union checks are strongest with HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY=sparse-rows" >&2
fi

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d)"
  cleanup() {
    rm -rf "$OUT_DIR"
  }
  trap cleanup EXIT
else
  mkdir -p "$OUT_DIR"
fi

TEXT_PATH="$OUT_DIR/memory-data.txt"
case "$DATA_SOURCE" in
  synthetic)
    python3 - "$TEXT_PATH" <<'PY'
from pathlib import Path
import sys

text = (
    "red key opens red room. blue key opens blue room. "
    "green key opens green room. yellow key opens yellow room. "
    "red room has warm soup. blue room has cold rain. "
    "green room has small trees. yellow room has bright sun. "
) * 12
Path(sys.argv[1]).write_text(text, encoding="utf-8")
PY
    ;;
  tinystories-valid)
    run cargo run --bin heirloom -- data tinystories-valid --out "$TEXT_PATH"
    ;;
  file)
    if [[ -z "$DATA_INPUT" ]]; then
      echo "HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_INPUT is required when DATA_SOURCE=file" >&2
      exit 1
    fi
    python3 - "$DATA_INPUT" "$TEXT_PATH" "$DATA_MAX_BYTES" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
max_bytes = int(sys.argv[3]) if sys.argv[3] else 0
data = source.read_bytes()
if max_bytes > 0:
    data = data[:max_bytes]
destination.write_text(data.decode("utf-8", errors="ignore"), encoding="utf-8")
PY
    ;;
  *)
    echo "HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_SOURCE must be synthetic, tinystories-valid, or file; got $DATA_SOURCE" >&2
    exit 2
    ;;
esac

if [[ "$DATA_MAX_BYTES" != "" && "$DATA_MAX_BYTES" != "0" && "$DATA_SOURCE" != "file" ]]; then
  python3 - "$TEXT_PATH" "$DATA_MAX_BYTES" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
max_bytes = int(sys.argv[2])
data = path.read_bytes()[:max_bytes]
path.write_text(data.decode("utf-8", errors="ignore"), encoding="utf-8")
PY
fi

run cargo run --bin heirloom -- tokenizer train \
  --input "$TEXT_PATH" \
  --out "$OUT_DIR/tokenizer.json" \
  --vocab-size "$VOCAB_SIZE"

run cargo run --bin heirloom -- data prepare \
  --input "$TEXT_PATH" \
  --tokenizer "$OUT_DIR/tokenizer.json" \
  --out-dir "$OUT_DIR/prepared" \
  --valid-fraction 0.2

TARGET_ARGS=()
if [[ "$DISTRIBUTED_MODE" == "1" ]]; then
  TARGET_ARGS=(
    --devices "$DEVICES"
    --distributed "$DISTRIBUTED"
    --ddp-init-timeout-secs "$DDP_INIT_TIMEOUT_SECS"
    --ddp-checksum-every "$DDP_CHECKSUM_EVERY"
  )
else
  TARGET_ARGS=(--device "$DEVICE")
fi

SMFT_ARGS=()
if [[ "$SMFT_ROW_MASK" == "auto" ]]; then
  SMFT_ROW_MASK="$OUT_DIR/smft-row-mask.json"
  python3 - "$SMFT_ROW_MASK" "$MEMORY_SLOTS" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
slots = int(sys.argv[2])
trainable = list(range(max(1, slots // 2)))
payload = {
    "memory_slots": slots,
    "trainable_rows": trainable,
    "frozen_rows": slots - len(trainable),
    "trainable_fraction": len(trainable) / slots,
    "scores": [],
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
fi
if [[ "$SMFT_MODE" != "disabled" ]]; then
  SMFT_ARGS+=(--smft-mode "$SMFT_MODE")
fi
if [[ -n "$SMFT_ROW_MASK" ]]; then
  SMFT_ARGS+=(--smft-row-mask "$SMFT_ROW_MASK")
fi

MODEL_ARGS=(
  --dataset-manifest "$OUT_DIR/prepared/manifest.json"
  --batch-size "$BATCH_SIZE"
  --block-size "$BLOCK_SIZE"
  --n-layers "$N_LAYERS"
  --d-model "$D_MODEL"
  --n-heads "$N_HEADS"
  --ff-hidden "$FF_HIDDEN"
  --memory-layer-indices "$MEMORY_LAYER_INDICES"
  --memory-slots "$MEMORY_SLOTS"
  --memory-key-dim "$MEMORY_KEY_DIM"
  --memory-value-dim "$MEMORY_VALUE_DIM"
  --memory-top-k "$MEMORY_TOP_K"
  --memory-heads "$MEMORY_HEADS"
  --memory-lookup "$MEMORY_LOOKUP"
  --shared-memory "$SHARED_MEMORY"
  --memory-plus "$MEMORY_PLUS"
  --memory-update-policy "$MEMORY_UPDATE_POLICY"
  "${SMFT_ARGS[@]}"
  --lr "$LR"
  --weight-decay 0.0
  --clip-norm 1.0
  --seed "$SEED"
  --precision "$PRECISION"
)

run cargo run --bin heirloom -- train-memory-lm \
  "${MODEL_ARGS[@]}" \
  --checkpoint "$OUT_DIR/checkpoint" \
  --steps "$STEPS" \
  "${TARGET_ARGS[@]}" \
  --log-every 10 \
  --report "$OUT_DIR/train-report.json"

if [[ "$DISTRIBUTED_MODE" == "1" && -d "$OUT_DIR/ddp-memory-ranks" ]]; then
  rm -rf "$OUT_DIR/train-ddp-memory-ranks"
  cp -a "$OUT_DIR/ddp-memory-ranks" "$OUT_DIR/train-ddp-memory-ranks"
fi

python3 - "$OUT_DIR/train-report.json" "$MIN_REDUCTION" "$STEPS" "$DEVICE" "$PRECISION" "$DEVICES" "$DISTRIBUTED" "$BATCH_SIZE" "$DDP_CHECKSUM_EVERY" "$MEMORY_UPDATE_POLICY" "$SMFT_MODE" "$SMFT_ROW_MASK" <<'PY'
import json
import math
import sys

(
    report_path,
    min_reduction,
    expected_steps,
    expected_device,
    expected_precision,
    expected_devices,
    expected_distributed,
    expected_batch_size,
    ddp_checksum_every,
    memory_update_policy,
    smft_mode,
    smft_row_mask,
) = sys.argv[1:13]
min_reduction = float(min_reduction)
expected_steps = int(expected_steps)
expected_batch_size = int(expected_batch_size)
ddp_checksum_every = int(ddp_checksum_every)
distributed = bool(expected_devices)
policy_names = {memory_update_policy, memory_update_policy.replace("-", ""), "SparseRows", "sparse_rows"}

with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)

if report.get("command") != "train-memory-lm":
    raise SystemExit(f"unexpected command in memory fixture report: {report.get('command')}")
if report.get("model_family") != "memory_transformer":
    raise SystemExit(f"unexpected model family: {report.get('model_family')}")
if report.get("start_step") != 0 or report.get("final_step") != expected_steps:
    raise SystemExit(f"unexpected train step range: {report.get('start_step')}->{report.get('final_step')}, expected 0->{expected_steps}")
if report.get("precision") != expected_precision:
    raise SystemExit(f"unexpected precision: {report.get('precision')} expected {expected_precision}")

initial = float(report["initial_loss"])
final = float(report["final_loss"])
reduction = float(report["loss_reduction"])
if not math.isfinite(initial) or not math.isfinite(final):
    raise SystemExit(f"non-finite memory fixture losses: initial={initial} final={final}")
if reduction < min_reduction:
    raise SystemExit(f"memory fixture loss reduction {reduction:.6f} below required {min_reduction:.6f}; initial={initial:.6f} final={final:.6f}")

if distributed:
    devices = [part.strip() for part in expected_devices.split(",") if part.strip()]
    if report.get("distributed") != expected_distributed:
        raise SystemExit(f"unexpected distributed mode: {report.get('distributed')} expected {expected_distributed}")
    if report.get("devices") != devices:
        raise SystemExit(f"unexpected devices: {report.get('devices')} expected {devices}")
    if int(report.get("world_size") or 0) != len(devices):
        raise SystemExit(f"unexpected world_size: {report.get('world_size')} expected {len(devices)}")
    if int(report.get("per_rank_batch_size") or 0) != expected_batch_size:
        raise SystemExit(f"unexpected per-rank batch size: {report.get('per_rank_batch_size')}")
    if int(report.get("global_batch_size") or 0) != expected_batch_size * len(devices):
        raise SystemExit(f"unexpected global batch size: {report.get('global_batch_size')}")
    if str(report.get("memory_update_policy")) not in policy_names:
        raise SystemExit(f"unexpected memory update policy: {report.get('memory_update_policy')} accepted={sorted(policy_names)}")
    if smft_row_mask:
        if str(report.get("smft_mode")) not in {smft_mode, "MaskedMemoryRows", "masked_memory_rows"}:
            raise SystemExit(f"unexpected SMFT mode in DDP report: {report.get('smft_mode')}")
        if not report.get("smft_row_mask"):
            raise SystemExit("expected aggregate DDP report to record smft_row_mask")
        if int(report.get("smft_row_mask_trainable_rows_rank0") or 0) <= 0:
            raise SystemExit("expected rank0 SMFT trainable row count in aggregate DDP report")
    if int(report.get("all_reduce_calls") or 0) <= 0 or int(report.get("all_reduce_bytes") or 0) <= 0:
        raise SystemExit("missing memory DDP gradient all-reduce stats")
    if memory_update_policy == "sparse-rows":
        if not report.get("compressed_sparse_gradient_transport"):
            raise SystemExit("expected sparse-row DDP report to enable compressed_sparse_gradient_transport")
        for field in [
            "row_union_all_reduce_calls",
            "row_union_all_reduce_bytes",
            "row_union_candidate_rows",
            "compact_gradient_all_reduce_calls",
            "compact_gradient_all_reduce_bytes",
        ]:
            if int(report.get(field) or 0) <= 0:
                raise SystemExit(f"missing sparse-row DDP compact transport evidence: {field}={report.get(field)}")
    tolerance = float(report.get("parameter_checksum_tolerance") or 0.0)
    for field in [
        "memory_table_checksum_sum_max_error",
        "memory_table_checksum_sumsq_max_error",
        "step_memory_checksum_sum_max_error",
        "step_memory_checksum_sumsq_max_error",
    ]:
        value = float(report.get(field) or 0.0)
        if value > tolerance:
            raise SystemExit(f"{field}={value:.6e} exceeds tolerance={tolerance:.6e}")
    if ddp_checksum_every > 0 and not report.get("step_checksum_drift"):
        raise SystemExit("expected per-step memory checksum drift entries in DDP report")
    ranks = report.get("ranks") or []
    if len(ranks) != len(devices):
        raise SystemExit(f"expected {len(devices)} rank reports, got {len(ranks)}")
    for rank in ranks:
        if int(rank.get("memory_gradient_parameter_count") or 0) <= 0:
            raise SystemExit(f"rank missing memory gradients: {rank}")
        if memory_update_policy == "sparse-rows" and int(rank.get("row_union_all_reduce_calls") or 0) <= 0:
            raise SystemExit(f"rank missing row-union calls: {rank}")
        if memory_update_policy == "sparse-rows":
            for field in ["compact_gradient_all_reduce_calls", "compact_gradient_all_reduce_bytes"]:
                if int(rank.get(field) or 0) <= 0:
                    raise SystemExit(f"rank missing compact sparse-gradient transport evidence {field}: {rank}")
            rank_kernels = (rank.get("cuda_memory_kernels") or {}).get("counters") or {}
            for field in ["bool_mask_to_indices_calls", "gather_selected_rows_calls", "sparse_adamw_compact_rows_calls"]:
                if int(rank_kernels.get(field) or 0) <= 0:
                    raise SystemExit(f"rank missing compact sparse optimizer evidence {field}: {rank_kernels}")
else:
    if report.get("device") != expected_device:
        raise SystemExit(f"unexpected device: {report.get('device')} expected {expected_device}")
    optimizer = report.get("memory_optimizer") or {}
    if smft_row_mask:
        row_mask = optimizer.get("smft_row_mask") or {}
        if not row_mask.get("source"):
            raise SystemExit(f"expected SMFT row-mask evidence in single-rank optimizer report: {optimizer}")
        if int(row_mask.get("trainable_rows") or 0) <= 0:
            raise SystemExit(f"expected trainable SMFT rows in optimizer report: {optimizer}")
        if int(optimizer.get("row_mask_attached_sparse_update_count") or 0) <= 0:
            raise SystemExit(f"expected SMFT row masks attached to sparse updates: {optimizer}")
    if memory_update_policy == "sparse-rows":
        if optimizer.get("applied_path") != "cuda_sparse_rows_selected_memory_tables":
            raise SystemExit(f"unexpected sparse optimizer path: {optimizer}")
        if int(optimizer.get("sparse_update_parameter_count") or 0) <= 0:
            raise SystemExit(f"missing sparse update descriptors: {optimizer}")
        if not optimizer.get("sparse_optimizer_gathers_compact_gradient_rows"):
            raise SystemExit(f"missing compact sparse optimizer report evidence: {optimizer}")
    kernels = (report.get("cuda_memory_kernels") or {}).get("counters") or {}
    for field in ["topk_calls", "weighted_value_forward_calls", "selected_rows"]:
        if int(kernels.get(field) or 0) <= 0:
            raise SystemExit(f"missing memory kernel evidence {field}: {kernels}")
    if memory_update_policy == "sparse-rows":
        for field in ["gather_selected_rows_calls", "sparse_adamw_compact_rows_calls"]:
            if int(kernels.get(field) or 0) <= 0:
                raise SystemExit(f"missing compact sparse optimizer kernel evidence {field}: {kernels}")

print(
    "cuda_train_memory_lm_fixture train_check "
    f"initial={initial:.6f} final={final:.6f} reduction={reduction:.6f} "
    f"distributed={distributed} policy={memory_update_policy}"
)
PY

if [[ "$RESUME_STEPS" != "0" ]]; then
  run cargo run --bin heirloom -- train-memory-lm \
    "${MODEL_ARGS[@]}" \
    --checkpoint "$OUT_DIR/checkpoint" \
    --steps "$RESUME_STEPS" \
    "${TARGET_ARGS[@]}" \
    --resume \
    --log-every 1 \
    --report "$OUT_DIR/resume-report.json"

  if [[ "$DISTRIBUTED_MODE" == "1" && -d "$OUT_DIR/ddp-memory-ranks" ]]; then
    rm -rf "$OUT_DIR/resume-ddp-memory-ranks"
    cp -a "$OUT_DIR/ddp-memory-ranks" "$OUT_DIR/resume-ddp-memory-ranks"
  fi

  python3 - "$OUT_DIR/resume-report.json" "$STEPS" "$RESUME_STEPS" "$PRECISION" "$DEVICES" "$DISTRIBUTED" "$BATCH_SIZE" "$DDP_CHECKSUM_EVERY" "$MEMORY_UPDATE_POLICY" "$SMFT_MODE" "$SMFT_ROW_MASK" <<'PY'
import json
import math
import sys

(
    report_path,
    start_step,
    resume_steps,
    expected_precision,
    expected_devices,
    expected_distributed,
    expected_batch_size,
    ddp_checksum_every,
    memory_update_policy,
    smft_mode,
    smft_row_mask,
) = sys.argv[1:12]
start_step = int(start_step)
resume_steps = int(resume_steps)
expected_batch_size = int(expected_batch_size)
ddp_checksum_every = int(ddp_checksum_every)
distributed = bool(expected_devices)

with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)

if report.get("command") != "train-memory-lm":
    raise SystemExit(f"unexpected command in memory resume report: {report.get('command')}")
if report.get("model_family") != "memory_transformer":
    raise SystemExit(f"unexpected model family in memory resume report: {report.get('model_family')}")
if report.get("start_step") != start_step or report.get("final_step") != start_step + resume_steps:
    raise SystemExit(f"unexpected resume step range: {report.get('start_step')}->{report.get('final_step')}, expected {start_step}->{start_step + resume_steps}")
if report.get("precision") != expected_precision:
    raise SystemExit(f"unexpected resume precision: {report.get('precision')} expected {expected_precision}")
for field in ["initial_loss", "final_loss"]:
    value = float(report[field])
    if not math.isfinite(value):
        raise SystemExit(f"non-finite {field} in memory resume report: {value}")

if distributed:
    devices = [part.strip() for part in expected_devices.split(",") if part.strip()]
    if report.get("distributed") != expected_distributed:
        raise SystemExit(f"unexpected resume distributed mode: {report.get('distributed')} expected {expected_distributed}")
    if int(report.get("world_size") or 0) != len(devices):
        raise SystemExit(f"unexpected resume world_size: {report.get('world_size')} expected {len(devices)}")
    if int(report.get("per_rank_batch_size") or 0) != expected_batch_size:
        raise SystemExit(f"unexpected resume batch size: {report.get('per_rank_batch_size')}")
    if int(report.get("all_reduce_calls") or 0) <= 0 or int(report.get("all_reduce_bytes") or 0) <= 0:
        raise SystemExit("missing memory resume DDP gradient all-reduce stats")
    if smft_row_mask and not report.get("smft_row_mask"):
        raise SystemExit("expected resume aggregate DDP report to record smft_row_mask")
    if memory_update_policy == "sparse-rows":
        if not report.get("compressed_sparse_gradient_transport"):
            raise SystemExit("expected sparse-row DDP resume report to enable compressed_sparse_gradient_transport")
        for field in [
            "row_union_all_reduce_calls",
            "row_union_all_reduce_bytes",
            "row_union_candidate_rows",
            "compact_gradient_all_reduce_calls",
            "compact_gradient_all_reduce_bytes",
        ]:
            if int(report.get(field) or 0) <= 0:
                raise SystemExit(f"missing resume sparse-row DDP compact transport evidence: {field}={report.get(field)}")
    tolerance = float(report.get("parameter_checksum_tolerance") or 0.0)
    for field in [
        "memory_table_checksum_sum_max_error",
        "memory_table_checksum_sumsq_max_error",
        "step_memory_checksum_sum_max_error",
        "step_memory_checksum_sumsq_max_error",
    ]:
        value = float(report.get(field) or 0.0)
        if value > tolerance:
            raise SystemExit(f"resume {field}={value:.6e} exceeds tolerance={tolerance:.6e}")
    if ddp_checksum_every > 0 and not report.get("step_checksum_drift"):
        raise SystemExit("expected resume per-step memory checksum drift entries in DDP report")

print(
    "cuda_train_memory_lm_fixture resume_check "
    f"start_step={start_step} final_step={start_step + resume_steps} "
    f"distributed={distributed} policy={memory_update_policy}"
)
PY
fi

python3 - "$OUT_DIR/summary.json" "$OUT_DIR" "$DEVICE" "$DEVICES" "$DISTRIBUTED" "$PRECISION" "$MEMORY_UPDATE_POLICY" "$SMFT_MODE" "$SMFT_ROW_MASK" "$STEPS" "$RESUME_STEPS" "$DATA_SOURCE" "$TEXT_PATH" <<'PY'
import json
from pathlib import Path
import sys

(
    summary_path,
    out_dir,
    device,
    devices,
    distributed,
    precision,
    memory_update_policy,
    smft_mode,
    smft_row_mask,
    steps,
    resume_steps,
    data_source,
    text_path,
) = sys.argv[1:14]

out = Path(out_dir)

def load_report(name):
    path = out / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

def kernel_counters(report):
    if not report:
        return None
    kernels = report.get("cuda_memory_kernels_rank0") or report.get("cuda_memory_kernels") or {}
    return kernels.get("counters") or kernels

def report_summary(report):
    if not report:
        return None
    keys = [
        "status",
        "distributed",
        "world_size",
        "per_rank_batch_size",
        "global_batch_size",
        "start_step",
        "final_step",
        "initial_loss",
        "final_loss",
        "loss_reduction",
        "all_reduce_calls",
        "all_reduce_bytes",
        "row_union_all_reduce_calls",
        "row_union_all_reduce_bytes",
        "row_union_candidate_rows",
        "compact_gradient_all_reduce_calls",
        "compact_gradient_all_reduce_bytes",
        "compressed_sparse_gradient_transport",
        "memory_gradient_parameter_counts",
        "memory_table_checksum_sum_max_error",
        "memory_table_checksum_sumsq_max_error",
        "step_memory_checksum_sum_max_error",
        "step_memory_checksum_sumsq_max_error",
        "parameter_checksum_tolerance",
        "smft_row_mask",
        "smft_row_mask_trainable_rows_rank0",
        "smft_row_mask_frozen_rows_rank0",
    ]
    summary = {key: report.get(key) for key in keys if key in report}
    ranks = report.get("ranks") or []
    if ranks:
        summary["rank_count"] = len(ranks)
        summary["rank_compact_gradient_all_reduce_calls"] = [
            rank.get("compact_gradient_all_reduce_calls") for rank in ranks
        ]
        summary["rank_compact_gradient_all_reduce_bytes"] = [
            rank.get("compact_gradient_all_reduce_bytes") for rank in ranks
        ]
    counters = kernel_counters(report)
    if counters:
        summary["cuda_memory_kernel_counters_rank0"] = {
            key: counters.get(key)
            for key in [
                "topk_calls",
                "product_key_calls",
                "weighted_value_forward_calls",
                "weighted_value_backward_calls",
                "selected_key_backward_calls",
                "scatter_add_rows_calls",
                "gather_selected_rows_calls",
                "bool_mask_to_indices_calls",
                "sparse_adamw_compact_rows_calls",
                "selected_rows",
            ]
            if key in counters
        }
    return summary

train = load_report("train-report.json")
resume = load_report("resume-report.json")
manifest = load_report("prepared/manifest.json") or {}
summary = {
    "fixture": "cuda_train_memory_lm_fixture",
    "device": device,
    "devices": devices,
    "distributed": distributed,
    "precision": precision,
    "data_source": data_source,
    "data_path": text_path,
    "source_bytes": manifest.get("source_bytes"),
    "train_tokens": manifest.get("train_tokens"),
    "valid_tokens": manifest.get("valid_tokens"),
    "memory_update_policy": memory_update_policy,
    "smft_mode": smft_mode,
    "smft_row_mask": smft_row_mask,
    "steps": int(steps),
    "resume_steps": int(resume_steps),
    "report": str(out / "train-report.json"),
    "resume_report": str(out / "resume-report.json"),
    "train": report_summary(train),
    "resume": report_summary(resume),
}

Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

echo "cuda_train_memory_lm_fixture complete out_dir=$OUT_DIR"
