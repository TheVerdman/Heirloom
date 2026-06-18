#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

workdir="${1:-runs/tinystories-valid}"
steps="${HEIRLOOM_TINYSTORIES_STEPS:-500}"
batch_size="${HEIRLOOM_TINYSTORIES_BATCH:-4}"
block_size="${HEIRLOOM_TINYSTORIES_BLOCK:-64}"
d_model="${HEIRLOOM_TINYSTORIES_D_MODEL:-64}"
n_heads="${HEIRLOOM_TINYSTORIES_HEADS:-4}"
ff_hidden="${HEIRLOOM_TINYSTORIES_FF:-256}"
lr="${HEIRLOOM_TINYSTORIES_LR:-0.001}"
min_reduction="${HEIRLOOM_TINYSTORIES_MIN_REDUCTION:-0.20}"
eval_batches="${HEIRLOOM_TINYSTORIES_EVAL_BATCHES:-200}"
device="${HEIRLOOM_TINYSTORIES_DEVICE:-cpu}"
devices="${HEIRLOOM_TINYSTORIES_DEVICES:-}"
distributed="${HEIRLOOM_TINYSTORIES_DISTRIBUTED:-}"
precision="${HEIRLOOM_TINYSTORIES_PRECISION:-f32}"
vocab_size="${HEIRLOOM_TINYSTORIES_VOCAB:-1024}"
max_bytes="${HEIRLOOM_TINYSTORIES_MAX_BYTES:-}"
run_label="${HEIRLOOM_TINYSTORIES_RUN_LABEL:-manual}"
generation_tokens="${HEIRLOOM_TINYSTORIES_GENERATION_TOKENS:-80}"
log_every="${HEIRLOOM_TINYSTORIES_LOG_EVERY:-25}"
resume_steps="${HEIRLOOM_TINYSTORIES_RESUME_STEPS:-0}"
ddp_init_timeout_secs="${HEIRLOOM_TINYSTORIES_DDP_INIT_TIMEOUT_SECS:-120}"
ddp_checksum_every="${HEIRLOOM_TINYSTORIES_DDP_CHECKSUM_EVERY:-1}"

mkdir -p "$workdir"

data_path="$workdir/TinyStories-valid.txt"
tokenizer_path="$workdir/tokenizer.json"
prepared_dir="$workdir/prepared"
manifest_path="$prepared_dir/manifest.json"
checkpoint_path="$workdir/checkpoint"
resume_checkpoint_path="$workdir/checkpoint-resume"
report_path="$workdir/report.json"
resume_report_path="$workdir/resume-report.json"
eval_report_path="$workdir/eval.json"
generation_report_path="$workdir/generation.json"
generation_path="$workdir/generation.txt"
summary_path="$workdir/summary.json"
timings_path="$workdir/timings.json"
stage_names=()
stage_millis=()

cleanup_distributed_ranks() {
  if [[ "$distributed" == "nccl" ]]; then
    pkill -f "heirloom train-lm-rank" 2>/dev/null || true
  fi
}
trap cleanup_distributed_ranks EXIT

now_ms() {
  python3 -c 'import time; print(time.time_ns() // 1_000_000)'
}

run_stage() {
  local name="$1"
  shift
  local start
  local end
  local status
  start="$(now_ms)"
  set +e
  "$@"
  status="$?"
  set -e
  end="$(now_ms)"
  stage_names+=("$name")
  stage_millis+=("$((end - start))")
  if [[ "$status" -ne 0 ]]; then
    cleanup_distributed_ranks
    return "$status"
  fi
}

if [[ ! -f "$data_path" ]]; then
  run_stage download cargo run --bin heirloom -- data tinystories-valid --out "$data_path"
fi

tokenizer_args=(
  cargo run --bin heirloom -- tokenizer train
  --input "$data_path"
  --out "$tokenizer_path"
  --vocab-size "$vocab_size"
)
if [[ -n "$max_bytes" ]]; then
  tokenizer_args+=(--max-bytes "$max_bytes")
fi
run_stage tokenizer "${tokenizer_args[@]}"

prepare_args=(
  cargo run --bin heirloom -- data prepare
  --input "$data_path" \
  --tokenizer "$tokenizer_path" \
  --out-dir "$prepared_dir" \
  --valid-fraction 0.05
)
if [[ -n "$max_bytes" ]]; then
  prepare_args+=(--max-bytes "$max_bytes")
fi
run_stage prepare "${prepare_args[@]}"

train_args=(
  cargo run --bin heirloom -- train-lm
  --dataset-manifest "$manifest_path" \
  --checkpoint "$checkpoint_path" \
  --steps "$steps" \
  --batch-size "$batch_size" \
  --block-size "$block_size" \
  --d-model "$d_model" \
  --n-heads "$n_heads" \
  --ff-hidden "$ff_hidden" \
  --lr "$lr" \
  --weight-decay 0.01 \
  --clip-norm 1.0 \
  --precision "$precision" \
  --log-every "$log_every" \
  --report "$report_path"
)
if [[ -n "$distributed" ]]; then
  train_args+=(
    --devices "$devices"
    --distributed "$distributed"
    --ddp-init-timeout-secs "$ddp_init_timeout_secs"
    --ddp-checksum-every "$ddp_checksum_every"
  )
else
  train_args+=(--device "$device")
fi
run_stage train "${train_args[@]}"

if [[ "$distributed" == "nccl" && -d "$workdir/ddp-ranks" ]]; then
  rm -rf "$workdir/train-ddp-ranks"
  cp -a "$workdir/ddp-ranks" "$workdir/train-ddp-ranks"
fi
if [[ "$distributed" == "nccl" && -f "$workdir/launcher-report.json" ]]; then
  cp "$workdir/launcher-report.json" "$workdir/train-launcher-report.json"
fi

if [[ "$resume_steps" -gt 0 ]]; then
  rm -rf "$resume_checkpoint_path"
  cp -a "$checkpoint_path" "$resume_checkpoint_path"
  resume_args=(
    cargo run --bin heirloom -- train-lm
    --dataset-manifest "$manifest_path" \
    --checkpoint "$resume_checkpoint_path" \
    --steps "$resume_steps" \
    --batch-size "$batch_size" \
    --resume \
    --precision "$precision" \
    --log-every 1 \
    --report "$resume_report_path"
  )
  if [[ -n "$distributed" ]]; then
    resume_args+=(
      --devices "$devices"
      --distributed "$distributed"
      --ddp-init-timeout-secs "$ddp_init_timeout_secs"
      --ddp-checksum-every "$ddp_checksum_every"
    )
  else
    resume_args+=(--device "$device")
  fi
  run_stage resume "${resume_args[@]}"
  if [[ "$distributed" == "nccl" && -d "$workdir/ddp-ranks" ]]; then
    rm -rf "$workdir/resume-ddp-ranks"
    cp -a "$workdir/ddp-ranks" "$workdir/resume-ddp-ranks"
  fi
  if [[ "$distributed" == "nccl" && -f "$workdir/launcher-report.json" ]]; then
    cp "$workdir/launcher-report.json" "$workdir/resume-launcher-report.json"
  fi
  python3 - "$report_path" "$resume_report_path" "$resume_steps" "$distributed" "$devices" "$batch_size" "$precision" "${HEIRLOOM_REQUIRE_TENSOR_CORES:-0}" <<'PY'
import json
import math
import sys

(
    train_path,
    resume_path,
    resume_steps,
    distributed,
    devices,
    batch_size,
    precision,
    require_tensor_cores,
) = sys.argv[1:9]
train = json.load(open(train_path, encoding="utf-8"))
resume = json.load(open(resume_path, encoding="utf-8"))
resume_steps = int(resume_steps)
expected_start = int(train["final_step"])
expected_final = expected_start + resume_steps
if int(resume["start_step"]) != expected_start or int(resume["final_step"]) != expected_final:
    raise SystemExit(
        f"TinyStories resume gate failed: expected {expected_start}->{expected_final}, "
        f"got {resume['start_step']}->{resume['final_step']}"
    )
if resume.get("precision") != precision:
    raise SystemExit(
        f"TinyStories resume gate failed: precision {resume.get('precision')} != {precision}"
    )
if not (math.isfinite(float(resume["initial_loss"])) and math.isfinite(float(resume["final_loss"]))):
    raise SystemExit(f"TinyStories resume gate failed: non-finite resume losses {resume}")
if distributed:
    expected_devices = [item for item in devices.split(",") if item]
    if resume.get("distributed") != distributed:
        raise SystemExit(
            f"TinyStories resume gate failed: distributed={resume.get('distributed')} expected {distributed}"
        )
    if resume.get("devices") != expected_devices:
        raise SystemExit(
            f"TinyStories resume gate failed: devices={resume.get('devices')} expected {expected_devices}"
        )
    if int(resume.get("world_size") or 0) != len(expected_devices):
        raise SystemExit(
            f"TinyStories resume gate failed: world_size={resume.get('world_size')}"
        )
    if int(resume.get("per_rank_batch_size") or 0) != int(batch_size):
        raise SystemExit(
            f"TinyStories resume gate failed: per_rank_batch_size={resume.get('per_rank_batch_size')}"
        )
    if int(resume.get("all_reduce_calls") or 0) <= 0 or int(resume.get("all_reduce_bytes") or 0) <= 0:
        raise SystemExit(f"TinyStories resume gate failed: missing all-reduce stats {resume}")
    tolerance = float(resume.get("parameter_checksum_tolerance") or 0.0)
    for field in [
        "parameter_checksum_sum_max_error",
        "parameter_checksum_sumsq_max_error",
        "step_checksum_sum_max_error",
        "step_checksum_sumsq_max_error",
    ]:
        value = float(resume.get(field) or 0.0)
        if value > tolerance:
            raise SystemExit(
                f"TinyStories resume gate failed: {field}={value:.6e} > tolerance={tolerance:.6e}"
            )
require = require_tensor_cores in {"1", "true", "TRUE", "yes", "YES"}
if require:
    tensor_core = resume.get("tensor_core") or {}
    coverage = resume.get("tensor_core_coverage") or {}
    linear_totals = coverage.get("linear_totals") or {}
    if int(tensor_core.get("bf16_tensor_core_matmul_forward_calls") or 0) <= 0:
        raise SystemExit(f"TinyStories resume gate failed: no Tensor Core forward calls {tensor_core}")
    if int(tensor_core.get("bf16_tensor_core_matmul_backward_calls") or 0) <= 0:
        raise SystemExit(f"TinyStories resume gate failed: no Tensor Core backward calls {tensor_core}")
    if int(tensor_core.get("bf16_scalar_matmul_fallback_calls") or 0) != 0:
        raise SystemExit(f"TinyStories resume gate failed: scalar fallbacks {tensor_core}")
    if int(linear_totals.get("tensor_core_calls") or 0) <= 0:
        raise SystemExit(f"TinyStories resume gate failed: missing Linear Tensor Core coverage {coverage}")
    if int(linear_totals.get("fallback_calls") or 0) != 0:
        raise SystemExit(f"TinyStories resume gate failed: Linear fallbacks {coverage}")
print(
    "TinyStories resume gate passed: "
    f"start_step={resume['start_step']} final_step={resume['final_step']} "
    f"all_reduce_calls={resume.get('all_reduce_calls')} "
    f"loss={resume['initial_loss']:.6f}->{resume['final_loss']:.6f}"
)
PY
fi

run_stage eval cargo run --bin heirloom -- eval-lm \
  --checkpoint "$checkpoint_path" \
  --dataset-manifest "$manifest_path" \
  --split valid \
  --device "$device" \
  --precision "$precision" \
  --batch-size "$batch_size" \
  --max-batches "$eval_batches" \
  --report "$eval_report_path"

generate_start="$(now_ms)"
cargo run --bin heirloom -- generate \
  --checkpoint "$checkpoint_path" \
  --prompt "Once upon a time" \
  --device "$device" \
  --precision "$precision" \
  --max-new-tokens "$generation_tokens" \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --frequency-penalty 0.02 \
  --presence-penalty 0.02 \
  --seed 2026 \
  --report "$generation_report_path" | tee "$generation_path"
generate_end="$(now_ms)"
stage_names+=(generate)
stage_millis+=("$((generate_end - generate_start))")

python3 - "$report_path" "$min_reduction" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
minimum = float(sys.argv[2])
actual = float(report["loss_reduction"])
if actual < minimum:
    raise SystemExit(
        f"TinyStories gate failed: loss reduction {actual:.3f} < required {minimum:.3f}"
    )
print(f"TinyStories gate passed: loss reduction {actual:.3f}")
PY

python3 - "$report_path" "${HEIRLOOM_REQUIRE_TENSOR_CORES:-0}" "${HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES:-0}" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
require = sys.argv[2] in {"1", "true", "TRUE", "yes", "YES"}
require_attention = sys.argv[3] in {"1", "true", "TRUE", "yes", "YES"}
if not require and not require_attention:
    raise SystemExit(0)

tensor_core = report.get("tensor_core") or {}
coverage = report.get("tensor_core_coverage") or {}
linear_totals = coverage.get("linear_totals") or {}
matmul_calls = int(tensor_core.get("bf16_tensor_core_matmul_calls") or 0)
forward_calls = int(tensor_core.get("bf16_tensor_core_matmul_forward_calls") or 0)
backward_calls = int(tensor_core.get("bf16_tensor_core_matmul_backward_calls") or 0)
fallback_calls = int(tensor_core.get("bf16_scalar_matmul_fallback_calls") or 0)
attention_forward_calls = int(tensor_core.get("bf16_tensor_core_attention_forward_calls") or 0)
attention_qk_calls = int(tensor_core.get("bf16_tensor_core_attention_qk_matmul_calls") or 0)
attention_av_calls = int(tensor_core.get("bf16_tensor_core_attention_av_matmul_calls") or 0)
attention_backward_calls = int(tensor_core.get("bf16_tensor_core_attention_backward_calls") or 0)
attention_score_grad_calls = int(tensor_core.get("bf16_tensor_core_attention_score_grad_matmul_calls") or 0)
attention_dq_calls = int(tensor_core.get("bf16_tensor_core_attention_dq_matmul_calls") or 0)
attention_dk_calls = int(tensor_core.get("bf16_tensor_core_attention_dk_matmul_calls") or 0)
attention_dv_calls = int(tensor_core.get("bf16_tensor_core_attention_dv_matmul_calls") or 0)
linear_tensor_core_calls = int(linear_totals.get("tensor_core_calls") or 0)
linear_fallback_calls = int(linear_totals.get("fallback_calls") or 0)
if require and matmul_calls <= 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_TENSOR_CORES=1 but train report has no Tensor Core matmul calls; "
        f"tensor_core={tensor_core} report_keys={sorted(report.keys())}"
    )
if require and forward_calls <= 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_TENSOR_CORES=1 but train report has no Tensor Core forward matmul calls; "
        f"tensor_core={tensor_core}"
    )
if require and backward_calls <= 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_TENSOR_CORES=1 but train report has no Tensor Core backward matmul calls; "
        f"tensor_core={tensor_core}"
    )
if require and fallback_calls != 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_TENSOR_CORES=1 but scalar Tensor Core fallback calls were recorded; "
        f"tensor_core={tensor_core}"
    )
if require and linear_tensor_core_calls <= 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_TENSOR_CORES=1 but module coverage has no Tensor Core Linear calls; "
        f"tensor_core_coverage={coverage}"
    )
if require and linear_fallback_calls != 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_TENSOR_CORES=1 but module coverage recorded Linear fallbacks; "
        f"tensor_core_coverage={coverage}"
    )
if require_attention and attention_forward_calls <= 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 but train report has no Tensor Core attention forward calls; "
        f"tensor_core={tensor_core}"
    )
if require_attention and (attention_qk_calls <= 0 or attention_av_calls <= 0):
    raise SystemExit(
        "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 but QK/AV Tensor Core attention calls are missing; "
        f"qk={attention_qk_calls} av={attention_av_calls} tensor_core={tensor_core}"
    )
if require_attention and attention_backward_calls <= 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 but train report has no Tensor Core attention backward calls; "
        f"tensor_core={tensor_core}"
    )
if require_attention and min(
    attention_score_grad_calls,
    attention_dq_calls,
    attention_dk_calls,
    attention_dv_calls,
) <= 0:
    raise SystemExit(
        "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 but score/dQ/dK/dV Tensor Core attention backward calls are missing; "
        f"score_grad={attention_score_grad_calls} dq={attention_dq_calls} "
        f"dk={attention_dk_calls} dv={attention_dv_calls} tensor_core={tensor_core}"
    )
print(
    "Tensor Core gate passed: "
    f"bf16_tensor_core_matmul_calls={matmul_calls} "
    f"forward_calls={forward_calls} "
    f"backward_calls={backward_calls} "
    f"linear_tensor_core_calls={linear_tensor_core_calls} "
    f"attention_forward_calls={attention_forward_calls} "
    f"attention_qk_calls={attention_qk_calls} "
    f"attention_av_calls={attention_av_calls} "
    f"attention_backward_calls={attention_backward_calls} "
    f"attention_score_grad_calls={attention_score_grad_calls} "
    f"attention_dq_calls={attention_dq_calls} "
    f"attention_dk_calls={attention_dk_calls} "
    f"attention_dv_calls={attention_dv_calls}"
)
PY

python3 - "$timings_path" "${stage_names[@]}" -- "${stage_millis[@]}" <<'PY'
import json
import sys
from pathlib import Path

separator = sys.argv.index("--")
names = sys.argv[2:separator]
millis = [int(value) for value in sys.argv[separator + 1:]]
Path(sys.argv[1]).write_text(
    json.dumps(dict(zip(names, millis)), indent=2) + "\n",
    encoding="utf-8",
)
PY

python3 - "$summary_path" "$workdir" "$device" "$devices" "$distributed" "$precision" "$vocab_size" "$max_bytes" "$run_label" "$steps" "$batch_size" "$block_size" "$d_model" "$n_heads" "$ff_hidden" "$lr" "$min_reduction" "$eval_batches" "$generation_tokens" "$report_path" "$eval_report_path" "$generation_report_path" "$timings_path" "$manifest_path" "$data_path" <<'PY'
import json
import sys
from pathlib import Path

(
    summary_path,
    workdir,
    device,
    devices,
    distributed,
    precision,
    vocab_size,
    max_bytes,
    run_label,
    steps,
    batch_size,
    block_size,
    d_model,
    n_heads,
    ff_hidden,
    lr,
    min_reduction,
    eval_batches,
    generation_tokens,
    train_path,
    eval_path,
    generation_path,
    timings_path,
    manifest_path,
    data_path,
) = sys.argv[1:26]
with open(train_path, encoding="utf-8") as handle:
    train = json.load(handle)
with open(eval_path, encoding="utf-8") as handle:
    eval_report = json.load(handle)
with open(generation_path, encoding="utf-8") as handle:
    generation = json.load(handle)
with open(timings_path, encoding="utf-8") as handle:
    timings = json.load(handle)
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)

summary = {
    "command": "tinystories-valid-reference",
    "run_label": run_label,
    "workdir": workdir,
    "device": device,
    "devices": [item for item in devices.split(",") if item] if devices else None,
    "distributed": distributed or None,
    "precision": precision,
    "vocab_size": int(vocab_size),
    "max_bytes": int(max_bytes) if max_bytes else None,
    "model": {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "per_rank_batch_size": train.get("per_rank_batch_size"),
        "global_batch_size": train.get("global_batch_size"),
        "block_size": int(block_size),
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "ff_hidden": int(ff_hidden),
        "lr": float(lr),
        "precision": precision,
        "min_reduction": float(min_reduction),
    },
    "data": {
        "path": str(Path(data_path)),
        "manifest": str(Path(manifest_path)),
        "source_hash": manifest.get("source_hash"),
        "tokenizer_hash": manifest.get("tokenizer_hash"),
        "train_tokens": manifest.get("train_tokens"),
        "valid_tokens": manifest.get("valid_tokens"),
        "max_bytes": int(max_bytes) if max_bytes else None,
    },
    "timings_millis": timings,
    "train": {
        "start_step": train["start_step"],
        "final_step": train["final_step"],
        "initial_loss": train["initial_loss"],
        "final_loss": train["final_loss"],
        "loss_reduction": train["loss_reduction"],
        "device": train.get("device"),
        "devices": train.get("devices"),
        "distributed": train.get("distributed"),
        "precision": train.get("precision"),
        "all_reduce_calls": train.get("all_reduce_calls"),
        "all_reduce_bytes": train.get("all_reduce_bytes"),
        "parameter_checksum_sum_max_error": train.get("parameter_checksum_sum_max_error"),
        "parameter_checksum_sumsq_max_error": train.get("parameter_checksum_sumsq_max_error"),
        "step_checksum_sum_max_error": train.get("step_checksum_sum_max_error"),
        "step_checksum_sumsq_max_error": train.get("step_checksum_sumsq_max_error"),
        "tensor_core": train.get("tensor_core"),
        "tensor_core_pad_crop": train.get("tensor_core_pad_crop"),
        "tensor_core_coverage": train.get("tensor_core_coverage"),
    },
    "eval": {
        "device": eval_report.get("device"),
        "precision": eval_report.get("precision"),
        "split": eval_report.get("split"),
        "metrics": eval_report.get("metrics"),
    },
    "generation": {
        "device": generation.get("device"),
        "precision": generation.get("precision"),
        "prompt": generation.get("prompt"),
        "generated_tokens": generation.get("generated_tokens"),
        "configured_max_new_tokens": int(generation_tokens),
        "finish_reason": generation.get("finish_reason"),
    },
    "artifacts": {
        "train_report": str(Path(train_path)),
        "eval_report": str(Path(eval_path)),
        "generation_report": str(Path(generation_path)),
        "timings_report": str(Path(timings_path)),
    },
}
Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

if [[ "$resume_steps" -gt 0 ]]; then
  python3 - "$summary_path" "$resume_report_path" "$resume_checkpoint_path" "$resume_steps" <<'PY'
import json
import sys
from pathlib import Path

summary_path, resume_path, resume_checkpoint_path, resume_steps = sys.argv[1:5]
summary_file = Path(summary_path)
summary = json.loads(summary_file.read_text(encoding="utf-8"))
resume = json.loads(Path(resume_path).read_text(encoding="utf-8"))
summary["model"]["resume_steps"] = int(resume_steps)
summary["resume"] = {
    "start_step": resume["start_step"],
    "final_step": resume["final_step"],
    "initial_loss": resume["initial_loss"],
    "final_loss": resume["final_loss"],
    "loss_reduction": resume["loss_reduction"],
    "devices": resume.get("devices"),
    "distributed": resume.get("distributed"),
    "precision": resume.get("precision"),
    "all_reduce_calls": resume.get("all_reduce_calls"),
    "all_reduce_bytes": resume.get("all_reduce_bytes"),
    "parameter_checksum_sum_max_error": resume.get("parameter_checksum_sum_max_error"),
    "parameter_checksum_sumsq_max_error": resume.get("parameter_checksum_sumsq_max_error"),
    "step_checksum_sum_max_error": resume.get("step_checksum_sum_max_error"),
    "step_checksum_sumsq_max_error": resume.get("step_checksum_sumsq_max_error"),
    "tensor_core": resume.get("tensor_core"),
    "tensor_core_pad_crop": resume.get("tensor_core_pad_crop"),
    "tensor_core_coverage": resume.get("tensor_core_coverage"),
    "checkpoint": str(Path(resume_checkpoint_path)),
}
summary["artifacts"]["resume_report"] = str(Path(resume_path))
summary["artifacts"]["resume_checkpoint"] = str(Path(resume_checkpoint_path))
summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
fi

echo "summary: $summary_path"
echo "timings: $timings_path"
echo "report: $report_path"
if [[ "$resume_steps" -gt 0 ]]; then
  echo "resume report: $resume_report_path"
fi
echo "eval: $eval_report_path"
echo "generation report: $generation_report_path"
echo "generation: $generation_path"
