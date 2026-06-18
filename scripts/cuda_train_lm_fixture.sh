#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${HEIRLOOM_CUDA_FIXTURE_DEVICE:-cuda:0}"
DEVICES="${HEIRLOOM_CUDA_FIXTURE_DEVICES:-}"
DISTRIBUTED="${HEIRLOOM_CUDA_FIXTURE_DISTRIBUTED:-}"
DDP_INIT_TIMEOUT_SECS="${HEIRLOOM_CUDA_FIXTURE_DDP_INIT_TIMEOUT_SECS:-120}"
DDP_CHECKSUM_EVERY="${HEIRLOOM_CUDA_FIXTURE_DDP_CHECKSUM_EVERY:-1}"
PRECISION="${HEIRLOOM_CUDA_FIXTURE_PRECISION:-f32}"
REQUIRE_TENSOR_CORES="${HEIRLOOM_REQUIRE_TENSOR_CORES:-0}"
REQUIRE_ATTENTION_TENSOR_CORES="${HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES:-0}"
EXPECT_TENSOR_CORES="${HEIRLOOM_EXPECT_TENSOR_CORES:-$REQUIRE_TENSOR_CORES}"
EXPECT_ATTENTION_TENSOR_CORES="${HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES:-$REQUIRE_ATTENTION_TENSOR_CORES}"
EXPECT_FLASH_BF16_ATTENTION="${HEIRLOOM_EXPECT_FLASH_BF16_ATTENTION:-0}"
EXPECT_TENSOR_CORE_PADDING="${HEIRLOOM_EXPECT_TENSOR_CORE_PADDING:-0}"
EXPECT_ATTENTION_TENSOR_CORE_PADDING="${HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORE_PADDING:-0}"
AUTO_TILE="${HEIRLOOM_CUDA_FIXTURE_AUTO_TILE:-1}"
RAGGED_TENSOR_CORES="${HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES:-0}"
RAGGED_ATTENTION_TENSOR_CORES="${HEIRLOOM_CUDA_FIXTURE_RAGGED_ATTENTION_TENSOR_CORES:-0}"
STEPS="${HEIRLOOM_CUDA_FIXTURE_STEPS:-40}"
RESUME_STEPS="${HEIRLOOM_CUDA_FIXTURE_RESUME_STEPS:-3}"
MIN_REDUCTION="${HEIRLOOM_CUDA_FIXTURE_MIN_REDUCTION:-0.02}"
BATCH_SIZE="${HEIRLOOM_CUDA_FIXTURE_BATCH_SIZE:-4}"
BLOCK_SIZE="${HEIRLOOM_CUDA_FIXTURE_BLOCK_SIZE:-4}"
D_MODEL="${HEIRLOOM_CUDA_FIXTURE_D_MODEL:-4}"
N_HEADS="${HEIRLOOM_CUDA_FIXTURE_N_HEADS:-2}"
FF_HIDDEN="${HEIRLOOM_CUDA_FIXTURE_FF_HIDDEN:-8}"
VOCAB_SIZE="${HEIRLOOM_CUDA_FIXTURE_VOCAB_SIZE:-280}"
LR="${HEIRLOOM_CUDA_FIXTURE_LR:-0.01}"
SEED="${HEIRLOOM_CUDA_FIXTURE_SEED:-123}"
OUT_DIR="${HEIRLOOM_CUDA_FIXTURE_OUT_DIR:-}"
DISTRIBUTED_MODE=0
if [[ -n "$DEVICES" || -n "$DISTRIBUTED" ]]; then
  DISTRIBUTED_MODE=1
  DISTRIBUTED="${DISTRIBUTED:-nccl}"
  if [[ -z "$DEVICES" ]]; then
    echo "HEIRLOOM_CUDA_FIXTURE_DEVICES is required when HEIRLOOM_CUDA_FIXTURE_DISTRIBUTED is set" >&2
    exit 1
  fi
fi
CUDA_TARGET_IS_CUDA=0
if [[ "$DEVICE" == cuda:* || -n "$DEVICES" ]]; then
  CUDA_TARGET_IS_CUDA=1
fi

if [[ "$RAGGED_TENSOR_CORES" == "1" && "$REQUIRE_ATTENTION_TENSOR_CORES" == "1" ]]; then
  echo "HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES=1 is intentionally incompatible with HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1" >&2
  exit 1
fi

if [[ "$RAGGED_TENSOR_CORES" == "1" && "$RAGGED_ATTENTION_TENSOR_CORES" == "1" ]]; then
  echo "Choose only one ragged fixture mode: HEIRLOOM_CUDA_FIXTURE_RAGGED_TENSOR_CORES or HEIRLOOM_CUDA_FIXTURE_RAGGED_ATTENTION_TENSOR_CORES" >&2
  exit 1
fi

if [[ "$RAGGED_ATTENTION_TENSOR_CORES" == "1" && "$CUDA_TARGET_IS_CUDA" == "1" && ( "$PRECISION" == "amp-bf16" || "$REQUIRE_ATTENTION_TENSOR_CORES" == "1" ) ]]; then
  BATCH_SIZE="${HEIRLOOM_CUDA_FIXTURE_BATCH_SIZE:-3}"
  BLOCK_SIZE="${HEIRLOOM_CUDA_FIXTURE_BLOCK_SIZE:-15}"
  D_MODEL="${HEIRLOOM_CUDA_FIXTURE_D_MODEL:-17}"
  N_HEADS="${HEIRLOOM_CUDA_FIXTURE_N_HEADS:-1}"
  FF_HIDDEN="${HEIRLOOM_CUDA_FIXTURE_FF_HIDDEN:-37}"
  VOCAB_SIZE="${HEIRLOOM_CUDA_FIXTURE_VOCAB_SIZE:-281}"
  EXPECT_TENSOR_CORES="${HEIRLOOM_EXPECT_TENSOR_CORES:-1}"
  EXPECT_ATTENTION_TENSOR_CORES="${HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES:-1}"
  EXPECT_TENSOR_CORE_PADDING="${HEIRLOOM_EXPECT_TENSOR_CORE_PADDING:-1}"
  EXPECT_ATTENTION_TENSOR_CORE_PADDING="${HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORE_PADDING:-1}"
elif [[ "$RAGGED_TENSOR_CORES" == "1" && "$CUDA_TARGET_IS_CUDA" == "1" && ( "$PRECISION" == "amp-bf16" || "$REQUIRE_TENSOR_CORES" == "1" ) ]]; then
  BATCH_SIZE="${HEIRLOOM_CUDA_FIXTURE_BATCH_SIZE:-3}"
  BLOCK_SIZE="${HEIRLOOM_CUDA_FIXTURE_BLOCK_SIZE:-5}"
  D_MODEL="${HEIRLOOM_CUDA_FIXTURE_D_MODEL:-18}"
  N_HEADS="${HEIRLOOM_CUDA_FIXTURE_N_HEADS:-3}"
  FF_HIDDEN="${HEIRLOOM_CUDA_FIXTURE_FF_HIDDEN:-37}"
  VOCAB_SIZE="${HEIRLOOM_CUDA_FIXTURE_VOCAB_SIZE:-281}"
  EXPECT_TENSOR_CORES="${HEIRLOOM_EXPECT_TENSOR_CORES:-1}"
  EXPECT_TENSOR_CORE_PADDING="${HEIRLOOM_EXPECT_TENSOR_CORE_PADDING:-1}"
elif [[ "$AUTO_TILE" != "0" && "$CUDA_TARGET_IS_CUDA" == "1" && ( "$PRECISION" == "amp-bf16" || "$REQUIRE_TENSOR_CORES" == "1" ) ]]; then
  BATCH_SIZE="${HEIRLOOM_CUDA_FIXTURE_BATCH_SIZE:-4}"
  BLOCK_SIZE="${HEIRLOOM_CUDA_FIXTURE_BLOCK_SIZE:-4}"
  D_MODEL="${HEIRLOOM_CUDA_FIXTURE_D_MODEL:-16}"
  N_HEADS="${HEIRLOOM_CUDA_FIXTURE_N_HEADS:-4}"
  FF_HIDDEN="${HEIRLOOM_CUDA_FIXTURE_FF_HIDDEN:-64}"
  VOCAB_SIZE="${HEIRLOOM_CUDA_FIXTURE_VOCAB_SIZE:-272}"
  EXPECT_TENSOR_CORES="${HEIRLOOM_EXPECT_TENSOR_CORES:-1}"
fi

if [[ "$AUTO_TILE" != "0" && "$CUDA_TARGET_IS_CUDA" == "1" && "$REQUIRE_ATTENTION_TENSOR_CORES" == "1" && "$RAGGED_ATTENTION_TENSOR_CORES" != "1" && "$RAGGED_TENSOR_CORES" != "1" ]]; then
  BATCH_SIZE="${HEIRLOOM_CUDA_FIXTURE_BATCH_SIZE:-4}"
  BLOCK_SIZE="${HEIRLOOM_CUDA_FIXTURE_BLOCK_SIZE:-16}"
  D_MODEL="${HEIRLOOM_CUDA_FIXTURE_D_MODEL:-16}"
  N_HEADS="${HEIRLOOM_CUDA_FIXTURE_N_HEADS:-1}"
  FF_HIDDEN="${HEIRLOOM_CUDA_FIXTURE_FF_HIDDEN:-64}"
  VOCAB_SIZE="${HEIRLOOM_CUDA_FIXTURE_VOCAB_SIZE:-272}"
  EXPECT_TENSOR_CORES="${HEIRLOOM_EXPECT_TENSOR_CORES:-1}"
  EXPECT_ATTENTION_TENSOR_CORES="${HEIRLOOM_EXPECT_ATTENTION_TENSOR_CORES:-1}"
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

python3 - "$OUT_DIR/tiny.txt" <<'PY'
from pathlib import Path
import sys

text = (
    "one little bird saw a red door. "
    "one little bird opened the red door. "
    "two little cats saw a blue door. "
    "two little cats opened the blue door. "
    "the red door led to a warm room. "
    "the blue door led to a cold room. "
) * 8
Path(sys.argv[1]).write_text(text, encoding="utf-8")
PY

run cargo run --bin heirloom -- tokenizer train \
  --input "$OUT_DIR/tiny.txt" \
  --out "$OUT_DIR/tokenizer.json" \
  --vocab-size "$VOCAB_SIZE"

run cargo run --bin heirloom -- data prepare \
  --input "$OUT_DIR/tiny.txt" \
  --tokenizer "$OUT_DIR/tokenizer.json" \
  --out-dir "$OUT_DIR/prepared" \
  --valid-fraction 0.2

TRAIN_TARGET_ARGS=()
if [[ "$DISTRIBUTED_MODE" == "1" ]]; then
  TRAIN_TARGET_ARGS=(
    --devices "$DEVICES"
    --distributed "$DISTRIBUTED"
    --ddp-init-timeout-secs "$DDP_INIT_TIMEOUT_SECS"
    --ddp-checksum-every "$DDP_CHECKSUM_EVERY"
  )
else
  TRAIN_TARGET_ARGS=(--device "$DEVICE")
fi

run cargo run --bin heirloom -- train-lm \
  --dataset-manifest "$OUT_DIR/prepared/manifest.json" \
  --checkpoint "$OUT_DIR/checkpoint" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --block-size "$BLOCK_SIZE" \
  --d-model "$D_MODEL" \
  --n-heads "$N_HEADS" \
  --ff-hidden "$FF_HIDDEN" \
  --lr "$LR" \
  --weight-decay 0.0 \
  --clip-norm 1.0 \
  --seed "$SEED" \
  "${TRAIN_TARGET_ARGS[@]}" \
  --precision "$PRECISION" \
  --log-every 10 \
  --report "$OUT_DIR/train-report.json"

if [[ "$DISTRIBUTED_MODE" == "1" && -d "$OUT_DIR/ddp-ranks" ]]; then
  rm -rf "$OUT_DIR/train-ddp-ranks"
  cp -a "$OUT_DIR/ddp-ranks" "$OUT_DIR/train-ddp-ranks"
fi

python3 - "$OUT_DIR/train-report.json" "$MIN_REDUCTION" "$STEPS" "$DEVICE" "$PRECISION" "$EXPECT_TENSOR_CORES" "$EXPECT_ATTENTION_TENSOR_CORES" "$EXPECT_FLASH_BF16_ATTENTION" "$EXPECT_TENSOR_CORE_PADDING" "$EXPECT_ATTENTION_TENSOR_CORE_PADDING" "$DEVICES" "$DISTRIBUTED" "$BATCH_SIZE" "$DDP_CHECKSUM_EVERY" <<'PY'
import json
import math
import os
import sys

(
    report_path,
    min_reduction,
    expected_steps,
    expected_device,
    expected_precision,
    expect_tensor_cores,
    expect_attention_tensor_cores,
    expect_flash_bf16_attention,
    expect_tensor_core_padding,
    expect_attention_tensor_core_padding,
    expected_devices,
    expected_distributed,
    expected_batch_size,
    ddp_checksum_every,
) = sys.argv[1:15]
min_reduction = float(min_reduction)
expected_steps = int(expected_steps)
expected_batch_size = int(expected_batch_size)
ddp_checksum_every = int(ddp_checksum_every)
expect_tensor_cores = expect_tensor_cores in {"1", "true", "TRUE", "yes", "YES"}
expect_attention_tensor_cores = expect_attention_tensor_cores in {"1", "true", "TRUE", "yes", "YES"}
expect_flash_bf16_attention = expect_flash_bf16_attention in {"1", "true", "TRUE", "yes", "YES"}
expect_tensor_core_padding = expect_tensor_core_padding in {"1", "true", "TRUE", "yes", "YES"}
expect_attention_tensor_core_padding = expect_attention_tensor_core_padding in {"1", "true", "TRUE", "yes", "YES"}
legacy_warp_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
global_cta_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
wide_swizzled_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
ldmatrix_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
cp_async_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
distributed = bool(expected_devices)

initial = float(report["initial_loss"])
final = float(report["final_loss"])
reduction = float(report["loss_reduction"])
if report["start_step"] != 0 or report["final_step"] != expected_steps:
    raise SystemExit(f"unexpected train step range: {report}")
if distributed:
    devices = [part.strip() for part in expected_devices.split(",") if part.strip()]
    if report.get("distributed") != expected_distributed:
        raise SystemExit(f"unexpected distributed mode in train report: {report.get('distributed')} expected {expected_distributed}")
    if report.get("devices") != devices:
        raise SystemExit(f"unexpected devices in train report: {report.get('devices')} expected {devices}")
    if int(report.get("world_size") or 0) != len(devices):
        raise SystemExit(f"unexpected world_size in train report: {report.get('world_size')} expected {len(devices)}")
    if int(report.get("per_rank_batch_size") or 0) != expected_batch_size:
        raise SystemExit(f"unexpected per-rank batch size in train report: {report.get('per_rank_batch_size')}")
    if int(report.get("global_batch_size") or 0) != expected_batch_size * len(devices):
        raise SystemExit(f"unexpected global batch size in train report: {report.get('global_batch_size')}")
    if int(report.get("all_reduce_calls") or 0) <= 0 or int(report.get("all_reduce_bytes") or 0) <= 0:
        raise SystemExit(f"missing distributed all-reduce stats in train report: {report}")
    tolerance = float(report.get("parameter_checksum_tolerance") or 0.0)
    for field in [
        "parameter_checksum_sum_max_error",
        "parameter_checksum_sumsq_max_error",
        "step_checksum_sum_max_error",
        "step_checksum_sumsq_max_error",
    ]:
        value = float(report.get(field) or 0.0)
        if value > tolerance:
            raise SystemExit(f"{field}={value:.6e} exceeds tolerance={tolerance:.6e}")
    if ddp_checksum_every > 0 and not report.get("step_checksum_drift"):
        raise SystemExit("expected per-step checksum drift entries in distributed train report")
    ranks = report.get("ranks") or []
    if len(ranks) != len(devices):
        raise SystemExit(f"expected {len(devices)} rank reports, found {len(ranks)}")
    for index, rank in enumerate(ranks):
        if int(rank.get("rank")) != index:
            raise SystemExit(f"rank report order mismatch: index={index} rank={rank}")
        if int(rank.get("final_step") or -1) != expected_steps:
            raise SystemExit(f"rank {index} final_step mismatch: {rank}")
        if int(rank.get("all_reduce_calls") or 0) <= 0:
            raise SystemExit(f"rank {index} has no all-reduce calls: {rank}")
        if bool(rank.get("checkpoint_saved")) != (index == 0):
            raise SystemExit(f"rank {index} checkpoint_saved mismatch: {rank.get('checkpoint_saved')}")
else:
    if report.get("device") != expected_device:
        raise SystemExit(f"unexpected device in train report: {report.get('device')} expected {expected_device}")
if report.get("precision") != expected_precision:
    raise SystemExit(f"unexpected precision in train report: {report.get('precision')} expected {expected_precision}")
if not (math.isfinite(initial) and math.isfinite(final) and math.isfinite(reduction)):
    raise SystemExit(f"non-finite train loss report: {report}")
def validate_amp_bf16_report(payload, label):
    if expected_precision != "amp-bf16":
        return
    validation = payload.get("amp_bf16_validation") or {}
    if validation.get("status") != "passed":
        raise SystemExit(f"expected passed AMP BF16 validation in {label} report: {validation}")
    if int(validation.get("unexpected_host_staging_count") or 0) != 0:
        raise SystemExit(f"unexpected CUDA host staging in {label} report: {validation}")
    if int(validation.get("finite_check_failure_count") or 0) != 0:
        raise SystemExit(f"AMP finite-check failure in {label} report: {validation}")
    if int(validation.get("finite_check_count") or 0) <= 0:
        raise SystemExit(f"missing AMP finite checks in {label} report: {validation}")
    if int(validation.get("op_decision_count") or 0) <= 0:
        raise SystemExit(f"missing AMP op decisions in {label} report: {validation}")
    decisions = payload.get("amp_bf16_op_decisions") or payload.get("amp_bf16_op_decisions_rank0") or []
    finite_checks = payload.get("amp_bf16_finite_checks") or payload.get("amp_bf16_finite_checks_rank0") or []
    if not decisions:
        raise SystemExit(f"missing AMP op-decision detail in {label} report")
    if not finite_checks:
        raise SystemExit(f"missing AMP finite-check detail in {label} report")
    for index, rank_validation in enumerate(payload.get("amp_bf16_validation_by_rank") or []):
        if (rank_validation or {}).get("status") != "passed":
            raise SystemExit(f"rank {index} AMP BF16 validation failed in {label} report: {rank_validation}")
validate_amp_bf16_report(report, "train")
if final >= initial * (1.0 - min_reduction):
    raise SystemExit(
        f"expected CUDA fixture loss reduction >= {min_reduction:.3f}; "
        f"initial={initial:.6f} final={final:.6f} reduction={reduction:.6f}"
    )
tensor_core = report.get("tensor_core") or {}
cuda_runtime = report.get("cuda_runtime") or {}
pad_crop = report.get("tensor_core_pad_crop") or {}
coverage = report.get("tensor_core_coverage") or {}
if distributed and not coverage and report.get("ranks"):
    coverage = report["ranks"][0].get("tensor_core_coverage") or {}
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
tensor_core_padded_tiles = int(pad_crop.get("padded_tiles") or cuda_runtime.get("tensor_core_padded_tiles") or 0)
tensor_core_remainder_tiles = int(pad_crop.get("remainder_tiles") or cuda_runtime.get("tensor_core_remainder_tiles") or 0)
tensor_core_cta_gemm_calls = int(cuda_runtime.get("tensor_core_cta_gemm_calls") or 0)
tensor_core_cta_tiles = int(cuda_runtime.get("tensor_core_cta_tiles") or 0)
tensor_core_mma_warp_tiles = int(cuda_runtime.get("tensor_core_mma_warp_tiles") or 0)
tensor_core_staged_cta_gemm_calls = int(cuda_runtime.get("tensor_core_staged_cta_gemm_calls") or 0)
tensor_core_shared_stage_tiles = int(cuda_runtime.get("tensor_core_shared_stage_tiles") or 0)
tensor_core_shared_stage_bytes = int(cuda_runtime.get("tensor_core_shared_stage_bytes") or 0)
tensor_core_wide_swizzled_cta_gemm_calls = int(cuda_runtime.get("tensor_core_wide_swizzled_cta_gemm_calls") or 0)
tensor_core_swizzled_stage_tiles = int(cuda_runtime.get("tensor_core_swizzled_stage_tiles") or 0)
tensor_core_swizzled_stage_bytes = int(cuda_runtime.get("tensor_core_swizzled_stage_bytes") or 0)
tensor_core_ldmatrix_gemm_executed_calls = int(cuda_runtime.get("tensor_core_ldmatrix_gemm_executed_calls") or 0)
tensor_core_ldmatrix_gemm_staged_fallback_calls = int(cuda_runtime.get("tensor_core_ldmatrix_gemm_staged_fallback_calls") or 0)
tensor_core_ldmatrix_gemm_hard_require_failures = int(cuda_runtime.get("tensor_core_ldmatrix_gemm_hard_require_failures") or 0)
tensor_core_ldmatrix_gemm_instructions = int(cuda_runtime.get("tensor_core_ldmatrix_gemm_instructions") or 0)
tensor_core_cp_async_gemm_executed_calls = int(cuda_runtime.get("tensor_core_cp_async_gemm_executed_calls") or 0)
tensor_core_cp_async_gemm_staged_fallback_calls = int(cuda_runtime.get("tensor_core_cp_async_gemm_staged_fallback_calls") or 0)
tensor_core_cp_async_gemm_hard_require_failures = int(cuda_runtime.get("tensor_core_cp_async_gemm_hard_require_failures") or 0)
tensor_core_cp_async_gemm_instructions = int(cuda_runtime.get("tensor_core_cp_async_gemm_instructions") or 0)
tensor_core_global_cta_gemm_calls = int(cuda_runtime.get("tensor_core_global_cta_gemm_calls") or 0)
tensor_core_legacy_warp_gemm_calls = int(cuda_runtime.get("tensor_core_legacy_warp_gemm_calls") or 0)
attention_padded_tiles = int(cuda_runtime.get("tensor_core_attention_padded_tiles") or 0)
attention_remainder_tiles = int(cuda_runtime.get("tensor_core_attention_remainder_tiles") or 0)
attention_scalar_fallbacks = int(cuda_runtime.get("tensor_core_attention_scalar_fallbacks") or 0)
flash_forward_executed = int(cuda_runtime.get("flash_bf16_tensor_core_executed_calls") or 0)
flash_forward_fallback = int(cuda_runtime.get("flash_bf16_tensor_core_fallback_calls") or 0)
flash_backward_requested = int(cuda_runtime.get("flash_bf16_tensor_core_backward_requested_calls") or 0)
flash_backward_executed = int(cuda_runtime.get("flash_bf16_tensor_core_backward_executed_calls") or 0)
flash_backward_fallback = int(cuda_runtime.get("flash_bf16_tensor_core_backward_fallback_calls") or 0)
flash_backward_qk_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls") or 0)
flash_backward_dp_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_dp_mma_tile_calls") or 0)
flash_backward_dq_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_dq_mma_tile_calls") or 0)
flash_backward_dk_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_dk_mma_tile_calls") or 0)
flash_backward_dv_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_dv_mma_tile_calls") or 0)
flash_backward_scalar_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_scalar_tile_calls") or 0)
flash_hard_failures = int(cuda_runtime.get("flash_bf16_attention_hard_require_failures") or 0)
materialized_reference_attention = int(cuda_runtime.get("bf16_attention_materialized_reference_calls") or 0)
flash_forward_elapsed_us = int(cuda_runtime.get("flash_bf16_tensor_core_elapsed_us") or 0)
flash_backward_elapsed_us = int(cuda_runtime.get("flash_bf16_tensor_core_backward_elapsed_us") or 0)
pad_crop_status = str(pad_crop.get("status") or "legacy_runtime_counters")
if expect_tensor_cores and matmul_calls <= 0:
    raise SystemExit(
        "expected Tensor Core matmul calls in train report but found none; "
        f"tensor_core={tensor_core} report={report}"
    )
if expect_tensor_cores and forward_calls <= 0:
    raise SystemExit(
        "expected Tensor Core forward matmul calls in train report but found none; "
        f"tensor_core={tensor_core} report={report}"
    )
if expect_tensor_cores and backward_calls <= 0:
    raise SystemExit(
        "expected Tensor Core backward matmul calls in train report but found none; "
        f"tensor_core={tensor_core} report={report}"
    )
if expect_tensor_cores and fallback_calls != 0:
    raise SystemExit(
        "expected zero scalar Tensor Core fallback calls under hard gate; "
        f"tensor_core={tensor_core}"
    )
if expect_tensor_cores and linear_tensor_core_calls <= 0:
    raise SystemExit(
        "expected Tensor Core Linear coverage in train report but found none; "
        f"tensor_core_coverage={coverage}"
    )
if expect_tensor_cores and linear_fallback_calls != 0:
    raise SystemExit(
        "expected zero Tensor Core Linear coverage fallbacks under hard gate; "
        f"tensor_core_coverage={coverage}"
    )
if expect_tensor_cores and not legacy_warp_gemm:
    if tensor_core_cta_gemm_calls <= 0 or tensor_core_cta_tiles <= 0 or tensor_core_mma_warp_tiles <= 0:
        raise SystemExit(
            "expected CTA Tensor Core GEMM counters in train report under hard gate; "
            f"cuda_runtime={cuda_runtime}"
        )
    if global_cta_gemm:
        if tensor_core_global_cta_gemm_calls <= 0:
            raise SystemExit(
                "expected global-fed CTA Tensor Core GEMM counter in train report when debug env is set; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_wide_swizzled_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero wide-swizzled CTA GEMM calls when global-fed debug env is set; "
                f"cuda_runtime={cuda_runtime}"
            )
    elif wide_swizzled_gemm:
        if (
            tensor_core_wide_swizzled_cta_gemm_calls <= 0
            or tensor_core_swizzled_stage_tiles <= 0
            or tensor_core_swizzled_stage_bytes <= 0
        ):
            raise SystemExit(
                "expected wide-swizzled CTA Tensor Core GEMM counters in train report when wide env is set; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_staged_cta_gemm_calls != 0 or tensor_core_global_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero default staged/global CTA GEMM calls in wide-swizzled train report; "
                f"cuda_runtime={cuda_runtime}"
            )
    elif cp_async_gemm:
        if (
            tensor_core_cp_async_gemm_executed_calls <= 0
            or tensor_core_cp_async_gemm_instructions <= 0
            or tensor_core_ldmatrix_gemm_instructions <= 0
        ):
            raise SystemExit(
                "expected double-buffered cp.async Tensor Core GEMM counters in train report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if (
            tensor_core_cp_async_gemm_staged_fallback_calls != 0
            or tensor_core_cp_async_gemm_hard_require_failures != 0
        ):
            raise SystemExit(
                "expected zero cp.async GEMM fallback/hard-failure counters in train report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_staged_cta_gemm_calls != 0 or tensor_core_global_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero staged/global CTA GEMM calls in cp.async train report; "
                f"cuda_runtime={cuda_runtime}"
            )
    elif ldmatrix_gemm:
        if (
            tensor_core_ldmatrix_gemm_executed_calls <= 0
            or tensor_core_ldmatrix_gemm_instructions <= 0
        ):
            raise SystemExit(
                "expected ldmatrix Tensor Core GEMM counters in train report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if (
            tensor_core_ldmatrix_gemm_staged_fallback_calls != 0
            or tensor_core_ldmatrix_gemm_hard_require_failures != 0
        ):
            raise SystemExit(
                "expected zero ldmatrix GEMM fallback/hard-failure counters in train report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_staged_cta_gemm_calls != 0 or tensor_core_global_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero staged/global CTA GEMM calls in ldmatrix train report; "
                f"cuda_runtime={cuda_runtime}"
            )
    else:
        if tensor_core_staged_cta_gemm_calls <= 0 or tensor_core_shared_stage_tiles <= 0 or tensor_core_shared_stage_bytes <= 0:
            raise SystemExit(
                "expected shared-memory staged CTA Tensor Core GEMM counters in train report under hard gate; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_global_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero global-fed CTA GEMM calls in default staged train report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_wide_swizzled_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero wide-swizzled CTA GEMM calls in default staged train report; "
                f"cuda_runtime={cuda_runtime}"
            )
    if tensor_core_legacy_warp_gemm_calls != 0:
        raise SystemExit(
            "expected zero legacy one-warp Tensor Core GEMM calls in train report; "
            f"cuda_runtime={cuda_runtime}"
        )
if expect_tensor_core_padding:
    if pad_crop:
        if not bool(pad_crop.get("used")) or pad_crop.get("status") != "passed":
            raise SystemExit(
                "expected named Tensor Core pad/crop report to pass in train report; "
                f"tensor_core_pad_crop={pad_crop}"
            )
    if tensor_core_padded_tiles <= 0:
        raise SystemExit(
            "expected positive Tensor Core padded tile count in train report; "
            f"tensor_core_pad_crop={pad_crop} cuda_runtime={cuda_runtime}"
        )
    if tensor_core_remainder_tiles <= 0:
        raise SystemExit(
            "expected positive Tensor Core remainder tile count in train report; "
            f"tensor_core_pad_crop={pad_crop} cuda_runtime={cuda_runtime}"
        )
if expect_attention_tensor_cores and attention_forward_calls <= 0:
    raise SystemExit(
        "expected Tensor Core attention forward calls in train report but found none; "
        f"tensor_core={tensor_core} report={report}"
    )
if expect_attention_tensor_cores and (attention_qk_calls <= 0 or attention_av_calls <= 0):
    raise SystemExit(
        "expected Tensor Core attention QK and AV matmul calls in train report; "
        f"qk={attention_qk_calls} av={attention_av_calls} tensor_core={tensor_core}"
    )
if expect_attention_tensor_cores and attention_backward_calls <= 0:
    raise SystemExit(
        "expected Tensor Core attention backward calls in train report but found none; "
        f"tensor_core={tensor_core} report={report}"
    )
if expect_attention_tensor_cores and min(
    attention_score_grad_calls,
    attention_dq_calls,
    attention_dk_calls,
    attention_dv_calls,
) <= 0:
    raise SystemExit(
        "expected Tensor Core attention backward score/dQ/dK/dV matmul calls in train report; "
        f"score_grad={attention_score_grad_calls} dq={attention_dq_calls} "
        f"dk={attention_dk_calls} dv={attention_dv_calls} tensor_core={tensor_core}"
    )
if expect_attention_tensor_cores and attention_scalar_fallbacks != 0:
    raise SystemExit(
        "expected zero Tensor Core attention scalar fallbacks in train report; "
        f"cuda_runtime={cuda_runtime}"
    )
if expect_attention_tensor_core_padding:
    if attention_padded_tiles <= 0:
        raise SystemExit(
            "expected positive Tensor Core attention padded tile count in train report; "
            f"cuda_runtime={cuda_runtime}"
        )
    if attention_remainder_tiles <= 0:
        raise SystemExit(
            "expected positive Tensor Core attention remainder tile count in train report; "
            f"cuda_runtime={cuda_runtime}"
        )
if expect_flash_bf16_attention:
    if flash_forward_executed <= 0:
        raise SystemExit(
            "expected Tensor Core flash attention forward execution in train report; "
            f"cuda_runtime={cuda_runtime}"
        )
    if flash_backward_requested <= 0 or flash_backward_executed <= 0:
        raise SystemExit(
            "expected flash attention backward request/execution in train report; "
            f"requested={flash_backward_requested} executed={flash_backward_executed} "
            f"cuda_runtime={cuda_runtime}"
        )
    if flash_forward_fallback != 0 or flash_backward_fallback != 0:
        raise SystemExit(
            "expected zero flash attention fallback counters in train report; "
            f"forward_fallback={flash_forward_fallback} backward_fallback={flash_backward_fallback} "
            f"cuda_runtime={cuda_runtime}"
        )
    if flash_hard_failures != 0:
        raise SystemExit(
            "expected zero flash attention hard-require failures in train report; "
            f"cuda_runtime={cuda_runtime}"
        )
    if materialized_reference_attention != 0:
        raise SystemExit(
            "expected zero materialized BF16 reference attention calls under flash train gate; "
            f"cuda_runtime={cuda_runtime}"
        )
    if flash_forward_elapsed_us <= 0 or flash_backward_elapsed_us <= 0:
        raise SystemExit(
            "expected positive CUDA-event flash forward/backward timing in train report; "
            f"cuda_runtime={cuda_runtime}"
        )
    if min(
        flash_backward_qk_mma_tiles,
        flash_backward_dp_mma_tiles,
        flash_backward_dq_mma_tiles,
        flash_backward_dk_mma_tiles,
        flash_backward_dv_mma_tiles,
    ) <= 0:
        raise SystemExit(
            "expected positive Tensor Core flash backward QK/dP/dQ/dK/dV MMA tile counts in train report; "
            f"qk={flash_backward_qk_mma_tiles} dp={flash_backward_dp_mma_tiles} "
            f"dq={flash_backward_dq_mma_tiles} dk={flash_backward_dk_mma_tiles} "
            f"dv={flash_backward_dv_mma_tiles} cuda_runtime={cuda_runtime}"
        )
    if flash_backward_scalar_tiles != 0:
        raise SystemExit(
            "expected zero scalar recompute flash backward tile count in train report; "
            f"cuda_runtime={cuda_runtime}"
        )
print(
    "cuda_train_lm_fixture loss_check "
    f"initial={initial:.6f} final={final:.6f} reduction={reduction:.6f} "
    f"tensor_core_forward_calls={forward_calls} "
    f"tensor_core_backward_calls={backward_calls} "
    f"tensor_core_matmul_calls={matmul_calls} "
    f"attention_tensor_core_forward_calls={attention_forward_calls} "
    f"attention_tensor_core_qk_calls={attention_qk_calls} "
    f"attention_tensor_core_av_calls={attention_av_calls} "
    f"attention_tensor_core_backward_calls={attention_backward_calls} "
    f"attention_tensor_core_score_grad_calls={attention_score_grad_calls} "
    f"attention_tensor_core_dq_calls={attention_dq_calls} "
    f"attention_tensor_core_dk_calls={attention_dk_calls} "
    f"attention_tensor_core_dv_calls={attention_dv_calls} "
    f"attention_padded_tiles={attention_padded_tiles} "
    f"attention_remainder_tiles={attention_remainder_tiles} "
    f"attention_scalar_fallbacks={attention_scalar_fallbacks} "
    f"flash_forward_executed={flash_forward_executed} "
    f"flash_backward_executed={flash_backward_executed} "
    f"flash_backward_qk_mma_tiles={flash_backward_qk_mma_tiles} "
    f"flash_backward_dp_mma_tiles={flash_backward_dp_mma_tiles} "
    f"flash_backward_dq_mma_tiles={flash_backward_dq_mma_tiles} "
    f"flash_backward_dk_mma_tiles={flash_backward_dk_mma_tiles} "
    f"flash_backward_dv_mma_tiles={flash_backward_dv_mma_tiles} "
    f"flash_backward_scalar_tiles={flash_backward_scalar_tiles} "
    f"linear_tensor_core_calls={linear_tensor_core_calls}",
    f"tensor_core_padded_tiles={tensor_core_padded_tiles} "
    f"tensor_core_remainder_tiles={tensor_core_remainder_tiles} "
    f"tensor_core_pad_crop_status={pad_crop_status}",
    f"staged_cta_gemm_calls={tensor_core_staged_cta_gemm_calls} "
    f"shared_stage_tiles={tensor_core_shared_stage_tiles} "
    f"shared_stage_bytes={tensor_core_shared_stage_bytes} "
    f"wide_swizzled_cta_gemm_calls={tensor_core_wide_swizzled_cta_gemm_calls} "
    f"swizzled_stage_tiles={tensor_core_swizzled_stage_tiles} "
    f"swizzled_stage_bytes={tensor_core_swizzled_stage_bytes} "
    f"ldmatrix_gemm_executed_calls={tensor_core_ldmatrix_gemm_executed_calls} "
    f"ldmatrix_gemm_instructions={tensor_core_ldmatrix_gemm_instructions} "
    f"cp_async_gemm_executed_calls={tensor_core_cp_async_gemm_executed_calls} "
    f"cp_async_gemm_instructions={tensor_core_cp_async_gemm_instructions} "
    f"global_cta_gemm_calls={tensor_core_global_cta_gemm_calls}",
    flush=True,
)
PY

run cargo run --bin heirloom -- train-lm \
  --dataset-manifest "$OUT_DIR/prepared/manifest.json" \
  --checkpoint "$OUT_DIR/checkpoint" \
  --steps "$RESUME_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --resume \
  "${TRAIN_TARGET_ARGS[@]}" \
  --precision "$PRECISION" \
  --log-every 1 \
  --report "$OUT_DIR/resume-report.json"

if [[ "$DISTRIBUTED_MODE" == "1" && -d "$OUT_DIR/ddp-ranks" ]]; then
  rm -rf "$OUT_DIR/resume-ddp-ranks"
  cp -a "$OUT_DIR/ddp-ranks" "$OUT_DIR/resume-ddp-ranks"
fi

python3 - "$OUT_DIR/train-report.json" "$OUT_DIR/resume-report.json" "$OUT_DIR/checkpoint/metadata.json" "$RESUME_STEPS" "$DEVICE" "$PRECISION" "$EXPECT_TENSOR_CORES" "$EXPECT_ATTENTION_TENSOR_CORES" "$EXPECT_FLASH_BF16_ATTENTION" "$EXPECT_TENSOR_CORE_PADDING" "$EXPECT_ATTENTION_TENSOR_CORE_PADDING" "$DEVICES" "$DISTRIBUTED" "$BATCH_SIZE" "$DDP_CHECKSUM_EVERY" <<'PY'
import json
import math
import os
import sys

(
    train_path,
    resume_path,
    metadata_path,
    resume_steps,
    expected_device,
    expected_precision,
    expect_tensor_cores,
    expect_attention_tensor_cores,
    expect_flash_bf16_attention,
    expect_tensor_core_padding,
    expect_attention_tensor_core_padding,
    expected_devices,
    expected_distributed,
    expected_batch_size,
    ddp_checksum_every,
) = sys.argv[1:16]
resume_steps = int(resume_steps)
expected_batch_size = int(expected_batch_size)
ddp_checksum_every = int(ddp_checksum_every)
expect_tensor_cores = expect_tensor_cores in {"1", "true", "TRUE", "yes", "YES"}
expect_attention_tensor_cores = expect_attention_tensor_cores in {"1", "true", "TRUE", "yes", "YES"}
expect_flash_bf16_attention = expect_flash_bf16_attention in {"1", "true", "TRUE", "yes", "YES"}
expect_tensor_core_padding = expect_tensor_core_padding in {"1", "true", "TRUE", "yes", "YES"}
expect_attention_tensor_core_padding = expect_attention_tensor_core_padding in {"1", "true", "TRUE", "yes", "YES"}
legacy_warp_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
global_cta_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
wide_swizzled_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
ldmatrix_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
cp_async_gemm = os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
with open(train_path, encoding="utf-8") as handle:
    train = json.load(handle)
with open(resume_path, encoding="utf-8") as handle:
    resume = json.load(handle)
with open(metadata_path, encoding="utf-8") as handle:
    metadata = json.load(handle)

expected_start = int(train["final_step"])
expected_final = expected_start + resume_steps
if resume["start_step"] != expected_start or resume["final_step"] != expected_final:
    raise SystemExit(f"unexpected resume step range: train={train} resume={resume}")
if metadata["step"] != expected_final:
    raise SystemExit(f"checkpoint metadata step {metadata['step']} != expected {expected_final}")
distributed = bool(expected_devices)
if distributed:
    devices = [part.strip() for part in expected_devices.split(",") if part.strip()]
    if resume.get("distributed") != expected_distributed:
        raise SystemExit(f"unexpected distributed mode in resume report: {resume.get('distributed')} expected {expected_distributed}")
    if resume.get("devices") != devices:
        raise SystemExit(f"unexpected devices in resume report: {resume.get('devices')} expected {devices}")
    if int(resume.get("world_size") or 0) != len(devices):
        raise SystemExit(f"unexpected world_size in resume report: {resume.get('world_size')} expected {len(devices)}")
    if int(resume.get("per_rank_batch_size") or 0) != expected_batch_size:
        raise SystemExit(f"unexpected per-rank batch size in resume report: {resume.get('per_rank_batch_size')}")
    if int(resume.get("global_batch_size") or 0) != expected_batch_size * len(devices):
        raise SystemExit(f"unexpected global batch size in resume report: {resume.get('global_batch_size')}")
    if int(resume.get("all_reduce_calls") or 0) <= 0 or int(resume.get("all_reduce_bytes") or 0) <= 0:
        raise SystemExit(f"missing distributed all-reduce stats in resume report: {resume}")
    tolerance = float(resume.get("parameter_checksum_tolerance") or 0.0)
    for field in [
        "parameter_checksum_sum_max_error",
        "parameter_checksum_sumsq_max_error",
        "step_checksum_sum_max_error",
        "step_checksum_sumsq_max_error",
    ]:
        value = float(resume.get(field) or 0.0)
        if value > tolerance:
            raise SystemExit(f"{field}={value:.6e} exceeds tolerance={tolerance:.6e}")
    if ddp_checksum_every > 0 and not resume.get("step_checksum_drift"):
        raise SystemExit("expected per-step checksum drift entries in distributed resume report")
    ranks = resume.get("ranks") or []
    if len(ranks) != len(devices):
        raise SystemExit(f"expected {len(devices)} rank reports, found {len(ranks)}")
    for index, rank in enumerate(ranks):
        if int(rank.get("rank")) != index:
            raise SystemExit(f"rank report order mismatch: index={index} rank={rank}")
        if int(rank.get("start_step") or -1) != expected_start or int(rank.get("final_step") or -1) != expected_final:
            raise SystemExit(f"rank {index} step range mismatch: {rank}")
        if int(rank.get("all_reduce_calls") or 0) <= 0:
            raise SystemExit(f"rank {index} has no all-reduce calls: {rank}")
        if bool(rank.get("checkpoint_saved")) != (index == 0):
            raise SystemExit(f"rank {index} checkpoint_saved mismatch: {rank.get('checkpoint_saved')}")
else:
    if resume.get("device") != expected_device:
        raise SystemExit(f"unexpected device in resume report: {resume.get('device')} expected {expected_device}")
if resume.get("precision") != expected_precision:
    raise SystemExit(f"unexpected precision in resume report: {resume.get('precision')} expected {expected_precision}")
if not (math.isfinite(float(resume["initial_loss"])) and math.isfinite(float(resume["final_loss"]))):
    raise SystemExit(f"non-finite resume loss report: {resume}")
def validate_amp_bf16_report(payload, label):
    if expected_precision != "amp-bf16":
        return
    validation = payload.get("amp_bf16_validation") or {}
    if validation.get("status") != "passed":
        raise SystemExit(f"expected passed AMP BF16 validation in {label} report: {validation}")
    if int(validation.get("unexpected_host_staging_count") or 0) != 0:
        raise SystemExit(f"unexpected CUDA host staging in {label} report: {validation}")
    if int(validation.get("finite_check_failure_count") or 0) != 0:
        raise SystemExit(f"AMP finite-check failure in {label} report: {validation}")
    if int(validation.get("finite_check_count") or 0) <= 0:
        raise SystemExit(f"missing AMP finite checks in {label} report: {validation}")
    if int(validation.get("op_decision_count") or 0) <= 0:
        raise SystemExit(f"missing AMP op decisions in {label} report: {validation}")
    decisions = payload.get("amp_bf16_op_decisions") or payload.get("amp_bf16_op_decisions_rank0") or []
    finite_checks = payload.get("amp_bf16_finite_checks") or payload.get("amp_bf16_finite_checks_rank0") or []
    if not decisions:
        raise SystemExit(f"missing AMP op-decision detail in {label} report")
    if not finite_checks:
        raise SystemExit(f"missing AMP finite-check detail in {label} report")
    for index, rank_validation in enumerate(payload.get("amp_bf16_validation_by_rank") or []):
        if (rank_validation or {}).get("status") != "passed":
            raise SystemExit(f"rank {index} AMP BF16 validation failed in {label} report: {rank_validation}")
validate_amp_bf16_report(resume, "resume")
tensor_core = resume.get("tensor_core") or {}
cuda_runtime = resume.get("cuda_runtime") or {}
pad_crop = resume.get("tensor_core_pad_crop") or {}
coverage = resume.get("tensor_core_coverage") or {}
if distributed and not coverage and resume.get("ranks"):
    coverage = resume["ranks"][0].get("tensor_core_coverage") or {}
linear_totals = coverage.get("linear_totals") or {}
matmul_calls = int(tensor_core.get("bf16_tensor_core_matmul_calls") or 0)
forward_calls = int(tensor_core.get("bf16_tensor_core_matmul_forward_calls") or 0)
backward_calls = int(tensor_core.get("bf16_tensor_core_matmul_backward_calls") or 0)
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
tensor_core_padded_tiles = int(pad_crop.get("padded_tiles") or cuda_runtime.get("tensor_core_padded_tiles") or 0)
tensor_core_remainder_tiles = int(pad_crop.get("remainder_tiles") or cuda_runtime.get("tensor_core_remainder_tiles") or 0)
tensor_core_cta_gemm_calls = int(cuda_runtime.get("tensor_core_cta_gemm_calls") or 0)
tensor_core_cta_tiles = int(cuda_runtime.get("tensor_core_cta_tiles") or 0)
tensor_core_mma_warp_tiles = int(cuda_runtime.get("tensor_core_mma_warp_tiles") or 0)
tensor_core_staged_cta_gemm_calls = int(cuda_runtime.get("tensor_core_staged_cta_gemm_calls") or 0)
tensor_core_shared_stage_tiles = int(cuda_runtime.get("tensor_core_shared_stage_tiles") or 0)
tensor_core_shared_stage_bytes = int(cuda_runtime.get("tensor_core_shared_stage_bytes") or 0)
tensor_core_wide_swizzled_cta_gemm_calls = int(cuda_runtime.get("tensor_core_wide_swizzled_cta_gemm_calls") or 0)
tensor_core_swizzled_stage_tiles = int(cuda_runtime.get("tensor_core_swizzled_stage_tiles") or 0)
tensor_core_swizzled_stage_bytes = int(cuda_runtime.get("tensor_core_swizzled_stage_bytes") or 0)
tensor_core_ldmatrix_gemm_executed_calls = int(cuda_runtime.get("tensor_core_ldmatrix_gemm_executed_calls") or 0)
tensor_core_ldmatrix_gemm_staged_fallback_calls = int(cuda_runtime.get("tensor_core_ldmatrix_gemm_staged_fallback_calls") or 0)
tensor_core_ldmatrix_gemm_hard_require_failures = int(cuda_runtime.get("tensor_core_ldmatrix_gemm_hard_require_failures") or 0)
tensor_core_ldmatrix_gemm_instructions = int(cuda_runtime.get("tensor_core_ldmatrix_gemm_instructions") or 0)
tensor_core_cp_async_gemm_executed_calls = int(cuda_runtime.get("tensor_core_cp_async_gemm_executed_calls") or 0)
tensor_core_cp_async_gemm_staged_fallback_calls = int(cuda_runtime.get("tensor_core_cp_async_gemm_staged_fallback_calls") or 0)
tensor_core_cp_async_gemm_hard_require_failures = int(cuda_runtime.get("tensor_core_cp_async_gemm_hard_require_failures") or 0)
tensor_core_cp_async_gemm_instructions = int(cuda_runtime.get("tensor_core_cp_async_gemm_instructions") or 0)
tensor_core_global_cta_gemm_calls = int(cuda_runtime.get("tensor_core_global_cta_gemm_calls") or 0)
tensor_core_legacy_warp_gemm_calls = int(cuda_runtime.get("tensor_core_legacy_warp_gemm_calls") or 0)
attention_padded_tiles = int(cuda_runtime.get("tensor_core_attention_padded_tiles") or 0)
attention_remainder_tiles = int(cuda_runtime.get("tensor_core_attention_remainder_tiles") or 0)
attention_scalar_fallbacks = int(cuda_runtime.get("tensor_core_attention_scalar_fallbacks") or 0)
flash_forward_executed = int(cuda_runtime.get("flash_bf16_tensor_core_executed_calls") or 0)
flash_forward_fallback = int(cuda_runtime.get("flash_bf16_tensor_core_fallback_calls") or 0)
flash_backward_requested = int(cuda_runtime.get("flash_bf16_tensor_core_backward_requested_calls") or 0)
flash_backward_executed = int(cuda_runtime.get("flash_bf16_tensor_core_backward_executed_calls") or 0)
flash_backward_fallback = int(cuda_runtime.get("flash_bf16_tensor_core_backward_fallback_calls") or 0)
flash_backward_qk_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls") or 0)
flash_backward_dp_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_dp_mma_tile_calls") or 0)
flash_backward_dq_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_dq_mma_tile_calls") or 0)
flash_backward_dk_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_dk_mma_tile_calls") or 0)
flash_backward_dv_mma_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_dv_mma_tile_calls") or 0)
flash_backward_scalar_tiles = int(cuda_runtime.get("flash_bf16_tensor_core_backward_scalar_tile_calls") or 0)
flash_hard_failures = int(cuda_runtime.get("flash_bf16_attention_hard_require_failures") or 0)
materialized_reference_attention = int(cuda_runtime.get("bf16_attention_materialized_reference_calls") or 0)
flash_forward_elapsed_us = int(cuda_runtime.get("flash_bf16_tensor_core_elapsed_us") or 0)
flash_backward_elapsed_us = int(cuda_runtime.get("flash_bf16_tensor_core_backward_elapsed_us") or 0)
pad_crop_status = str(pad_crop.get("status") or "legacy_runtime_counters")
if expect_tensor_cores and matmul_calls <= 0:
    raise SystemExit(
        "expected Tensor Core matmul calls in resume report but found none; "
        f"tensor_core={tensor_core} resume={resume}"
    )
if expect_tensor_cores and forward_calls <= 0:
    raise SystemExit(
        "expected Tensor Core forward matmul calls in resume report but found none; "
        f"tensor_core={tensor_core} resume={resume}"
    )
if expect_tensor_cores and backward_calls <= 0:
    raise SystemExit(
        "expected Tensor Core backward matmul calls in resume report but found none; "
        f"tensor_core={tensor_core} resume={resume}"
    )
if expect_tensor_cores and linear_tensor_core_calls <= 0:
    raise SystemExit(
        "expected Tensor Core Linear coverage in resume report but found none; "
        f"tensor_core_coverage={coverage}"
    )
if expect_tensor_cores and linear_fallback_calls != 0:
    raise SystemExit(
        "expected zero Tensor Core Linear coverage fallbacks in resume report under hard gate; "
        f"tensor_core_coverage={coverage}"
    )
if expect_tensor_cores and not legacy_warp_gemm:
    if tensor_core_cta_gemm_calls <= 0 or tensor_core_cta_tiles <= 0 or tensor_core_mma_warp_tiles <= 0:
        raise SystemExit(
            "expected CTA Tensor Core GEMM counters in resume report under hard gate; "
            f"cuda_runtime={cuda_runtime}"
        )
    if global_cta_gemm:
        if tensor_core_global_cta_gemm_calls <= 0:
            raise SystemExit(
                "expected global-fed CTA Tensor Core GEMM counter in resume report when debug env is set; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_wide_swizzled_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero wide-swizzled CTA GEMM calls when global-fed debug env is set; "
                f"cuda_runtime={cuda_runtime}"
            )
    elif wide_swizzled_gemm:
        if (
            tensor_core_wide_swizzled_cta_gemm_calls <= 0
            or tensor_core_swizzled_stage_tiles <= 0
            or tensor_core_swizzled_stage_bytes <= 0
        ):
            raise SystemExit(
                "expected wide-swizzled CTA Tensor Core GEMM counters in resume report when wide env is set; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_staged_cta_gemm_calls != 0 or tensor_core_global_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero default staged/global CTA GEMM calls in wide-swizzled resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
    elif cp_async_gemm:
        if (
            tensor_core_cp_async_gemm_executed_calls <= 0
            or tensor_core_cp_async_gemm_instructions <= 0
            or tensor_core_ldmatrix_gemm_instructions <= 0
        ):
            raise SystemExit(
                "expected double-buffered cp.async Tensor Core GEMM counters in resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if (
            tensor_core_cp_async_gemm_staged_fallback_calls != 0
            or tensor_core_cp_async_gemm_hard_require_failures != 0
        ):
            raise SystemExit(
                "expected zero cp.async GEMM fallback/hard-failure counters in resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_staged_cta_gemm_calls != 0 or tensor_core_global_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero staged/global CTA GEMM calls in cp.async resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
    elif ldmatrix_gemm:
        if (
            tensor_core_ldmatrix_gemm_executed_calls <= 0
            or tensor_core_ldmatrix_gemm_instructions <= 0
        ):
            raise SystemExit(
                "expected ldmatrix Tensor Core GEMM counters in resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if (
            tensor_core_ldmatrix_gemm_staged_fallback_calls != 0
            or tensor_core_ldmatrix_gemm_hard_require_failures != 0
        ):
            raise SystemExit(
                "expected zero ldmatrix GEMM fallback/hard-failure counters in resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_staged_cta_gemm_calls != 0 or tensor_core_global_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero staged/global CTA GEMM calls in ldmatrix resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
    else:
        if tensor_core_staged_cta_gemm_calls <= 0 or tensor_core_shared_stage_tiles <= 0 or tensor_core_shared_stage_bytes <= 0:
            raise SystemExit(
                "expected shared-memory staged CTA Tensor Core GEMM counters in resume report under hard gate; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_global_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero global-fed CTA GEMM calls in default staged resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
        if tensor_core_wide_swizzled_cta_gemm_calls != 0:
            raise SystemExit(
                "expected zero wide-swizzled CTA GEMM calls in default staged resume report; "
                f"cuda_runtime={cuda_runtime}"
            )
    if tensor_core_legacy_warp_gemm_calls != 0:
        raise SystemExit(
            "expected zero legacy one-warp Tensor Core GEMM calls in resume report; "
            f"cuda_runtime={cuda_runtime}"
        )
if expect_tensor_core_padding:
    if pad_crop:
        if not bool(pad_crop.get("used")) or pad_crop.get("status") != "passed":
            raise SystemExit(
                "expected named Tensor Core pad/crop report to pass in resume report; "
                f"tensor_core_pad_crop={pad_crop}"
            )
    if tensor_core_padded_tiles <= 0:
        raise SystemExit(
            "expected positive Tensor Core padded tile count in resume report; "
            f"tensor_core_pad_crop={pad_crop} cuda_runtime={cuda_runtime}"
        )
    if tensor_core_remainder_tiles <= 0:
        raise SystemExit(
            "expected positive Tensor Core remainder tile count in resume report; "
            f"tensor_core_pad_crop={pad_crop} cuda_runtime={cuda_runtime}"
        )
if expect_attention_tensor_cores and attention_forward_calls <= 0:
    raise SystemExit(
        "expected Tensor Core attention forward calls in resume report but found none; "
        f"tensor_core={tensor_core} resume={resume}"
    )
if expect_attention_tensor_cores and (attention_qk_calls <= 0 or attention_av_calls <= 0):
    raise SystemExit(
        "expected Tensor Core attention QK and AV matmul calls in resume report; "
        f"qk={attention_qk_calls} av={attention_av_calls} tensor_core={tensor_core}"
    )
if expect_attention_tensor_cores and attention_backward_calls <= 0:
    raise SystemExit(
        "expected Tensor Core attention backward calls in resume report but found none; "
        f"tensor_core={tensor_core} resume={resume}"
    )
if expect_attention_tensor_cores and min(
    attention_score_grad_calls,
    attention_dq_calls,
    attention_dk_calls,
    attention_dv_calls,
) <= 0:
    raise SystemExit(
        "expected Tensor Core attention backward score/dQ/dK/dV matmul calls in resume report; "
        f"score_grad={attention_score_grad_calls} dq={attention_dq_calls} "
        f"dk={attention_dk_calls} dv={attention_dv_calls} tensor_core={tensor_core}"
    )
if expect_attention_tensor_cores and attention_scalar_fallbacks != 0:
    raise SystemExit(
        "expected zero Tensor Core attention scalar fallbacks in resume report; "
        f"cuda_runtime={cuda_runtime}"
    )
if expect_attention_tensor_core_padding:
    if attention_padded_tiles <= 0:
        raise SystemExit(
            "expected positive Tensor Core attention padded tile count in resume report; "
            f"cuda_runtime={cuda_runtime}"
        )
    if attention_remainder_tiles <= 0:
        raise SystemExit(
            "expected positive Tensor Core attention remainder tile count in resume report; "
            f"cuda_runtime={cuda_runtime}"
        )
if expect_flash_bf16_attention:
    if flash_forward_executed <= 0:
        raise SystemExit(
            "expected Tensor Core flash attention forward execution in resume report; "
            f"cuda_runtime={cuda_runtime}"
        )
    if flash_backward_requested <= 0 or flash_backward_executed <= 0:
        raise SystemExit(
            "expected flash attention backward request/execution in resume report; "
            f"requested={flash_backward_requested} executed={flash_backward_executed} "
            f"cuda_runtime={cuda_runtime}"
        )
    if flash_forward_fallback != 0 or flash_backward_fallback != 0:
        raise SystemExit(
            "expected zero flash attention fallback counters in resume report; "
            f"forward_fallback={flash_forward_fallback} backward_fallback={flash_backward_fallback} "
            f"cuda_runtime={cuda_runtime}"
        )
    if flash_hard_failures != 0:
        raise SystemExit(
            "expected zero flash attention hard-require failures in resume report; "
            f"cuda_runtime={cuda_runtime}"
        )
    if materialized_reference_attention != 0:
        raise SystemExit(
            "expected zero materialized BF16 reference attention calls under flash resume gate; "
            f"cuda_runtime={cuda_runtime}"
        )
    if flash_forward_elapsed_us <= 0 or flash_backward_elapsed_us <= 0:
        raise SystemExit(
            "expected positive CUDA-event flash forward/backward timing in resume report; "
            f"cuda_runtime={cuda_runtime}"
        )
    if min(
        flash_backward_qk_mma_tiles,
        flash_backward_dp_mma_tiles,
        flash_backward_dq_mma_tiles,
        flash_backward_dk_mma_tiles,
        flash_backward_dv_mma_tiles,
    ) <= 0:
        raise SystemExit(
            "expected positive Tensor Core flash backward QK/dP/dQ/dK/dV MMA tile counts in resume report; "
            f"qk={flash_backward_qk_mma_tiles} dp={flash_backward_dp_mma_tiles} "
            f"dq={flash_backward_dq_mma_tiles} dk={flash_backward_dk_mma_tiles} "
            f"dv={flash_backward_dv_mma_tiles} cuda_runtime={cuda_runtime}"
        )
    if flash_backward_scalar_tiles != 0:
        raise SystemExit(
            "expected zero scalar recompute flash backward tile count in resume report; "
            f"cuda_runtime={cuda_runtime}"
        )
print(
    "cuda_train_lm_fixture resume_check "
    f"start_step={resume['start_step']} final_step={resume['final_step']} "
    f"tensor_core_matmul_calls={matmul_calls} "
    f"tensor_core_forward_calls={forward_calls} "
    f"tensor_core_backward_calls={backward_calls} "
    f"attention_tensor_core_forward_calls={attention_forward_calls} "
    f"attention_tensor_core_qk_calls={attention_qk_calls} "
    f"attention_tensor_core_av_calls={attention_av_calls} "
    f"attention_tensor_core_backward_calls={attention_backward_calls} "
    f"attention_tensor_core_score_grad_calls={attention_score_grad_calls} "
    f"attention_tensor_core_dq_calls={attention_dq_calls} "
    f"attention_tensor_core_dk_calls={attention_dk_calls} "
    f"attention_tensor_core_dv_calls={attention_dv_calls} "
    f"attention_padded_tiles={attention_padded_tiles} "
    f"attention_remainder_tiles={attention_remainder_tiles} "
    f"attention_scalar_fallbacks={attention_scalar_fallbacks} "
    f"flash_forward_executed={flash_forward_executed} "
    f"flash_backward_executed={flash_backward_executed} "
    f"flash_backward_qk_mma_tiles={flash_backward_qk_mma_tiles} "
    f"flash_backward_dp_mma_tiles={flash_backward_dp_mma_tiles} "
    f"flash_backward_dq_mma_tiles={flash_backward_dq_mma_tiles} "
    f"flash_backward_dk_mma_tiles={flash_backward_dk_mma_tiles} "
    f"flash_backward_dv_mma_tiles={flash_backward_dv_mma_tiles} "
    f"flash_backward_scalar_tiles={flash_backward_scalar_tiles} "
    f"linear_tensor_core_calls={linear_tensor_core_calls} "
    f"tensor_core_padded_tiles={tensor_core_padded_tiles} "
    f"tensor_core_remainder_tiles={tensor_core_remainder_tiles} "
    f"tensor_core_pad_crop_status={pad_crop_status}",
    f"staged_cta_gemm_calls={tensor_core_staged_cta_gemm_calls} "
    f"shared_stage_tiles={tensor_core_shared_stage_tiles} "
    f"shared_stage_bytes={tensor_core_shared_stage_bytes} "
    f"wide_swizzled_cta_gemm_calls={tensor_core_wide_swizzled_cta_gemm_calls} "
    f"swizzled_stage_tiles={tensor_core_swizzled_stage_tiles} "
    f"swizzled_stage_bytes={tensor_core_swizzled_stage_bytes} "
    f"ldmatrix_gemm_executed_calls={tensor_core_ldmatrix_gemm_executed_calls} "
    f"ldmatrix_gemm_instructions={tensor_core_ldmatrix_gemm_instructions} "
    f"cp_async_gemm_executed_calls={tensor_core_cp_async_gemm_executed_calls} "
    f"cp_async_gemm_instructions={tensor_core_cp_async_gemm_instructions} "
    f"global_cta_gemm_calls={tensor_core_global_cta_gemm_calls}",
    flush=True,
)
PY

python3 - "$OUT_DIR/train-report.json" "$OUT_DIR/resume-report.json" "$OUT_DIR/tensor-core-pad-crop-summary.json" "$OUT_DIR/tensor-core-pad-crop-summary.txt" <<'PY'
import json
import sys
from pathlib import Path

train_path, resume_path, summary_path, text_path = map(Path, sys.argv[1:5])

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def compact(report):
    pad_crop = report.get("tensor_core_pad_crop") or {}
    runtime = report.get("cuda_runtime") or {}
    tensor_core = report.get("tensor_core") or {}
    coverage = report.get("tensor_core_coverage") or {}
    linear_totals = coverage.get("linear_totals") or {}
    return {
        "status": pad_crop.get("status", "legacy_runtime_counters"),
        "used": bool(pad_crop.get("used")),
        "passed": bool(pad_crop.get("passed")),
        "padded_tiles": int(pad_crop.get("padded_tiles") or runtime.get("tensor_core_padded_tiles") or 0),
        "remainder_tiles": int(pad_crop.get("remainder_tiles") or runtime.get("tensor_core_remainder_tiles") or 0),
        "scalar_fallbacks": int(
            pad_crop.get("scalar_fallbacks")
            or tensor_core.get("bf16_scalar_matmul_fallback_calls")
            or 0
        ),
        "linear_fallbacks": int(pad_crop.get("linear_fallbacks") or linear_totals.get("fallback_calls") or 0),
        "linear_tensor_core_calls": int(
            pad_crop.get("linear_tensor_core_calls") or linear_totals.get("tensor_core_calls") or 0
        ),
        "tensor_core_cta_gemm_calls": int(runtime.get("tensor_core_cta_gemm_calls") or 0),
        "tensor_core_cta_tiles": int(runtime.get("tensor_core_cta_tiles") or 0),
        "tensor_core_cta_warps_launched": int(runtime.get("tensor_core_cta_warps_launched") or 0),
        "tensor_core_mma_warp_tiles": int(runtime.get("tensor_core_mma_warp_tiles") or 0),
        "tensor_core_staged_cta_gemm_calls": int(
            runtime.get("tensor_core_staged_cta_gemm_calls") or 0
        ),
        "tensor_core_shared_stage_tiles": int(runtime.get("tensor_core_shared_stage_tiles") or 0),
        "tensor_core_shared_stage_bytes": int(runtime.get("tensor_core_shared_stage_bytes") or 0),
        "tensor_core_wide_swizzled_cta_gemm_calls": int(
            runtime.get("tensor_core_wide_swizzled_cta_gemm_calls") or 0
        ),
        "tensor_core_swizzled_stage_tiles": int(runtime.get("tensor_core_swizzled_stage_tiles") or 0),
        "tensor_core_swizzled_stage_bytes": int(runtime.get("tensor_core_swizzled_stage_bytes") or 0),
        "tensor_core_global_cta_gemm_calls": int(
            runtime.get("tensor_core_global_cta_gemm_calls") or 0
        ),
        "tensor_core_legacy_warp_gemm_calls": int(
            runtime.get("tensor_core_legacy_warp_gemm_calls") or 0
        ),
        "linear_modules": pad_crop.get("linear_modules") or [],
    }

train = compact(load(train_path))
resume = compact(load(resume_path))
status = "passed" if all(
    entry["status"] in {"passed", "not_used", "legacy_runtime_counters"}
    and entry["scalar_fallbacks"] == 0
    and entry["linear_fallbacks"] == 0
    for entry in [train, resume]
) else "failed"
summary = {
    "status": status,
    "train": train,
    "resume": resume,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
text_path.write_text(
    "\n".join([
        f"tensor_core_pad_crop_summary status={status}",
        (
            "train "
            f"status={train['status']} used={train['used']} "
            f"padded_tiles={train['padded_tiles']} remainder_tiles={train['remainder_tiles']} "
            f"scalar_fallbacks={train['scalar_fallbacks']} linear_fallbacks={train['linear_fallbacks']} "
            f"cta_gemm_calls={train['tensor_core_cta_gemm_calls']} cta_tiles={train['tensor_core_cta_tiles']} "
            f"cta_warps_launched={train['tensor_core_cta_warps_launched']} "
            f"mma_warp_tiles={train['tensor_core_mma_warp_tiles']} "
            f"staged_cta_gemm_calls={train['tensor_core_staged_cta_gemm_calls']} "
            f"shared_stage_tiles={train['tensor_core_shared_stage_tiles']} "
            f"shared_stage_bytes={train['tensor_core_shared_stage_bytes']} "
            f"wide_swizzled_cta_gemm_calls={train['tensor_core_wide_swizzled_cta_gemm_calls']} "
            f"swizzled_stage_tiles={train['tensor_core_swizzled_stage_tiles']} "
            f"swizzled_stage_bytes={train['tensor_core_swizzled_stage_bytes']} "
            f"global_cta_gemm_calls={train['tensor_core_global_cta_gemm_calls']} "
            f"legacy_warp_gemm_calls={train['tensor_core_legacy_warp_gemm_calls']}"
        ),
        (
            "resume "
            f"status={resume['status']} used={resume['used']} "
            f"padded_tiles={resume['padded_tiles']} remainder_tiles={resume['remainder_tiles']} "
            f"scalar_fallbacks={resume['scalar_fallbacks']} linear_fallbacks={resume['linear_fallbacks']} "
            f"cta_gemm_calls={resume['tensor_core_cta_gemm_calls']} cta_tiles={resume['tensor_core_cta_tiles']} "
            f"cta_warps_launched={resume['tensor_core_cta_warps_launched']} "
            f"mma_warp_tiles={resume['tensor_core_mma_warp_tiles']} "
            f"staged_cta_gemm_calls={resume['tensor_core_staged_cta_gemm_calls']} "
            f"shared_stage_tiles={resume['tensor_core_shared_stage_tiles']} "
            f"shared_stage_bytes={resume['tensor_core_shared_stage_bytes']} "
            f"wide_swizzled_cta_gemm_calls={resume['tensor_core_wide_swizzled_cta_gemm_calls']} "
            f"swizzled_stage_tiles={resume['tensor_core_swizzled_stage_tiles']} "
            f"swizzled_stage_bytes={resume['tensor_core_swizzled_stage_bytes']} "
            f"global_cta_gemm_calls={resume['tensor_core_global_cta_gemm_calls']} "
            f"legacy_warp_gemm_calls={resume['tensor_core_legacy_warp_gemm_calls']}"
        ),
        "",
    ]),
    encoding="utf-8",
)
print(text_path.read_text(encoding="utf-8"), end="")
PY

if [[ "$DISTRIBUTED_MODE" == "1" ]]; then
  printf '\ncuda train-lm fixture complete: devices=%s distributed=%s precision=%s expect_tensor_cores=%s expect_attention_tensor_cores=%s expect_flash_bf16_attention=%s expect_tensor_core_padding=%s expect_attention_tensor_core_padding=%s ragged_tensor_cores=%s ragged_attention_tensor_cores=%s out_dir=%s\n' "$DEVICES" "$DISTRIBUTED" "$PRECISION" "$EXPECT_TENSOR_CORES" "$EXPECT_ATTENTION_TENSOR_CORES" "$EXPECT_FLASH_BF16_ATTENTION" "$EXPECT_TENSOR_CORE_PADDING" "$EXPECT_ATTENTION_TENSOR_CORE_PADDING" "$RAGGED_TENSOR_CORES" "$RAGGED_ATTENTION_TENSOR_CORES" "$OUT_DIR"
else
  printf '\ncuda train-lm fixture complete: device=%s precision=%s expect_tensor_cores=%s expect_attention_tensor_cores=%s expect_flash_bf16_attention=%s expect_tensor_core_padding=%s expect_attention_tensor_core_padding=%s ragged_tensor_cores=%s ragged_attention_tensor_cores=%s out_dir=%s\n' "$DEVICE" "$PRECISION" "$EXPECT_TENSOR_CORES" "$EXPECT_ATTENTION_TENSOR_CORES" "$EXPECT_FLASH_BF16_ATTENTION" "$EXPECT_TENSOR_CORE_PADDING" "$EXPECT_ATTENTION_TENSOR_CORE_PADDING" "$RAGGED_TENSOR_CORES" "$RAGGED_ATTENTION_TENSOR_CORES" "$OUT_DIR"
fi
