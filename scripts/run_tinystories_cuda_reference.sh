#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

workdir="${1:-runs/tinystories-cuda-reference}"
mode="${HEIRLOOM_TINYSTORIES_CUDA_MODE:-reference}"

case "$mode" in
  smoke)
    default_steps=40
    default_batch=2
    default_block=16
    default_d_model=16
    default_heads=4
    default_ff=64
    default_lr=0.003
    default_min_reduction=0.0
    default_eval_batches=10
    default_vocab=512
    default_max_bytes=1048576
    default_generation_tokens=80
    default_resume_steps=0
    ;;
  reference)
    default_steps=80
    default_batch=2
    default_block=16
    default_d_model=16
    default_heads=4
    default_ff=64
    default_lr=0.003
    default_min_reduction=0.01
    default_eval_batches=20
    default_vocab=512
    default_max_bytes=1048576
    default_generation_tokens=80
    default_resume_steps=0
    ;;
  full)
    default_steps=500
    default_batch=4
    default_block=64
    default_d_model=64
    default_heads=4
    default_ff=256
    default_lr=0.001
    default_min_reduction=0.20
    default_eval_batches=200
    default_vocab=1024
    default_max_bytes=""
    default_generation_tokens=120
    default_resume_steps=5
    ;;
  *)
    echo "HEIRLOOM_TINYSTORIES_CUDA_MODE must be smoke, reference, or full; got $mode" >&2
    exit 2
    ;;
esac

case "${HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES:-0}" in
  1|true|TRUE|yes|YES)
    default_block=16
    default_d_model=16
    default_heads=1
    default_ff=64
    ;;
esac

export HEIRLOOM_TINYSTORIES_DEVICE="${HEIRLOOM_TINYSTORIES_CUDA_DEVICE:-cuda:0}"
export HEIRLOOM_TINYSTORIES_DEVICES="${HEIRLOOM_TINYSTORIES_CUDA_DEVICES:-}"
export HEIRLOOM_TINYSTORIES_DISTRIBUTED="${HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED:-}"
export HEIRLOOM_TINYSTORIES_PRECISION="${HEIRLOOM_TINYSTORIES_CUDA_PRECISION:-f32}"
export HEIRLOOM_TINYSTORIES_STEPS="${HEIRLOOM_TINYSTORIES_CUDA_STEPS:-$default_steps}"
export HEIRLOOM_TINYSTORIES_BATCH="${HEIRLOOM_TINYSTORIES_CUDA_BATCH:-$default_batch}"
export HEIRLOOM_TINYSTORIES_BLOCK="${HEIRLOOM_TINYSTORIES_CUDA_BLOCK:-$default_block}"
export HEIRLOOM_TINYSTORIES_D_MODEL="${HEIRLOOM_TINYSTORIES_CUDA_D_MODEL:-$default_d_model}"
export HEIRLOOM_TINYSTORIES_HEADS="${HEIRLOOM_TINYSTORIES_CUDA_HEADS:-$default_heads}"
export HEIRLOOM_TINYSTORIES_FF="${HEIRLOOM_TINYSTORIES_CUDA_FF:-$default_ff}"
export HEIRLOOM_TINYSTORIES_LR="${HEIRLOOM_TINYSTORIES_CUDA_LR:-$default_lr}"
export HEIRLOOM_TINYSTORIES_MIN_REDUCTION="${HEIRLOOM_TINYSTORIES_CUDA_MIN_REDUCTION:-$default_min_reduction}"
export HEIRLOOM_TINYSTORIES_EVAL_BATCHES="${HEIRLOOM_TINYSTORIES_CUDA_EVAL_BATCHES:-$default_eval_batches}"
export HEIRLOOM_TINYSTORIES_VOCAB="${HEIRLOOM_TINYSTORIES_CUDA_VOCAB:-$default_vocab}"
export HEIRLOOM_TINYSTORIES_MAX_BYTES="${HEIRLOOM_TINYSTORIES_CUDA_MAX_BYTES:-$default_max_bytes}"
export HEIRLOOM_TINYSTORIES_GENERATION_TOKENS="${HEIRLOOM_TINYSTORIES_CUDA_GENERATION_TOKENS:-$default_generation_tokens}"
export HEIRLOOM_TINYSTORIES_RESUME_STEPS="${HEIRLOOM_TINYSTORIES_CUDA_RESUME_STEPS:-$default_resume_steps}"
export HEIRLOOM_TINYSTORIES_RUN_LABEL="cuda-$mode"

exec scripts/run_tinystories_valid.sh "$workdir"
