#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

timestamp="$(date +%Y%m%d-%H%M%S)"
workdir="${1:-runs/learning-sanity-ladder-local-$timestamp}"
steps="${HEIRLOOM_LEARNING_SANITY_STEPS:-32}"
memory_steps="${HEIRLOOM_LEARNING_SANITY_MEMORY_STEPS:-32}"
batch_size="${HEIRLOOM_LEARNING_SANITY_BATCH:-4}"
block_size="${HEIRLOOM_LEARNING_SANITY_BLOCK:-8}"
d_model="${HEIRLOOM_LEARNING_SANITY_D_MODEL:-16}"
n_heads="${HEIRLOOM_LEARNING_SANITY_HEADS:-2}"
ff_hidden="${HEIRLOOM_LEARNING_SANITY_FF:-32}"
lr="${HEIRLOOM_LEARNING_SANITY_LR:-0.01}"
dense_min_loss_reduction="${HEIRLOOM_LEARNING_SANITY_DENSE_MIN_REDUCTION:-0.0}"
memory_disabled_min_loss_reduction="${HEIRLOOM_LEARNING_SANITY_MEMORY_DISABLED_MIN_REDUCTION:-0.0}"
memory_min_loss_reduction="${HEIRLOOM_LEARNING_SANITY_MEMORY_MIN_REDUCTION:-0.0}"
memory_smft_min_loss_reduction="${HEIRLOOM_LEARNING_SANITY_MEMORY_SMFT_MIN_REDUCTION:-0.0}"
product_key_min_loss_reduction="${HEIRLOOM_LEARNING_SANITY_PRODUCT_KEY_MIN_REDUCTION:-0.0}"

mkdir -p "$workdir"

corpus="$workdir/fixed-shard.txt"
tokenizer="$workdir/tokenizer.json"
prepared="$workdir/prepared"
dense_report="$workdir/dense-train-report.json"
memory_disabled_report="$workdir/memory-layers-disabled-report.json"
memory_report="$workdir/memory-exact-smft-disabled-report.json"
memory_smft_report="$workdir/memory-exact-smft-enabled-report.json"
memory_smft_access_counts="$workdir/memory-exact-smft-enabled-access-counts.json"
memory_smft_mask="$workdir/memory-exact-smft-enabled-mask.json"
product_key_report="$workdir/memory-product-key-report.json"
manifest="$workdir/learning-sanity-ladder.json"
validation_report="$workdir/learning-sanity-validation.json"

: > "$corpus"
for _ in {1..96}; do
  printf 'alpha beta gamma delta. alpha beta gamma delta. ' >> "$corpus"
  printf 'memory slots learn stable rows. exact lookup follows evidence. ' >> "$corpus"
  printf 'one plus one is two. two plus two is four. ' >> "$corpus"
done

run cargo run --bin heirloom -- tokenizer train \
  --input "$corpus" \
  --out "$tokenizer" \
  --vocab-size 280

run cargo run --bin heirloom -- data prepare \
  --input "$corpus" \
  --tokenizer "$tokenizer" \
  --out-dir "$prepared" \
  --format binary-shard \
  --valid-fraction 0.25 \
  --shard-tokens 512

run cargo run --bin heirloom -- train-lm \
  --dataset-manifest "$prepared/manifest.json" \
  --checkpoint "$workdir/dense-checkpoint" \
  --steps "$steps" \
  --batch-size "$batch_size" \
  --block-size "$block_size" \
  --d-model "$d_model" \
  --n-heads "$n_heads" \
  --ff-hidden "$ff_hidden" \
  --lr "$lr" \
  --log-every "$steps" \
  --report "$dense_report"

run cargo run --bin heirloom -- train-memory-lm \
  --dataset-manifest "$prepared/manifest.json" \
  --checkpoint "$workdir/memory-layers-disabled-checkpoint" \
  --steps "$memory_steps" \
  --batch-size "$batch_size" \
  --block-size "$block_size" \
  --n-layers 2 \
  --d-model "$d_model" \
  --n-heads "$n_heads" \
  --ff-hidden "$ff_hidden" \
  --disable-memory-layers \
  --memory-slots 32 \
  --memory-key-dim 8 \
  --memory-value-dim "$d_model" \
  --memory-top-k 2 \
  --memory-heads 1 \
  --memory-lookup exact \
  --memory-update-policy full \
  --smft-mode disabled \
  --lr "$lr" \
  --log-every "$memory_steps" \
  --report "$memory_disabled_report"

run cargo run --bin heirloom -- train-memory-lm \
  --dataset-manifest "$prepared/manifest.json" \
  --checkpoint "$workdir/memory-exact-smft-disabled-checkpoint" \
  --steps "$memory_steps" \
  --batch-size "$batch_size" \
  --block-size "$block_size" \
  --n-layers 2 \
  --d-model "$d_model" \
  --n-heads "$n_heads" \
  --ff-hidden "$ff_hidden" \
  --memory-layer-indices 1 \
  --memory-slots 32 \
  --memory-key-dim 8 \
  --memory-value-dim "$d_model" \
  --memory-top-k 2 \
  --memory-heads 1 \
  --memory-lookup exact \
  --memory-update-policy full \
  --smft-mode disabled \
  --lr "$lr" \
  --log-every "$memory_steps" \
  --report "$memory_report"

run cargo run --bin heirloom -- train-memory-lm \
  --dataset-manifest "$prepared/manifest.json" \
  --checkpoint "$workdir/memory-exact-smft-enabled-checkpoint" \
  --steps "$memory_steps" \
  --batch-size "$batch_size" \
  --block-size "$block_size" \
  --n-layers 2 \
  --d-model "$d_model" \
  --n-heads "$n_heads" \
  --ff-hidden "$ff_hidden" \
  --memory-layer-indices 1 \
  --memory-slots 32 \
  --memory-key-dim 8 \
  --memory-value-dim "$d_model" \
  --memory-top-k 2 \
  --memory-heads 1 \
  --memory-lookup exact \
  --memory-update-policy sparse-rows \
  --smft-mode masked-memory-rows \
  --smft-refresh-every 1 \
  --smft-trainable-fraction 0.5 \
  --smft-min-rows 2 \
  --smft-access-counts-out "$memory_smft_access_counts" \
  --smft-mask-out "$memory_smft_mask" \
  --lr "$lr" \
  --log-every "$memory_steps" \
  --report "$memory_smft_report"

run cargo run --bin heirloom -- train-memory-lm \
  --dataset-manifest "$prepared/manifest.json" \
  --checkpoint "$workdir/memory-product-key-checkpoint" \
  --steps "$memory_steps" \
  --batch-size "$batch_size" \
  --block-size "$block_size" \
  --n-layers 2 \
  --d-model "$d_model" \
  --n-heads "$n_heads" \
  --ff-hidden "$ff_hidden" \
  --memory-layer-indices 1 \
  --memory-slots 16 \
  --memory-key-dim 8 \
  --memory-value-dim "$d_model" \
  --memory-top-k 2 \
  --memory-heads 1 \
  --memory-lookup product-key \
  --memory-update-policy full \
  --smft-mode disabled \
  --lr "$lr" \
  --log-every "$memory_steps" \
  --report "$product_key_report"

cat > "$manifest" <<JSON
{
  "format": "heirloom.learning_sanity_ladder",
  "version": 0,
  "stages": [
    {
      "stage_id": "dense_fixed_shard",
      "report": "dense-train-report.json",
      "expected_command": "train-lm",
      "expected_model_family": "tiny_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "min_loss_reduction": $dense_min_loss_reduction
    },
    {
      "stage_id": "memory_layers_disabled",
      "report": "memory-layers-disabled-report.json",
      "expected_command": "train-memory-lm",
      "expected_model_family": "memory_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "expected_memory_lookup": "exact",
      "expected_smft_mode": "disabled",
      "min_loss_reduction": $memory_disabled_min_loss_reduction
    },
    {
      "stage_id": "memory_exact_smft_disabled",
      "report": "memory-exact-smft-disabled-report.json",
      "expected_command": "train-memory-lm",
      "expected_model_family": "memory_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "expected_memory_lookup": "exact",
      "expected_smft_mode": "disabled",
      "min_loss_reduction": $memory_min_loss_reduction
    },
    {
      "stage_id": "memory_exact_smft_enabled",
      "report": "memory-exact-smft-enabled-report.json",
      "expected_command": "train-memory-lm",
      "expected_model_family": "memory_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "expected_memory_lookup": "exact",
      "expected_memory_update_policy": "sparse_rows",
      "expected_smft_mode": "masked_memory_rows",
      "min_loss_reduction": $memory_smft_min_loss_reduction
    },
    {
      "stage_id": "product_key_parity",
      "report": "memory-product-key-report.json",
      "expected_command": "train-memory-lm",
      "expected_model_family": "memory_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "expected_memory_lookup": "product_key",
      "expected_memory_update_policy": "full",
      "expected_smft_mode": "disabled",
      "min_loss_reduction": $product_key_min_loss_reduction
    }
  ]
}
JSON

run cargo run --bin heirloom -- readiness validate-learning-sanity \
  --manifest "$manifest" \
  --report "$validation_report"

printf '\nlearning sanity ladder artifacts:\n'
printf '  manifest: %s\n' "$manifest"
printf '  validation: %s\n' "$validation_report"
printf '  dense report: %s\n' "$dense_report"
printf '  memory disabled report: %s\n' "$memory_disabled_report"
printf '  memory report: %s\n' "$memory_report"
printf '  memory SMFT report: %s\n' "$memory_smft_report"
printf '  memory SMFT access counts: %s\n' "$memory_smft_access_counts"
printf '  memory SMFT mask: %s\n' "$memory_smft_mask"
printf '  product-key report: %s\n' "$product_key_report"
