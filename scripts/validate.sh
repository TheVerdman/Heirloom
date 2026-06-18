#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run cargo fmt --all --check
run bash -n scripts/cuda_train_memory_lm_fixture.sh
run bash -n scripts/run_learning_sanity_ladder.sh
run bash -n scripts/run_qb_data_hardpath.sh
run zsh -n scripts/gcp/submit_vertex_heirloom_validate.sh
run python3 -m py_compile scripts/validate_memory_fixture_artifacts.py
run python3 -m py_compile scripts/validate_qb_data_hardpath_artifacts.py
run cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/ladder-valid.json
run cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/lr-grad-sweep-valid.json
run cargo run --bin heirloom -- readiness validate-learning-sanity --manifest tests/fixtures/learning_sanity/longer-32k-blend-valid.json
run cargo run --bin heirloom -- padawan validate --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures
run cargo run --bin heirloom -- padawan verify --episodes padawan/fixtures/episode_valid.jsonl --artifact-root padawan/fixtures
run cargo test --workspace
run cargo clippy --workspace --all-targets -- -D warnings
run cargo run --bin train
run cargo run --bin classify
run cargo run --bin custom
run cargo run --example microgpt_heirloom

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
printf 'the cat sat. the cat ran. the dog sat. the dog ran. ' > "$tmpdir/tiny.txt"
run cargo run --bin heirloom -- tokenizer train --input "$tmpdir/tiny.txt" --out "$tmpdir/tokenizer.json" --vocab-size 280
run cargo run --bin heirloom -- data prepare --input "$tmpdir/tiny.txt" --tokenizer "$tmpdir/tokenizer.json" --out-dir "$tmpdir/prepared" --valid-fraction 0.25
run cargo run --bin heirloom -- train-lm --dataset-manifest "$tmpdir/prepared/manifest.json" --checkpoint "$tmpdir/checkpoint" --steps 20 --batch-size 4 --block-size 6 --d-model 8 --n-heads 2 --ff-hidden 16 --lr 0.01 --log-every 10
run cargo run --bin heirloom -- train-lm --dataset-manifest "$tmpdir/prepared/manifest.json" --checkpoint "$tmpdir/checkpoint-bf16" --steps 3 --batch-size 4 --block-size 6 --d-model 8 --n-heads 2 --ff-hidden 16 --lr 0.01 --precision bf16 --log-every 1 --report "$tmpdir/train-bf16.json"
run cargo run --bin heirloom -- eval-lm --checkpoint "$tmpdir/checkpoint-bf16" --dataset-manifest "$tmpdir/prepared/manifest.json" --split valid --batch-size 2 --max-batches 1 --precision bf16 --report "$tmpdir/eval-bf16.json"
run cargo run --bin heirloom -- generate --checkpoint "$tmpdir/checkpoint-bf16" --prompt "the" --max-new-tokens 2 --precision bf16 --temperature 0.0 --report "$tmpdir/generation-bf16.json"
run cargo run --bin heirloom -- train-lm --dataset-manifest "$tmpdir/prepared/manifest.json" --checkpoint "$tmpdir/checkpoint" --steps 1 --batch-size 4 --resume --log-every 1
run cargo run --bin heirloom -- eval-lm --checkpoint "$tmpdir/checkpoint" --dataset-manifest "$tmpdir/prepared/manifest.json" --split valid --batch-size 2 --max-batches 1 --report "$tmpdir/eval.json"
run cargo run --bin heirloom -- generate --checkpoint "$tmpdir/checkpoint" --prompt "the" --max-new-tokens 8 --temperature 0.8 --top-k 8 --top-p 0.9 --repetition-penalty 1.1 --frequency-penalty 0.05 --presence-penalty 0.05 --seed 99 --report "$tmpdir/generation.json"

if [[ "${HEIRLOOM_RUN_PYTHON_PARITY:-0}" == "1" ]]; then
  run ./scripts/python_parity.sh
fi

printf '\nvalidation complete\n'
