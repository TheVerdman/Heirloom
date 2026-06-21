#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_TS="${HEIRLOOM_QB1_TOKENIZER_RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${HEIRLOOM_QB1_TOKENIZER_OUT_DIR:-$REPO_ROOT/runs/qb1-tokenizer-local-$RUN_TS}"

: "${HEIRLOOM_QB1_TOKENIZER_UPLOAD_PREFIX:=gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/tokenizers/qb1-32k-$RUN_TS}"
: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MODE:=full}"
: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_VOCAB:=32768}"
: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_SAMPLE_BYTES:=6000000000}"
: "${HEIRLOOM_QB_TOKENIZER_HARDPATH_TOKENIZER_ONLY:=1}"
: "${HEIRLOOM_TOKENIZER_TRAIN_PROGRESS_EVERY_MERGES:=1000}"
: "${HEIRLOOM_GCS_COPY_TOOL:=gcloud}"

QB1_SOURCE_ROOT="gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices-1b-rehearsal"
: "${HEIRLOOM_QB_TOKENIZER_DOLMA_PATH:=${QB1_SOURCE_ROOT}/dolma-v1_7/}"
: "${HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH:=${QB1_SOURCE_ROOT}/nemotron-cc-high-actual/nemotron-cc-high-actual-sample.jsonl}"
: "${HEIRLOOM_QB_TOKENIZER_OLMO3_PATH:=${QB1_SOURCE_ROOT}/dolma3-dolmino-mix-100b-1125/}"
: "${HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH:=${QB1_SOURCE_ROOT}/nemotron-cc-math/nemotron-cc-math-sample.jsonl}"
: "${HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH:=${QB1_SOURCE_ROOT}/vecl-qb-v1-hard/corpus.jsonl}"

export HEIRLOOM_QB_TOKENIZER_HARDPATH_MODE
export HEIRLOOM_QB_TOKENIZER_HARDPATH_VOCAB
export HEIRLOOM_QB_TOKENIZER_HARDPATH_SAMPLE_BYTES
export HEIRLOOM_QB_TOKENIZER_HARDPATH_TOKENIZER_ONLY
export HEIRLOOM_TOKENIZER_TRAIN_PROGRESS_EVERY_MERGES
export HEIRLOOM_GCS_COPY_TOOL
export HEIRLOOM_QB_TOKENIZER_DOLMA_PATH
export HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH
export HEIRLOOM_QB_TOKENIZER_OLMO3_PATH
export HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH
export HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH

mkdir -p "$OUT_DIR"

copy_to_gcs() {
  local source="$1"
  local destination="$2"
  if [[ "${HEIRLOOM_GCS_COPY_TOOL:-}" == "gcloud" && "$(command -v gcloud || true)" != "" ]]; then
    gcloud storage cp "$source" "$destination"
  elif command -v gsutil >/dev/null 2>&1; then
    gsutil cp "$source" "$destination"
  else
    gcloud storage cp "$source" "$destination"
  fi
}

upload_outputs() {
  local prefix="${HEIRLOOM_QB1_TOKENIZER_UPLOAD_PREFIX%/}"
  if [[ -z "$prefix" || "$prefix" == "__HEIRLOOM_UNSET__" ]]; then
    return
  fi
  for relative in \
    corpus-blend.json \
    tokenizer.json \
    tokenizer-train-report.json \
    tokenizer-fertility-report.json \
    summary.json \
    run.log; do
    if [[ -f "$OUT_DIR/$relative" ]]; then
      copy_to_gcs "$OUT_DIR/$relative" "$prefix/$relative"
    fi
  done
  python3 - "$OUT_DIR/upload-summary.json" "$prefix" "$RUN_TS" <<'PY'
import json
import sys

out, prefix, run_ts = sys.argv[1:]
summary = {
    "status": "uploaded",
    "run_ts": run_ts,
    "upload_prefix": prefix,
    "tokenizer_uri": f"{prefix}/tokenizer.json",
    "tokenizer_report_uri": f"{prefix}/tokenizer-train-report.json",
    "fertility_report_uri": f"{prefix}/tokenizer-fertility-report.json",
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")
PY
  copy_to_gcs "$OUT_DIR/upload-summary.json" "$prefix/upload-summary.json"
}

cd "$REPO_ROOT"
bash "$REPO_ROOT/scripts/run_qb_tokenizer_hardpath.sh" "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run.log"
upload_outputs
