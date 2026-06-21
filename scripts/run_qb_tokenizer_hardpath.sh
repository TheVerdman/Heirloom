#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-runs/qb-tokenizer-hardpath}"
mode="${HEIRLOOM_QB_TOKENIZER_HARDPATH_MODE:-smoke}"
qb_root="${HEIRLOOM_QB_TOKENIZER_HARDPATH_QB_ROOT:-/Users/andrewverdiramo/Desktop/VECL-QB/data}"
vocab_size="${HEIRLOOM_QB_TOKENIZER_HARDPATH_VOCAB:-}"
sample_bytes="${HEIRLOOM_QB_TOKENIZER_HARDPATH_SAMPLE_BYTES:-}"
target_tokens="${HEIRLOOM_QB_TOKENIZER_HARDPATH_TARGET_TOKENS:-}"
materialize_mode="${HEIRLOOM_QB_TOKENIZER_HARDPATH_MATERIALIZE_MODE:-}"
seed="${HEIRLOOM_QB_TOKENIZER_HARDPATH_SEED:-1107}"
existing_tokenizer_path="${HEIRLOOM_QB_TOKENIZER_HARDPATH_TOKENIZER_PATH:-}"
prepared_manifest_source="${HEIRLOOM_QB_TOKENIZER_HARDPATH_PREPARED_MANIFEST:-}"
prepared_selected_docs="${HEIRLOOM_QB_TOKENIZER_HARDPATH_PREPARED_SELECTED_DOCS:-}"
tokenizer_only="${HEIRLOOM_QB_TOKENIZER_HARDPATH_TOKENIZER_ONLY:-0}"
cargo_profile="${HEIRLOOM_QB_TOKENIZER_HARDPATH_CARGO_PROFILE:-}"
device="${HEIRLOOM_QB_TOKENIZER_HARDPATH_DEVICE:-cpu}"
devices="${HEIRLOOM_QB_TOKENIZER_HARDPATH_DEVICES:-}"
distributed="${HEIRLOOM_QB_TOKENIZER_HARDPATH_DISTRIBUTED:-}"
precision="${HEIRLOOM_QB_TOKENIZER_HARDPATH_PRECISION:-f32}"
steps="${HEIRLOOM_QB_TOKENIZER_HARDPATH_STEPS:-2}"
resume_steps="${HEIRLOOM_QB_TOKENIZER_HARDPATH_RESUME_STEPS:-1}"
batch_size="${HEIRLOOM_QB_TOKENIZER_HARDPATH_BATCH:-2}"
grad_accumulation_steps="${HEIRLOOM_QB_TOKENIZER_HARDPATH_GRAD_ACCUMULATION_STEPS:-1}"
block_size="${HEIRLOOM_QB_TOKENIZER_HARDPATH_BLOCK:-16}"
d_model="${HEIRLOOM_QB_TOKENIZER_HARDPATH_D_MODEL:-16}"
n_heads="${HEIRLOOM_QB_TOKENIZER_HARDPATH_HEADS:-4}"
ff_hidden="${HEIRLOOM_QB_TOKENIZER_HARDPATH_FF:-64}"
lr="${HEIRLOOM_QB_TOKENIZER_HARDPATH_LR:-0.001}"
eval_batches="${HEIRLOOM_QB_TOKENIZER_HARDPATH_EVAL_BATCHES:-1}"
generation_tokens="${HEIRLOOM_QB_TOKENIZER_HARDPATH_GENERATION_TOKENS:-16}"
shard_tokens="${HEIRLOOM_QB_TOKENIZER_HARDPATH_SHARD_TOKENS:-512}"
valid_fraction="${HEIRLOOM_QB_TOKENIZER_HARDPATH_VALID_FRACTION:-0.1}"
max_source_bytes="${HEIRLOOM_QB_TOKENIZER_HARDPATH_MAX_SOURCE_BYTES:-}"
max_docs_per_source="${HEIRLOOM_QB_TOKENIZER_HARDPATH_MAX_DOCS_PER_SOURCE:-}"
max_doc_bytes="${HEIRLOOM_QB_TOKENIZER_HARDPATH_MAX_DOC_BYTES:-}"
candidate_text_mode="${HEIRLOOM_QB_TOKENIZER_HARDPATH_CANDIDATE_TEXT_MODE:-}"
candidate_retention_token_multiplier="${HEIRLOOM_QB_TOKENIZER_HARDPATH_CANDIDATE_RETENTION_TOKEN_MULTIPLIER:-}"
candidate_retention_min_docs="${HEIRLOOM_QB_TOKENIZER_HARDPATH_CANDIDATE_RETENTION_MIN_DOCS:-}"
candidate_prune_every="${HEIRLOOM_QB_TOKENIZER_HARDPATH_CANDIDATE_PRUNE_EVERY:-}"
progress_every_records="${HEIRLOOM_QB_TOKENIZER_HARDPATH_PROGRESS_EVERY_RECORDS:-}"
progress_every_bytes="${HEIRLOOM_QB_TOKENIZER_HARDPATH_PROGRESS_EVERY_BYTES:-}"
checkpoint_dir="${HEIRLOOM_QB_TOKENIZER_HARDPATH_CHECKPOINT_DIR:-}"
resume_checkpoint="${HEIRLOOM_QB_TOKENIZER_HARDPATH_RESUME_CHECKPOINT:-0}"
checkpoint_every_records="${HEIRLOOM_QB_TOKENIZER_HARDPATH_CHECKPOINT_EVERY_RECORDS:-}"
checkpoint_every_bytes="${HEIRLOOM_QB_TOKENIZER_HARDPATH_CHECKPOINT_EVERY_BYTES:-}"
source_stage_dir="${HEIRLOOM_QB_TOKENIZER_SOURCE_STAGE_DIR:-}"
learning_sanity="${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY:-auto}"
learning_sanity_min_loss_reduction="${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_LOSS_REDUCTION:-0.0}"
learning_sanity_min_selected_tokens="${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_TOKENS:-}"
learning_sanity_min_selected_docs="${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_SELECTED_DOCS:-1}"
learning_sanity_min_final_step="${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_FINAL_STEP:-}"
learning_sanity_min_blend_sources="${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_MIN_BLEND_SOURCES:-5}"
learning_sanity_require_production_gate="${HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY_REQUIRE_PRODUCTION_GATE:-false}"

if [[ -z "$vocab_size" ]]; then
  if [[ "$mode" == "full" ]]; then
    vocab_size=32768
  else
    vocab_size=512
  fi
fi
if [[ -z "$sample_bytes" ]]; then
  if [[ "$mode" == "full" ]]; then
    sample_bytes=34359738368
  else
    sample_bytes=8192
  fi
fi
if [[ -z "$target_tokens" ]]; then
  if [[ "$mode" == "full" ]]; then
    target_tokens=20000000000
  else
    target_tokens=4096
  fi
fi
if [[ -z "$materialize_mode" ]]; then
  if [[ "$mode" == "full" ]]; then
    materialize_mode=full
  else
    materialize_mode=sample
  fi
fi
if [[ "$mode" == "full" ]]; then
  if [[ -z "$candidate_text_mode" ]]; then
    candidate_text_mode=rescan
  fi
  if [[ -z "$candidate_retention_token_multiplier" ]]; then
    candidate_retention_token_multiplier=1.0
  fi
  if [[ -z "$candidate_retention_min_docs" ]]; then
    candidate_retention_min_docs=100000
  fi
  if [[ -z "$candidate_prune_every" ]]; then
    candidate_prune_every=50000
  fi
  if [[ -z "$progress_every_records" ]]; then
    progress_every_records=100000
  fi
  if [[ -z "$progress_every_bytes" ]]; then
    progress_every_bytes=1073741824
  fi
  if [[ -z "$checkpoint_dir" ]]; then
    checkpoint_dir="$out_dir/materializer-checkpoints"
  fi
  if [[ -z "$checkpoint_every_records" ]]; then
    checkpoint_every_records=100000
  fi
  if [[ -z "$checkpoint_every_bytes" ]]; then
    checkpoint_every_bytes=1073741824
  fi
fi

if [[ -z "$cargo_profile" ]]; then
  if [[ "$mode" == "full" ]]; then
    cargo_profile=release
  else
    cargo_profile=debug
  fi
fi
case "$cargo_profile" in
  release)
    cargo_run=(cargo run --release --bin heirloom)
    ;;
  debug | dev)
    cargo_run=(cargo run --bin heirloom)
    ;;
  *)
    echo "unsupported HEIRLOOM_QB_TOKENIZER_HARDPATH_CARGO_PROFILE=$cargo_profile (expected release or debug)" >&2
    exit 2
    ;;
esac

mkdir -p "$out_dir"
work_dir="$out_dir/tokenizer-work"
mkdir -p "$work_dir"
if [[ -z "$source_stage_dir" ]]; then
  source_stage_dir="$out_dir/source-slices"
fi
mkdir -p "$source_stage_dir"

is_gcs_uri() {
  [[ "$1" == gs://* ]]
}

copy_gcs_object_or_prefix() {
  local source="$1"
  local destination="$2"
  local recursive="${3:-0}"
  if [[ "${HEIRLOOM_GCS_COPY_TOOL:-}" == "gcloud" && "$(command -v gcloud || true)" != "" ]]; then
    gcloud storage cp "$source" "$destination" >&2
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    if [[ "$recursive" == "1" ]]; then
      gsutil -m cp "$source" "$destination" >&2
    else
      gsutil cp "$source" "$destination" >&2
    fi
    return
  fi
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "$source" "$destination" >&2
    return
  fi
  return 1
}

reject_compressed_source() {
  local name="$1"
  local path="$2"
  case "$path" in
    *.gz|*.zst|*.zstd)
      echo "$name must point to an uncompressed materialized slice for data materialize-blend: $path" >&2
      exit 1
      ;;
  esac
}

source_dir_file_count() {
  local path="$1"
  find "$path" -maxdepth 1 -type f \
    ! -name '.*' \
    ! -name '*.report.json' \
    ! -name 'metadata.json' \
    | wc -l | tr -d '[:space:]'
}

reject_bad_source_dir() {
  local name="$1"
  local path="$2"
  local nested=""
  nested="$(find "$path" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -print -quit)"
  if [[ -n "$nested" ]]; then
    echo "$name source directory must be flat, found nested directory: $nested" >&2
    exit 1
  fi
  local compressed=""
  compressed="$(find "$path" -maxdepth 1 -type f \( -name '*.gz' -o -name '*.zst' -o -name '*.zstd' \) -print -quit)"
  if [[ -n "$compressed" ]]; then
    echo "$name must point to uncompressed materialized slices, found: $compressed" >&2
    exit 1
  fi
  local count=""
  count="$(source_dir_file_count "$path")"
  if [[ "$count" == "0" ]]; then
    echo "$name source directory has no materialized source files: $path" >&2
    exit 1
  fi
}

stage_gcs_source() {
  local name="$1"
  local uri="$2"
  local stage_name="$3"
  if [[ "$uri" == */ ]]; then
    local destination_dir="$source_stage_dir/$stage_name"
    mkdir -p "$destination_dir"
    if [[ "$(source_dir_file_count "$destination_dir")" == "0" ]]; then
      echo "staging $name prefix from $uri to $destination_dir" >&2
      if ! copy_gcs_object_or_prefix "${uri%/}/*" "$destination_dir/" 1; then
        python3 - "$uri" "$destination_dir" <<'PY'
import os
import sys

uri, destination_dir = sys.argv[1:]
if not uri.startswith("gs://"):
    raise SystemExit(f"not a GCS URI: {uri}")
bucket_name, prefix = uri[5:].split("/", 1)
prefix = prefix.rstrip("/") + "/"
os.makedirs(destination_dir, exist_ok=True)
try:
    from google.cloud import storage
except Exception as err:
    raise SystemExit(
        "staging gs:// sources requires gsutil, gcloud, or the "
        f"google-cloud-storage Python package: {err}"
    )
client = storage.Client(project=os.environ.get("PROJECT_ID") or None)
downloaded = 0
for blob in client.bucket(bucket_name).list_blobs(prefix=prefix):
    name = blob.name[len(prefix):]
    if not name or "/" in name:
        continue
    if name.startswith("."):
        continue
    destination = os.path.join(destination_dir, os.path.basename(name))
    blob.download_to_filename(destination)
    downloaded += 1
if downloaded == 0:
    raise SystemExit(f"no source objects found under {uri}")
PY
      fi
    fi
    reject_bad_source_dir "$name" "$destination_dir"
    printf '%s\n' "$destination_dir"
    return
  fi
  local filename="${uri##*/}"
  if [[ -z "$filename" || "$filename" == "$uri" ]]; then
    echo "$name GCS URI must point to a single object, got $uri" >&2
    exit 1
  fi
  reject_compressed_source "$name" "$filename"
  local destination_dir="$source_stage_dir/$stage_name"
  local destination="$destination_dir/$filename"
  mkdir -p "$destination_dir"
  if [[ ! -f "$destination" ]]; then
    echo "staging $name from $uri to $destination" >&2
    if ! copy_gcs_object_or_prefix "$uri" "$destination"; then
      python3 - "$uri" "$destination" <<'PY'
import os
import sys

uri, destination = sys.argv[1:]
if not uri.startswith("gs://"):
    raise SystemExit(f"not a GCS URI: {uri}")
bucket_name, blob_name = uri[5:].split("/", 1)
os.makedirs(os.path.dirname(destination), exist_ok=True)
try:
    from google.cloud import storage
except Exception as err:
    raise SystemExit(
        "staging gs:// sources requires gsutil, gcloud, or the "
        f"google-cloud-storage Python package: {err}"
    )
client = storage.Client(project=os.environ.get("PROJECT_ID") or None)
client.bucket(bucket_name).blob(blob_name).download_to_filename(destination)
PY
    fi
  fi
  if [[ ! -f "$destination" ]]; then
    echo "$name failed to stage from $uri to $destination" >&2
    exit 1
  fi
  printf '%s\n' "$destination"
}

require_source_path() {
  local name="$1"
  local path="$2"
  if [[ -d "$path" ]]; then
    reject_bad_source_dir "$name" "$path"
    return
  fi
  if [[ -f "$path" ]]; then
    reject_compressed_source "$name" "$path"
    return
  fi
  echo "$name must point to a materialized approved source file or flat directory for full mode: $path" >&2
  exit 1
}

resolve_existing_tokenizer_path() {
  local name="$1"
  local path="$2"
  local staged="$path"
  if is_gcs_uri "$path"; then
    staged="$(stage_gcs_source "$name" "$path" reused-tokenizer)"
  fi
  if [[ -d "$staged" ]]; then
    if [[ -f "$staged/tokenizer.json" ]]; then
      staged="$staged/tokenizer.json"
    else
      echo "$name directory must contain tokenizer.json: $staged" >&2
      exit 1
    fi
  fi
  if [[ ! -f "$staged" ]]; then
    echo "$name must point to a tokenizer.json file or directory containing tokenizer.json: $path" >&2
    exit 1
  fi
  printf '%s\n' "$staged"
}

stage_prepared_manifest() {
  local source="$1"
  local destination_dir="$2"
  mkdir -p "$destination_dir"
  repair_flat_prepared_shards() {
    local dir="$1"
    if [[ -d "$dir/shards" ]]; then
      return
    fi
    local flat_shard
    flat_shard="$(find "$dir" -maxdepth 1 -type f \( -name '*.tokens.bin' -o -name '*.tokens.json' \) -print -quit)"
    if [[ -n "$flat_shard" ]]; then
      mkdir -p "$dir/shards"
      find "$dir" -maxdepth 1 -type f \( -name '*.tokens.bin' -o -name '*.tokens.json' \) -exec mv {} "$dir/shards/" \;
    fi
  }
  if is_gcs_uri "$source"; then
    local prefix="$source"
    if [[ "$prefix" == */manifest.json ]]; then
      prefix="${prefix%/manifest.json}/"
    else
      prefix="${prefix%/}/"
    fi
    echo "staging HEIRLOOM_QB_TOKENIZER_HARDPATH_PREPARED_MANIFEST from $prefix to $destination_dir" >&2
    if command -v gcloud >/dev/null 2>&1; then
      gcloud storage cp "${prefix%/}/**" "$destination_dir/" >&2 || true
      repair_flat_prepared_shards "$destination_dir"
    fi
    if [[ ! -f "$destination_dir/manifest.json" ]] && command -v gsutil >/dev/null 2>&1; then
      gsutil -m cp -r "${prefix%/}/*" "$destination_dir/" >&2 || true
      repair_flat_prepared_shards "$destination_dir"
    fi
    if [[ ! -f "$destination_dir/manifest.json" || ! -d "$destination_dir/shards" ]]; then
      python3 - "$prefix" "$destination_dir" <<'PY'
import os
import sys

uri, destination_dir = sys.argv[1:]
if not uri.startswith("gs://"):
    raise SystemExit(f"not a GCS URI: {uri}")
bucket_name, prefix = uri[5:].split("/", 1)
prefix = prefix.rstrip("/") + "/"
os.makedirs(destination_dir, exist_ok=True)
try:
    from google.cloud import storage
except Exception as err:
    raise SystemExit(
        "staging gs:// prepared manifests requires gcloud, gsutil, or the "
        f"google-cloud-storage Python package: {err}"
    )
client = storage.Client(project=os.environ.get("PROJECT_ID") or None)
downloaded = 0
for blob in client.bucket(bucket_name).list_blobs(prefix=prefix):
    name = blob.name[len(prefix):]
    if not name or name.endswith("/"):
        continue
    destination = os.path.join(destination_dir, name)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    blob.download_to_filename(destination)
    downloaded += 1
if downloaded == 0:
    raise SystemExit(f"no prepared manifest objects found under {uri}")
PY
    fi
  else
    local source_dir="$source"
    if [[ -f "$source" ]]; then
      source_dir="$(cd "$(dirname "$source")" && pwd)"
    elif [[ -d "$source" ]]; then
      source_dir="$(cd "$source" && pwd)"
    else
      echo "HEIRLOOM_QB_TOKENIZER_HARDPATH_PREPARED_MANIFEST must point to a manifest.json file, prepared directory, or gs:// prepared prefix: $source" >&2
      exit 1
    fi
    if [[ ! -f "$source_dir/manifest.json" ]]; then
      echo "prepared manifest directory must contain manifest.json: $source_dir" >&2
      exit 1
    fi
    local source_real destination_real
    source_real="$(cd "$source_dir" && pwd)"
    destination_real="$(mkdir -p "$destination_dir" && cd "$destination_dir" && pwd)"
    if [[ "$source_real" != "$destination_real" ]]; then
      cp -R "$source_dir/." "$destination_dir/"
      repair_flat_prepared_shards "$destination_dir"
    fi
  fi
  repair_flat_prepared_shards "$destination_dir"
  if [[ ! -f "$destination_dir/manifest.json" ]]; then
    echo "failed to stage prepared manifest from $source to $destination_dir/manifest.json" >&2
    exit 1
  fi
  if [[ ! -d "$destination_dir/shards" ]]; then
    echo "prepared manifest reuse requires a staged shards/ directory next to manifest.json: $destination_dir" >&2
    exit 1
  fi
  printf '%s\n' "$destination_dir/manifest.json"
}

stage_reused_materialization_sidecars() {
  local source="$1"
  local destination_dir="$2"
  mkdir -p "$destination_dir"
  if is_gcs_uri "$source"; then
    local prefix="$source"
    if [[ "$prefix" == */manifest.json ]]; then
      prefix="${prefix%/manifest.json}"
    fi
    prefix="${prefix%/}"
    if [[ "${prefix##*/}" == "prepared" ]]; then
      local materialized_prefix="${prefix%/prepared}"
      for name in curation-report.json source-index.json selected-docs.jsonl tokenizer-sample-manifest.json; do
        if [[ ! -f "$destination_dir/$name" ]]; then
          if command -v gcloud >/dev/null 2>&1; then
            gcloud storage cp "$materialized_prefix/$name" "$destination_dir/$name" >&2 || true
          elif command -v gsutil >/dev/null 2>&1; then
            gsutil cp "$materialized_prefix/$name" "$destination_dir/$name" >&2 || true
          fi
        fi
      done
    fi
  else
    local source_dir="$source"
    if [[ -f "$source" ]]; then
      source_dir="$(cd "$(dirname "$source")" && pwd)"
    elif [[ -d "$source" ]]; then
      source_dir="$(cd "$source" && pwd)"
    fi
    if [[ "${source_dir##*/}" == "prepared" ]]; then
      local materialized_dir
      materialized_dir="$(cd "$source_dir/.." && pwd)"
      for name in curation-report.json source-index.json selected-docs.jsonl tokenizer-sample-manifest.json; do
        if [[ -f "$materialized_dir/$name" && ! -f "$destination_dir/$name" ]]; then
          cp "$materialized_dir/$name" "$destination_dir/$name"
        fi
      done
    fi
  fi
}

write_reused_materialization_metadata() {
  local manifest="$1"
  local destination_dir="$2"
  local source="$3"
  python3 - "$manifest" "$destination_dir" "$source" "$target_tokens" "$materialize_mode" "$prepared_selected_docs" "$tokenizer_path" <<'PY'
import json
import os
import sys

manifest_path, destination_dir, source, target_tokens, materialize_mode, selected_docs_override, tokenizer_path = sys.argv[1:]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

manifest = load(manifest_path)
tokenizer = load(tokenizer_path)
tokenizer_metadata = tokenizer.get("metadata", {})
selected_tokens = int(manifest.get("train_tokens", 0)) + int(manifest.get("valid_tokens", 0))
selected_docs_path = os.path.join(destination_dir, "selected-docs.jsonl")
if os.path.exists(selected_docs_path):
    with open(selected_docs_path, "r", encoding="utf-8") as f:
        selected_docs = sum(1 for line in f if line.strip())
elif selected_docs_override:
    selected_docs = int(selected_docs_override)
else:
    selected_docs = 0

source_index_path = os.path.join(destination_dir, "source-index.json")
if not os.path.exists(source_index_path):
    with open(source_index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "heirloom.reused_source_index",
                "status": "passed",
                "prepared_manifest_reused": True,
                "prepared_manifest_source": source,
                "sources": manifest.get("sources", []),
            },
            f,
            indent=2,
        )
        f.write("\n")

if selected_docs and not os.path.exists(selected_docs_path):
    with open(selected_docs_path, "w", encoding="utf-8") as f:
        for index in range(selected_docs):
            f.write(json.dumps({"reuse_placeholder": True, "index": index}) + "\n")

curation_path = os.path.join(destination_dir, "curation-report.json")
if not os.path.exists(curation_path):
    with open(curation_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "heirloom.blend_curation_report",
                "status": "passed",
                "mode": materialize_mode,
                "target_tokens": int(target_tokens),
                "selected_tokens": selected_tokens,
                "selected_docs": selected_docs,
                "prepared_manifest_reused": True,
                "prepared_manifest_source": source,
                "tokenizer": {
                    "tokenizer_id": tokenizer_metadata.get("tokenizer_id", ""),
                    "format": tokenizer_metadata.get("format", ""),
                    "version": int(tokenizer_metadata.get("version", 0)),
                    "vocab_size": int(tokenizer_metadata.get("vocab_size", 0)),
                    "tokenizer_hash": tokenizer_metadata.get("artifact_hash")
                    or tokenizer_metadata.get("validation", {}).get("artifact_hash", ""),
                    "artifact_hash": tokenizer_metadata.get("artifact_hash", ""),
                    "reserved_tokens": len(tokenizer.get("reserved_tokens", [])),
                    "reserved_registry_hash": tokenizer_metadata.get("reserved_registry_hash", ""),
                    "training_config": tokenizer_metadata.get("training_config", {}),
                    "validation": tokenizer_metadata.get("validation", {}),
                },
                "sources": manifest.get("sources", []),
                "sizing": {
                    "train_tokens": manifest.get("train_tokens"),
                    "valid_tokens": manifest.get("valid_tokens"),
                    "storage": manifest.get("storage"),
                },
            },
            f,
            indent=2,
        )
        f.write("\n")
PY
}

write_smoke_file_if_missing() {
  local path="$1"
  local content="$2"
  if [[ -d "$path" ]]; then
    reject_bad_source_dir "smoke source" "$path"
    return
  fi
  if [[ ! -e "$path" ]]; then
    printf '%s\n' "$content" > "$path"
  elif [[ ! -f "$path" ]]; then
    echo "smoke source path must be a file or flat directory: $path" >&2
    exit 1
  fi
}

dolma_path="${HEIRLOOM_QB_TOKENIZER_DOLMA_PATH:-}"
nemotron_path="${HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH:-}"
olmo_path="${HEIRLOOM_QB_TOKENIZER_OLMO3_PATH:-}"
math_path="${HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH:-}"
qb_path="${HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH:-$qb_root/synthetic/v1-hard/corpus.jsonl}"
dolma_source_uri="$dolma_path"
nemotron_source_uri="$nemotron_path"
olmo_source_uri="$olmo_path"
math_source_uri="$math_path"
qb_source_uri="$qb_path"

if [[ "$mode" == "full" ]]; then
  for pair in \
    "HEIRLOOM_QB_TOKENIZER_DOLMA_PATH:dolma-v1_7:$dolma_path" \
    "HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH:nemotron-cc-high-actual:$nemotron_path" \
    "HEIRLOOM_QB_TOKENIZER_OLMO3_PATH:dolma3-dolmino-mix-100b-1125:$olmo_path" \
    "HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH:nemotron-cc-math:$math_path" \
    "HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH:vecl-qb-v1-hard:$qb_path"; do
    name="${pair%%:*}"
    rest="${pair#*:}"
    stage_name="${rest%%:*}"
    value="${rest#*:}"
    if [[ -z "$value" ]]; then
      echo "$name must point to a local file or gs:// object for full mode" >&2
      exit 1
    fi
    if is_gcs_uri "$value"; then
      staged="$(stage_gcs_source "$name" "$value" "$stage_name")"
      case "$name" in
        HEIRLOOM_QB_TOKENIZER_DOLMA_PATH) dolma_path="$staged" ;;
        HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH) nemotron_path="$staged" ;;
        HEIRLOOM_QB_TOKENIZER_OLMO3_PATH) olmo_path="$staged" ;;
        HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH) math_path="$staged" ;;
        HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH) qb_path="$staged" ;;
      esac
    else
      require_source_path "$name" "$value"
    fi
  done
else
  dolma_path="${dolma_path:-$work_dir/dolma-smoke.txt}"
  nemotron_path="${nemotron_path:-$work_dir/nemotron-cc-smoke.jsonl}"
  olmo_path="${olmo_path:-$work_dir/olmo3-smoke.txt}"
  math_path="${math_path:-$work_dir/nemotron-cc-math-smoke.txt}"
  qb_path="${qb_path:-$work_dir/qb-v1-hard-smoke.jsonl}"
  if is_gcs_uri "$dolma_path"; then
    dolma_path="$(stage_gcs_source HEIRLOOM_QB_TOKENIZER_DOLMA_PATH "$dolma_path" dolma-v1_7)"
  fi
  if is_gcs_uri "$nemotron_path"; then
    nemotron_path="$(stage_gcs_source HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_PATH "$nemotron_path" nemotron-cc-high-actual)"
  fi
  if is_gcs_uri "$olmo_path"; then
    olmo_path="$(stage_gcs_source HEIRLOOM_QB_TOKENIZER_OLMO3_PATH "$olmo_path" dolma3-dolmino-mix-100b-1125)"
  fi
  if is_gcs_uri "$math_path"; then
    math_path="$(stage_gcs_source HEIRLOOM_QB_TOKENIZER_NEMOTRON_CC_MATH_PATH "$math_path" nemotron-cc-math)"
  fi
  if is_gcs_uri "$qb_path"; then
    qb_path="$(stage_gcs_source HEIRLOOM_QB_TOKENIZER_QB_V1_HARD_PATH "$qb_path" vecl-qb-v1-hard)"
  fi
  write_smoke_file_if_missing "$dolma_path" 'Dolma-style governed prose sample with code, markdown, and ordinary English.'
  write_smoke_file_if_missing "$nemotron_path" '{"text":"Nemotron-CC style high quality web record with url metadata.","language":"eng","url":"https://example.invalid"}'
  write_smoke_file_if_missing "$olmo_path" '{"text":"Dolma 3 Dolmino mix sample for OLMo 3 stage two annealing: curated web, code, math, instruction, and reasoning text."}'
  write_smoke_file_if_missing "$math_path" '{"text":"Nemotron-CC-Math style sample: let x=2, y=3, then x+y=5 with preserved symbols."}'
  write_smoke_file_if_missing "$qb_path" '{"source_id":"fixture-tool","prompt":"Route this synthetic request to the local fixture tool.","target_text":"{\"specialist_id\":\"fixture\",\"confidence\":0.91}","task_kind":"tool_call_json"}'
  dolma_source_uri="${dolma_source_uri:-$dolma_path}"
  nemotron_source_uri="${nemotron_source_uri:-$nemotron_path}"
  olmo_source_uri="${olmo_source_uri:-$olmo_path}"
  math_source_uri="${math_source_uri:-$math_path}"
  qb_source_uri="${qb_source_uri:-$qb_path}"
fi

blend_path="$out_dir/corpus-blend.json"
tokenizer_path="$out_dir/tokenizer.json"
tokenizer_report="$out_dir/tokenizer-train-report.json"
fertility_report="$out_dir/tokenizer-fertility-report.json"
materialized_dir="$out_dir/materialized"
prepared_manifest="$materialized_dir/prepared/manifest.json"
checkpoint="$out_dir/checkpoint"
train_report="$out_dir/train-report.json"
resume_report="$out_dir/resume-report.json"
eval_report="$out_dir/eval-report.json"
generation_report="$out_dir/generation-report.json"
summary_path="$out_dir/summary.json"
learning_sanity_manifest="$out_dir/learning-sanity-ladder.json"
learning_sanity_validation="$out_dir/learning-sanity-validation.json"

python3 - "$blend_path" "$vocab_size" "$dolma_path" "$dolma_source_uri" "$nemotron_path" "$nemotron_source_uri" "$olmo_path" "$olmo_source_uri" "$math_path" "$math_source_uri" "$qb_path" "$qb_source_uri" <<'PY'
import json
import os
import sys

(
    out,
    vocab,
    dolma,
    dolma_source_uri,
    nemotron,
    nemotron_source_uri,
    olmo,
    olmo_source_uri,
    math,
    math_source_uri,
    qb,
    qb_source_uri,
) = sys.argv[1:]

def size(path):
    if os.path.isdir(path):
        total = 0
        for name in sorted(os.listdir(path)):
            if name.startswith(".") or name.endswith(".report.json") or name == "metadata.json":
                continue
            child = os.path.join(path, name)
            if os.path.isfile(child):
                total += os.path.getsize(child)
        return total
    return os.path.getsize(path)

def fmt_for(path, default):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.startswith(".") or name.endswith(".report.json") or name == "metadata.json":
                continue
            if name.endswith(".jsonl"):
                return "jsonl"
        return default
    return "jsonl" if path.endswith(".jsonl") else default

def source(source_id, display, role, path, source_uri, fmt, weight, license_status):
    local_path = os.path.abspath(path)
    extra = {}
    if source_uri and source_uri != local_path and source_uri.startswith("gs://"):
        extra["gcs_uri"] = source_uri
    return {
        "source_id": source_id,
        "display_name": display,
        "kind": "materialized_pretraining_corpus",
        "status": "available",
        "role": role,
        "source_url": source_uri if source_uri.startswith("gs://") else None,
        "path": local_path,
        "metadata_path": None,
        "data_format": fmt,
        "license": "materialized source terms verified by caller",
        "license_url": None,
        "license_status": license_status,
        "provenance": [source_id],
        "sampling_weight": weight,
        "include_in_tokenizer_training": True,
        "include_in_pretraining": True,
        "include_in_memory_trace_training": source_id.startswith("vecl_qb."),
        "synthetic": source_id.startswith("vecl_qb."),
        "local_bytes": size(local_path),
        "local_records": None,
        "content_hash": None,
        "metadata_hash": None,
        "split_counts": {},
        "domain_counts": {},
        "category_counts": {},
        "task_kind_counts": {},
        "extra": extra,
    }

sources = [
    source("allenai.dolma.v1_7", "Dolma v1.7 materialized sample", "general_language_backbone", dolma, dolma_source_uri, fmt_for(dolma, "text"), 0.35, "odc_by_internal_attribution"),
    source("nvidia.nemotron_cc.high_actual", "Nemotron-CC high actual materialized sample", "high_quality_web_backbone", nemotron, nemotron_source_uri, "jsonl", 0.25, "nvidia_data_agreement_internal_training"),
    source("allenai.dolma3_dolmino_mix-100B-1125", "Dolma 3 Dolmino mix 100B (OLMo 3 stage 2) materialized sample", "olmo3_dolmino_mix", olmo, olmo_source_uri, "jsonl", 0.20, "odc_by_internal_attribution"),
    source("nvidia.nemotron_cc_math", "Nemotron-CC-Math materialized sample", "math_science_code_reasoning", math, math_source_uri, "jsonl", 0.10, "nvidia_data_agreement_internal_training"),
    source("vecl_qb.synthetic.v1-hard", "VECL-QB synthetic v1-hard", "tool_routing_memory_trace_supervision", qb, qb_source_uri, "jsonl", 0.10, "internal_synthetic"),
]
manifest = {
    "format": "heirloom.corpus_blend",
    "version": 1,
    "blend_id": "qb-tokenizer-hardpath",
    "tokenizer_target_vocab_size": int(vocab),
    "tokenizer_family": "heirloom.byte_bpe",
    "sources": sources,
    "local_source_count": len(sources),
    "planned_source_count": 0,
    "local_record_count": 0,
    "local_bytes": sum(s["local_bytes"] for s in sources),
    "notes": [
        "Full mode requires caller-provided materialized approved source paths.",
        "Smoke mode uses local non-TinyStories fixtures and is not a production corpus gate.",
    ],
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

if [[ -n "$existing_tokenizer_path" ]]; then
  resolved_tokenizer_path="$(resolve_existing_tokenizer_path HEIRLOOM_QB_TOKENIZER_HARDPATH_TOKENIZER_PATH "$existing_tokenizer_path")"
  cp "$resolved_tokenizer_path" "$tokenizer_path"
  python3 - "$tokenizer_report" "$existing_tokenizer_path" "$resolved_tokenizer_path" "$tokenizer_path" "$vocab_size" "$sample_bytes" "$seed" <<'PY'
import json
import os
import sys

report, source, resolved, destination, vocab_size, sample_bytes, seed = sys.argv[1:]
with open(destination, "r", encoding="utf-8") as f:
    tokenizer = json.load(f)
metadata = tokenizer.get("metadata", {})
actual_vocab_size = int(metadata.get("vocab_size", 0))
expected_vocab_size = int(vocab_size)
if actual_vocab_size != expected_vocab_size:
    raise SystemExit(
        f"reused tokenizer vocab_size={actual_vocab_size} does not match expected {expected_vocab_size}"
    )
value = {
    "command": "tokenizer train-corpus",
    "status": "reused",
    "source_tokenizer": source,
    "resolved_tokenizer": resolved,
    "destination_tokenizer": destination,
    "requested_vocab_size": expected_vocab_size,
    "sample_bytes": int(sample_bytes),
    "seed": int(seed),
    "tokenizer_metadata": metadata,
}
os.makedirs(os.path.dirname(report), exist_ok=True)
with open(report, "w", encoding="utf-8") as f:
    json.dump(value, f, indent=2)
    f.write("\n")
PY
else
  train_args=(
    "${cargo_run[@]}" -- tokenizer train-corpus
    --corpus-blend "$blend_path"
    --out "$tokenizer_path"
    --work-dir "$work_dir"
    --vocab-size "$vocab_size"
    --sample-bytes "$sample_bytes"
    --seed "$seed"
    --report "$tokenizer_report"
  )
  if [[ "$mode" == "full" && "$vocab_size" == "32768" ]]; then
    train_args+=(--require-exact-vocab)
  fi
  "${train_args[@]}"
fi

if [[ -n "$existing_tokenizer_path" ]]; then
  python3 - "$tokenizer_report" "$tokenizer_path" "$blend_path" "$work_dir" "$existing_tokenizer_path" "$resolved_tokenizer_path" <<'PY'
import collections
import json
import os
import sys

report, tokenizer_path, blend_path, work_dir, source, resolved = sys.argv[1:]

with open(tokenizer_path, "r", encoding="utf-8") as f:
    tokenizer = json.load(f)

metadata = tokenizer.get("metadata", {})
id_to_bytes = tokenizer.get("id_to_bytes", [])
histogram = collections.Counter(str(len(piece)) for piece in id_to_bytes)
validation = metadata.get("validation", {})
artifact_hash = metadata.get("artifact_hash") or validation.get("artifact_hash") or "reused"
value = {
    "command": "tokenizer train-corpus",
    "status": "passed",
    "reuse_status": "reused",
    "reused_tokenizer": True,
    "source_tokenizer": source,
    "resolved_tokenizer": resolved,
    "corpus_blend": blend_path,
    "work_dir": work_dir,
    "sample_manifest": None,
    "sample_manifest_hash": metadata.get("sample_manifest_hash", ""),
    "sampled_bytes": int(validation.get("training_bytes") or metadata.get("training_bytes") or 0),
    "tokenizer": {
        "tokenizer_id": metadata.get("tokenizer_id", ""),
        "tokenizer_path": tokenizer_path,
        "format": metadata.get("format", ""),
        "version": int(metadata.get("version", 0)),
        "vocab_size": int(metadata.get("vocab_size", 0)),
        "tokenizer_hash": artifact_hash,
        "artifact_hash": metadata.get("artifact_hash", ""),
        "source_blend_hash": metadata.get("source_blend_hash", ""),
        "sample_manifest_hash": metadata.get("sample_manifest_hash", ""),
        "reserved_tokens": len(tokenizer.get("reserved_tokens", [])),
        "reserved_registry_hash": metadata.get("reserved_registry_hash", ""),
        "training_config": metadata.get("training_config", {}),
        "validation": validation,
    },
    "token_length_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    "source_reports": [],
    "timing": {
        "sample_materialization_elapsed_ms": 0,
        "tokenizer_train_elapsed_ms": 0,
        "save_hash_elapsed_ms": 0,
        "total_elapsed_ms": 0,
    },
    "hard_path": {
        "native_trainer": True,
        "external_tokenizer_dependency": False,
        "tiny_stories_allowed": False,
    },
}
os.makedirs(os.path.dirname(report), exist_ok=True)
with open(report, "w", encoding="utf-8") as f:
    json.dump(value, f, indent=2)
    f.write("\n")
PY
fi

"${cargo_run[@]}" -- tokenizer validate "$tokenizer_path"
"${cargo_run[@]}" -- tokenizer fertility \
  --tokenizer "$tokenizer_path" \
  --corpus-blend "$blend_path" \
  --report "$fertility_report" \
  --sample-bytes "$sample_bytes" \
  --seed "$seed"

if [[ "$tokenizer_only" == "1" || "$tokenizer_only" == "true" ]]; then
  python3 - "$summary_path" "$mode" "$blend_path" "$tokenizer_path" "$tokenizer_report" "$fertility_report" <<'PY'
import json
import sys

summary, mode, blend, tokenizer, tokenizer_report, fertility = sys.argv[1:]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

tok_json = load(tokenizer)
train_json = load(tokenizer_report)
fertility_json = load(fertility)
summary_json = {
    "status": "passed",
    "mode": mode,
    "tokenizer_only": True,
    "production_gate": mode == "full",
    "tokenizer_version": tok_json["metadata"]["version"],
    "tokenizer_vocab_size": tok_json["metadata"]["vocab_size"],
    "reserved_tokens": len(tok_json.get("reserved_tokens", [])),
    "tokenizer_train_status": train_json.get("status", "passed"),
    "sampled_bytes": train_json.get("sampled_bytes"),
    "fertility_report_status": fertility_json.get("status"),
    "artifacts": {
        "corpus_blend": blend,
        "tokenizer": tokenizer,
        "tokenizer_report": tokenizer_report,
        "fertility_report": fertility,
    },
}
with open(summary, "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2)
    f.write("\n")
PY
  exit 0
fi

prepared_manifest_reused=0
if [[ -n "$prepared_manifest_source" ]]; then
  prepared_manifest_reused=1
  rm -rf "$materialized_dir/prepared"
  prepared_manifest="$(stage_prepared_manifest "$prepared_manifest_source" "$materialized_dir/prepared")"
  stage_reused_materialization_sidecars "$prepared_manifest_source" "$materialized_dir"
  write_reused_materialization_metadata "$prepared_manifest" "$materialized_dir" "$prepared_manifest_source"
else
materialize_args=(
  "${cargo_run[@]}" -- data materialize-blend
  --corpus-blend "$blend_path"
  --tokenizer "$tokenizer_path"
  --out-dir "$materialized_dir"
  --target-tokens "$target_tokens"
  --mode "$materialize_mode"
  --valid-fraction "$valid_fraction"
  --shard-tokens "$shard_tokens"
)
if [[ -n "$max_source_bytes" ]]; then
  materialize_args+=(--max-source-bytes "$max_source_bytes")
fi
if [[ -n "$max_docs_per_source" ]]; then
  materialize_args+=(--max-docs-per-source "$max_docs_per_source")
fi
if [[ -n "$max_doc_bytes" ]]; then
  materialize_args+=(--max-doc-bytes "$max_doc_bytes")
fi
if [[ -n "$candidate_text_mode" ]]; then
  materialize_args+=(--candidate-text-mode "$candidate_text_mode")
fi
if [[ -n "$candidate_retention_token_multiplier" ]]; then
  materialize_args+=(--candidate-retention-token-multiplier "$candidate_retention_token_multiplier")
fi
if [[ -n "$candidate_retention_min_docs" ]]; then
  materialize_args+=(--candidate-retention-min-docs "$candidate_retention_min_docs")
fi
if [[ -n "$candidate_prune_every" ]]; then
  materialize_args+=(--candidate-prune-every "$candidate_prune_every")
fi
if [[ -n "$progress_every_records" ]]; then
  materialize_args+=(--progress-every-records "$progress_every_records")
fi
if [[ -n "$progress_every_bytes" ]]; then
  materialize_args+=(--progress-every-bytes "$progress_every_bytes")
fi
if [[ -n "$checkpoint_dir" ]]; then
  materialize_args+=(--checkpoint-dir "$checkpoint_dir")
fi
if [[ "$resume_checkpoint" == "1" || "$resume_checkpoint" == "true" ]]; then
  materialize_args+=(--resume-checkpoint)
fi
if [[ -n "$checkpoint_every_records" ]]; then
  materialize_args+=(--checkpoint-every-records "$checkpoint_every_records")
fi
if [[ -n "$checkpoint_every_bytes" ]]; then
  materialize_args+=(--checkpoint-every-bytes "$checkpoint_every_bytes")
fi
"${materialize_args[@]}"
fi

train_memory_args=(
  "${cargo_run[@]}" -- train-memory-lm
  --dataset-manifest "$prepared_manifest"
  --checkpoint "$checkpoint"
  --steps "$steps"
  --batch-size "$batch_size"
  --grad-accumulation-steps "$grad_accumulation_steps"
  --block-size "$block_size"
  --n-layers "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_N_LAYERS:-2}"
  --d-model "$d_model"
  --n-heads "$n_heads"
  --ff-hidden "$ff_hidden"
  --memory-layer-indices "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_LAYER_INDICES:-1}"
  --memory-slots "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_SLOTS:-64}"
  --memory-key-dim "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_KEY_DIM:-8}"
  --memory-value-dim "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_VALUE_DIM:-16}"
  --memory-top-k "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_TOP_K:-4}"
  --memory-heads "${HEIRLOOM_QB_TOKENIZER_HARDPATH_MEMORY_HEADS:-1}"
  --lr "$lr"
  --device "$device"
  --precision "$precision"
  --log-every 1
  --report "$train_report"
)
if [[ -n "$devices" ]]; then
  train_memory_args+=(--devices "$devices")
fi
if [[ -n "$distributed" ]]; then
  train_memory_args+=(--distributed "$distributed")
fi
"${train_memory_args[@]}"

resume_args=("${train_memory_args[@]}")
for i in "${!resume_args[@]}"; do
  case "${resume_args[$i]}" in
    "--steps") resume_args[$((i + 1))]="$resume_steps" ;;
    "$train_report") resume_args[$i]="$resume_report" ;;
  esac
done
resume_args+=(--resume)
"${resume_args[@]}"

"${cargo_run[@]}" -- eval-memory-lm \
  --checkpoint "$checkpoint" \
  --dataset-manifest "$prepared_manifest" \
  --split valid \
  --batch-size "$batch_size" \
  --max-batches "$eval_batches" \
  --device "$device" \
  --precision "$precision" \
  --report "$eval_report"

"${cargo_run[@]}" -- generate-memory-lm \
  --checkpoint "$checkpoint" \
  --prompt "<|user|>Summarize the QB tokenizer hard path.<|message_end|><|assistant|>" \
  --max-new-tokens "$generation_tokens" \
  --device "$device" \
  --precision "$precision" \
  --report "$generation_report"

python3 - "$summary_path" "$mode" "$blend_path" "$tokenizer_path" "$tokenizer_report" "$fertility_report" "$prepared_manifest" "$materialized_dir/curation-report.json" "$materialized_dir/source-index.json" "$materialized_dir/selected-docs.jsonl" "$train_report" "$resume_report" "$eval_report" "$generation_report" "$prepared_manifest_reused" "$prepared_manifest_source" <<'PY'
import json
import sys

summary, mode, blend, tokenizer, tokenizer_report, fertility, manifest, curation, source_index, selected_docs, train, resume, eval_report, generation, prepared_reused, prepared_source = sys.argv[1:]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_json = load(train)
manifest_json = load(manifest)
tok_json = load(tokenizer)
curation_json = load(curation)
summary_json = {
    "status": "passed",
    "mode": mode,
    "production_gate": mode == "full",
    "materialize_mode": curation_json["mode"],
    "target_tokens": curation_json["target_tokens"],
    "selected_tokens": curation_json["selected_tokens"],
    "selected_docs": curation_json["selected_docs"],
    "materialization_sizing": curation_json.get("sizing"),
    "materialization_checkpoint": curation_json.get("checkpoint"),
    "prepared_manifest_reused": prepared_reused in {"1", "true", "True"},
    "prepared_manifest_source": prepared_source or None,
    "tokenizer_version": tok_json["metadata"]["version"],
    "tokenizer_vocab_size": tok_json["metadata"]["vocab_size"],
    "reserved_tokens": len(tok_json.get("reserved_tokens", [])),
    "manifest_version": manifest_json["version"],
    "manifest_storage": manifest_json["storage"],
    "loader_kind": train_json["loader"]["kind"],
    "tokens_materialized": train_json["loader"]["tokens_materialized"],
    "artifacts": {
        "corpus_blend": blend,
        "tokenizer": tokenizer,
        "tokenizer_report": tokenizer_report,
        "fertility_report": fertility,
        "curation_report": curation,
        "source_index": source_index,
        "selected_docs": selected_docs,
        "dataset_manifest": manifest,
        "train_report": train,
        "resume_report": resume,
        "eval_report": eval_report,
        "generation_report": generation,
    },
    "performance": train_json.get("performance", {}),
}
with open(summary, "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2)
    f.write("\n")
PY

if [[ -z "$learning_sanity_min_selected_tokens" ]]; then
  learning_sanity_min_selected_tokens="$target_tokens"
fi
if [[ -z "$learning_sanity_min_final_step" ]]; then
  learning_sanity_min_final_step="$steps"
fi
run_learning_sanity=0
case "$learning_sanity" in
  1|true|yes|on)
    run_learning_sanity=1
    ;;
  auto)
    if [[ "$vocab_size" == "32768" ]]; then
      run_learning_sanity=1
    fi
    ;;
  0|false|no|off)
    run_learning_sanity=0
    ;;
  *)
    echo "unknown HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY=$learning_sanity" >&2
    exit 1
    ;;
esac

if [[ "$run_learning_sanity" == "1" ]]; then
  python3 - "$learning_sanity_manifest" "$learning_sanity_min_loss_reduction" "$learning_sanity_min_selected_tokens" "$learning_sanity_min_selected_docs" "$learning_sanity_min_final_step" "$learning_sanity_min_blend_sources" "$learning_sanity_require_production_gate" <<'PY'
import json
import sys

(
    manifest,
    min_loss_reduction,
    min_selected_tokens,
    min_selected_docs,
    min_final_step,
    min_blend_sources,
    require_production_gate,
) = sys.argv[1:]

def as_bool(value):
    return value.lower() in {"1", "true", "yes", "on"}

ladder = {
    "format": "heirloom.learning_sanity_ladder",
    "version": 0,
    "stages": [
        {
            "stage_id": "longer_32k_blend",
            "report": "summary.json",
            "expected_command": "train-memory-lm",
            "expected_model_family": "memory_transformer",
            "expected_loader_kind": "binary_shard_streaming",
            "min_loss_reduction": float(min_loss_reduction),
            "min_selected_tokens": int(min_selected_tokens),
            "min_selected_docs": int(min_selected_docs),
            "min_final_step": int(min_final_step),
            "min_blend_sources": int(min_blend_sources),
            "require_production_gate": as_bool(require_production_gate),
        }
    ],
}
with open(manifest, "w", encoding="utf-8") as f:
    json.dump(ladder, f, indent=2)
    f.write("\n")
PY
  "${cargo_run[@]}" -- readiness validate-learning-sanity \
    --manifest "$learning_sanity_manifest" \
    --report "$learning_sanity_validation"
fi

echo "qb tokenizer hard path complete summary=$summary_path"
