#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

workdir="${1:-runs/qb-data-hardpath}"
mode="${HEIRLOOM_QB_DATA_HARDPATH_MODE:-smoke}"
model_kind="${HEIRLOOM_QB_DATA_HARDPATH_MODEL:-dense}"

case "$model_kind" in
  dense|memory)
    ;;
  *)
    echo "HEIRLOOM_QB_DATA_HARDPATH_MODEL must be dense or memory; got $model_kind" >&2
    exit 2
    ;;
esac

case "$mode" in
  smoke)
    default_steps=4
    default_batch=2
    default_grad_accumulation_steps=1
    default_block=16
    default_d_model=16
    default_heads=4
    default_ff=64
    default_lr=0.003
    default_min_reduction=-999.0
    default_eval_batches=2
    default_vocab=512
    default_max_bytes=1048576
    default_shard_tokens=100000
    default_generation_tokens=32
    default_resume_steps=1
    default_memory_n_layers=4
    default_memory_layer_indices="1,3"
    default_memory_slots=16
    default_memory_key_dim=8
    default_memory_value_dim=16
    default_memory_top_k=2
    default_memory_heads=1
    default_memory_lookup="exact"
    default_memory_shared_memory="true"
    default_memory_plus="true"
    default_memory_update_policy="full"
    default_memory_smft_mode="disabled"
    default_memory_smft_row_mask=""
    ;;
  full)
    default_steps=500
    default_batch=4
    default_grad_accumulation_steps=1
    default_block=64
    default_d_model=64
    default_heads=4
    default_ff=256
    default_lr=0.001
    default_min_reduction=0.20
    default_eval_batches=200
    default_vocab=1024
    default_max_bytes=""
    default_shard_tokens=1000000
    default_generation_tokens=120
    default_resume_steps=5
    default_memory_n_layers=32
    default_memory_layer_indices="8,16,24"
    default_memory_slots=1024
    default_memory_key_dim=32
    default_memory_value_dim=64
    default_memory_top_k=4
    default_memory_heads=1
    default_memory_lookup="exact"
    default_memory_shared_memory="true"
    default_memory_plus="true"
    default_memory_update_policy="sparse-rows"
    default_memory_smft_mode="masked-memory-rows"
    default_memory_smft_row_mask="auto"
    ;;
  *)
    echo "HEIRLOOM_QB_DATA_HARDPATH_MODE must be smoke or full; got $mode" >&2
    exit 2
    ;;
esac

if [[ "$model_kind" == "memory" && "$mode" == "full" ]]; then
  default_steps=100
  default_batch=1
  default_lr=0.0003
  default_min_reduction=-1.0
  default_resume_steps=5
fi

steps="${HEIRLOOM_QB_DATA_HARDPATH_STEPS:-${HEIRLOOM_TINYSTORIES_CUDA_STEPS:-$default_steps}}"
batch_size="${HEIRLOOM_QB_DATA_HARDPATH_BATCH:-${HEIRLOOM_TINYSTORIES_CUDA_BATCH:-$default_batch}}"
grad_accumulation_steps="${HEIRLOOM_QB_DATA_HARDPATH_GRAD_ACCUMULATION_STEPS:-${HEIRLOOM_TINYSTORIES_CUDA_GRAD_ACCUMULATION_STEPS:-$default_grad_accumulation_steps}}"
block_size="${HEIRLOOM_QB_DATA_HARDPATH_BLOCK:-${HEIRLOOM_TINYSTORIES_CUDA_BLOCK:-$default_block}}"
d_model="${HEIRLOOM_QB_DATA_HARDPATH_D_MODEL:-${HEIRLOOM_TINYSTORIES_CUDA_D_MODEL:-$default_d_model}}"
n_heads="${HEIRLOOM_QB_DATA_HARDPATH_HEADS:-${HEIRLOOM_TINYSTORIES_CUDA_HEADS:-$default_heads}}"
ff_hidden="${HEIRLOOM_QB_DATA_HARDPATH_FF:-${HEIRLOOM_TINYSTORIES_CUDA_FF:-$default_ff}}"
lr="${HEIRLOOM_QB_DATA_HARDPATH_LR:-${HEIRLOOM_TINYSTORIES_CUDA_LR:-$default_lr}}"
min_reduction="${HEIRLOOM_QB_DATA_HARDPATH_MIN_REDUCTION:-${HEIRLOOM_TINYSTORIES_CUDA_MIN_REDUCTION:-$default_min_reduction}}"
eval_batches="${HEIRLOOM_QB_DATA_HARDPATH_EVAL_BATCHES:-${HEIRLOOM_TINYSTORIES_CUDA_EVAL_BATCHES:-$default_eval_batches}}"
vocab_size="${HEIRLOOM_QB_DATA_HARDPATH_VOCAB:-${HEIRLOOM_TINYSTORIES_CUDA_VOCAB:-$default_vocab}}"
max_bytes="${HEIRLOOM_QB_DATA_HARDPATH_MAX_BYTES:-${HEIRLOOM_TINYSTORIES_CUDA_MAX_BYTES:-$default_max_bytes}}"
shard_tokens="${HEIRLOOM_QB_DATA_HARDPATH_SHARD_TOKENS:-$default_shard_tokens}"
generation_tokens="${HEIRLOOM_QB_DATA_HARDPATH_GENERATION_TOKENS:-${HEIRLOOM_TINYSTORIES_CUDA_GENERATION_TOKENS:-$default_generation_tokens}}"
resume_steps="${HEIRLOOM_QB_DATA_HARDPATH_RESUME_STEPS:-${HEIRLOOM_TINYSTORIES_CUDA_RESUME_STEPS:-$default_resume_steps}}"
precision="${HEIRLOOM_QB_DATA_HARDPATH_PRECISION:-${HEIRLOOM_TINYSTORIES_CUDA_PRECISION:-f32}}"
devices="${HEIRLOOM_QB_DATA_HARDPATH_DEVICES:-${HEIRLOOM_TINYSTORIES_CUDA_DEVICES:-}}"
distributed="${HEIRLOOM_QB_DATA_HARDPATH_DISTRIBUTED:-${HEIRLOOM_TINYSTORIES_CUDA_DISTRIBUTED:-}}"
device="${HEIRLOOM_QB_DATA_HARDPATH_DEVICE:-${HEIRLOOM_TINYSTORIES_CUDA_DEVICE:-cpu}}"
if [[ -n "$distributed" && -n "$devices" && "$device" == "cpu" ]]; then
  device="${devices%%,*}"
fi
log_every="${HEIRLOOM_QB_DATA_HARDPATH_LOG_EVERY:-25}"
resume_log_every="${HEIRLOOM_QB_DATA_HARDPATH_RESUME_LOG_EVERY:-1}"
ddp_init_timeout_secs="${HEIRLOOM_QB_DATA_HARDPATH_DDP_INIT_TIMEOUT_SECS:-120}"
ddp_checksum_every="${HEIRLOOM_QB_DATA_HARDPATH_DDP_CHECKSUM_EVERY:-1}"
expect_flash_bf16_attention="${HEIRLOOM_EXPECT_FLASH_BF16_ATTENTION:-0}"
hardpath_profile="${HEIRLOOM_QB_DATA_HARDPATH_PROFILE:-gate}"
skip_eval_generation="${HEIRLOOM_QB_DATA_HARDPATH_SKIP_EVAL_GENERATION:-0}"
throughput_report="${HEIRLOOM_QB_DATA_HARDPATH_THROUGHPUT_REPORT:-}"
cargo_profile="${HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE:-dev}"
memory_n_layers="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_N_LAYERS:-$default_memory_n_layers}"
memory_layer_indices="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LAYER_INDICES:-$default_memory_layer_indices}"
memory_slots="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SLOTS:-$default_memory_slots}"
memory_key_dim="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_KEY_DIM:-$default_memory_key_dim}"
memory_value_dim="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_VALUE_DIM:-$default_memory_value_dim}"
memory_top_k="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_TOP_K:-$default_memory_top_k}"
memory_heads="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_HEADS:-$default_memory_heads}"
memory_lookup="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_LOOKUP:-$default_memory_lookup}"
memory_shared_memory="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SHARED_MEMORY:-$default_memory_shared_memory}"
memory_plus="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_PLUS:-$default_memory_plus}"
memory_update_policy="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_UPDATE_POLICY:-$default_memory_update_policy}"
memory_smft_mode="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_MODE:-$default_memory_smft_mode}"
memory_smft_row_mask="${HEIRLOOM_QB_DATA_HARDPATH_MEMORY_SMFT_ROW_MASK:-$default_memory_smft_row_mask}"

mkdir -p "$workdir"

tinystories_path="$workdir/TinyStories-valid.txt"
tinystories_train_path="$tinystories_path"
qb_jsonl_path="$workdir/qb-traces.jsonl"
qb_text_path="$workdir/qb-traces.txt"
tokenizer_corpus_path="$workdir/tokenizer-corpus.txt"
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
smft_row_mask_path="$workdir/smft-row-mask.json"
stage_names=()
stage_millis=()

cargo_profile_args=()
case "$cargo_profile" in
  dev|debug)
    cargo_profile="dev"
    ;;
  release)
    cargo_profile_args=(--release)
    ;;
  *)
    echo "HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE must be dev or release; got $cargo_profile" >&2
    exit 2
    ;;
esac
heirloom_cmd=(cargo run "${cargo_profile_args[@]}" --bin heirloom --)

cleanup_distributed_ranks() {
  if [[ "$distributed" == "nccl" ]]; then
    pkill -f "heirloom train-lm-rank" 2>/dev/null || true
    pkill -f "heirloom train-memory-lm-rank" 2>/dev/null || true
  fi
}
trap cleanup_distributed_ranks EXIT

now_ms() {
  python3 -c 'import time; print(time.time_ns() // 1_000_000)'
}

is_truthy() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
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

if [[ ! -f "$tinystories_path" ]]; then
  run_stage download "${heirloom_cmd[@]}" data tinystories-valid --out "$tinystories_path"
fi

if [[ -n "$max_bytes" ]]; then
  tinystories_train_path="$workdir/TinyStories-valid.slice.txt"
  python3 - "$tinystories_path" "$tinystories_train_path" "$max_bytes" <<'PY'
import sys
source, target, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(source, "rb") as handle:
    data = handle.read(limit)
with open(target, "wb") as handle:
    handle.write(data)
PY
fi

cat > "$qb_jsonl_path" <<'EOF'
{"trace_id":"qbtrace_000001","temporal_context":{"current_time":"2026-06-10T14:03:12-04:00","timezone":"America/New_York"},"state":{"known":["local report"],"unknown":["whether gate passed"]},"delta":{"new_constraints":["read evidence before answering"]},"route":{"mode":"CODEBASE","tool_needed":true},"action":{"tool":"read_file","arguments":{"path":"runs/report.json"}},"evidence":[{"ref":"artifact://runs/report.json","claim":"loss decreased","verdict":"supports"}],"answer":{"direct":"The gate passed.","caveats":["This is a runtime gate, not a quality claim."]}}
{"trace_id":"qbtrace_000002","temporal_context":{"current_time":"2026-06-13T09:00:00-04:00","timezone":"America/New_York","previous_user_message_at":"2026-06-10T14:03:12-04:00","elapsed_since_previous_user_message":"2d18h56m48s"},"state":{"known":["thread resumed later"],"unknown":["whether external facts changed"]},"delta":{"new_constraints":["use current time for relative dates"]},"route":{"mode":"DIRECT","tool_needed":false},"action":{"tool":null,"arguments":{}},"evidence":[{"ref":"temporal_context","claim":"a few days later is true","verdict":"supports"}],"answer":{"direct":"It is now a few days later in the thread.","caveats":["Search is still needed for unstable outside facts."]}}
EOF

python3 - "$qb_jsonl_path" "$qb_text_path" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as src, open(sys.argv[2], "w", encoding="utf-8") as out:
    for line in src:
        item = json.loads(line)
        out.write(
            "TRACE {trace_id} STATE {state} DELTA {delta} ROUTE {route} ACTION {action} "
            "EVIDENCE {evidence} ANSWER {answer}\n".format(
                trace_id=item["trace_id"],
                state=json.dumps(item["state"], sort_keys=True),
                delta=json.dumps(item["delta"], sort_keys=True),
                route=json.dumps(item["route"], sort_keys=True),
                action=json.dumps(item["action"], sort_keys=True),
                evidence=json.dumps(item["evidence"], sort_keys=True),
                answer=json.dumps(item["answer"], sort_keys=True),
            )
        )
PY

cp "$tinystories_train_path" "$tokenizer_corpus_path"
printf '\n' >> "$tokenizer_corpus_path"
cat "$qb_text_path" >> "$tokenizer_corpus_path"

run_stage tokenizer "${heirloom_cmd[@]}" tokenizer train \
  --input "$tokenizer_corpus_path" \
  --out "$tokenizer_path" \
  --vocab-size "$vocab_size"

run_stage prepare "${heirloom_cmd[@]}" data prepare \
  --input "$tinystories_train_path" \
  --input "$qb_text_path" \
  --tokenizer "$tokenizer_path" \
  --out-dir "$prepared_dir" \
  --format binary-shard \
  --valid-fraction 0.05 \
  --shard-tokens "$shard_tokens"

memory_smft_args=()
memory_smft_arg_count=0
if [[ "$model_kind" == "memory" && "$memory_smft_row_mask" == "auto" ]]; then
  python3 - "$smft_row_mask_path" "$memory_slots" <<'PY'
import json
import sys
from pathlib import Path

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
  memory_smft_row_mask="$smft_row_mask_path"
fi
if [[ "$model_kind" == "memory" && "$memory_smft_mode" != "disabled" ]]; then
  memory_smft_args+=(--smft-mode "$memory_smft_mode")
  memory_smft_arg_count=$((memory_smft_arg_count + 2))
fi
if [[ "$model_kind" == "memory" && -n "$memory_smft_row_mask" ]]; then
  memory_smft_args+=(--smft-row-mask "$memory_smft_row_mask")
  memory_smft_arg_count=$((memory_smft_arg_count + 2))
fi

target_args=()
if [[ -n "$distributed" ]]; then
  target_args=(
    --devices "$devices"
    --distributed "$distributed"
    --ddp-init-timeout-secs "$ddp_init_timeout_secs"
    --ddp-checksum-every "$ddp_checksum_every"
  )
else
  target_args=(--device "$device")
fi

if [[ "$model_kind" == "memory" ]]; then
  train_args=(
    "${heirloom_cmd[@]}" train-memory-lm
    --dataset-manifest "$manifest_path"
    --checkpoint "$checkpoint_path"
    --steps "$steps"
    --batch-size "$batch_size"
    --grad-accumulation-steps "$grad_accumulation_steps"
    --block-size "$block_size"
    --n-layers "$memory_n_layers"
    --d-model "$d_model"
    --n-heads "$n_heads"
    --ff-hidden "$ff_hidden"
    --memory-layer-indices "$memory_layer_indices"
    --memory-slots "$memory_slots"
    --memory-key-dim "$memory_key_dim"
    --memory-value-dim "$memory_value_dim"
    --memory-top-k "$memory_top_k"
    --memory-heads "$memory_heads"
    --memory-lookup "$memory_lookup"
    --shared-memory "$memory_shared_memory"
    --memory-plus "$memory_plus"
    --memory-update-policy "$memory_update_policy"
    --lr "$lr"
    --weight-decay 0.01
    --clip-norm 1.0
    --precision "$precision"
    --log-every "$log_every"
    --report "$report_path"
    "${target_args[@]}"
  )
  if [[ "$memory_smft_arg_count" -gt 0 ]]; then
    train_args+=("${memory_smft_args[@]}")
  fi
else
  train_args=(
    "${heirloom_cmd[@]}" train-lm
    --dataset-manifest "$manifest_path"
    --checkpoint "$checkpoint_path"
    --steps "$steps"
    --batch-size "$batch_size"
    --grad-accumulation-steps "$grad_accumulation_steps"
    --block-size "$block_size"
    --d-model "$d_model"
    --n-heads "$n_heads"
    --ff-hidden "$ff_hidden"
    --lr "$lr"
    --weight-decay 0.01
    --clip-norm 1.0
    --precision "$precision"
    --log-every "$log_every"
    --report "$report_path"
    "${target_args[@]}"
  )
fi
run_stage train "${train_args[@]}"

if [[ "$model_kind" == "memory" && "$distributed" == "nccl" && -d "$workdir/ddp-memory-ranks" ]]; then
  rm -rf "$workdir/train-ddp-memory-ranks"
  cp -a "$workdir/ddp-memory-ranks" "$workdir/train-ddp-memory-ranks"
elif [[ "$distributed" == "nccl" && -d "$workdir/ddp-ranks" ]]; then
  rm -rf "$workdir/train-ddp-ranks"
  cp -a "$workdir/ddp-ranks" "$workdir/train-ddp-ranks"
fi
if [[ "$distributed" == "nccl" && -f "$workdir/launcher-report.json" ]]; then
  cp "$workdir/launcher-report.json" "$workdir/train-launcher-report.json"
fi

if [[ "$resume_steps" -gt 0 ]]; then
  rm -rf "$resume_checkpoint_path"
  cp -a "$checkpoint_path" "$resume_checkpoint_path"
  if [[ "$model_kind" == "memory" ]]; then
    resume_args=(
      "${heirloom_cmd[@]}" train-memory-lm
      --dataset-manifest "$manifest_path"
      --checkpoint "$resume_checkpoint_path"
      --steps "$resume_steps"
      --batch-size "$batch_size"
      --grad-accumulation-steps "$grad_accumulation_steps"
      --block-size "$block_size"
      --n-layers "$memory_n_layers"
      --d-model "$d_model"
      --n-heads "$n_heads"
      --ff-hidden "$ff_hidden"
      --memory-layer-indices "$memory_layer_indices"
      --memory-slots "$memory_slots"
      --memory-key-dim "$memory_key_dim"
      --memory-value-dim "$memory_value_dim"
      --memory-top-k "$memory_top_k"
      --memory-heads "$memory_heads"
      --memory-lookup "$memory_lookup"
      --shared-memory "$memory_shared_memory"
      --memory-plus "$memory_plus"
      --memory-update-policy "$memory_update_policy"
      --resume
      --precision "$precision"
      --log-every "$resume_log_every"
      --report "$resume_report_path"
      "${target_args[@]}"
    )
    if [[ "$memory_smft_arg_count" -gt 0 ]]; then
      resume_args+=("${memory_smft_args[@]}")
    fi
  else
    resume_args=(
      "${heirloom_cmd[@]}" train-lm
      --dataset-manifest "$manifest_path"
      --checkpoint "$resume_checkpoint_path"
      --steps "$resume_steps"
      --batch-size "$batch_size"
      --grad-accumulation-steps "$grad_accumulation_steps"
      --resume
      --precision "$precision"
      --log-every "$resume_log_every"
      --report "$resume_report_path"
      "${target_args[@]}"
    )
  fi
  run_stage resume "${resume_args[@]}"
  if [[ "$model_kind" == "memory" && "$distributed" == "nccl" && -d "$workdir/ddp-memory-ranks" ]]; then
    rm -rf "$workdir/resume-ddp-memory-ranks"
    cp -a "$workdir/ddp-memory-ranks" "$workdir/resume-ddp-memory-ranks"
  elif [[ "$distributed" == "nccl" && -d "$workdir/ddp-ranks" ]]; then
    rm -rf "$workdir/resume-ddp-ranks"
    cp -a "$workdir/ddp-ranks" "$workdir/resume-ddp-ranks"
  fi
  if [[ "$distributed" == "nccl" && -f "$workdir/launcher-report.json" ]]; then
    cp "$workdir/launcher-report.json" "$workdir/resume-launcher-report.json"
  fi
fi

eval_command="eval-lm"
generate_command="generate"
if [[ "$model_kind" == "memory" ]]; then
  eval_command="eval-memory-lm"
  generate_command="generate-memory-lm"
fi

if ! is_truthy "$skip_eval_generation"; then
  run_stage eval "${heirloom_cmd[@]}" "$eval_command" \
    --checkpoint "$checkpoint_path" \
    --dataset-manifest "$manifest_path" \
    --split valid \
    --device "$device" \
    --precision "$precision" \
    --batch-size "$batch_size" \
    --max-batches "$eval_batches" \
    --report "$eval_report_path"

  generate_start="$(now_ms)"
  "${heirloom_cmd[@]}" "$generate_command" \
    --checkpoint "$checkpoint_path" \
    --prompt "STATE known thread resumed DELTA current time ACTION" \
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
else
  echo "skipping eval/generation because HEIRLOOM_QB_DATA_HARDPATH_SKIP_EVAL_GENERATION=$skip_eval_generation"
fi

python3 - "$report_path" "$resume_report_path" "$manifest_path" "$model_kind" "$min_reduction" "$distributed" "${HEIRLOOM_REQUIRE_TENSOR_CORES:-0}" "${HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES:-0}" "$expect_flash_bf16_attention" "${HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING:-0}" "${HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM:-0}" "${HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM:-0}" <<'PY'
import json
import sys
(
    report_path,
    resume_report_path,
    manifest_path,
    model_kind,
    min_reduction,
    distributed,
    require_tc,
    require_attn,
    expect_flash,
    expect_flash_timing,
    expect_cp_async_gemm,
    require_cp_async_gemm,
) = sys.argv[1:13]
report = json.load(open(report_path, encoding="utf-8"))
resume_report = json.load(open(resume_report_path, encoding="utf-8")) if resume_report_path and __import__("pathlib").Path(resume_report_path).exists() else None
manifest = json.load(open(manifest_path, encoding="utf-8"))
truthy = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}

def require_flash_attention_evidence(candidate, label):
    runtime = candidate.get("cuda_runtime") or candidate.get("cuda_runtime_rank0") or {}
    tensor_core = candidate.get("tensor_core") or candidate.get("tensor_core_rank0") or {}
    for field in [
        "flash_bf16_tensor_core_requested_calls",
        "flash_bf16_tensor_core_executed_calls",
        "flash_bf16_tensor_core_backward_requested_calls",
        "flash_bf16_tensor_core_backward_executed_calls",
        "flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls",
        "flash_bf16_tensor_core_backward_dp_mma_tile_calls",
        "flash_bf16_tensor_core_backward_dq_mma_tile_calls",
        "flash_bf16_tensor_core_backward_dk_mma_tile_calls",
        "flash_bf16_tensor_core_backward_dv_mma_tile_calls",
    ]:
        if int(runtime.get(field) or 0) <= 0:
            raise SystemExit(f"{label} missing flash attention field {field}: {runtime}")
    for field in [
        "flash_bf16_tensor_core_fallback_calls",
        "flash_bf16_tensor_core_backward_fallback_calls",
        "flash_bf16_tensor_core_backward_scalar_tile_calls",
        "flash_bf16_attention_hard_require_failures",
        "bf16_attention_materialized_reference_calls",
    ]:
        if int(runtime.get(field) or 0) != 0:
            raise SystemExit(f"{label} flash attention fallback/materialization field {field} is non-zero: {runtime}")
    for field in [
        "bf16_tensor_core_attention_forward_calls",
        "bf16_tensor_core_attention_backward_calls",
        "bf16_tensor_core_attention_qk_matmul_calls",
        "bf16_tensor_core_attention_score_grad_matmul_calls",
        "bf16_tensor_core_attention_dq_matmul_calls",
        "bf16_tensor_core_attention_dk_matmul_calls",
        "bf16_tensor_core_attention_dv_matmul_calls",
    ]:
        if int(tensor_core.get(field) or 0) <= 0:
            raise SystemExit(f"{label} missing flash Tensor Core field {field}: {tensor_core}")
    if expect_flash_timing in truthy:
        for field in [
            "flash_bf16_tensor_core_elapsed_us",
            "flash_bf16_tensor_core_backward_elapsed_us",
        ]:
            if int(runtime.get(field) or 0) <= 0:
                raise SystemExit(f"{label} missing flash timing field {field}: {runtime}")

def require_cp_async_evidence(candidate, label):
    runtime = candidate.get("cuda_runtime") or candidate.get("cuda_runtime_rank0") or {}
    if int(runtime.get("tensor_core_cp_async_gemm_executed_calls") or 0) <= 0:
        raise SystemExit(f"{label} missing cp.async GEMM executions: {runtime}")
    if int(runtime.get("tensor_core_cp_async_gemm_instructions") or 0) <= 0:
        raise SystemExit(f"{label} missing cp.async instruction counter: {runtime}")
    if int(runtime.get("tensor_core_ldmatrix_gemm_instructions") or 0) <= 0:
        raise SystemExit(f"{label} missing ldmatrix instruction counter under cp.async GEMM: {runtime}")
    for field in [
        "tensor_core_cp_async_gemm_staged_fallback_calls",
        "tensor_core_cp_async_gemm_hard_require_failures",
    ]:
        if int(runtime.get(field) or 0) != 0:
            raise SystemExit(f"{label} cp.async fallback/hard-failure field {field} is non-zero: {runtime}")

if manifest.get("version") != 2 or manifest.get("storage") != "binary_shards":
    raise SystemExit(f"manifest is not v2 binary_shards: {manifest}")
source_ids = {source.get("source_id") for source in manifest.get("sources", [])}
if not any("TinyStories" in (source.get("path") or "") for source in manifest.get("sources", [])):
    raise SystemExit(f"TinyStories source missing: {manifest.get('sources')}")
if "qb-traces" not in source_ids:
    raise SystemExit(f"QB trace source missing: {manifest.get('sources')}")
loader = report.get("loader") or {}
if loader.get("kind") != "binary_shard_streaming" or loader.get("tokens_materialized") is not False:
    raise SystemExit(f"train report did not use streaming v2 loader: {loader}")
if model_kind == "memory":
    if report.get("command") != "train-memory-lm" or report.get("model_family") != "memory_transformer":
        raise SystemExit(f"memory hard path did not run train-memory-lm: {report.get('command')} {report.get('model_family')}")
    memory_config = report.get("memory_config") or {}
    if not memory_config.get("memory_layer_indices"):
        raise SystemExit(f"memory hard path has no memory layers: {memory_config}")
    if distributed:
        if int(report.get("memory_table_parameter_count") or 0) <= 0:
            raise SystemExit(f"distributed memory report missing memory table parameters: {report}")
        if any(int(value or 0) <= 0 for value in (report.get("memory_gradient_parameter_counts") or [])):
            raise SystemExit(f"distributed memory report missing per-rank memory gradients: {report.get('memory_gradient_parameter_counts')}")
    else:
        selection = report.get("memory_selection") or {}
        if int(selection.get("captured_memory_layers") or 0) <= 0 or int(selection.get("selected_row_events") or 0) <= 0:
            raise SystemExit(f"memory selection evidence missing: {selection}")
        optimizer = report.get("memory_optimizer") or {}
        if int(optimizer.get("memory_table_parameter_count") or 0) <= 0:
            raise SystemExit(f"memory optimizer evidence missing: {optimizer}")
        smft_access = ((report.get("smft_access") or {}).get("counts") or {})
        if int(smft_access.get("total_events") or 0) <= 0:
            raise SystemExit(f"SMFT access evidence missing: {report.get('smft_access')}")
elif report.get("command") != "train-lm":
    raise SystemExit(f"dense hard path did not run train-lm: {report.get('command')}")
performance = report.get("performance") or {}
for field in [
    "tokens_seen",
    "train_elapsed_ms",
    "dataloader_elapsed_ms",
    "host_to_device_elapsed_ms",
    "forward_backward_elapsed_ms",
    "optimizer_elapsed_ms",
    "dataloader_host_elapsed_ms",
    "host_to_device_host_elapsed_ms",
    "forward_backward_host_elapsed_ms",
    "optimizer_host_elapsed_ms",
    "host_to_device_cuda_elapsed_ms",
    "forward_backward_cuda_elapsed_ms",
    "optimizer_cuda_elapsed_ms",
    "cuda_event_timing_available",
    "tokens_per_second",
    "active_dense_flops_per_token_estimate",
    "dense_core_mfu_estimate",
    "end_to_end_mfu_estimate",
    "micro_batch_size",
    "grad_accumulation_steps",
    "effective_batch_size",
    "global_effective_batch_size",
]:
    if field not in performance:
        raise SystemExit(f"performance field missing: {field} in {performance}")
if float(report.get("loss_reduction", 0.0)) < float(min_reduction):
    raise SystemExit(
        f"loss reduction {report.get('loss_reduction')} < required {min_reduction}"
    )
if distributed:
    if int(report.get("all_reduce_calls") or 0) <= 0 or int(report.get("all_reduce_bytes") or 0) <= 0:
        raise SystemExit(f"missing DDP all-reduce evidence: {report}")
    if model_kind == "memory":
        tolerance = float(report.get("parameter_checksum_tolerance") or 0.0)
        for field in [
            "memory_table_checksum_sum_max_error",
            "memory_table_checksum_sumsq_max_error",
            "step_memory_checksum_sum_max_error",
            "step_memory_checksum_sumsq_max_error",
        ]:
            if float(report.get(field) or 0.0) > tolerance:
                raise SystemExit(f"{field} exceeds tolerance: {report.get(field)} > {tolerance}")
        if str(report.get("memory_update_policy")) in {"SparseRows", "sparse_rows", "sparse-rows"}:
            for field in [
                "row_union_all_reduce_calls",
                "row_union_all_reduce_bytes",
                "row_union_candidate_rows",
                "compact_gradient_all_reduce_calls",
                "compact_gradient_all_reduce_bytes",
            ]:
                if int(report.get(field) or 0) <= 0:
                    raise SystemExit(f"missing sparse memory DDP evidence {field}: {report}")
    if performance.get("cuda_event_timing_available") is not True:
        raise SystemExit(f"missing CUDA event timing evidence: {performance}")
    if float(performance.get("forward_backward_cuda_elapsed_ms") or 0.0) <= 0.0:
        raise SystemExit(f"missing forward/backward CUDA event timing: {performance}")
    runtime = report.get("cuda_runtime") or report.get("cuda_runtime_rank0") or {}
    if int(runtime.get("event_elapsed_calls") or 0) <= 0:
        raise SystemExit(f"missing CUDA event elapsed counter: {runtime}")
    if model_kind == "dense":
        tolerance = float(report.get("parameter_checksum_tolerance") or 0.0)
        for field in [
            "parameter_checksum_sum_max_error",
            "parameter_checksum_sumsq_max_error",
            "step_checksum_sum_max_error",
            "step_checksum_sumsq_max_error",
        ]:
            if float(report.get(field) or 0.0) > tolerance:
                raise SystemExit(f"{field} exceeds tolerance: {report.get(field)} > {tolerance}")
tensor_core = report.get("tensor_core") or report.get("tensor_core_rank0") or {}
coverage = report.get("tensor_core_coverage") or {}
linear_totals = coverage.get("linear_totals") or {}
if require_tc in {"1", "true", "TRUE", "yes", "YES"}:
    if int(tensor_core.get("bf16_tensor_core_matmul_forward_calls") or 0) <= 0:
        raise SystemExit(f"missing Tensor Core forward calls: {tensor_core}")
    if int(tensor_core.get("bf16_tensor_core_matmul_backward_calls") or 0) <= 0:
        raise SystemExit(f"missing Tensor Core backward calls: {tensor_core}")
    if int(tensor_core.get("bf16_scalar_matmul_fallback_calls") or 0) != 0:
        raise SystemExit(f"Tensor Core scalar fallbacks present: {tensor_core}")
    if int(linear_totals.get("fallback_calls") or 0) != 0:
        raise SystemExit(f"Linear Tensor Core fallbacks present: {coverage}")
if require_attn in {"1", "true", "TRUE", "yes", "YES"}:
    for field in [
        "bf16_tensor_core_attention_forward_calls",
        "bf16_tensor_core_attention_qk_matmul_calls",
        "bf16_tensor_core_attention_av_matmul_calls",
        "bf16_tensor_core_attention_backward_calls",
        "bf16_tensor_core_attention_score_grad_matmul_calls",
        "bf16_tensor_core_attention_dq_matmul_calls",
        "bf16_tensor_core_attention_dk_matmul_calls",
        "bf16_tensor_core_attention_dv_matmul_calls",
    ]:
        if int(tensor_core.get(field) or 0) <= 0:
            raise SystemExit(f"missing attention Tensor Core field {field}: {tensor_core}")
if expect_flash in truthy:
    require_flash_attention_evidence(report, "train")
    if resume_report is not None:
        require_flash_attention_evidence(resume_report, "resume")
if expect_cp_async_gemm in truthy or require_cp_async_gemm in truthy:
    require_cp_async_evidence(report, "train")
    if resume_report is not None:
        require_cp_async_evidence(resume_report, "resume")
print("QB data hard-path train gate passed")
PY

if ! is_truthy "$skip_eval_generation"; then
python3 - "$eval_report_path" "$generation_report_path" "$model_kind" <<'PY'
import json
import sys
eval_report = json.load(open(sys.argv[1], encoding="utf-8"))
generation_report = json.load(open(sys.argv[2], encoding="utf-8"))
model_kind = sys.argv[3]
expected_eval = "eval-memory-lm" if model_kind == "memory" else "eval-lm"
expected_generation = "generate-memory-lm" if model_kind == "memory" else "generate"
expected_family = "memory_transformer" if model_kind == "memory" else "tiny_transformer"
if eval_report.get("command") != expected_eval or eval_report.get("model_family") != expected_family:
    raise SystemExit(f"unexpected eval report identity: {eval_report.get('command')} {eval_report.get('model_family')}")
if generation_report.get("command") != expected_generation or generation_report.get("model_family") != expected_family:
    raise SystemExit(f"unexpected generation report identity: {generation_report.get('command')} {generation_report.get('model_family')}")
loader = eval_report.get("loader") or {}
if loader.get("kind") != "binary_shard_streaming" or loader.get("tokens_materialized") is not False:
    raise SystemExit(f"eval report did not use streaming v2 loader: {loader}")
print("QB data hard-path eval/generation gate passed")
PY
fi

python3 - "$timings_path" "${stage_names[@]}" -- "${stage_millis[@]}" <<'PY'
import json
import sys
from pathlib import Path
separator = sys.argv.index("--")
names = sys.argv[2:separator]
millis = [int(value) for value in sys.argv[separator + 1:]]
Path(sys.argv[1]).write_text(json.dumps(dict(zip(names, millis)), indent=2) + "\n", encoding="utf-8")
PY

python3 - "$summary_path" "$workdir" "$model_kind" "$mode" "$device" "$devices" "$distributed" "$precision" "$vocab_size" "$max_bytes" "$shard_tokens" "$steps" "$batch_size" "$grad_accumulation_steps" "$block_size" "$d_model" "$n_heads" "$ff_hidden" "$lr" "$min_reduction" "$eval_batches" "$generation_tokens" "$expect_flash_bf16_attention" "$report_path" "$resume_report_path" "$eval_report_path" "$generation_report_path" "$timings_path" "$manifest_path" "$tinystories_train_path" "$qb_jsonl_path" "$qb_text_path" "$hardpath_profile" "$skip_eval_generation" "$resume_log_every" "$throughput_report" <<'PY'
import json
import os
import sys
from pathlib import Path
(
    summary_path,
    workdir,
    model_kind,
    mode,
    device,
    devices,
    distributed,
    precision,
    vocab_size,
    max_bytes,
    shard_tokens,
    steps,
    batch_size,
    grad_accumulation_steps,
    block_size,
    d_model,
    n_heads,
    ff_hidden,
    lr,
    min_reduction,
    eval_batches,
    generation_tokens,
    expect_flash_bf16_attention,
    train_path,
    resume_path,
    eval_path,
    generation_path,
    timings_path,
    manifest_path,
    tinystories_path,
    qb_jsonl_path,
    qb_text_path,
    hardpath_profile,
    skip_eval_generation,
    resume_log_every,
    throughput_report,
) = sys.argv[1:37]
load = lambda path: json.load(open(path, encoding="utf-8")) if Path(path).exists() else None
train = load(train_path)
resume = load(resume_path)
eval_report = load(eval_path)
generation = load(generation_path)
timings = load(timings_path)
manifest = load(manifest_path)
def as_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

def safe_div(numerator, denominator):
    return (float(numerator) / float(denominator)) if denominator else None

def report_runtime(report):
    if not report:
        return {}
    return report.get("cuda_runtime") or report.get("cuda_runtime_rank0") or {}

def report_performance(report):
    if not report:
        return {}
    return report.get("performance") or {}

def flash_metrics(report):
    runtime = report_runtime(report)
    performance = report_performance(report)
    tokens_seen = as_int(performance.get("tokens_seen"))
    forward_calls = as_int(runtime.get("flash_bf16_tensor_core_executed_calls"))
    backward_calls = as_int(runtime.get("flash_bf16_tensor_core_backward_executed_calls"))
    forward_us = as_int(runtime.get("flash_bf16_tensor_core_elapsed_us"))
    backward_us = as_int(runtime.get("flash_bf16_tensor_core_backward_elapsed_us"))
    total_us = forward_us + backward_us
    return {
        "tokens_seen": tokens_seen,
        "forward_executed_calls": forward_calls,
        "backward_executed_calls": backward_calls,
        "forward_elapsed_us": forward_us,
        "backward_elapsed_us": backward_us,
        "total_elapsed_us": total_us,
        "forward_us_per_call": safe_div(forward_us, forward_calls),
        "backward_us_per_call": safe_div(backward_us, backward_calls),
        "total_us_per_token": safe_div(total_us, tokens_seen),
        "forward_qk_mma_tile_calls": as_int(runtime.get("flash_bf16_tensor_core_qk_mma_tile_calls")),
        "forward_av_mma_tile_calls": as_int(runtime.get("flash_bf16_tensor_core_av_mma_tile_calls")),
        "backward_qk_recompute_mma_tile_calls": as_int(runtime.get("flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls")),
        "backward_dp_mma_tile_calls": as_int(runtime.get("flash_bf16_tensor_core_backward_dp_mma_tile_calls")),
        "backward_dq_mma_tile_calls": as_int(runtime.get("flash_bf16_tensor_core_backward_dq_mma_tile_calls")),
        "backward_dk_mma_tile_calls": as_int(runtime.get("flash_bf16_tensor_core_backward_dk_mma_tile_calls")),
        "backward_dv_mma_tile_calls": as_int(runtime.get("flash_bf16_tensor_core_backward_dv_mma_tile_calls")),
        "fallback_calls": as_int(runtime.get("flash_bf16_tensor_core_fallback_calls")),
        "backward_fallback_calls": as_int(runtime.get("flash_bf16_tensor_core_backward_fallback_calls")),
        "backward_scalar_tile_calls": as_int(runtime.get("flash_bf16_tensor_core_backward_scalar_tile_calls")),
        "materialized_reference_calls": as_int(runtime.get("bf16_attention_materialized_reference_calls")),
    }

def cp_async_metrics(report):
    runtime = report_runtime(report)
    elapsed_us = as_int(runtime.get("tensor_core_cp_async_gemm_elapsed_us"))
    executed_calls = as_int(runtime.get("tensor_core_cp_async_gemm_executed_calls"))
    forward_backward_us = as_int(
        float(report_performance(report).get("forward_backward_cuda_elapsed_ms") or 0.0) * 1000.0
    )
    return {
        "executed_calls": executed_calls,
        "requested_calls": as_int(runtime.get("tensor_core_cp_async_gemm_requested_calls")),
        "instructions": as_int(runtime.get("tensor_core_cp_async_gemm_instructions")),
        "elapsed_us": elapsed_us,
        "us_per_call": safe_div(elapsed_us, executed_calls),
        "share_of_forward_backward_cuda": safe_div(elapsed_us, forward_backward_us),
        "staged_fallback_calls": as_int(runtime.get("tensor_core_cp_async_gemm_staged_fallback_calls")),
        "hard_require_failures": as_int(runtime.get("tensor_core_cp_async_gemm_hard_require_failures")),
        "ldmatrix_instructions": as_int(runtime.get("tensor_core_ldmatrix_gemm_instructions")),
        "ldmatrix_elapsed_us": as_int(runtime.get("tensor_core_ldmatrix_gemm_elapsed_us")),
        "staged_elapsed_us": as_int(runtime.get("tensor_core_staged_cta_gemm_elapsed_us")),
        "wide_swizzled_elapsed_us": as_int(runtime.get("tensor_core_wide_swizzled_cta_gemm_elapsed_us")),
    }

def kernel_launch_family_metrics(report, limit=16):
    runtime = report_runtime(report)
    families = runtime.get("kernel_launch_families") or {}
    total_calls = as_int(runtime.get("kernel_launch_calls"))
    total_elements = as_int(runtime.get("kernel_launch_elements"))
    rows = []
    if isinstance(families, dict):
        for label, stats in families.items():
            if not isinstance(stats, dict):
                continue
            calls = as_int(stats.get("calls"))
            elements = as_int(stats.get("elements"))
            rows.append({
                "label": label,
                "calls": calls,
                "elements": elements,
                "call_share": safe_div(calls, total_calls),
                "element_share": safe_div(elements, total_elements),
            })
    rows.sort(key=lambda row: (-row["calls"], -row["elements"], row["label"]))
    return rows[:limit]

truthy = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
throughput_enabled = hardpath_profile == "throughput" or throughput_report in {"train", "resume"}
selected_throughput_report = throughput_report
if not selected_throughput_report and throughput_enabled:
    selected_throughput_report = "resume" if resume else "train"
if selected_throughput_report == "resume" and resume:
    throughput_source = resume
elif selected_throughput_report == "train" or not resume:
    selected_throughput_report = "train"
    throughput_source = train
else:
    selected_throughput_report = "train"
    throughput_source = train

def step_delta(report):
    if not report:
        return None
    start = report.get("start_step")
    final = report.get("final_step")
    if isinstance(start, int) and isinstance(final, int):
        return max(0, final - start)
    return None

model_block_size = int(block_size)
model_d_model = int(d_model)
model_n_heads = int(n_heads)
model_head_dim = model_d_model // model_n_heads if model_n_heads else None
exact_tile_flash_shape = (
    model_head_dim is not None
    and model_block_size % 16 == 0
    and model_head_dim % 16 == 0
)
summary = {
    "command": "qb-data-hardpath",
    "profile": hardpath_profile,
    "model_kind": model_kind,
    "mode": mode,
    "status": "passed",
    "workdir": workdir,
    "device": device,
    "devices": [item for item in devices.split(",") if item] if devices else None,
    "distributed": distributed or None,
    "precision": precision,
    "cargo_profile": os.environ.get("HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE", "dev"),
    "skip_eval_generation": skip_eval_generation in truthy,
    "vocab_size": int(vocab_size),
    "max_bytes": int(max_bytes) if max_bytes else None,
    "shard_tokens": int(shard_tokens),
    "grad_accumulation_steps": int(grad_accumulation_steps),
    "manifest": {
        "path": str(Path(manifest_path)),
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "storage": manifest.get("storage"),
        "sources": manifest.get("sources"),
        "train_shards": manifest.get("train_shards"),
        "valid_shards": manifest.get("valid_shards"),
        "train_tokens": manifest.get("train_tokens"),
        "valid_tokens": manifest.get("valid_tokens"),
        "shard_tokens": int(shard_tokens),
    },
    "loader": train.get("loader") if train else None,
    "eval_loader": eval_report.get("loader") if eval_report else None,
    "performance": train.get("performance") if train else None,
    "model": {
        "kind": model_kind,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "grad_accumulation_steps": int(grad_accumulation_steps),
        "effective_batch_size": int(batch_size) * int(grad_accumulation_steps),
        "block_size": int(block_size),
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "ff_hidden": int(ff_hidden),
        "lr": float(lr),
        "min_reduction": float(min_reduction),
        "shard_tokens": int(shard_tokens),
        "memory_config": train.get("memory_config") if train and model_kind == "memory" else None,
    },
    "throughput": {
        "enabled": throughput_enabled,
        "selected_report": selected_throughput_report if throughput_enabled else None,
        "warmup_report": "train" if selected_throughput_report == "resume" else None,
        "warmup_steps": step_delta(train) if selected_throughput_report == "resume" else 0,
        "measured_steps": step_delta(throughput_source),
        "resume_log_every": int(resume_log_every),
        "performance": report_performance(throughput_source) if throughput_enabled else None,
        "flash_attention_metrics": flash_metrics(throughput_source) if throughput_enabled else None,
        "cp_async_gemm_metrics": cp_async_metrics(throughput_source) if throughput_enabled else None,
        "kernel_launch_families": kernel_launch_family_metrics(throughput_source) if throughput_enabled else None,
        "all_reduce_calls": throughput_source.get("all_reduce_calls") if throughput_source and throughput_enabled else None,
        "all_reduce_bytes": throughput_source.get("all_reduce_bytes") if throughput_source and throughput_enabled else None,
        "row_union_all_reduce_calls": throughput_source.get("row_union_all_reduce_calls") if throughput_source and throughput_enabled else None,
        "row_union_all_reduce_bytes": throughput_source.get("row_union_all_reduce_bytes") if throughput_source and throughput_enabled else None,
        "compact_gradient_all_reduce_calls": throughput_source.get("compact_gradient_all_reduce_calls") if throughput_source and throughput_enabled else None,
        "compact_gradient_all_reduce_bytes": throughput_source.get("compact_gradient_all_reduce_bytes") if throughput_source and throughput_enabled else None,
    },
    "shape_gate": {
        "block_size": model_block_size,
        "d_model": model_d_model,
        "n_heads": model_n_heads,
        "head_dim": model_head_dim,
        "exact_tile_flash_attention": exact_tile_flash_shape,
        "exact_tile_flash_attention_rule": "block_size%16==0 and head_dim%16==0",
    },
    "flash_attention_gate": {
        "expected": expect_flash_bf16_attention in {"1", "true", "TRUE", "yes", "YES", "on", "ON"},
        "cuda_flash_bf16_attention": os.environ.get("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION"),
        "cuda_flash_bf16_attention_tensor_core": os.environ.get("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE"),
        "cuda_flash_bf16_attention_backward": os.environ.get("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD"),
        "cuda_flash_bf16_attention_timing": os.environ.get("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING"),
        "require_flash_bf16_attention": os.environ.get("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION"),
        "cuda_tensor_core_cp_async_gemm": os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM"),
        "cuda_tensor_core_gemm_timing": os.environ.get("HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING"),
        "require_cuda_tensor_core_cp_async_gemm": os.environ.get("HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM"),
    },
    "flash_attention_metrics": {
        "train": flash_metrics(train),
        "resume": flash_metrics(resume) if resume else None,
    },
    "cp_async_gemm_metrics": {
        "train": cp_async_metrics(train),
        "resume": cp_async_metrics(resume) if resume else None,
    },
    "data": {
        "manifest": str(Path(manifest_path)),
        "manifest_version": manifest.get("version"),
        "storage": manifest.get("storage"),
        "sources": manifest.get("sources"),
        "train_tokens": manifest.get("train_tokens"),
        "valid_tokens": manifest.get("valid_tokens"),
        "tinystories_path": str(Path(tinystories_path)),
        "qb_jsonl_path": str(Path(qb_jsonl_path)),
        "qb_text_path": str(Path(qb_text_path)),
    },
    "timings_millis": timings,
    "train": train,
    "resume": resume,
    "eval": eval_report,
    "generation": generation,
    "artifacts": {
        "train_report": str(Path(train_path)),
        "resume_report": str(Path(resume_path)) if Path(resume_path).exists() else None,
        "eval_report": str(Path(eval_path)) if Path(eval_path).exists() else None,
        "generation_report": str(Path(generation_path)) if Path(generation_path).exists() else None,
        "timings_report": str(Path(timings_path)),
        "manifest": str(Path(manifest_path)),
    },
}
Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

echo "summary: $summary_path"
echo "timings: $timings_path"
echo "report: $report_path"
if [[ "$resume_steps" -gt 0 ]]; then
  echo "resume report: $resume_report_path"
fi
if ! is_truthy "$skip_eval_generation"; then
  echo "eval: $eval_report_path"
  echo "generation report: $generation_report_path"
  echo "generation: $generation_path"
fi
