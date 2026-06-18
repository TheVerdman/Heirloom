#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

out_dir="${1:-runs/gpu-smoke}"
device="${HEIRLOOM_GPU_DEVICE:-0}"
len="${HEIRLOOM_GPU_SMOKE_LEN:-4096}"
collect_topology="${HEIRLOOM_COLLECT_GPU_TOPOLOGY:-0}"
run_tensor_core_probe="${HEIRLOOM_RUN_TENSOR_CORE_PROBE:-0}"
run_nccl_probe="${HEIRLOOM_RUN_NCCL_PROBE:-0}"
nccl_probe_devices="${HEIRLOOM_NCCL_PROBE_DEVICES:-}"
nccl_probe_len="${HEIRLOOM_NCCL_PROBE_LEN:-1024}"

mkdir -p "$out_dir"

cargo run --bin heirloom -- gpu info | tee "$out_dir/gpu-info.txt"
if [[ "$collect_topology" == "1" ]]; then
  cargo run --bin heirloom -- gpu topology \
    --report "$out_dir/heirloom-gpu-topology.json" \
    | tee "$out_dir/heirloom-gpu-topology.txt"
  echo "gpu topology report: $out_dir/heirloom-gpu-topology.json"
fi

cargo run --bin heirloom -- gpu smoke \
  --device "$device" \
  --len "$len" \
  --report "$out_dir/gpu-smoke-device-${device}.json"

echo "gpu smoke report: $out_dir/gpu-smoke-device-${device}.json"

if [[ "$run_tensor_core_probe" == "1" ]]; then
  cargo run --bin heirloom -- gpu tensor-core-probe \
    --device "$device" \
    --report "$out_dir/tensor-core-probe-device-${device}.json" \
    | tee "$out_dir/tensor-core-probe-device-${device}.txt"
  echo "tensor core probe report: $out_dir/tensor-core-probe-device-${device}.json"
fi

if [[ "$run_nccl_probe" == "1" ]]; then
  if [[ -z "$nccl_probe_devices" ]]; then
    echo "HEIRLOOM_NCCL_PROBE_DEVICES is required when HEIRLOOM_RUN_NCCL_PROBE=1" >&2
    exit 2
  fi
  cargo run --bin heirloom -- gpu nccl-probe \
    --devices "$nccl_probe_devices" \
    --len "$nccl_probe_len" \
    --report "$out_dir/nccl-probe.json" \
    | tee "$out_dir/nccl-probe.txt"
  echo "NCCL probe report: $out_dir/nccl-probe.json"
fi
