#!/usr/bin/env python3
"""Validate QB data hard-path artifacts from a local directory or GCS prefix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_PERFORMANCE_FIELDS = [
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
]

ATTENTION_TENSOR_CORE_FIELDS = [
    "bf16_tensor_core_attention_forward_calls",
    "bf16_tensor_core_attention_qk_matmul_calls",
    "bf16_tensor_core_attention_av_matmul_calls",
    "bf16_tensor_core_attention_backward_calls",
    "bf16_tensor_core_attention_score_grad_matmul_calls",
    "bf16_tensor_core_attention_dq_matmul_calls",
    "bf16_tensor_core_attention_dk_matmul_calls",
    "bf16_tensor_core_attention_dv_matmul_calls",
]

FLASH_ATTENTION_POSITIVE_RUNTIME_FIELDS = [
    "flash_bf16_tensor_core_requested_calls",
    "flash_bf16_tensor_core_executed_calls",
    "flash_bf16_tensor_core_backward_requested_calls",
    "flash_bf16_tensor_core_backward_executed_calls",
    "flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls",
    "flash_bf16_tensor_core_backward_dp_mma_tile_calls",
    "flash_bf16_tensor_core_backward_dq_mma_tile_calls",
    "flash_bf16_tensor_core_backward_dk_mma_tile_calls",
    "flash_bf16_tensor_core_backward_dv_mma_tile_calls",
]

FLASH_ATTENTION_ZERO_RUNTIME_FIELDS = [
    "flash_bf16_tensor_core_fallback_calls",
    "flash_bf16_tensor_core_backward_fallback_calls",
    "flash_bf16_tensor_core_backward_scalar_tile_calls",
    "flash_bf16_attention_hard_require_failures",
    "bf16_attention_materialized_reference_calls",
]

FLASH_ATTENTION_TIMING_FIELDS = [
    "flash_bf16_tensor_core_elapsed_us",
    "flash_bf16_tensor_core_backward_elapsed_us",
]


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summary_location(location: str) -> str:
    location = location.rstrip("/")
    if location.endswith(".json"):
        return location
    if location.startswith("gs://"):
        return f"{location}/summary.json"
    return str(Path(location) / "summary.json")


def read_text(location: str) -> str:
    if location.startswith("gs://"):
        return subprocess.check_output(["gsutil", "cat", location], text=True)
    return Path(location).read_text(encoding="utf-8")


def load_json(location: str) -> dict[str, Any]:
    return json.loads(read_text(location))


def resolve_hardpath_summary(location: str) -> tuple[str, dict[str, Any]]:
    resolved = summary_location(location)
    summary = load_json(resolved)
    if summary.get("command") == "qb-data-hardpath":
        return resolved, summary

    uris = summary.get("qb_data_hardpath_report_uris") or []
    for uri in uris:
        if str(uri).endswith("/qb-data-hardpath/summary.json"):
            return str(uri), load_json(str(uri))
        if str(uri).endswith("/summary.json") and "qb-data-hardpath" in str(uri):
            return str(uri), load_json(str(uri))
    raise ValidationError(
        f"{resolved} is not a QB hard-path summary and did not reference one"
    )


def source_ids_and_paths(manifest: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for source in manifest.get("sources") or []:
        if isinstance(source, dict):
            for key in ("source_id", "path"):
                value = source.get(key)
                if value:
                    values.add(str(value))
    return values


def validate_loader(summary: dict[str, Any], train: dict[str, Any]) -> None:
    loader = summary.get("loader") or train.get("loader") or {}
    require(loader.get("kind") == "binary_shard_streaming", f"bad train loader: {loader}")
    require(
        loader.get("tokens_materialized") is False,
        f"train loader materialized tokens: {loader}",
    )
    eval_loader = summary.get("eval_loader") or (summary.get("eval") or {}).get("loader") or {}
    require(eval_loader.get("kind") == "binary_shard_streaming", f"bad eval loader: {eval_loader}")
    require(
        eval_loader.get("tokens_materialized") is False,
        f"eval loader materialized tokens: {eval_loader}",
    )


def validate_eval_generation(summary: dict[str, Any], model_kind: str) -> None:
    eval_report = summary.get("eval") or {}
    generation = summary.get("generation") or {}
    require(eval_report, "eval report missing from summary")
    require(generation, "generation report missing from summary")
    expected_eval = "eval-memory-lm" if model_kind == "memory" else "eval-lm"
    expected_generation = "generate-memory-lm" if model_kind == "memory" else "generate"
    expected_family = "memory_transformer" if model_kind == "memory" else "tiny_transformer"
    require(
        eval_report.get("command") == expected_eval,
        f"unexpected eval command: {eval_report.get('command')}",
    )
    require(
        generation.get("command") == expected_generation,
        f"unexpected generation command: {generation.get('command')}",
    )
    require(
        eval_report.get("model_family") == expected_family,
        f"unexpected eval model_family: {eval_report.get('model_family')}",
    )
    require(
        generation.get("model_family") == expected_family,
        f"unexpected generation model_family: {generation.get('model_family')}",
    )


def validate_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    manifest = summary.get("manifest") or {}
    require(manifest.get("version") == 2, f"manifest version is not 2: {manifest}")
    require(
        manifest.get("storage") == "binary_shards",
        f"manifest storage is not binary_shards: {manifest}",
    )
    require(manifest.get("train_shards"), "manifest has no train_shards")
    require(manifest.get("valid_shards"), "manifest has no valid_shards")
    sources = source_ids_and_paths(manifest)
    require(any("TinyStories" in item for item in sources), f"TinyStories source missing: {sources}")
    require(any("qb-traces" in item for item in sources), f"QB trace source missing: {sources}")
    return manifest


def validate_performance(summary: dict[str, Any], train: dict[str, Any]) -> dict[str, Any]:
    performance = summary.get("performance") or train.get("performance") or {}
    missing = [field for field in REQUIRED_PERFORMANCE_FIELDS if field not in performance]
    require(not missing, f"performance fields missing: {missing}")
    require(as_int(performance.get("tokens_seen")) > 0, f"bad tokens_seen: {performance}")
    require(
        as_float(performance.get("tokens_per_second")) > 0.0,
        f"bad tokens_per_second: {performance}",
    )
    return performance


def validate_memory_model(summary: dict[str, Any], train: dict[str, Any]) -> None:
    require(train.get("command") == "train-memory-lm", f"not train-memory-lm: {train.get('command')}")
    require(
        train.get("model_family") == "memory_transformer",
        f"not a memory-transformer report: {train.get('model_family')}",
    )
    memory_config = train.get("memory_config") or {}
    require(memory_config.get("memory_layer_indices"), f"memory layers missing: {memory_config}")
    if train.get("distributed") == "nccl":
        require(
            as_int(train.get("memory_table_parameter_count")) > 0,
            f"memory table parameter count missing: {train}",
        )
        gradient_counts = train.get("memory_gradient_parameter_counts") or []
        require(gradient_counts, "memory gradient parameter counts missing")
        require(
            all(as_int(value) > 0 for value in gradient_counts),
            f"memory gradients missing on at least one rank: {gradient_counts}",
        )
    else:
        selection = train.get("memory_selection") or {}
        require(
            as_int(selection.get("captured_memory_layers")) > 0,
            f"memory layers were not captured: {selection}",
        )
        require(
            as_int(selection.get("selected_row_events")) > 0,
            f"memory selected-row events missing: {selection}",
        )
        optimizer = train.get("memory_optimizer") or {}
        require(
            as_int(optimizer.get("memory_table_parameter_count")) > 0,
            f"memory optimizer table evidence missing: {optimizer}",
        )
        smft_counts = (train.get("smft_access") or {}).get("counts") or {}
        require(
            as_int(smft_counts.get("total_events")) > 0,
            f"SMFT access counts missing: {train.get('smft_access')}",
        )


def validate_full_gate(
    summary: dict[str, Any],
    train: dict[str, Any],
    expected_world_size: int | None,
    require_tensor_cores: bool,
    require_attention_tensor_cores: bool,
) -> None:
    model_kind = summary.get("model_kind", "dense")
    world_size = as_int(train.get("world_size") or len(summary.get("devices") or []))
    if expected_world_size is not None:
        require(
            world_size == expected_world_size,
            f"world_size {world_size} != expected {expected_world_size}",
        )
    require(summary.get("distributed") == "nccl", f"distributed is not nccl: {summary.get('distributed')}")
    require(as_int(train.get("all_reduce_calls")) > 0, "all_reduce_calls is not positive")
    require(as_int(train.get("all_reduce_bytes")) > 0, "all_reduce_bytes is not positive")
    performance = summary.get("performance") or train.get("performance") or {}
    runtime = train.get("cuda_runtime") or train.get("cuda_runtime_rank0") or {}
    require(
        performance.get("cuda_event_timing_available") is True,
        f"CUDA event timing missing from full gate performance: {performance}",
    )
    require(
        as_float(performance.get("forward_backward_cuda_elapsed_ms")) > 0.0,
        f"forward/backward CUDA event time is not positive: {performance}",
    )
    require(
        as_int(runtime.get("event_elapsed_calls")) > 0,
        f"CUDA event elapsed counter is not positive: {runtime}",
    )

    tolerance = as_float(train.get("parameter_checksum_tolerance"), 1.0e-3)
    checksum_fields = [
        "memory_table_checksum_sum_max_error",
        "memory_table_checksum_sumsq_max_error",
        "step_memory_checksum_sum_max_error",
        "step_memory_checksum_sumsq_max_error",
    ] if model_kind == "memory" else [
        "parameter_checksum_sum_max_error",
        "parameter_checksum_sumsq_max_error",
        "step_checksum_sum_max_error",
        "step_checksum_sumsq_max_error",
    ]
    for field in checksum_fields:
        require(as_float(train.get(field)) <= tolerance, f"{field} exceeds {tolerance}")
    if model_kind == "memory" and str(train.get("memory_update_policy")) in {
        "SparseRows",
        "sparse_rows",
        "sparse-rows",
    }:
        for field in [
            "row_union_all_reduce_calls",
            "row_union_all_reduce_bytes",
            "row_union_candidate_rows",
            "compact_gradient_all_reduce_calls",
            "compact_gradient_all_reduce_bytes",
        ]:
            require(as_int(train.get(field)) > 0, f"{field} is not positive")

    tensor_core = train.get("tensor_core") or train.get("tensor_core_rank0") or {}
    coverage = train.get("tensor_core_coverage") or {}
    linear_totals = coverage.get("linear_totals") or {}
    if require_tensor_cores:
        require(
            as_int(tensor_core.get("bf16_tensor_core_matmul_forward_calls")) > 0,
            f"missing Tensor Core forward calls: {tensor_core}",
        )
        require(
            as_int(tensor_core.get("bf16_tensor_core_matmul_backward_calls")) > 0,
            f"missing Tensor Core backward calls: {tensor_core}",
        )
        require(
            as_int(tensor_core.get("bf16_scalar_matmul_fallback_calls")) == 0,
            f"Tensor Core scalar fallbacks present: {tensor_core}",
        )
        require(
            as_int(linear_totals.get("fallback_calls")) == 0,
            f"linear Tensor Core fallbacks present: {linear_totals}",
        )
    if require_attention_tensor_cores:
        for field in ATTENTION_TENSOR_CORE_FIELDS:
            require(as_int(tensor_core.get(field)) > 0, f"missing {field}: {tensor_core}")
        require(
            as_int(runtime.get("tensor_core_attention_scalar_fallbacks")) == 0,
            f"attention scalar fallbacks present: {runtime}",
        )


def flash_attention_expected(summary: dict[str, Any]) -> bool:
    gate = summary.get("flash_attention_gate") or {}
    return gate.get("expected") is True


def validate_flash_attention_report(
    report: dict[str, Any],
    *,
    label: str,
    require_timing: bool,
) -> None:
    runtime = report.get("cuda_runtime") or report.get("cuda_runtime_rank0") or {}
    tensor_core = report.get("tensor_core") or report.get("tensor_core_rank0") or {}
    for field in FLASH_ATTENTION_POSITIVE_RUNTIME_FIELDS:
        require(as_int(runtime.get(field)) > 0, f"{label} missing flash runtime field {field}: {runtime}")
    for field in FLASH_ATTENTION_ZERO_RUNTIME_FIELDS:
        require(
            as_int(runtime.get(field)) == 0,
            f"{label} flash runtime field {field} is non-zero: {runtime}",
        )
    for field in ATTENTION_TENSOR_CORE_FIELDS:
        require(as_int(tensor_core.get(field)) > 0, f"{label} missing flash Tensor Core field {field}: {tensor_core}")
    if require_timing:
        for field in FLASH_ATTENTION_TIMING_FIELDS:
            require(as_int(runtime.get(field)) > 0, f"{label} missing flash timing field {field}: {runtime}")


def validate_flash_attention_gate(summary: dict[str, Any], train: dict[str, Any], require_timing: bool) -> None:
    validate_flash_attention_report(train, label="train", require_timing=require_timing)
    resume = summary.get("resume")
    if resume:
        validate_flash_attention_report(resume, label="resume", require_timing=require_timing)


def validate_scale_requirements(
    summary: dict[str, Any],
    performance: dict[str, Any],
    *,
    min_block_size: int | None,
    min_d_model: int | None,
    min_head_dim: int | None,
    min_grad_accumulation_steps: int | None,
    min_tokens_seen: int | None,
    min_tokens_per_second: float | None,
    min_dense_core_mfu: float | None,
    require_exact_tile_shape: bool,
) -> None:
    model = summary.get("model") or {}
    shape = summary.get("shape_gate") or {}
    block_size = as_int(shape.get("block_size") or model.get("block_size"))
    d_model = as_int(shape.get("d_model") or model.get("d_model"))
    n_heads = as_int(shape.get("n_heads") or model.get("n_heads"))
    head_dim = shape.get("head_dim")
    if head_dim is None and n_heads > 0:
        require(d_model % n_heads == 0, f"d_model {d_model} is not divisible by n_heads {n_heads}")
        head_dim = d_model // n_heads
    head_dim = as_int(head_dim)
    grad_accumulation_steps = as_int(
        performance.get("grad_accumulation_steps")
        or model.get("grad_accumulation_steps")
        or summary.get("grad_accumulation_steps")
    )
    if min_block_size is not None:
        require(block_size >= min_block_size, f"block_size {block_size} < {min_block_size}")
    if min_d_model is not None:
        require(d_model >= min_d_model, f"d_model {d_model} < {min_d_model}")
    if min_head_dim is not None:
        require(head_dim >= min_head_dim, f"head_dim {head_dim} < {min_head_dim}")
    if min_grad_accumulation_steps is not None:
        require(
            grad_accumulation_steps >= min_grad_accumulation_steps,
            f"grad_accumulation_steps {grad_accumulation_steps} < {min_grad_accumulation_steps}",
        )
    if min_tokens_seen is not None:
        require(
            as_int(performance.get("tokens_seen")) >= min_tokens_seen,
            f"tokens_seen {performance.get('tokens_seen')} < {min_tokens_seen}",
        )
    if min_tokens_per_second is not None:
        require(
            as_float(performance.get("tokens_per_second")) >= min_tokens_per_second,
            f"tokens_per_second {performance.get('tokens_per_second')} < {min_tokens_per_second}",
        )
    if min_dense_core_mfu is not None:
        require(
            as_float(performance.get("dense_core_mfu_estimate")) >= min_dense_core_mfu,
            f"dense_core_mfu_estimate {performance.get('dense_core_mfu_estimate')} < {min_dense_core_mfu}",
        )
    if require_exact_tile_shape:
        exact_tile = shape.get("exact_tile_flash_attention")
        if exact_tile is None:
            exact_tile = block_size % 16 == 0 and head_dim % 16 == 0
        require(
            exact_tile is True,
            f"shape is not exact-tile flash-compatible: block_size={block_size} head_dim={head_dim}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("location", help="Local run dir/summary.json or gs:// prefix/summary.json")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--no-tensor-core-gates", action="store_true")
    parser.add_argument("--require-flash-attention", action="store_true")
    parser.add_argument("--require-flash-timing", action="store_true")
    parser.add_argument("--require-exact-tile-shape", action="store_true")
    parser.add_argument("--min-block-size", type=int)
    parser.add_argument("--min-d-model", type=int)
    parser.add_argument("--min-head-dim", type=int)
    parser.add_argument("--min-grad-accumulation-steps", type=int)
    parser.add_argument("--min-tokens-seen", type=int)
    parser.add_argument("--min-tokens-per-second", type=float)
    parser.add_argument("--min-dense-core-mfu", type=float)
    args = parser.parse_args()

    try:
        resolved, summary = resolve_hardpath_summary(args.location)
        require(summary.get("status") == "passed", f"summary status is {summary.get('status')}")
        manifest = validate_manifest(summary)
        train = summary.get("train") or {}
        require(train, "train report missing from summary")
        model_kind = summary.get("model_kind", "dense")
        require(model_kind in {"dense", "memory"}, f"unknown model_kind: {model_kind}")
        if model_kind == "memory":
            validate_memory_model(summary, train)
        else:
            require(train.get("command") == "train-lm", f"not train-lm: {train.get('command')}")
        validate_loader(summary, train)
        validate_eval_generation(summary, model_kind)
        performance = validate_performance(summary, train)
        if args.profile == "full":
            require(summary.get("resume"), "resume report missing from full summary")
            validate_full_gate(
                summary,
                train,
                args.expected_world_size,
                not args.no_tensor_core_gates,
                not args.no_tensor_core_gates,
            )
        if args.require_flash_attention or flash_attention_expected(summary):
            validate_flash_attention_gate(
                summary,
                train,
                args.require_flash_timing,
            )
        validate_scale_requirements(
            summary,
            performance,
            min_block_size=args.min_block_size,
            min_d_model=args.min_d_model,
            min_head_dim=args.min_head_dim,
            min_grad_accumulation_steps=args.min_grad_accumulation_steps,
            min_tokens_seen=args.min_tokens_seen,
            min_tokens_per_second=args.min_tokens_per_second,
            min_dense_core_mfu=args.min_dense_core_mfu,
            require_exact_tile_shape=args.require_exact_tile_shape,
        )
        shape = summary.get("shape_gate") or {}
        model = summary.get("model") or {}
        result_block_size = shape.get("block_size") or model.get("block_size")
        result_d_model = shape.get("d_model") or model.get("d_model")
        result_n_heads = shape.get("n_heads") or model.get("n_heads")
        result_head_dim = shape.get("head_dim")
        if result_head_dim is None and result_d_model and result_n_heads:
            if as_int(result_d_model) % as_int(result_n_heads) == 0:
                result_head_dim = as_int(result_d_model) // as_int(result_n_heads)
        result_exact_tile = shape.get("exact_tile_flash_attention")
        if result_exact_tile is None and result_block_size and result_head_dim:
            result_exact_tile = as_int(result_block_size) % 16 == 0 and as_int(result_head_dim) % 16 == 0
        flash_metrics = (summary.get("flash_attention_metrics") or {}).get("train") or {}
        result = {
            "status": "passed",
            "summary": resolved,
            "profile": args.profile,
            "model_kind": model_kind,
            "flash_attention": args.require_flash_attention or flash_attention_expected(summary),
            "shape": {
                "block_size": result_block_size,
                "d_model": result_d_model,
                "n_heads": result_n_heads,
                "head_dim": result_head_dim,
                "exact_tile_flash_attention": result_exact_tile,
            },
            "train_tokens": manifest.get("train_tokens"),
            "valid_tokens": manifest.get("valid_tokens"),
            "tokens_seen": performance.get("tokens_seen"),
            "tokens_per_second": performance.get("tokens_per_second"),
            "dense_core_mfu_estimate": performance.get("dense_core_mfu_estimate"),
            "grad_accumulation_steps": performance.get("grad_accumulation_steps"),
            "flash_attention_total_us_per_token": flash_metrics.get("total_us_per_token"),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, subprocess.CalledProcessError, ValidationError, json.JSONDecodeError) as err:
        print(f"qb_data_hardpath_artifacts status=failed reason={err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
