#!/usr/bin/env python3
"""Validate QB memory throughput artifacts from a local directory or GCS prefix."""

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
    "all_reduce_elapsed_ms",
    "optimizer_elapsed_ms",
    "dataloader_host_elapsed_ms",
    "host_to_device_host_elapsed_ms",
    "forward_backward_host_elapsed_ms",
    "all_reduce_host_elapsed_ms",
    "optimizer_host_elapsed_ms",
    "host_to_device_cuda_elapsed_ms",
    "forward_backward_cuda_elapsed_ms",
    "all_reduce_cuda_elapsed_ms",
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

FLASH_POSITIVE_RUNTIME_FIELDS = [
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

FLASH_ZERO_RUNTIME_FIELDS = [
    "flash_bf16_tensor_core_fallback_calls",
    "flash_bf16_tensor_core_backward_fallback_calls",
    "flash_bf16_tensor_core_backward_scalar_tile_calls",
    "flash_bf16_attention_hard_require_failures",
    "bf16_attention_materialized_reference_calls",
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


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


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


def resolve_summary(location: str) -> tuple[str, dict[str, Any]]:
    resolved = summary_location(location)
    summary = load_json(resolved)
    if summary.get("command") == "qb-data-hardpath":
        return resolved, summary

    for uri in summary.get("qb_data_hardpath_report_uris") or []:
        text = str(uri)
        if text.endswith("/qb-data-hardpath/summary.json"):
            return text, load_json(text)
        if text.endswith("/summary.json") and "qb-data-hardpath" in text:
            return text, load_json(text)
    raise ValidationError(
        f"{resolved} is not a QB hard-path summary and did not reference one"
    )


def source_ids_and_paths(manifest: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for source in manifest.get("sources") or []:
        if not isinstance(source, dict):
            continue
        for key in ("source_id", "path"):
            value = source.get(key)
            if value:
                values.add(str(value))
    return values


def runtime(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("cuda_runtime") or report.get("cuda_runtime_rank0") or {}


def tensor_core(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("tensor_core") or report.get("tensor_core_rank0") or {}


def step_delta(report: dict[str, Any]) -> int:
    return max(0, as_int(report.get("final_step")) - as_int(report.get("start_step")))


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


def validate_streaming_loader(report: dict[str, Any], label: str) -> None:
    loader = report.get("loader") or {}
    require(loader.get("kind") == "binary_shard_streaming", f"{label} bad loader: {loader}")
    require(
        loader.get("tokens_materialized") is False,
        f"{label} loader materialized tokens: {loader}",
    )


def validate_performance(report: dict[str, Any]) -> dict[str, Any]:
    performance = report.get("performance") or {}
    missing = [field for field in REQUIRED_PERFORMANCE_FIELDS if field not in performance]
    require(not missing, f"performance fields missing: {missing}")
    require(as_int(performance.get("tokens_seen")) > 0, f"bad tokens_seen: {performance}")
    require(
        as_float(performance.get("tokens_per_second")) > 0.0,
        f"bad tokens_per_second: {performance}",
    )
    require(
        performance.get("cuda_event_timing_available") is True,
        f"CUDA event timing missing: {performance}",
    )
    require(
        as_float(performance.get("forward_backward_cuda_elapsed_ms")) > 0.0,
        f"forward/backward CUDA timing missing: {performance}",
    )
    return performance


def validate_memory_ddp(report: dict[str, Any], expected_world_size: int | None) -> None:
    require(report.get("command") == "train-memory-lm", f"not train-memory-lm: {report.get('command')}")
    require(
        report.get("model_family") == "memory_transformer",
        f"not memory_transformer: {report.get('model_family')}",
    )
    require(report.get("distributed") == "nccl", f"distributed is not nccl: {report.get('distributed')}")
    if expected_world_size is not None:
        require(
            as_int(report.get("world_size")) == expected_world_size,
            f"world_size {report.get('world_size')} != {expected_world_size}",
        )
    require(as_int(report.get("all_reduce_calls")) > 0, "all_reduce_calls is not positive")
    require(as_int(report.get("all_reduce_bytes")) > 0, "all_reduce_bytes is not positive")
    require(
        as_int(report.get("memory_table_parameter_count")) > 0,
        "memory_table_parameter_count is not positive",
    )
    gradient_counts = report.get("memory_gradient_parameter_counts") or []
    require(gradient_counts, "memory_gradient_parameter_counts missing")
    require(
        all(as_int(value) > 0 for value in gradient_counts),
        f"memory gradients missing on at least one rank: {gradient_counts}",
    )
    if str(report.get("memory_update_policy")) in {"SparseRows", "sparse_rows", "sparse-rows"}:
        for field in [
            "row_union_all_reduce_calls",
            "row_union_all_reduce_bytes",
            "row_union_candidate_rows",
            "compact_gradient_all_reduce_calls",
            "compact_gradient_all_reduce_bytes",
        ]:
            require(as_int(report.get(field)) > 0, f"{field} is not positive")


def validate_flash(report: dict[str, Any], require_timing: bool) -> None:
    rt = runtime(report)
    tc = tensor_core(report)
    for field in FLASH_POSITIVE_RUNTIME_FIELDS:
        require(as_int(rt.get(field)) > 0, f"missing flash field {field}: {rt}")
    for field in FLASH_ZERO_RUNTIME_FIELDS:
        require(as_int(rt.get(field)) == 0, f"flash field {field} non-zero: {rt}")
    for field in ATTENTION_TENSOR_CORE_FIELDS:
        require(as_int(tc.get(field)) > 0, f"missing attention Tensor Core field {field}: {tc}")
    if require_timing:
        for field in [
            "flash_bf16_tensor_core_elapsed_us",
            "flash_bf16_tensor_core_backward_elapsed_us",
        ]:
            require(as_int(rt.get(field)) > 0, f"missing flash timing {field}: {rt}")


def validate_cp_async(report: dict[str, Any], require_timing: bool) -> None:
    rt = runtime(report)
    require(
        as_int(rt.get("tensor_core_cp_async_gemm_executed_calls")) > 0,
        f"cp.async GEMM did not execute: {rt}",
    )
    require(
        as_int(rt.get("tensor_core_cp_async_gemm_instructions")) > 0,
        f"cp.async instruction count missing: {rt}",
    )
    require(
        as_int(rt.get("tensor_core_ldmatrix_gemm_instructions")) > 0,
        f"ldmatrix instruction count missing: {rt}",
    )
    require(
        as_int(rt.get("tensor_core_cp_async_gemm_staged_fallback_calls")) == 0,
        f"cp.async fallback non-zero: {rt}",
    )
    require(
        as_int(rt.get("tensor_core_cp_async_gemm_hard_require_failures")) == 0,
        f"cp.async hard failures non-zero: {rt}",
    )
    if require_timing:
        require(
            as_int(rt.get("tensor_core_cp_async_gemm_elapsed_us")) > 0,
            f"cp.async GEMM elapsed timing missing: {rt}",
        )


def validate_kernel_launch_families(
    report: dict[str, Any], *, require_timing: bool = False
) -> list[dict[str, Any]]:
    rt = runtime(report)
    families = rt.get("kernel_launch_families") or {}
    require(isinstance(families, dict), f"kernel_launch_families is not an object: {families}")
    require(families, f"kernel_launch_families is empty: {rt}")
    rows: list[dict[str, Any]] = []
    for label, stats in families.items():
        require(isinstance(stats, dict), f"kernel launch family {label} is not an object: {stats}")
        calls = as_int(stats.get("calls"))
        elements = as_int(stats.get("elements"))
        elapsed_us = as_int(stats.get("elapsed_us"))
        require(calls > 0, f"kernel launch family {label} has non-positive calls: {stats}")
        rows.append(
            {"label": label, "calls": calls, "elements": elements, "elapsed_us": elapsed_us}
        )
    has_elapsed_timing = any(row["elapsed_us"] > 0 for row in rows)
    if require_timing:
        require(has_elapsed_timing, f"kernel launch family timing missing: {families}")
        require(
            as_int(rt.get("kernel_launch_family_elapsed_us")) > 0,
            f"kernel_launch_family_elapsed_us missing: {rt}",
        )
    rows.sort(
        key=(
            (lambda row: (-row["elapsed_us"], -row["calls"], -row["elements"], row["label"]))
            if has_elapsed_timing
            else (lambda row: (-row["calls"], -row["elements"], row["label"]))
        )
    )
    total_calls = sum(row["calls"] for row in rows)
    total_elements = sum(row["elements"] for row in rows)
    total_elapsed_us = sum(row["elapsed_us"] for row in rows)
    require(
        total_calls == as_int(rt.get("kernel_launch_calls")),
        f"kernel launch family calls {total_calls} != kernel_launch_calls {rt.get('kernel_launch_calls')}",
    )
    require(
        total_elements == as_int(rt.get("kernel_launch_elements")),
        "kernel launch family elements "
        + f"{total_elements} != kernel_launch_elements {rt.get('kernel_launch_elements')}",
    )
    if as_int(rt.get("kernel_launch_family_elapsed_us")) > 0:
        require(
            total_elapsed_us == as_int(rt.get("kernel_launch_family_elapsed_us")),
            "kernel launch family elapsed_us "
            + f"{total_elapsed_us} != kernel_launch_family_elapsed_us {rt.get('kernel_launch_family_elapsed_us')}",
        )
    return rows[:16]


def validate_shape(
    summary: dict[str, Any],
    performance: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    shape = summary.get("shape_gate") or {}
    model = summary.get("model") or {}
    block_size = as_int(shape.get("block_size") or model.get("block_size"))
    d_model = as_int(shape.get("d_model") or model.get("d_model"))
    n_heads = as_int(shape.get("n_heads") or model.get("n_heads"))
    head_dim = shape.get("head_dim")
    if head_dim is None and n_heads > 0:
        require(d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}")
        head_dim = d_model // n_heads
    head_dim = as_int(head_dim)
    grad_accumulation_steps = as_int(
        performance.get("grad_accumulation_steps") or model.get("grad_accumulation_steps")
    )

    if args.min_block_size is not None:
        require(block_size >= args.min_block_size, f"block_size {block_size} < {args.min_block_size}")
    if args.min_d_model is not None:
        require(d_model >= args.min_d_model, f"d_model {d_model} < {args.min_d_model}")
    if args.min_head_dim is not None:
        require(head_dim >= args.min_head_dim, f"head_dim {head_dim} < {args.min_head_dim}")
    if args.min_grad_accumulation_steps is not None:
        require(
            grad_accumulation_steps >= args.min_grad_accumulation_steps,
            "grad_accumulation_steps "
            + f"{grad_accumulation_steps} < {args.min_grad_accumulation_steps}",
        )
    if args.require_exact_tile_shape:
        exact_tile = shape.get("exact_tile_flash_attention")
        if exact_tile is None:
            exact_tile = block_size % 16 == 0 and head_dim % 16 == 0
        require(
            exact_tile is True,
            f"shape is not exact-tile flash-compatible: block_size={block_size} head_dim={head_dim}",
        )
    return {
        "block_size": block_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "grad_accumulation_steps": grad_accumulation_steps,
    }


def bucket_summary(report: dict[str, Any], throughput: dict[str, Any]) -> dict[str, Any]:
    performance = report.get("performance") or {}
    flash = throughput.get("flash_attention_metrics") or {}
    cp_async = throughput.get("cp_async_gemm_metrics") or {}
    forward_backward_ms = as_float(performance.get("forward_backward_cuda_elapsed_ms"))
    train_ms = as_float(performance.get("train_elapsed_ms"))
    flash_ms = as_float(flash.get("total_elapsed_us")) / 1000.0
    cp_async_ms = as_float(cp_async.get("elapsed_us")) / 1000.0
    all_reduce_ms = as_float(performance.get("all_reduce_cuda_elapsed_ms"))
    optimizer_ms = as_float(performance.get("optimizer_cuda_elapsed_ms"))
    h2d_ms = as_float(performance.get("host_to_device_cuda_elapsed_ms"))
    return {
        "flash_cuda_ms": flash_ms,
        "flash_share_of_forward_backward_cuda": safe_div(flash_ms, forward_backward_ms),
        "cp_async_gemm_cuda_ms": cp_async_ms,
        "cp_async_gemm_share_of_forward_backward_cuda": safe_div(cp_async_ms, forward_backward_ms),
        "forward_backward_cuda_share_of_train": safe_div(forward_backward_ms, train_ms),
        "all_reduce_cuda_share_of_train": safe_div(all_reduce_ms, train_ms),
        "optimizer_cuda_share_of_train": safe_div(optimizer_ms, train_ms),
        "host_to_device_cuda_share_of_train": safe_div(h2d_ms, train_ms),
    }


def top_kernel_launch_families(report: dict[str, Any]) -> list[dict[str, Any]]:
    rt = runtime(report)
    families = rt.get("kernel_launch_families") or {}
    total_calls = as_int(rt.get("kernel_launch_calls"))
    total_elements = as_int(rt.get("kernel_launch_elements"))
    total_elapsed_us = as_int(rt.get("kernel_launch_family_elapsed_us"))
    rows: list[dict[str, Any]] = []
    if isinstance(families, dict):
        for label, stats in families.items():
            if not isinstance(stats, dict):
                continue
            calls = as_int(stats.get("calls"))
            elements = as_int(stats.get("elements"))
            elapsed_us = as_int(stats.get("elapsed_us"))
            rows.append(
                {
                    "label": label,
                    "calls": calls,
                    "elements": elements,
                    "elapsed_us": elapsed_us,
                    "call_share": safe_div(calls, total_calls),
                    "element_share": safe_div(elements, total_elements),
                    "elapsed_share": safe_div(elapsed_us, total_elapsed_us),
                }
            )
    if any(row["elapsed_us"] > 0 for row in rows):
        rows.sort(
            key=lambda row: (-row["elapsed_us"], -row["calls"], -row["elements"], row["label"])
        )
    else:
        rows.sort(key=lambda row: (-row["calls"], -row["elements"], row["label"]))
    return rows[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("location", help="Local run dir/summary.json or gs:// prefix/summary.json")
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--require-flash-attention", action="store_true")
    parser.add_argument("--require-flash-timing", action="store_true")
    parser.add_argument("--require-cp-async-gemm", action="store_true")
    parser.add_argument("--require-cp-async-gemm-timing", action="store_true")
    parser.add_argument("--require-kernel-launch-families", action="store_true")
    parser.add_argument("--require-kernel-launch-family-timing", action="store_true")
    parser.add_argument("--require-exact-tile-shape", action="store_true")
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--min-block-size", type=int)
    parser.add_argument("--min-d-model", type=int)
    parser.add_argument("--min-head-dim", type=int)
    parser.add_argument("--min-grad-accumulation-steps", type=int)
    parser.add_argument("--min-warmup-steps", type=int)
    parser.add_argument("--min-measured-steps", type=int)
    parser.add_argument("--min-tokens-seen", type=int)
    parser.add_argument("--min-tokens-per-second", type=float)
    parser.add_argument("--min-dense-core-mfu", type=float)
    args = parser.parse_args()

    try:
        resolved, summary = resolve_summary(args.location)
        require(summary.get("status") == "passed", f"summary status is {summary.get('status')}")
        require(summary.get("profile") == "throughput", f"profile is not throughput: {summary.get('profile')}")
        if args.require_release:
            require(
                summary.get("cargo_profile") == "release",
                f"cargo_profile is not release: {summary.get('cargo_profile')}",
            )
        require(summary.get("model_kind") == "memory", f"model_kind is not memory: {summary.get('model_kind')}")
        require(summary.get("skip_eval_generation") is True, "throughput run did not skip eval/generation")
        manifest = validate_manifest(summary)

        train = summary.get("train") or {}
        resume = summary.get("resume") or {}
        throughput = summary.get("throughput") or {}
        require(throughput.get("enabled") is True, f"throughput summary is not enabled: {throughput}")
        selected_report = throughput.get("selected_report")
        require(selected_report in {"train", "resume"}, f"bad selected_report: {selected_report}")
        measured = resume if selected_report == "resume" else train
        require(measured, f"selected measured report {selected_report} is missing")
        if selected_report == "resume":
            require(train, "warmup train report missing")
            validate_streaming_loader(train, "warmup")
        validate_streaming_loader(measured, "measured")
        validate_memory_ddp(measured, args.expected_world_size)
        performance = validate_performance(measured)

        warmup_steps = step_delta(train) if selected_report == "resume" else 0
        measured_steps = step_delta(measured)
        if args.min_warmup_steps is not None:
            require(warmup_steps >= args.min_warmup_steps, f"warmup_steps {warmup_steps} < {args.min_warmup_steps}")
        if args.min_measured_steps is not None:
            require(
                measured_steps >= args.min_measured_steps,
                f"measured_steps {measured_steps} < {args.min_measured_steps}",
            )
        if args.min_tokens_seen is not None:
            require(
                as_int(performance.get("tokens_seen")) >= args.min_tokens_seen,
                f"tokens_seen {performance.get('tokens_seen')} < {args.min_tokens_seen}",
            )
        if args.min_tokens_per_second is not None:
            require(
                as_float(performance.get("tokens_per_second")) >= args.min_tokens_per_second,
                f"tokens_per_second {performance.get('tokens_per_second')} < {args.min_tokens_per_second}",
            )
        if args.min_dense_core_mfu is not None:
            require(
                as_float(performance.get("dense_core_mfu_estimate")) >= args.min_dense_core_mfu,
                "dense_core_mfu_estimate "
                + f"{performance.get('dense_core_mfu_estimate')} < {args.min_dense_core_mfu}",
            )

        shape = validate_shape(summary, performance, args)
        if args.require_flash_attention:
            validate_flash(measured, args.require_flash_timing)
        if args.require_cp_async_gemm or args.require_cp_async_gemm_timing:
            validate_cp_async(measured, args.require_cp_async_gemm_timing)
        kernel_launch_families = (
            validate_kernel_launch_families(
                measured, require_timing=args.require_kernel_launch_family_timing
            )
            if args.require_kernel_launch_families or args.require_kernel_launch_family_timing
            else top_kernel_launch_families(measured)
        )

        result = {
            "status": "passed",
            "summary": resolved,
            "cargo_profile": summary.get("cargo_profile"),
            "selected_report": selected_report,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "shape": shape,
            "train_tokens": manifest.get("train_tokens"),
            "valid_tokens": manifest.get("valid_tokens"),
            "tokens_seen": performance.get("tokens_seen"),
            "tokens_per_second": performance.get("tokens_per_second"),
            "dense_core_mfu_estimate": performance.get("dense_core_mfu_estimate"),
            "end_to_end_mfu_estimate": performance.get("end_to_end_mfu_estimate"),
            "bucket_summary": bucket_summary(measured, throughput),
            "flash_attention_total_us_per_token": (
                throughput.get("flash_attention_metrics") or {}
            ).get("total_us_per_token"),
            "cp_async_gemm_executed_calls": (
                throughput.get("cp_async_gemm_metrics") or {}
            ).get("executed_calls"),
            "cp_async_gemm_elapsed_us": (
                throughput.get("cp_async_gemm_metrics") or {}
            ).get("elapsed_us"),
            "kernel_launch_top_families": kernel_launch_families,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, subprocess.CalledProcessError, ValidationError, json.JSONDecodeError) as err:
        print(f"qb_memory_throughput_artifacts status=failed reason={err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
