#!/usr/bin/env python3
"""Validate Heirloom memory CUDA fixture artifacts after a local or Vertex run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SPARSE_TRANSPORT_FIELDS = [
    "row_union_all_reduce_calls",
    "row_union_all_reduce_bytes",
    "row_union_candidate_rows",
    "compact_gradient_all_reduce_calls",
    "compact_gradient_all_reduce_bytes",
]

CHECKSUM_FIELDS = [
    "memory_table_checksum_sum_max_error",
    "memory_table_checksum_sumsq_max_error",
    "step_memory_checksum_sum_max_error",
    "step_memory_checksum_sumsq_max_error",
]

KERNEL_EVIDENCE_FIELDS = [
    "bool_mask_to_indices_calls",
    "gather_selected_rows_calls",
    "sparse_adamw_compact_rows_calls",
]

CORE_MEMORY_KERNEL_FIELDS = [
    "query_key_score_calls",
    "topk_calls",
    "weighted_value_forward_calls",
    "weighted_value_backward_calls",
    "selected_key_backward_calls",
    "scatter_add_rows_calls",
    "selected_rows",
]


def normalize_token(value: Any) -> str:
    return str(value).lower().replace("_", "-")


def parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    parsed = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed.append(int(part))
        except ValueError as err:
            raise SystemExit(f"invalid integer list value {value!r}") from err
    return parsed


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SystemExit(f"invalid JSON in {path}: {err}") from err


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SystemExit(f"invalid JSON in {path}: {err}") from err


def format_rank_failure(rank: dict[str, Any]) -> str:
    rank_id = rank.get("rank", "?")
    device = rank.get("device", "?")
    exit_status = rank.get("exit_status") or "<missing exit_status>"
    stage = rank.get("last_stage") or {}
    stage_name = stage.get("stage") or "<missing stage>"
    message = stage.get("message") or "<missing message>"
    report_exists = rank.get("rank_report_exists")
    return (
        f"rank={rank_id} device={device} exit={exit_status} "
        f"stage={stage_name} rank_report_exists={report_exists} message={message}"
    )


def diagnose_launcher_failure(fixture_dir: Path, launcher: dict[str, Any]) -> None:
    status = launcher.get("status")
    reason = launcher.get("reason") or "<missing reason>"
    ranks = launcher.get("ranks") or []
    rank_lines = [format_rank_failure(rank) for rank in ranks]
    unsupported_cuda = [
        line
        for line in rank_lines
        if "has no registered CUDA kernel yet" in line
        or "copy explicitly to CPU or implement the CUDA kernel path" in line
    ]
    message = [
        f"memory fixture incomplete at {fixture_dir}",
        f"launcher status={status!r} reason={reason}",
    ]
    if rank_lines:
        message.append("rank stages:")
        message.extend(f"  - {line}" for line in rank_lines)
    if unsupported_cuda:
        message.append(
            "diagnosis: at least one rank hit an unsupported CUDA operator path; "
            "this is a real device-residency blocker, not a missing report artifact."
        )
    raise SystemExit("\n".join(message))


def as_positive_int(report: dict[str, Any], field: str, label: str) -> int:
    value = int(report.get(field) or 0)
    if value <= 0:
        raise SystemExit(f"{label} missing positive {field}: {report.get(field)}")
    return value


def rank0_kernel_counters(report: dict[str, Any]) -> dict[str, Any]:
    kernels = report.get("cuda_memory_kernels_rank0") or report.get("cuda_memory_kernels") or {}
    counters = kernels.get("counters") if isinstance(kernels, dict) else None
    return counters or kernels or {}


def rank0_tensor_core_counters(report: dict[str, Any]) -> dict[str, Any]:
    counters = report.get("tensor_core_rank0") or report.get("tensor_core") or {}
    return counters if isinstance(counters, dict) else {}


def validate_checksum_fields(report: dict[str, Any], label: str) -> None:
    tolerance = float(report.get("parameter_checksum_tolerance") or 0.0)
    for field in CHECKSUM_FIELDS:
        if field not in report:
            continue
        value = float(report.get(field) or 0.0)
        if not math.isfinite(value):
            raise SystemExit(f"{label} has non-finite {field}: {value}")
        if value > tolerance:
            raise SystemExit(
                f"{label} {field}={value:.6e} exceeds tolerance={tolerance:.6e}"
            )


def validate_summary_mirror(
    summary: dict[str, Any], report: dict[str, Any], section: str, label: str
) -> None:
    mirrored = summary.get(section)
    if not isinstance(mirrored, dict):
        raise SystemExit(f"summary.json missing {section} object")
    for field in [
        "all_reduce_calls",
        "all_reduce_bytes",
        *SPARSE_TRANSPORT_FIELDS,
        "compressed_sparse_gradient_transport",
    ]:
        if field in report and mirrored.get(field) != report.get(field):
            raise SystemExit(
                f"summary {section}.{field}={mirrored.get(field)!r} does not match "
                f"{label}.{field}={report.get(field)!r}"
            )


def validate_report(
    report: dict[str, Any],
    label: str,
    *,
    expect_distributed: bool,
    expect_sparse_rows: bool,
    expected_precision: str | None,
    expected_world_size: int | None,
    expected_n_layers: int | None,
    expected_memory_layer_indices: list[int] | None,
    expected_memory_update_policy: str | None,
    expected_smft_mode: str | None,
    expected_start_step: int | None,
    expected_final_step: int | None,
    require_memory_kernel_counters: bool,
    require_tensor_core_counters: bool,
    require_attention_tensor_core_counters: bool,
) -> None:
    if report.get("command") != "train-memory-lm":
        raise SystemExit(f"{label} unexpected command: {report.get('command')}")
    if report.get("model_family") != "memory_transformer":
        raise SystemExit(f"{label} unexpected model family: {report.get('model_family')}")
    if expected_precision is not None and report.get("precision") != expected_precision:
        raise SystemExit(
            f"{label} expected precision={expected_precision}, got {report.get('precision')}"
        )
    if expected_start_step is not None and report.get("start_step") != expected_start_step:
        raise SystemExit(
            f"{label} expected start_step={expected_start_step}, got {report.get('start_step')}"
        )
    if expected_final_step is not None and report.get("final_step") != expected_final_step:
        raise SystemExit(
            f"{label} expected final_step={expected_final_step}, got {report.get('final_step')}"
        )
    if expected_memory_update_policy is not None and normalize_token(
        report.get("memory_update_policy")
    ) != normalize_token(expected_memory_update_policy):
        raise SystemExit(
            f"{label} expected memory_update_policy={expected_memory_update_policy}, "
            f"got {report.get('memory_update_policy')}"
        )
    if expected_smft_mode is not None and normalize_token(report.get("smft_mode")) != normalize_token(
        expected_smft_mode
    ):
        raise SystemExit(
            f"{label} expected smft_mode={expected_smft_mode}, got {report.get('smft_mode')}"
        )

    memory_config = report.get("memory_config") or {}
    if expected_n_layers is not None:
        actual = int(memory_config.get("n_layers") or 0)
        if actual != expected_n_layers:
            raise SystemExit(f"{label} expected memory_config.n_layers={expected_n_layers}, got {actual}")
    if expected_memory_layer_indices is not None:
        actual = memory_config.get("memory_layer_indices")
        if actual != expected_memory_layer_indices:
            raise SystemExit(
                f"{label} expected memory_config.memory_layer_indices="
                f"{expected_memory_layer_indices}, got {actual}"
            )
    if expected_memory_update_policy is not None and normalize_token(
        memory_config.get("memory_update_policy")
    ) != normalize_token(expected_memory_update_policy):
        raise SystemExit(
            f"{label} expected memory_config.memory_update_policy="
            f"{expected_memory_update_policy}, got {memory_config.get('memory_update_policy')}"
        )
    if expected_smft_mode is not None and normalize_token(memory_config.get("smft_mode")) != normalize_token(
        expected_smft_mode
    ):
        raise SystemExit(
            f"{label} expected memory_config.smft_mode={expected_smft_mode}, "
            f"got {memory_config.get('smft_mode')}"
        )

    for field in ["initial_loss", "final_loss"]:
        value = float(report.get(field))
        if not math.isfinite(value):
            raise SystemExit(f"{label} non-finite {field}: {value}")

    if expect_distributed:
        if report.get("distributed") != "nccl":
            raise SystemExit(f"{label} expected distributed=nccl, got {report.get('distributed')}")
        as_positive_int(report, "all_reduce_calls", label)
        as_positive_int(report, "all_reduce_bytes", label)
        validate_checksum_fields(report, label)
        world_size = int(report.get("world_size") or 0)
        if expected_world_size is not None and world_size != expected_world_size:
            raise SystemExit(f"{label} expected world_size={expected_world_size}, got {world_size}")
        ranks = report.get("ranks") or []
        if len(ranks) != world_size:
            raise SystemExit(f"{label} expected {world_size} rank reports, got {len(ranks)}")

    if require_memory_kernel_counters:
        counters = rank0_kernel_counters(report)
        for field in CORE_MEMORY_KERNEL_FIELDS:
            value = int(counters.get(field) or 0)
            if value <= 0:
                raise SystemExit(f"{label} rank0 missing core memory kernel evidence {field}: {counters}")

    if require_tensor_core_counters:
        counters = rank0_tensor_core_counters(report)
        total = int(counters.get("bf16_tensor_core_matmul_calls") or 0)
        forward = int(counters.get("bf16_tensor_core_matmul_forward_calls") or 0)
        if total <= 0 and forward <= 0:
            raise SystemExit(f"{label} missing Tensor Core BF16 matmul evidence: {counters}")

    if require_attention_tensor_core_counters:
        counters = rank0_tensor_core_counters(report)
        for field in [
            "bf16_tensor_core_attention_forward_calls",
            "bf16_tensor_core_attention_qk_matmul_calls",
            "bf16_tensor_core_attention_av_matmul_calls",
            "bf16_tensor_core_attention_backward_calls",
        ]:
            value = int(counters.get(field) or 0)
            if value <= 0:
                raise SystemExit(f"{label} missing attention Tensor Core evidence {field}: {counters}")

    if expect_sparse_rows:
        if not report.get("compressed_sparse_gradient_transport"):
            raise SystemExit(f"{label} missing compressed_sparse_gradient_transport=true")
        for field in SPARSE_TRANSPORT_FIELDS:
            as_positive_int(report, field, label)
        counters = rank0_kernel_counters(report)
        for field in KERNEL_EVIDENCE_FIELDS:
            value = int(counters.get(field) or 0)
            if value <= 0:
                raise SystemExit(f"{label} rank0 missing memory kernel evidence {field}: {counters}")
        for index, rank in enumerate(report.get("ranks") or []):
            for field in ["compact_gradient_all_reduce_calls", "compact_gradient_all_reduce_bytes"]:
                value = int(rank.get(field) or 0)
                if value <= 0:
                    raise SystemExit(f"{label} rank {index} missing positive {field}: {rank}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate cuda_train_memory_lm_fixture reports after a local or Vertex run."
    )
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--expect-distributed", action="store_true")
    parser.add_argument("--expect-sparse-rows", action="store_true")
    parser.add_argument("--require-resume", action="store_true")
    parser.add_argument("--expect-precision")
    parser.add_argument("--expect-world-size", type=int)
    parser.add_argument("--expect-n-layers", type=int)
    parser.add_argument("--expect-memory-layer-indices")
    parser.add_argument("--expect-memory-update-policy")
    parser.add_argument("--expect-smft-mode")
    parser.add_argument("--expect-steps", type=int)
    parser.add_argument("--expect-resume-steps", type=int)
    parser.add_argument("--expect-data-source")
    parser.add_argument("--expect-min-source-bytes", type=int)
    parser.add_argument("--expect-min-train-tokens", type=int)
    parser.add_argument("--require-memory-kernel-counters", action="store_true")
    parser.add_argument("--require-tensor-core-counters", action="store_true")
    parser.add_argument("--require-attention-tensor-core-counters", action="store_true")
    args = parser.parse_args()

    fixture_dir = args.fixture_dir
    launcher = load_optional_json(fixture_dir / "launcher-report.json")
    if not (fixture_dir / "summary.json").exists() and launcher is not None:
        diagnose_launcher_failure(fixture_dir, launcher)

    summary = load_json(fixture_dir / "summary.json")
    train = load_json(fixture_dir / "train-report.json")
    resume_path = fixture_dir / "resume-report.json"
    resume = load_json(resume_path) if resume_path.exists() else None
    if args.require_resume and resume is None:
        raise SystemExit(f"missing required artifact: {resume_path}")

    expect_sparse_rows = args.expect_sparse_rows or str(
        train.get("memory_update_policy") or summary.get("memory_update_policy") or ""
    ).lower().replace("_", "-") == "sparse-rows"
    expect_distributed = args.expect_distributed or train.get("distributed") == "nccl"
    expected_memory_layer_indices = (
        parse_int_list(args.expect_memory_layer_indices)
        if args.expect_memory_layer_indices is not None
        else None
    )

    if summary.get("fixture") != "cuda_train_memory_lm_fixture":
        raise SystemExit(f"unexpected fixture summary: {summary.get('fixture')}")
    if args.expect_precision is not None and summary.get("precision") != args.expect_precision:
        raise SystemExit(
            f"summary expected precision={args.expect_precision}, got {summary.get('precision')}"
        )
    if args.expect_steps is not None and int(summary.get("steps") or -1) != args.expect_steps:
        raise SystemExit(f"summary expected steps={args.expect_steps}, got {summary.get('steps')}")
    if args.expect_resume_steps is not None and int(summary.get("resume_steps") or -1) != args.expect_resume_steps:
        raise SystemExit(
            f"summary expected resume_steps={args.expect_resume_steps}, got {summary.get('resume_steps')}"
        )
    if args.expect_data_source is not None and summary.get("data_source") != args.expect_data_source:
        raise SystemExit(
            f"summary expected data_source={args.expect_data_source}, got {summary.get('data_source')}"
        )
    if args.expect_min_source_bytes is not None:
        source_bytes = int(summary.get("source_bytes") or 0)
        if source_bytes < args.expect_min_source_bytes:
            raise SystemExit(
                f"summary source_bytes={source_bytes} below required {args.expect_min_source_bytes}"
            )
    if args.expect_min_train_tokens is not None:
        train_tokens = int(summary.get("train_tokens") or 0)
        if train_tokens < args.expect_min_train_tokens:
            raise SystemExit(
                f"summary train_tokens={train_tokens} below required {args.expect_min_train_tokens}"
            )
    validate_report(
        train,
        "train-report.json",
        expect_distributed=expect_distributed,
        expect_sparse_rows=expect_sparse_rows,
        expected_precision=args.expect_precision,
        expected_world_size=args.expect_world_size,
        expected_n_layers=args.expect_n_layers,
        expected_memory_layer_indices=expected_memory_layer_indices,
        expected_memory_update_policy=args.expect_memory_update_policy,
        expected_smft_mode=args.expect_smft_mode,
        expected_start_step=0 if args.expect_steps is not None else None,
        expected_final_step=args.expect_steps,
        require_memory_kernel_counters=args.require_memory_kernel_counters,
        require_tensor_core_counters=args.require_tensor_core_counters,
        require_attention_tensor_core_counters=args.require_attention_tensor_core_counters,
    )
    validate_summary_mirror(summary, train, "train", "train-report.json")
    if resume is not None:
        validate_report(
            resume,
            "resume-report.json",
            expect_distributed=expect_distributed,
            expect_sparse_rows=expect_sparse_rows,
            expected_precision=args.expect_precision,
            expected_world_size=args.expect_world_size,
            expected_n_layers=args.expect_n_layers,
            expected_memory_layer_indices=expected_memory_layer_indices,
            expected_memory_update_policy=args.expect_memory_update_policy,
            expected_smft_mode=args.expect_smft_mode,
            expected_start_step=args.expect_steps if args.expect_steps is not None else None,
            expected_final_step=(
                args.expect_steps + args.expect_resume_steps
                if args.expect_steps is not None and args.expect_resume_steps is not None
                else None
            ),
            require_memory_kernel_counters=args.require_memory_kernel_counters,
            require_tensor_core_counters=args.require_tensor_core_counters,
            require_attention_tensor_core_counters=args.require_attention_tensor_core_counters,
        )
        validate_summary_mirror(summary, resume, "resume", "resume-report.json")

    print(
        "memory_fixture_artifacts status=passed "
        f"distributed={expect_distributed} sparse_rows={expect_sparse_rows} "
        f"train_compact_gradient_all_reduce_calls={train.get('compact_gradient_all_reduce_calls')}"
    )


if __name__ == "__main__":
    main()
