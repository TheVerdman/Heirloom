#!/usr/bin/env python3
"""Stage governed QB-native pretraining source slices into GCS.

This script intentionally treats GCS as the durable source-slice store and
developer machines as temporary staging/inspection surfaces. It uploads only
explicit local uncompressed slice files and writes an inventory that the
tokenizer/materializer hard path can use as launch input documentation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


DEFAULT_GCS_PREFIX = (
    "gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/"
    "heirloom/qb-native-pretraining-v1/source-slices"
)
DEFAULT_QB_CORPUS = (
    "/Users/andrewverdiramo/Desktop/VECL-QB/data/synthetic/v1-hard/corpus.jsonl"
)
DEFAULT_QB_METADATA = (
    "/Users/andrewverdiramo/Desktop/VECL-QB/data/synthetic/v1-hard/metadata.json"
)
APPROVED_LICENSE_STATUSES = {
    "approved",
    "source_terms_verified",
    "redistribution_allowed",
    "odc_by_verified",
    "odc_by_internal_attribution",
    "nvidia_data_agreement_internal_training",
    "internal_synthetic",
}
COMPRESSED_SUFFIXES = (".gz", ".zst", ".zstd")
GOVERNANCE_EVIDENCE: dict[str, dict[str, Any]] = {
    "allenai.dolma.v1_7": {
        "review_status": "conditionally_approved_internal_training_with_attribution",
        "primary_urls": [
            "https://huggingface.co/datasets/allenai/dolma",
            "https://opendatacommons.org/licenses/by/1-0/",
        ],
        "evidence_summary": (
            "Dolma is listed as ODC-BY on Hugging Face; its dataset card says "
            "original source licenses and terms also apply. For this project, "
            "Dolma slices are approved only for private/internal model training "
            "with ODC-BY attribution recorded and no raw public redistribution."
        ),
        "required_before_upload": [
            "Record ODC-BY attribution plan.",
            "Keep staged raw slices private/internal and do not redistribute them.",
            "Record the exact selected Dolma objects/sub-sources and hashes.",
        ],
    },
    "nvidia.nemotron_cc.high_actual": {
        "review_status": "conditionally_approved_internal_training_sample_artifact",
        "primary_urls": [
            "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample",
            "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/blob/main/LICENSE.md",
            "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html",
            "https://commoncrawl.org/terms-of-use",
        ],
        "evidence_summary": (
            "The nvidia/Nemotron-Pretraining-Dataset-sample artifact includes "
            "Nemotron-CC high-quality subsets and is governed by NVIDIA's Data "
            "Agreement for Model Training, not Apache-2.0. The agreement allows "
            "internal training use of Company AI Solutions, but prohibits making "
            "the raw datasets available to others and preserves third-party "
            "rights caveats. Full-size Nemotron-CC objects outside this sample "
            "artifact still need their own artifact/license check."
        ),
        "required_before_upload": [
            "Use only files/subsets from the nvidia/Nemotron-Pretraining-Dataset-sample artifact unless a separate full-artifact license is recorded.",
            "Keep staged slices internal/private for model training only.",
            "Record acceptance of NVIDIA Data Agreement terms and Common Crawl/source-term caveats.",
        ],
    },
    "allenai.dolma3_dolmino_mix-100B-1125": {
        "review_status": "conditionally_approved_internal_training_with_attribution",
        "primary_urls": [
            "https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125",
            "https://arxiv.org/abs/2512.13961",
            "https://opendatacommons.org/licenses/by/1-0/",
        ],
        "evidence_summary": (
            "The OLMo 3 second-stage annealing pool is published as "
            "allenai/dolma3_dolmino_mix-100B-1125 on Hugging Face with license "
            "odc-by and an OLMo 3 paper reference. The card describes it as a "
            "100B-token mixed-down high-quality pool, making it the preferred "
            "default for our OLMo/Dolma 3 curated slice. For this project, "
            "Dolmino slices are approved only for private/internal model "
            "training with ODC-BY attribution recorded and no raw public "
            "redistribution."
        ),
        "required_before_upload": [
            "Record ODC-BY attribution plan.",
            "Keep staged raw slices private/internal and do not redistribute them.",
            "Record the exact selected Dolmino mix objects/sub-sources and hashes.",
        ],
    },
    "nvidia.nemotron_cc_math": {
        "review_status": "conditionally_approved_internal_training_sample_artifact",
        "primary_urls": [
            "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample",
            "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/blob/main/LICENSE.md",
            "https://arxiv.org/abs/2508.15096",
        ],
        "evidence_summary": (
            "The nvidia/Nemotron-Pretraining-Dataset-sample artifact includes "
            "a Nemotron-CC-MATH subset and is governed by NVIDIA's Data "
            "Agreement for Model Training, not Apache-2.0. The agreement allows "
            "internal training use but prohibits public/raw dataset sharing and "
            "does not grant rights to underlying third-party copyrighted "
            "material. Full Nemotron-CC-Math releases outside this sample "
            "artifact still need exact artifact/license review."
        ),
        "required_before_upload": [
            "Use only files/subsets from the nvidia/Nemotron-Pretraining-Dataset-sample artifact unless a separate full-artifact license is recorded.",
            "Keep staged slices internal/private for model training only.",
            "Record acceptance of NVIDIA Data Agreement terms and third-party-rights caveats.",
        ],
    },
    "vecl_qb.synthetic.v1-hard": {
        "review_status": "approved_internal_synthetic",
        "primary_urls": [
            "file:///Users/andrewverdiramo/Desktop/VECL-QB/data/synthetic/v1-hard/metadata.json",
        ],
        "evidence_summary": (
            "Project-internal synthetic VECL-QB corpus with no production user "
            "data, local metadata, record counts, validation fields, and hashes."
        ),
        "required_before_upload": [],
    },
}


@dataclass(frozen=True)
class SourceSpec:
    arg_name: str
    slug: str
    source_id: str
    display_name: str
    data_format: str
    role: str
    sampling_weight: float
    license_status: str
    default_path: str | None = None
    default_metadata: str | None = None


SOURCE_SPECS = [
    SourceSpec(
        arg_name="dolma",
        slug="dolma-v1_7",
        source_id="allenai.dolma.v1_7",
        display_name="Dolma v1.7 materialized slice",
        data_format="text",
        role="general_language_backbone",
        sampling_weight=0.35,
        license_status="odc_by_internal_attribution",
    ),
    SourceSpec(
        arg_name="nemotron_cc",
        slug="nemotron-cc-high-actual",
        source_id="nvidia.nemotron_cc.high_actual",
        display_name="Nemotron-CC high/actual materialized slice",
        data_format="jsonl",
        role="high_quality_web_backbone",
        sampling_weight=0.25,
        license_status="nvidia_data_agreement_internal_training",
    ),
    SourceSpec(
        arg_name="olmo3",
        slug="dolma3-dolmino-mix-100b-1125",
        source_id="allenai.dolma3_dolmino_mix-100B-1125",
        display_name="Dolma 3 Dolmino mix 100B (OLMo 3 stage 2) materialized slice",
        data_format="jsonl",
        role="olmo3_dolmino_mix",
        sampling_weight=0.20,
        license_status="odc_by_internal_attribution",
    ),
    SourceSpec(
        arg_name="nemotron_cc_math",
        slug="nemotron-cc-math",
        source_id="nvidia.nemotron_cc_math",
        display_name="Nemotron-CC-Math materialized slice",
        data_format="jsonl",
        role="math_science_code_reasoning",
        sampling_weight=0.10,
        license_status="nvidia_data_agreement_internal_training",
    ),
    SourceSpec(
        arg_name="qb_v1_hard",
        slug="vecl-qb-v1-hard",
        source_id="vecl_qb.synthetic.v1-hard",
        display_name="VECL-QB synthetic v1-hard",
        data_format="jsonl",
        role="tool_routing_memory_trace_supervision",
        sampling_weight=0.10,
        license_status="internal_synthetic",
        default_path=DEFAULT_QB_CORPUS,
        default_metadata=DEFAULT_QB_METADATA,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gcs-prefix",
        default=DEFAULT_GCS_PREFIX,
        help="Destination GCS prefix for source-slice objects.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/qb-native-pretraining-v1/source-slices",
        help="Local directory for inventory and staging reports.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload available slice files and inventory to GCS.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("PROJECT_ID", ""),
        help="Optional GCP project for gcloud storage commands.",
    )
    parser.add_argument(
        "--include-pending",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include missing external sources as pending inventory entries.",
    )
    for spec in SOURCE_SPECS:
        flag = "--" + spec.arg_name.replace("_", "-")
        parser.add_argument(
            flag,
            default=spec.default_path,
            help=f"Local uncompressed slice file for {spec.source_id}.",
        )
        parser.add_argument(
            flag + "-metadata",
            default=spec.default_metadata,
            help=f"Optional local metadata file for {spec.source_id}.",
        )
        parser.add_argument(
            flag + "-license-status",
            default=spec.license_status,
            help=f"License status for {spec.source_id}.",
        )
    return parser.parse_args()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_records(path: pathlib.Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def reject_compressed(path: pathlib.Path, source_id: str) -> None:
    if path.name.endswith(COMPRESSED_SUFFIXES):
        raise SystemExit(
            f"{source_id} slice must be uncompressed for materialize-blend: {path}"
        )


def source_slice_files(path: pathlib.Path, source_id: str) -> list[pathlib.Path]:
    if path.is_file():
        reject_compressed(path, source_id)
        return [path]
    if not path.is_dir():
        raise SystemExit(f"{source_id} slice path is not a file or directory: {path}")
    files: list[pathlib.Path] = []
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            raise SystemExit(
                f"{source_id} slice directory must be flat; nested directory: {child}"
            )
        if not child.is_file():
            continue
        name = child.name
        if name.endswith(".report.json") or name == "metadata.json":
            continue
        reject_compressed(child, source_id)
        files.append(child)
    if not files:
        raise SystemExit(f"{source_id} slice directory contains no source files: {path}")
    return files


def source_set_hash(files: list[pathlib.Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def gcs_join(prefix: str, *parts: str) -> str:
    return "/".join([prefix.rstrip("/"), *[part.strip("/") for part in parts]])


def copy_to_gcs(source: pathlib.Path, destination: str, project: str) -> None:
    gcloud = shutil.which("gcloud")
    gsutil = shutil.which("gsutil")
    if gcloud:
        command = ["gcloud", "storage", "cp", str(source), destination]
        if project:
            command.append(f"--project={project}")
    elif gsutil:
        command = ["gsutil", "cp", str(source), destination]
    else:
        raise SystemExit("upload requires gcloud or gsutil on PATH")
    print("upload", source, "->", destination, flush=True)
    subprocess.run(command, check=True)


def read_metadata_summary(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"path": str(path), "parse_error": True}
    summary: dict[str, Any] = {"path": str(path)}
    for key in [
        "actual_count",
        "target_count",
        "dataset_hash",
        "seed",
        "size",
        "validation",
        "duplicate_stats",
        "semantic_diversity",
        "training_export_diversity",
    ]:
        if key in metadata:
            summary[key] = metadata[key]
    return summary


def source_entry(
    spec: SourceSpec,
    path_text: str | None,
    metadata_text: str | None,
    license_status: str,
    gcs_prefix: str,
    upload: bool,
    project: str,
) -> dict[str, Any] | None:
    if not path_text:
        return {
            "source_id": spec.source_id,
            "slug": spec.slug,
            "display_name": spec.display_name,
            "status": "pending",
            "role": spec.role,
            "sampling_weight": spec.sampling_weight,
            "license_status": license_status,
            "data_format": spec.data_format,
            "governance_evidence": GOVERNANCE_EVIDENCE.get(spec.source_id, {}),
            "reason": "no local approved slice path supplied",
        }
    path = pathlib.Path(path_text).expanduser()
    metadata_path = pathlib.Path(metadata_text).expanduser() if metadata_text else None
    if not path.exists():
        return {
            "source_id": spec.source_id,
            "slug": spec.slug,
            "display_name": spec.display_name,
            "status": "pending",
            "role": spec.role,
            "sampling_weight": spec.sampling_weight,
            "license_status": license_status,
            "data_format": spec.data_format,
            "governance_evidence": GOVERNANCE_EVIDENCE.get(spec.source_id, {}),
            "reason": f"slice path does not exist: {path}",
        }
    if license_status not in APPROVED_LICENSE_STATUSES:
        raise SystemExit(
            f"{spec.source_id} has unapproved license status {license_status!r}; "
            f"approved={sorted(APPROVED_LICENSE_STATUSES)}"
        )
    source_files = source_slice_files(path, spec.source_id)
    if metadata_path and metadata_path.exists():
        reject_compressed(metadata_path, spec.source_id)
    file_entries = []
    for source_file in source_files:
        file_destination = gcs_join(gcs_prefix, spec.slug, source_file.name)
        file_entry = {
            "local_path": str(source_file),
            "gcs_uri": file_destination,
            "bytes": source_file.stat().st_size,
            "records": count_records(source_file),
            "sha256": sha256_file(source_file),
        }
        file_entries.append(file_entry)
    byte_count = sum(int(entry["bytes"]) for entry in file_entries)
    record_count = sum(int(entry["records"]) for entry in file_entries)
    content_hash = file_entries[0]["sha256"] if len(file_entries) == 1 else source_set_hash(source_files)
    destination = (
        gcs_join(gcs_prefix, spec.slug, source_files[0].name)
        if len(source_files) == 1
        else gcs_join(gcs_prefix, spec.slug)
    )
    metadata_destination = None
    metadata_hash = None
    metadata_summary = None
    if metadata_path and metadata_path.exists():
        metadata_hash = sha256_file(metadata_path)
        metadata_destination = gcs_join(gcs_prefix, spec.slug, metadata_path.name)
        metadata_summary = read_metadata_summary(metadata_path)
    if upload:
        for source_file in source_files:
            copy_to_gcs(
                source_file,
                gcs_join(gcs_prefix, spec.slug, source_file.name),
                project,
            )
        if metadata_path and metadata_path.exists() and metadata_destination:
            copy_to_gcs(metadata_path, metadata_destination, project)
    return {
        "source_id": spec.source_id,
        "slug": spec.slug,
        "display_name": spec.display_name,
        "status": "available",
        "role": spec.role,
        "sampling_weight": spec.sampling_weight,
        "license_status": license_status,
        "data_format": spec.data_format,
        "governance_evidence": GOVERNANCE_EVIDENCE.get(spec.source_id, {}),
        "local_path": str(path),
        "gcs_uri": destination,
        "source_file_count": len(file_entries),
        "source_files": file_entries,
        "bytes": byte_count,
        "records": record_count,
        "sha256": content_hash,
        "metadata_local_path": str(metadata_path) if metadata_path and metadata_path.exists() else None,
        "metadata_gcs_uri": metadata_destination,
        "metadata_sha256": metadata_hash,
        "metadata_summary": metadata_summary,
    }


def main() -> int:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        entry = source_entry(
            spec,
            getattr(args, spec.arg_name),
            getattr(args, spec.arg_name + "_metadata"),
            getattr(args, spec.arg_name + "_license_status"),
            args.gcs_prefix,
            args.upload,
            args.project,
        )
        if entry is not None and (args.include_pending or entry["status"] != "pending"):
            entries.append(entry)
    available = [entry for entry in entries if entry["status"] == "available"]
    inventory = {
        "format": "heirloom.source_slice_inventory",
        "version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gcs_prefix": args.gcs_prefix.rstrip("/"),
        "upload_performed": args.upload,
        "approved_license_statuses": sorted(APPROVED_LICENSE_STATUSES),
        "source_count": len(entries),
        "available_source_count": len(available),
        "pending_source_count": len(entries) - len(available),
        "total_bytes": sum(int(entry.get("bytes", 0)) for entry in available),
        "total_records": sum(int(entry.get("records", 0)) for entry in available),
        "sources": entries,
    }
    inventory_path = out_dir / "source-slice-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    if args.upload:
        copy_to_gcs(
            inventory_path,
            gcs_join(args.gcs_prefix, "source-slice-inventory.json"),
            args.project,
        )
    print(
        "source slice inventory "
        f"available={inventory['available_source_count']} "
        f"pending={inventory['pending_source_count']} "
        f"bytes={inventory['total_bytes']} "
        f"records={inventory['total_records']} "
        f"path={inventory_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
