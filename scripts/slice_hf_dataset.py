#!/usr/bin/env python3
"""Create deterministic private source slices from Hugging Face datasets.

The output is an uncompressed JSONL file with a `text` field, plus a report
that records source files, hashes, byte counts, selection settings, and upload
locations. Secrets are only read from the environment or an env file at runtime
and are never written to the report.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fnmatch
import gzip
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any


DEFAULT_GCS_PREFIX = (
    "gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/"
    "heirloom/qb-native-pretraining-v1/source-slices"
)
DEFAULT_VECL_QB_ENV = "/Users/andrewverdiramo/Desktop/VECL-QB/.env"
COMPRESSED_SUFFIXES = (".gz", ".zst", ".zstd")
SECRET_MARKERS = (
    "begin rsa private key",
    "begin openssh private key",
    "password=",
    "api_key=",
    "secret_key=",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id.")
    parser.add_argument("--source-id", required=True, help="Governed source id.")
    parser.add_argument("--slug", required=True, help="Stable destination slug.")
    parser.add_argument(
        "--preset",
        choices=["none", "dolma-qb", "dolmino-qb"],
        default="none",
        help="Apply source-specific include/exclude defaults.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision/branch/commit to read.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Repo file glob to include. Repeatable. Default includes *.jsonl*.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Repo file glob to exclude. Repeatable.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Explicit repo file path. Skips tree listing when supplied.",
    )
    parser.add_argument(
        "--url-list",
        help="Local text file containing one source URL per line.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Explicit source URL. Repeatable.",
    )
    parser.add_argument(
        "--local-file",
        action="append",
        default=[],
        help="Local fixture/input file for smoke tests or already-downloaded HF shards.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output uncompressed JSONL slice path, or output directory when --shard-output-bytes is set.",
    )
    parser.add_argument(
        "--shard-output-bytes",
        type=int,
        default=0,
        help="Rotate selected output into part-XXXXX.jsonl files under --out once each part reaches this many bytes. 0 writes one file.",
    )
    parser.add_argument(
        "--report",
        help="Output report path. Defaults to <out>.report.json.",
    )
    parser.add_argument(
        "--work-dir",
        help="Download/cache directory. Defaults to a temp dir next to the output.",
    )
    parser.add_argument(
        "--target-text-bytes",
        type=int,
        default=0,
        help="Stop after selected text bytes reach this limit. 0 means no byte limit.",
    )
    parser.add_argument(
        "--target-tokens-estimate",
        type=int,
        default=0,
        help="Stop after estimated tokens reach this limit. Uses --chars-per-token.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Approximate UTF-8 text bytes per token for target-token slicing.",
    )
    parser.add_argument(
        "--max-input-files",
        type=int,
        default=0,
        help="Maximum matching input files to scan. 0 means no explicit limit.",
    )
    parser.add_argument("--seed", type=int, default=1107, help="Deterministic seed.")
    parser.add_argument(
        "--no-shuffle-files",
        action="store_true",
        help="Use lexicographic file order instead of deterministic shuffled order.",
    )
    parser.add_argument(
        "--record-sample-mod",
        type=int,
        default=1,
        help="Keep only records whose deterministic hash modulo this value matches.",
    )
    parser.add_argument(
        "--record-sample-remainder",
        type=int,
        default=None,
        help="Remainder for --record-sample-mod. Defaults to seed %% mod.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="JSONL field to use as document text.",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Optional JSONL field to preserve as source_record_id.",
    )
    parser.add_argument(
        "--metadata-fields",
        default="source,version,created,added,metadata",
        help="Comma-separated JSON fields to preserve in output metadata.",
    )
    parser.add_argument(
        "--min-text-bytes",
        type=int,
        default=64,
        help="Reject records with less text than this.",
    )
    parser.add_argument(
        "--max-text-bytes",
        type=int,
        default=0,
        help="Reject records with more text than this. 0 disables the check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected files and write report without downloading records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output slice/report.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload output JSONL and report to GCS.",
    )
    parser.add_argument(
        "--gcs-prefix",
        default=DEFAULT_GCS_PREFIX,
        help="Destination GCS prefix for uploaded slices.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("PROJECT_ID", ""),
        help="Optional GCP project for gcloud storage commands.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face token.",
    )
    parser.add_argument(
        "--hf-env-file",
        default=os.environ.get("HF_ENV_FILE", DEFAULT_VECL_QB_ENV),
        help="Optional env file to read if --hf-token-env is unset.",
    )
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "dolma-qb":
        if not args.include:
            args.include.extend(
                [
                    "dolma-v1_7/books/*.json.gz",
                    "dolma-v1_7/c4-filtered/*.json.gz",
                    "dolma-v1_7/cc_en_head/*.json.gz",
                    "dolma-v1_7/cc_news_head/*.json.gz",
                    "dolma-v1_7/cc_news_middle/*.json.gz",
                    "dolma-v1_7/falcon-refinedweb-filtered/*.json.gz",
                    "dolma-v1_7/pes2o/*.json.gz",
                    "dolma-v1_7/proof_pile_2-algebraic_stack/*.json.gz",
                    "dolma-v1_7/proof_pile_2-open_web_math/*.json.gz",
                    "dolma-v1_7/redpajama-arxiv/*.json.gz",
                    "dolma-v1_7/redpajama-stackexchange/*.json.gz",
                    "dolma-v1_7/wiki/*.json.gz",
                    "dolma-v1_7/wikiref_megawika/*.json.gz",
                ]
            )
        for pattern in [
            "*reddit*",
            "*cc_en_tail*",
        ]:
            if pattern not in args.exclude:
                args.exclude.append(pattern)
    if args.preset == "dolmino-qb":
        if not args.include:
            args.include.extend(["data/**/*.jsonl.zst"])
        for pattern in [
            "*adult_content*",
        ]:
            if pattern not in args.exclude:
                args.exclude.append(pattern)


def load_env_file(path_text: str | None) -> dict[str, str]:
    if not path_text:
        return {}
    path = pathlib.Path(path_text).expanduser()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def hf_token(args: argparse.Namespace) -> tuple[str | None, str]:
    token = os.environ.get(args.hf_token_env)
    if token:
        return token, f"env:{args.hf_token_env}"
    values = load_env_file(args.hf_env_file)
    token = values.get(args.hf_token_env) or values.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token, f"env_file:{pathlib.Path(args.hf_env_file).name}"
    return None, "none"


def http_json(url: str, token: str | None) -> Any:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    request.add_header("User-Agent", "heirloom-qb-source-slicer/1")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(f"HTTP {err.code} for {url}: {body}") from err


def hf_tree_url(repo_id: str, revision: str) -> str:
    repo = urllib.parse.quote(repo_id, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/datasets/{repo}/tree/{rev}?recursive=1"


def hf_resolve_url(repo_id: str, revision: str, path: str) -> str:
    repo = urllib.parse.quote(repo_id, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    file_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{file_path}"


def default_includes() -> list[str]:
    return [
        "*.jsonl",
        "*.json.gz",
        "*.jsonl.gz",
        "*.jsonl.zst",
        "*.jsonl.zstd",
        "*.parquet",
    ]


def list_hf_files(args: argparse.Namespace, token: str | None) -> list[dict[str, Any]]:
    if args.local_file:
        return [
            {
                "path": pathlib.Path(path).name,
                "local_path": str(pathlib.Path(path).expanduser()),
                "size": pathlib.Path(path).expanduser().stat().st_size,
            }
            for path in args.local_file
        ]
    urls: list[str] = []
    if args.url_list:
        url_list_path = pathlib.Path(args.url_list).expanduser()
        urls.extend(
            line.strip()
            for line in url_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    urls.extend(args.url)
    if urls:
        items = []
        for url in urls:
            path = urllib.parse.urlparse(url).path.lstrip("/") or url
            items.append({"path": path, "url": url, "size": None})
        includes = args.include or default_includes()
        return [
            item
            for item in items
            if any(fnmatch.fnmatch(item["path"], pattern) for pattern in includes)
            and not any(fnmatch.fnmatch(item["path"], pattern) for pattern in args.exclude)
        ]
    if args.file:
        return [{"path": path, "size": None} for path in args.file]
    tree = http_json(hf_tree_url(args.repo_id, args.revision), token)
    if not isinstance(tree, list):
        raise SystemExit("unexpected Hugging Face tree response")
    files = [
        item
        for item in tree
        if isinstance(item, dict) and item.get("type") == "file" and item.get("path")
    ]
    includes = args.include or default_includes()
    matched = []
    for item in files:
        path = item["path"]
        if not any(fnmatch.fnmatch(path, pattern) for pattern in includes):
            continue
        if any(fnmatch.fnmatch(path, pattern) for pattern in args.exclude):
            continue
        matched.append(item)
    return matched


def deterministic_file_order(
    files: list[dict[str, Any]], seed: int, no_shuffle: bool
) -> list[dict[str, Any]]:
    ordered = sorted(files, key=lambda item: item["path"])
    if not no_shuffle:
        random.Random(seed).shuffle(ordered)
    return ordered


def download_hf_file(
    args: argparse.Namespace,
    item: dict[str, Any],
    token: str | None,
    cache_dir: pathlib.Path,
) -> pathlib.Path:
    local_path = item.get("local_path")
    if local_path:
        return pathlib.Path(local_path)
    cache_name = hashlib.sha256(
        f"{args.repo_id}@{args.revision}:{item.get('url', item['path'])}".encode("utf-8")
    ).hexdigest()[:16] + "-" + pathlib.Path(item["path"]).name
    destination = cache_dir / cache_name
    if destination.exists():
        return destination
    url = item.get("url") or hf_resolve_url(args.repo_id, args.revision, item["path"])
    request = urllib.request.Request(url)
    if token and not item.get("url"):
        request.add_header("Authorization", f"Bearer {token}")
    request.add_header("User-Agent", "heirloom-qb-source-slicer/1")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with destination.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(f"HTTP {err.code} while downloading {item['path']}: {body}") from err
    return destination


def iter_lines(path: pathlib.Path) -> Iterator[str]:
    name = path.name
    if name.endswith((".zst", ".zstd")):
        zstd = shutil.which("zstd")
        if not zstd:
            raise SystemExit("reading .zst input requires the zstd binary on PATH")
        process = subprocess.Popen(
            [zstd, "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        exhausted = False
        status = 0
        try:
            for line in process.stdout:
                yield line
            exhausted = True
        finally:
            if not exhausted and process.poll() is None:
                process.terminate()
                try:
                    status = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    status = process.wait()
            else:
                status = process.wait()
            stderr = process.stderr.read() if process.stderr else ""
            if exhausted and status != 0:
                raise SystemExit(f"zstd failed for {path}: {stderr[-500:]}")
    elif name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle


def iter_records(path: pathlib.Path) -> Iterator[tuple[int, str, Any | None]]:
    if path.name.endswith(".parquet"):
        try:
            import pyarrow.parquet as parquet
        except ImportError as err:
            raise SystemExit(
                "reading .parquet input requires pyarrow; install it in the active "
                "Python environment or set PYTHONPATH to a pyarrow target directory"
            ) from err
        parquet_file = parquet.ParquetFile(path)
        ordinal = 0
        for batch in parquet_file.iter_batches(batch_size=1024):
            for value in batch.to_pylist():
                ordinal += 1
                raw = json.dumps(value, sort_keys=True, default=str)
                yield ordinal, raw, value
        return
    for line_no, line in enumerate(iter_lines(path), start=1):
        yield line_no, line, None


def record_hash(seed: int, source_path: str, line_no: int, line: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(source_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(line_no).encode("ascii"))
    digest.update(b"\0")
    digest.update(line[:256].encode("utf-8", errors="replace"))
    return int.from_bytes(digest.digest()[:8], "big")


def should_keep_record(args: argparse.Namespace, source_path: str, line_no: int, line: str) -> bool:
    modulus = max(1, args.record_sample_mod)
    if modulus == 1:
        return True
    remainder = args.record_sample_remainder
    if remainder is None:
        remainder = args.seed % modulus
    return record_hash(args.seed, source_path, line_no, line) % modulus == remainder


def has_secret_like_text(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in SECRET_MARKERS)


def has_pathological_repetition(text: str) -> bool:
    if len(text) < 256:
        return False
    sample = text[:4096]
    unique = len(set(sample))
    if unique <= 8:
        return True
    chunks = [sample[index : index + 32] for index in range(0, len(sample), 32)]
    if not chunks:
        return False
    most_common = max(chunks.count(chunk) for chunk in set(chunks))
    return most_common / len(chunks) > 0.35


def render_record(
    args: argparse.Namespace,
    item: dict[str, Any],
    line_no: int,
    line: str,
    parsed_value: Any | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if parsed_value is None:
        raw = line.strip()
        if not raw:
            return None, "empty"
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            text = raw
            value = {}
        else:
            if not isinstance(value, dict):
                return None, "malformed"
            text_value = value.get(args.text_field)
            if not isinstance(text_value, str):
                return None, "missing_text"
            text = text_value
    else:
        value = parsed_value
        if not isinstance(value, dict):
            return None, "malformed"
        text_value = value.get(args.text_field)
        if not isinstance(text_value, str):
            return None, "missing_text"
        text = text_value
    text_bytes = len(text.encode("utf-8"))
    if text_bytes < args.min_text_bytes:
        return None, "too_small"
    if args.max_text_bytes and text_bytes > args.max_text_bytes:
        return None, "too_large"
    if has_secret_like_text(text):
        return None, "secret_like"
    if has_pathological_repetition(text):
        return None, "repetition"
    metadata_fields = [field.strip() for field in args.metadata_fields.split(",") if field.strip()]
    metadata = {
        field: value[field]
        for field in metadata_fields
        if isinstance(value, dict) and field in value
    }
    source_record_id = value.get(args.id_field) if isinstance(value, dict) else None
    record = {
        "text": text,
        "source_id": args.source_id,
        "source_dataset": args.repo_id,
        "source_revision": args.revision,
        "source_file": item["path"],
        "source_line": line_no,
    }
    if item.get("url"):
        record["source_url"] = item["url"]
    if isinstance(source_record_id, str):
        record["source_record_id"] = source_record_id
    if metadata:
        record["metadata"] = metadata
    return record, None


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_set(paths: list[pathlib.Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


class SliceOutputWriter:
    def __init__(self, out_path: pathlib.Path, shard_output_bytes: int, slug: str):
        self.out_path = out_path
        self.shard_output_bytes = max(0, shard_output_bytes)
        self.slug = slug
        self.handle = None
        self.current_path: pathlib.Path | None = None
        self.current_bytes = 0
        self.index = 0
        self.paths: list[pathlib.Path] = []

    @property
    def sharded(self) -> bool:
        return self.shard_output_bytes > 0

    def __enter__(self) -> "SliceOutputWriter":
        if self.sharded:
            self.out_path.mkdir(parents=True, exist_ok=True)
            self._open_next_part()
        else:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.current_path = self.out_path
            self.paths.append(self.out_path)
            self.handle = self.out_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _open_next_part(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.current_path = self.out_path / f"{self.slug}-part-{self.index:05}.jsonl"
        self.index += 1
        self.current_bytes = 0
        self.paths.append(self.current_path)
        self.handle = self.current_path.open("w", encoding="utf-8")

    def write_record(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        encoded_len = len(line.encode("utf-8"))
        if (
            self.sharded
            and self.current_bytes > 0
            and self.current_bytes + encoded_len > self.shard_output_bytes
        ):
            self._open_next_part()
        assert self.handle is not None
        self.handle.write(line)
        self.current_bytes += encoded_len

    def output_files(self) -> list[dict[str, Any]]:
        return [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in self.paths
            if path.exists()
        ]


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


def reached_target(args: argparse.Namespace, text_bytes: int) -> bool:
    if args.target_text_bytes and text_bytes >= args.target_text_bytes:
        return True
    if args.target_tokens_estimate:
        estimate = int(text_bytes / max(args.chars_per_token, 0.1))
        return estimate >= args.target_tokens_estimate
    return False


def main() -> int:
    args = parse_args()
    apply_preset(args)
    out_path = pathlib.Path(args.out).expanduser()
    report_path = pathlib.Path(args.report).expanduser() if args.report else out_path.with_suffix(out_path.suffix + ".report.json")
    if args.shard_output_bytes and out_path.exists() and any(out_path.glob("*.jsonl")) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"sharded output directory contains JSONL files; use --overwrite: {out_path}")
    if not args.shard_output_bytes and out_path.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"output exists; use --overwrite: {out_path}")
    if report_path.exists() and not args.overwrite:
        raise SystemExit(f"report exists; use --overwrite: {report_path}")
    if args.shard_output_bytes:
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.local_file:
        token, token_source = None, "not_needed:local_file"
    elif args.url_list or args.url:
        token, token_source = None, "not_needed:url_input"
    else:
        token, token_source = hf_token(args)
    files = list_hf_files(args, token)
    ordered_files = deterministic_file_order(files, args.seed, args.no_shuffle_files)
    if args.max_input_files:
        ordered_files = ordered_files[: args.max_input_files]
    if not ordered_files:
        raise SystemExit("no matching input files")
    work_dir = (
        pathlib.Path(args.work_dir).expanduser()
        if args.work_dir
        else out_path.parent / ".hf-slice-cache"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(tempfile.mkdtemp(prefix="downloads-", dir=work_dir))
    started = dt.datetime.now(dt.timezone.utc)
    stats: dict[str, Any] = {
        "scanned_files": 0,
        "downloaded_compressed_bytes": 0,
        "scanned_records": 0,
        "selected_records": 0,
        "selected_text_bytes": 0,
        "rejections": {},
    }
    source_files: list[dict[str, Any]] = []
    writer = None if args.dry_run else SliceOutputWriter(out_path, args.shard_output_bytes, args.slug)
    try:
        with writer if writer is not None else contextlib.nullcontext():
            for item in ordered_files:
                if reached_target(args, stats["selected_text_bytes"]):
                    break
                local = download_hf_file(args, item, token, cache_dir) if not args.dry_run else None
                file_report: dict[str, Any] = {
                    "path": item["path"],
                    "url": item.get("url"),
                    "repo_size": item.get("size"),
                    "local_path": str(local) if local else None,
                }
                if local:
                    file_report["compressed_bytes"] = local.stat().st_size
                    file_report["compressed_sha256"] = sha256_file(local)
                    stats["downloaded_compressed_bytes"] += local.stat().st_size
                stats["scanned_files"] += 1
                file_selected = 0
                file_scanned = 0
                if local:
                    for line_no, line, parsed_value in iter_records(local):
                        if reached_target(args, stats["selected_text_bytes"]):
                            break
                        file_scanned += 1
                        stats["scanned_records"] += 1
                        if not should_keep_record(args, item["path"], line_no, line):
                            continue
                        record, rejection = render_record(args, item, line_no, line, parsed_value)
                        if rejection:
                            stats["rejections"][rejection] = stats["rejections"].get(rejection, 0) + 1
                            continue
                        assert record is not None
                        assert writer is not None
                        writer.write_record(record)
                        stats["selected_records"] += 1
                        file_selected += 1
                        stats["selected_text_bytes"] += len(record["text"].encode("utf-8"))
                file_report["scanned_records"] = file_scanned
                file_report["selected_records"] = file_selected
                source_files.append(file_report)
    finally:
        pass
    output_files = [] if writer is None else writer.output_files()
    output_paths = [pathlib.Path(entry["path"]) for entry in output_files]
    output_sha = (
        sha256_file_set(output_paths)
        if args.shard_output_bytes and output_paths
        else sha256_file(out_path) if out_path.exists() and not args.dry_run else None
    )
    output_bytes = sum(int(entry["bytes"]) for entry in output_files) if output_files else (out_path.stat().st_size if out_path.exists() and not args.dry_run else 0)
    gcs_slice_uri = None
    gcs_report_uri = None
    uploaded_output_files = []
    if args.upload and not args.dry_run:
        gcs_slice_uri = gcs_join(args.gcs_prefix, args.slug, out_path.name) if not args.shard_output_bytes else gcs_join(args.gcs_prefix, args.slug)
        gcs_report_uri = gcs_join(args.gcs_prefix, args.slug, report_path.name)
        if args.shard_output_bytes:
            for entry in output_files:
                local_path = pathlib.Path(entry["path"])
                destination = gcs_join(args.gcs_prefix, args.slug, local_path.name)
                copy_to_gcs(local_path, destination, args.project)
                uploaded_output_files.append({**entry, "gcs_uri": destination})
        else:
            copy_to_gcs(out_path, gcs_slice_uri, args.project)
            uploaded_output_files.append({**(output_files[0] if output_files else {}), "gcs_uri": gcs_slice_uri})
    finished = dt.datetime.now(dt.timezone.utc)
    report = {
        "format": "heirloom.hf_source_slice_report",
        "version": 1,
        "created_at_utc": finished.isoformat(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "source_id": args.source_id,
        "slug": args.slug,
        "output_path": None if args.dry_run else str(out_path),
        "output_gcs_uri": gcs_slice_uri,
        "output_bytes": output_bytes,
        "output_sha256": output_sha,
        "output_sharded": bool(args.shard_output_bytes),
        "output_files": uploaded_output_files if uploaded_output_files else output_files,
        "token_source": token_source,
        "hf_token_redacted": bool(token),
        "settings": {
            "include": args.include or default_includes(),
            "exclude": args.exclude,
            "seed": args.seed,
            "shuffle_files": not args.no_shuffle_files,
            "target_text_bytes": args.target_text_bytes,
            "target_tokens_estimate": args.target_tokens_estimate,
            "chars_per_token": args.chars_per_token,
            "record_sample_mod": args.record_sample_mod,
            "record_sample_remainder": args.record_sample_remainder,
            "min_text_bytes": args.min_text_bytes,
            "max_text_bytes": args.max_text_bytes,
            "dry_run": args.dry_run,
        },
        "stats": stats,
        "estimated_selected_tokens": int(stats["selected_text_bytes"] / max(args.chars_per_token, 0.1)),
        "selected_files": source_files,
        "elapsed_seconds": (finished - started).total_seconds(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.upload and not args.dry_run and gcs_report_uri:
        copy_to_gcs(report_path, gcs_report_uri, args.project)
    print(
        "hf source slice "
        f"repo={args.repo_id} "
        f"source_id={args.source_id} "
        f"records={stats['selected_records']} "
        f"text_bytes={stats['selected_text_bytes']} "
        f"estimated_tokens={report['estimated_selected_tokens']} "
        f"out={out_path if not args.dry_run else '<dry-run>'} "
        f"report={report_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
