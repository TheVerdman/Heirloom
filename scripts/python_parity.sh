#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv 0.11.28 is required; see https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${HEIRLOOM_PYTHON_ENV:-.venv-parity}"
export UV_CACHE_DIR="${HEIRLOOM_UV_CACHE_DIR:-${UV_CACHE_DIR:-.cache/uv}}"

uv sync --frozen --group parity --no-install-project
uv run --no-sync maturin develop --manifest-path heirloom-python/Cargo.toml --features extension-module
uv run --no-sync python -m pytest python/tests
