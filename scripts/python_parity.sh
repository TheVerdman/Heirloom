#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV="${HEIRLOOM_PYTHON_VENV:-.venv}"
PYTHON="$VENV/bin/python"
MATURIN="$VENV/bin/maturin"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python interpreter at $PYTHON" >&2
  exit 1
fi

if [[ ! -x "$MATURIN" ]]; then
  echo "missing maturin at $MATURIN" >&2
  exit 1
fi

"$MATURIN" develop --manifest-path heirloom-python/Cargo.toml --features extension-module
"$PYTHON" -m pytest python/tests
