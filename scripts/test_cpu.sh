#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Hardware tests are explicitly ignored and reported as such by this command.
cargo test --workspace
