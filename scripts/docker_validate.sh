#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

image="${1:-heirloom:dev}"

docker build -t "$image" .
docker run --rm "$image" ./scripts/validate.sh
