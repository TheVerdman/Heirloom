#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

expected="$(awk -F'"' '/^channel = / { print $2; exit }' rust-toolchain.toml)"
if [[ -z "$expected" ]]; then
  printf 'rust-toolchain.toml does not declare a channel\n' >&2
  exit 1
fi

expected_rust_version="${expected%.*}"
for manifest in Cargo.toml heirloom-kernels/Cargo.toml heirloom-python/Cargo.toml heirloom-skill/Cargo.toml; do
  declared="$(awk -F'"' '/^rust-version = / { print $2; exit }' "$manifest")"
  if [[ "$declared" != "$expected_rust_version" ]]; then
    printf '%s rust-version mismatch: expected %s, got %s\n' \
      "$manifest" "$expected_rust_version" "${declared:-missing}" >&2
    exit 1
  fi
done

if ! grep -Fq "FROM rust:${expected}-bookworm" Dockerfile; then
  printf 'Dockerfile is not pinned to rust:%s-bookworm\n' "$expected" >&2
  exit 1
fi

if ! grep -Fq "uses: dtolnay/rust-toolchain@${expected}" .github/workflows/ci.yml; then
  printf 'CI is not pinned to dtolnay/rust-toolchain@%s\n' "$expected" >&2
  exit 1
fi

actual="$(rustc --version | awk '{print $2}')"

if [[ "$actual" != "$expected" ]]; then
  printf 'unsupported Rust toolchain: expected %s, got %s\n' "$expected" "$actual" >&2
  printf 'rust-toolchain.toml should select the supported toolchain automatically.\n' >&2
  exit 1
fi

printf 'Rust toolchain %s aligned across rust-toolchain.toml, Cargo, Docker, and CI\n' "$actual"
