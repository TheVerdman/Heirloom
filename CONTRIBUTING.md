# Contributing

Heirloom accepts focused changes that strengthen the tensor/autograd → CUDA → memory-transformer system or its reproducible evidence. New model features should come with a concrete systems reason and are intentionally lower priority than correctness, safety, and testability.

## Setup

1. Install `rustup`; the repository selects Rust 1.95.0 automatically.
2. Run `./scripts/validate.sh` for the CPU/software gate.
3. Install `uv` 0.11.28 only if changing Python parity code, then run `./scripts/python_parity.sh`.

No cloud account or GPU is required for ordinary contributions.

## Change expectations

- Keep public behavior changes small and add a regression test.
- Treat shape, stride, allocation-size, pointer-stride, FFI, CUDA, and NCCL changes as safety-boundary work. Validate dimensions before entering unsafe code and document the invariant at the boundary.
- Do not make hardware tests return success when hardware is absent. Keep them explicitly ignored and use `scripts/test_gpu.sh`.
- Preserve user data and historical evidence. Generated runs belong under ignored `runs/`; small reviewer evidence belongs under `docs/evidence/`.
- Run `cargo fmt --all`, Clippy with `-D warnings`, relevant tests, and Rustdoc for public API changes.
- Keep experimental Padawan, skill/compiler, corpus, and cloud work clearly labeled and out of the default review path.

See `TESTING.md` for exact commands and `docs/API_DOCS.md` for the staged public
API documentation policy.
