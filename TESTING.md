# Testing

## CPU and software

```bash
./scripts/test_cpu.sh
./scripts/validate.sh
```

`test_cpu.sh` runs the Rust workspace. Hardware integration tests are compiled but reported as ignored. `validate.sh` is the advertised non-hardware gate: formatting, the Rust 1.95.0 drift check, fixture validation, CPU tests, Clippy, examples, and bounded CLI workflows.

## Python and PyTorch parity

Install `uv` 0.11.28, then run:

```bash
./scripts/python_parity.sh
```

The script creates `.venv-parity`, syncs the committed `uv.lock` under Python 3.13.12, installs CPU-only PyTorch, builds the PyO3 extension, and runs `python/tests`. It does not use or replace a developer's `.venv`.

Python lint is:

```bash
.venv-parity/bin/ruff check python scripts tools
```

## CUDA and NCCL

```bash
./scripts/test_gpu.sh cuda
./scripts/test_gpu.sh nccl
```

These commands run only explicitly ignored hardware tests. CUDA tests require a working CUDA Driver API; BF16 tests additionally require BF16 Tensor Core support. NCCL tests require a loadable NCCL library. Missing requirements fail the dedicated run rather than being counted as success.

No CUDA or NCCL command is part of the CPU gate. See `docs/evidence/gpu-validation.md` for the limits of retained historical hardware evidence.

## Documentation, security, and containers

```bash
cargo test --workspace --doc
cargo audit
shellcheck scripts/check_toolchain.sh scripts/docker_validate.sh scripts/gpu_smoke.sh \
  scripts/python_parity.sh scripts/test_cpu.sh scripts/test_gpu.sh scripts/validate.sh
./scripts/docker_validate.sh
```

CI runs Rust validation, Python parity/Ruff, ShellCheck, RustSec, and Docker validation as distinct jobs so failures identify the affected contract.
