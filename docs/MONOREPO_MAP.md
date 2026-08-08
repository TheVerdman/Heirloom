# Monorepo map

Heirloom grew as an R&D workspace. The repository remains a monorepo to preserve useful work and provenance, but only one path is the primary portfolio story: tensor/autograd → CUDA kernels → memory transformer.

| Path | Classification | Why it exists | Primary review path? |
| --- | --- | --- | --- |
| `src/tensor.rs`, `src/tensor/` | Core | Tensor metadata, storage views, operations, and reverse-mode autograd | Yes |
| `src/dispatch.rs`, `heirloom-kernels/` | Core | Checked CPU boundary, CUDA Driver/NCCL wrappers, and PTX kernels | Yes |
| `src/nn.rs`, `src/checkpoint.rs`, `src/memory_transformer.rs` | Core | Modules, optimizers, persistence, and the memory-transformer integration target | Yes |
| `tests/`, `examples/` | Supporting | Contract, regression, parity-fixture, and runnable evidence | Yes, selectively |
| `heirloom-python/`, `python/`, `tools/generate_pytorch_fixtures.py` | Supporting | Thin live PyTorch parity harness | Evidence only |
| `scripts/validate.sh`, `scripts/test_*.sh`, Docker and CI files | Supporting | Reproducible validation and hardware classification | Evidence only |
| `src/bin/heirloom.rs`, `src/bin/heirloom/` | Supporting | Thin CLI dispatcher plus responsibility-scoped training, inference, data, GPU, and experimental handlers | Optional |
| `src/data.rs`, `src/tokenizer.rs` | Supporting | Bounded data/tokenizer path needed to exercise training and checkpoint flows | Optional |
| `heirloom-skill/` | Experimental | Separate SKILL.md compiler/runtime investigation | No |
| `padawan/`, `docs/experimental/padawan/` | Experimental | Agent/verifier workflow research | No |
| `docs/experimental/qb/`, corpus and GCP scripts | Experimental infrastructure | Governed-corpus and cloud-scale validation work retained for provenance | No |
| `docs/history/` | Historical | Work log, investigations, and frozen milestone notes | No |

Experimental does not mean disposable or fictitious. It means the component is not required to understand, build, or evaluate the core ML runtime and should not be interpreted as a finished product surface.
