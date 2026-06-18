# Padawan Loop

Sidecar artifacts for Padawan Loop post-training and continual-learning
experiments.

This directory is intentionally separate from the synchronous QB-native
pretraining readiness path. Nothing here changes the 20B-token materializer,
tokenizer defaults, corpus blend weights, or `train-memory-lm` behavior.

## Contents

- `schemas/`: JSON Schema-style contracts for v0 Padawan episode records,
  teacher briefs, verifier results, and artifact bundles.
- `fixtures/`: one tiny valid episode with teacher context, compressed
  rationale, artifact hashes, hidden-test audit metadata, and SMFT evidence.
- `heirloom padawan validate`: native Rust contract validator for sidecar
  episodes and artifact bundles.
- `heirloom padawan verify`: native Rust P3 verifier harness for code patches,
  JSON/tool-call records, evidence/final-validation support, and memory/SMFT
  telemetry.

## Validation

Run the sidecar validator:

```bash
cargo run --bin heirloom -- padawan validate \
  --episodes padawan/fixtures/episode_valid.jsonl \
  --artifact-root padawan/fixtures

cargo run --bin heirloom -- padawan verify \
  --episodes padawan/fixtures/episode_valid.jsonl \
  --artifact-root padawan/fixtures
```

Expected output:

```text
validated padawan episodes=1 sft_eligible=1 smft_eligible=1 max_guidance_level=1
padawan verifier harness status=passed episodes=1 families=4 passed=4 failed=0 skipped=0
```

The validators are implemented in the Heirloom Rust CLI. The contract validator
checks that teacher tokens are masked from loss, artifact references resolve,
content hashes match, hidden-test audit metadata is opaque, compressed
rationales stay compressed, and SMFT-eligible episodes include
foreground/background lift plus replay stability. The verifier harness first
runs the contract validator, then grades the fixture with deterministic local
verifier families without changing the pretraining path.
