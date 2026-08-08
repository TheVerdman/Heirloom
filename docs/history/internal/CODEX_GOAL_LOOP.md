# Codex Goal Loop

Status: working operating loop
Date: 2026-06-16

This loop is for Codex work inside the current Heirloom workspace. It is based
on the repository state as of the QB-native pretraining readiness and Padawan
sidecar documents.

## Current State

Heirloom is a correctness-first Rust tensor/autograd runtime that now reaches a
guarded CUDA/QB memory-transformer hard path. The strongest current evidence is:

- manifest v2 binary shards train/resume/eval/generate on CPU and 4x A100;
- memory-transformer sparse-row updates, SMFT evidence, NCCL, AMP BF16,
  Tensor Core flash attention, and `cp.async` GEMM have passed guarded A100
  gates;
- the release measured-throughput lane is established at roughly `6078.8`
  tokens/sec and `0.00157` dense-core MFU for the target QB memory shape;
- launch-family diagnostics show the main runtime pressure is now dense
  forward/backward launch structure, not flash attention or the existing
  `cp.async` GEMM kernel;
- Padawan Loop is intentionally sidecar-only, with schemas, fixtures, and a
  validator in place, and must not interfere with the pretraining readiness
  lane.

## Primary Goal

Advance Heirloom from a proven hard-path prototype toward a reproducible
pretraining system by choosing the smallest next slice that improves one of:

1. learning sanity under the production tokenizer/materializer path;
2. governed source/materialization readiness;
3. measured CUDA throughput or launch-structure clarity;
4. Padawan sidecar verifier readiness, without touching the base pretraining
   path.

When those compete, prefer the QB-native pretraining readiness exit criteria
over speculative side quests.

## Loop

1. Orient.

   Read `README.md`, `QB_PRETRAINING_READINESS.md`, `PROGRESS.md`, and any
   touched module docs before acting. Treat recorded Vertex job IDs, artifact
   paths, counters, and validation commands as state, not decoration.

2. Pick one bounded slice.

   Choose the next slice from the current bottleneck list:

   - normal/strided RHS Tensor Core GEMM should become an `ldmatrix`/`cp.async`
     instruction-path kernel before it is considered for default routing;
   - remaining `matmul_strided_f32_reference`, `layer_norm`, and high-count
     elementwise families are valid launch-reduction targets;
   - larger approved source slices, materializer progress/throughput, and
     learning sanity ladder runs are readiness targets;
   - Padawan P3 verifier harnesses are safe sidecar targets only after keeping
     production tokenizer, blend, and checkpoint behavior unchanged.

3. Preserve contracts.

   Do not silently change corpus weights, tokenizer reserved-token behavior,
   manifest v2 semantics, checkpoint formats, SMFT rules, A100 gate meanings,
   or Padawan non-interference rules. If a change must touch a contract, update
   the owning document and add a focused validation.

4. Implement narrowly.

   Match existing Rust, shell, Python, and JSON-report patterns. Keep guard
   flags for experimental CUDA paths. Add counters before making performance
   claims. Prefer artifact validators over prose conclusions.

5. Validate locally first.

   Use the smallest relevant gate, then expand only as risk requires:

   ```bash
   cargo fmt --all --check
   cargo check --bin heirloom
   bash -n scripts/run_learning_sanity_ladder.sh
   cargo test --workspace
   cargo clippy --workspace --exclude heirloom-python --all-targets -- -D warnings
   bash scripts/run_learning_sanity_ladder.sh
   cargo run --bin heirloom -- readiness validate-learning-sanity \
     --manifest tests/fixtures/learning_sanity/ladder-valid.json
   cargo run --bin heirloom -- padawan validate \
     --episodes padawan/fixtures/episode_valid.jsonl \
     --artifact-root padawan/fixtures
   cargo run --bin heirloom -- padawan verify \
     --episodes padawan/fixtures/episode_valid.jsonl \
     --artifact-root padawan/fixtures
   ```

   CUDA and Vertex gates remain opt-in. For throughput work, require launch
   families and compare against the latest measured baseline before calling a
   change a win.

6. Record evidence.

   Update `PROGRESS.md` and the relevant readiness/design document with:

   - exact command or launcher;
   - pass/fail result;
   - artifact path;
   - before/after counters;
   - what remains guarded, experimental, or pending.

7. Decide the next move.

   If the slice improved the target metric and preserved contracts, consider
   promoting it toward the default path. If it only cleaned up launch count,
   keep the claim modest. If it regressed throughput, keep it guarded or revert
   only my own changes.

## Stop Rules

Stop and ask before:

- using scarce remote A100/Vertex budget;
- downloading or staging new external data;
- changing production corpus weights or tokenizer defaults;
- mutating a base checkpoint in place;
- mixing Padawan traces into the current 20B pretraining target;
- taking destructive filesystem or git actions.

## Working Memory

The current best next engineering instincts are:

- performance: replace the experimental normal-RHS staged GEMM with a true
  instruction-path kernel, or attack `matmul_strided_f32_reference`,
  `layer_norm`, and fused elementwise chains;
- readiness: materializer/source-scale progress is observable, and the
  learning sanity ladder now has a native Rust evidence validator plus a tiny
  local producer for dense fixed-shard, memory-layers-disabled, and
  exact-memory reports with SMFT disabled/enabled plus product-key parity. The
  LR/grad-accumulation sweep now has a native Rust nested-manifest validator
  and checked fixture, and the 32K tokenizer/blend hard-path summary now has a
  native Rust artifact-bundle validator, checked fixture, auto-emitted
  hard-path manifest support, and validated bounded local digit-isolated 32K
  rehearsal evidence. Vertex job `7741721827130474496` passed the remote quick
  Rust gate with explicit learning-sanity fixture validation and uploaded
  `learning-sanity-fixtures/` reports. Actual paid 4x A100 LR/grad sweep
  evidence and longer production-data 32K blend evidence remain pending before
  larger quality claims;
- Padawan: P2 validation and the first P3 local verifier harness are native
  Rust CLI paths under `heirloom padawan`; next sidecar work should add more
  hard-verifier fixtures or wait for the learning sanity ladder before running
  local pilots. Keep every artifact under `padawan/` or `runs/padawan-loop/`.

The project rewards honest measurement. Do not round a cleanup into a
breakthrough.
