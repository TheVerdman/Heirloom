# Heirloom Skill Compiler

`heirloom-skill` treats `SKILL.md` as procedural source code. A skill is parsed into typed `SkillIr`, linted, compiled into a deterministic `.hskill` object file, routed at runtime, sliced into compact context, checked against symbolic tool policy, validated, logged as JSONL traces, and exported into SFT-style records.

This is not better prompting. It is a Rust-native compiler and training-data pipeline for turning scattered human-written agent procedures into durable, testable, reusable procedural competence inside Heirloom.

## Quickstart

```text
cargo run -p heirloom-skill --bin heirloom-skill -- compile \
  --source heirloom-skill/examples/skills/spreadsheets/SKILL.md \
  --out build/skills/spreadsheets.hskill \
  --registry heirloom-skill/examples/registry/core_skills.json

cargo run -p heirloom-skill --bin heirloom-skill -- route \
  --skills build/skills \
  --request "Turn this CSV into a styled Excel dashboard" \
  --top-k 3

cargo run -p heirloom-skill --bin heirloom-skill -- context \
  --skills build/skills \
  --request "Turn this CSV into a styled Excel dashboard" \
  --budget-chars 6000

cargo run -p heirloom-skill --bin heirloom-skill -- synthetic-tasks \
  --skill build/skills/spreadsheets.hskill \
  --out build/tasks/spreadsheets.jsonl

cargo run -p heirloom-skill --bin heirloom-skill -- eval \
  --skills build/skills \
  --out reports/skill_eval.json

cargo run -p heirloom-skill --bin heirloom-skill -- sft-export \
  --traces build/traces.jsonl \
  --out build/sft_cloud.jsonl

cargo run -p heirloom-skill --bin heirloom-skill -- trace audit \
  --traces build/traces.jsonl \
  --out reports/trace_hygiene.json

cargo run -p heirloom-skill --bin heirloom-skill -- registry lint \
  --registry heirloom-skill/examples/registry/core_skills.json
```

## Artifact Model

`SKILL.md` is source text. `SkillIr` is the typed intermediate representation. `.hskill` is the compiled object file for the Heirloom agent runtime.

The `.hskill` container uses:

- magic bytes `HSKILL01`
- format version `1`
- sorted length-prefixed sections
- canonical JSON manifest and IR sections
- little-endian `f32` embedding matrices
- final SHA-256 checksum over the payload

Hard constraints, tool policy, preflight steps, and validators remain symbolic. Embeddings are only used for routing and retrieval.

## Routing And Context

The router combines trigger phrase boosts, hashed bag-of-words cosine similarity over triggers/examples/sections, and negative-trigger penalties. Runtime context then retrieves only the compact slices needed for execution: activated skills, hard constraints, required and forbidden tools, validation steps, preflight steps, relevant triggers, and nearest examples.

## Traces To SFT

Successful executions can be logged as JSONL `SkillTrace` records. `sft-export` converts successful and export-eligible traces into chat-style JSONL records for future skill-cloud training. Failed traces are skipped by default and can be included with `--include-failures` for future contrastive datasets.

SFT export is hygiene-gated. `write_trace` records `metadata.hygiene_v1`, `trace audit` emits an audit report, and `sft-export` recomputes hygiene before writing records. Blocking findings such as API tokens, private keys, `.env` references, and secret assignments are skipped by default; warning findings such as local paths and email-like strings are recorded but do not block export.
