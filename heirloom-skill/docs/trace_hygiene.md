# Trace Hygiene And Safe SFT Export

Skill traces are potential training data, so they must be audited before export.
The hygiene layer is deterministic, Rust-only, and intentionally conservative.

## Default Policy

- Trace logging is allowed for debugging evidence.
- `write_trace` validates trace shape and adds `metadata.hygiene_v1`.
- SFT export recomputes hygiene from the trace before writing records.
- Blocking findings are skipped by default and are never exported in this slice.
- Warning findings are recorded but do not block export.

## Blocking Detectors

- OpenAI-style `sk-...` tokens
- GitHub-style `ghp_...`, `gho_...`, `ghu_...`, `ghs_...`, `ghr_...`, and `github_pat_...` tokens
- AWS access keys beginning with `AKIA` or `ASIA`
- private key blocks
- `.env` references
- `password=`, `token=`, `secret=`, and `api_key=` assignments

## Warning Detectors

- email-like strings
- phone-like digit sequences
- local absolute paths
- user-home paths

Audit output does not include raw matched text. It records the field path, kind,
severity, message, and a SHA-256 fingerprint of the matched content.

## Commands

```text
cargo run -p heirloom-skill --bin heirloom-skill -- trace audit \
  --traces build/traces.jsonl \
  --out reports/trace_hygiene.json

cargo run -p heirloom-skill --bin heirloom-skill -- sft-export \
  --traces build/traces.jsonl \
  --out build/sft_cloud.jsonl
```

Exact artifact paths in final responses are normalized in exported SFT records
to stable placeholders such as:

```text
artifact://trace/{trace_id}/{index}
```

This keeps local sandbox paths out of training records while preserving the fact
that a trace returned an artifact reference.
