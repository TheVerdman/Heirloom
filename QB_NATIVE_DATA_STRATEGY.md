# QB-Native Data Strategy

Status: draft
Date: 2026-06-10

This document turns the Heirloom + VECL-QB direction into an executable data plan
for a small memory-native model. The target is not a miniature frontier chatbot.
The target is a 1B-3B parameter operational model that routes precisely, uses
tools cleanly, tracks provenance, handles time correctly, and learns sparse
updates in a substrate where sparse state is native.

## Target

Train a small "QB-native" model from scratch with Heirloom's memory-transformer
runtime:

- Dense backbone: roughly 300M-500M parameters.
- Sparse memory pool: roughly 1M slots, about 1B sparse parameters.
- Total parameter count: 1B-2B initially, expandable toward 3B.
- Active FLOPs: sized for 4x NVIDIA A100 80GB on Vertex.
- Runtime substrate: Heirloom memory transformer, memory layers, SMFT, and sparse
  row updates.
- Kernel stance: preserve the no-cuBLAS-dependency design and continue using
  Heirloom-owned CUDA Driver API / hand-rolled PTX kernels.
- Performance target: optimize toward roughly 40% MFU on the available 4x A100
  80GB infrastructure once correctness, data scale, and memory semantics are
  stable.

The data strategy must therefore optimize for:

- Compact operational behavior instead of encyclopedic memorization.
- High-quality routing, tool calls, evidence handling, and refusal/escalation.
- Native memory supervision, including row access, update masks, and provenance.
- Scale-ready tokenization and input storage before we attempt a serious run.

## Current Heirloom State

Heirloom already has the important runtime proof points:

- 4x A100 DDP training has succeeded on TinyStories.
- Memory-transformer execution has succeeded on 4x A100 with memory layers,
  shared memory, Memory+, sparse-row optimizer paths, SMFT masks, and memory
  kernel counters.
- The CLI has a `train-memory-lm` path with memory layer, lookup, SMFT, precision,
  device, distributed, checkpoint, and reporting options.

The current data path has crossed the first production hard-path gate:

- `heirloom data prepare --format binary-shard` emits manifest v2 with
  `format="heirloom.token_dataset"`, `storage="binary_shards"`, tokenizer
  fingerprint metadata, source records, split token counts, shard metadata, and
  per-shard hashes.
- `train-lm`, `train-memory-lm`, DDP rank helpers, `eval-lm`, and
  `eval-memory-lm` select the v2 streaming binary-shard loader without
  materializing all tokens.
- `scripts/run_qb_data_hardpath.sh` now has both dense and memory modes. Dense
  mode remains the full train/resume/eval/generation gate; memory mode runs
  `train-memory-lm`, `eval-memory-lm`, and `generate-memory-lm` on manifest v2
  and validates memory selection/SMFT evidence.
- The 4x A100 QB data hard-path gate passed on Vertex at
  `2026-06-11T01:36:13Z` (`2026-06-10` US/Eastern) using full
  TinyStories-valid plus a deterministic QB trace fixture, DDP train,
  checkpoint resume, eval, generation, Tensor Core hard gates, checksum drift
  checks, and MFU/timing reports.
- v1 JSON token manifests remain compatible as small correctness fixtures.
- Data preparation can now split each train/valid split into multiple shard
  files with `--shard-tokens`.
- Training supports `--grad-accumulation-steps` while keeping `steps` as
  optimizer steps and reporting micro/effective batch sizes.
- Native tokenizer artifact v2 is implemented with byte fallback, stable
  PAD/BOS/EOS and byte-token IDs, reserved-token registry, corpus-sample
  metadata, validation, fertility reporting, and train/eval/generation report
  identity fields.
- `heirloom data materialize-blend` now bridges `heirloom.corpus_blend` source
  governance into curated text shards plus a manifest v2 binary-shard token
  dataset, with license gates, deterministic weighted quotas, dedupe/filtering,
  curation reports, source indexes, and selected-document manifests.

The remaining data path is intentionally narrow:

- Full-mode source acquisition is not done yet: Dolma, Nemotron-CC, Dolmino 100B,
  Nemotron-CC-Math, and VECL-QB v1-hard must be provided as approved governed
  slices before the 20B-token production materializer gate can run. Dolma is an
  internal-training attribution source, NVIDIA sample-artifact slices are
  internal-training-only, and OLMo 3 pretraining data is tracked as the
  `allenai/dolma3_dolmino_mix-100B-1125` ODC-BY mixed-down artifact.
- The generated default blend now records the production weights
  `0.35/0.25/0.20/0.10/0.10` for Dolma, Nemotron-CC high/actual, Dolmino 100B,
  Nemotron-CC-Math, and VECL-QB v1-hard. See
  `QB_PRETRAINING_READINESS.md`.
- The first 4x A100 hard-path gate used a tiny dense LM. The memory-transformer
  v2 hard path now also has live 4x A100 evidence: job
  `683395937506164736`, display `heirloom-validate-quick-20260610-231715`,
  with full artifact validation passing against the uploaded summary.

That makes the next scale-readiness priorities clear:

1. Acquire and materialize approved bounded source slices for the 20B-token
   production blend.
2. Run the 32K tokenizer and materializer hard path in full mode.
3. Product-key lookup hardening and large-memory stress sets.
4. Memory/SMFT trace data that teaches sparse updates as native behavior.
5. Grad-accumulated memory-model scale gates on the v2 shard path.
6. PTX GEMM/attention/memory-kernel throughput measurement and MFU improvement.

## Performance Doctrine

The model should be designed for the hardware we actually have: 4x NVIDIA A100
80GB on GCP. Once the scale path is correct, performance work should become
ruthless and measurement-driven.

Non-negotiables:

- No cuBLAS dependency in the training runtime.
- Keep kernel ownership inside Heirloom through hand-rolled PTX and the CUDA
  Driver API boundary already used by `heirloom-kernels`.
- Treat Tensor Core utilization, memory movement, launch overhead, DDP all-reduce
  volume, and dataloader stalls as first-class training metrics.
- Optimize for end-to-end training throughput, not isolated microbenchmarks.
- Preserve correctness gates while replacing correctness-first kernels with
  tuned kernels.

MFU reporting should be explicit:

- Report estimated active FLOPs/token for the dense backbone.
- Report sparse memory lookup/update work separately so memory parameters do not
  inflate the denominator.
- Report dense-core MFU against A100 BF16 Tensor Core peak, using CUDA event
  forward/backward timing when available and clearly labeling any host-timing
  fallback.
- Report end-to-end MFU including attention, MLP, memory lookup, optimizer,
  all-reduce, and dataloader time.
- Keep host elapsed buckets and CUDA event elapsed buckets separate so
  dataloader/host movement stalls do not get mistaken for GPU kernel time.
- Keep a kernel-level roofline view for GEMM, attention, memory lookup/top-k,
  sparse row updates, and communication.

The 40% MFU target is not a first milestone. It is the optimization direction
after the sharded data path, 32K tokenizer, memory lookup correctness, and
training reports are stable.

## Data Thesis

Open foundation corpora can teach the model language, code, math, and document
structure. They will not, by themselves, teach QB-native cognition.

The differentiating data must be created by us:

- Temporal context envelopes.
- `STATE -> DELTA -> ACTION -> EVIDENCE -> ANSWER` traces.
- Tool-call plans and validated tool results.
- Negative traces for unnecessary tools, bad citations, stale-time assumptions,
  false certainty, and over-answering.
- Specialist handoff traces from VECL-QB domains.
- Memory access and SMFT supervision traces.
- Provenance-preserving summaries and updates.

The foundation corpus teaches the model to read and write. The QB corpus teaches
it to route, verify, remember, cite, refuse, compress, and act.

## Open Data To Reuse

Use public/open corpora as the foundation layer, with license and redistribution
checks before any artifact is mirrored into `vecl-qb-artifacts`.

### OLMo / Dolma / Tulu Lineage

Useful because it is unusually transparent about data, checkpoints, tooling, and
recipes:

- Dolma-style open pretraining mixture for broad web, papers, code, books,
  social, and encyclopedic coverage.
- OLMo-style full-lifecycle reproducibility discipline.
- OLMo 3-style emphasis on long-context reasoning, function calling, coding,
  instruction following, and knowledge recall.
- Tulu 3-style post-training recipe for SFT, preference data, verifiable rewards,
  decontamination, and transparent evaluation.

Primary references:

- Dolma: https://arxiv.org/abs/2402.00159
- OLMo: https://arxiv.org/abs/2402.00838
- OLMo 3: https://arxiv.org/abs/2512.13961
- Tulu 3: https://arxiv.org/abs/2411.15124

### NVIDIA Nemotron Lineage

Useful because the recent Nemotron work is explicitly agentic, tool-oriented,
hardware-conscious, and open about at least part of its data and recipe surface:

- Nemotron-CC-style Common Crawl refinement for long-horizon pretraining.
- Nemotron-CC-Math-style extraction that preserves math and code structure.
- Nemotron 3-style agentic post-training and multi-environment RL as a reference
  pattern, even though the architecture is much larger than our target.

Primary references:

- Nemotron-CC: https://arxiv.org/abs/2412.02595
- Nemotron-CC-Math: https://arxiv.org/abs/2508.15096
- Nemotron 3: https://arxiv.org/abs/2512.20856
- Nemotron 3 Nano: https://arxiv.org/abs/2512.20848

### Code, API, Math, And Structured Data

We should add carefully licensed sources for:

- Code and repository text.
- API documentation and examples.
- Math, proofs, problem solutions, and symbolic traces.
- Logs, tables, manifests, JSON, TOML, YAML, SQL, shell, and config files.
- Bug reports, issue threads, and patch explanations where licenses permit.

This category is crucial because the model's real job is operational precision.
It should be comfortable reading boring structured inputs without turning them
into chatty summaries.

## Data We Need To Create

### Temporal Awareness Corpus

Every chat-style training sample should be able to include a temporal envelope:

```json
{
  "current_time": "2026-06-10T14:03:12-04:00",
  "timezone": "America/New_York",
  "thread_created_at": "2026-06-03T09:15:00-04:00",
  "previous_user_message_at": "2026-06-07T18:41:22-04:00",
  "elapsed_since_previous_user_message": "2d19h21m50s"
}
```

Training goals:

- Interpret "today", "yesterday", "tomorrow", "last week", and "a few days
  later" from the envelope.
- Notice when a long-running thread has resumed after a gap.
- Distinguish temporal grounding from factual freshness.
- Ask for search only when the answer depends on unstable outside facts.
- Avoid stale-context errors without needing the user to say "it is later now."

### QB Trace Corpus

The core supervised format should teach the operational loop directly:

```json
{
  "trace_id": "qbtrace_000001",
  "temporal_context": {
    "current_time": "2026-06-10T14:03:12-04:00",
    "timezone": "America/New_York"
  },
  "task": {
    "user_message": "Review this run report and tell me whether the gate passed.",
    "context_refs": ["artifact://runs/memory_32_block/report.json"]
  },
  "state": {
    "known": ["The report is local and structured."],
    "unknown": ["Whether the success counters are positive."]
  },
  "delta": {
    "new_constraints": ["Do not infer pass/fail without reading the report."]
  },
  "route": {
    "mode": "CODEBASE",
    "tool_needed": true,
    "specialist": null
  },
  "action": {
    "tool": "read_file",
    "arguments": {
      "path": "runs/memory_32_block/report.json"
    }
  },
  "evidence": [
    {
      "ref": "artifact://runs/memory_32_block/report.json",
      "claim": "kernel counters are positive",
      "verdict": "supports"
    }
  ],
  "answer": {
    "direct": "The runtime gate passed.",
    "caveats": ["Loss did not improve; this is a runtime proof, not quality proof."]
  },
  "labels": {
    "schema_valid": true,
    "tool_call_valid": true,
    "evidence_faithful": true,
    "over_answered": false
  }
}
```

Training goals:

- Choose direct answering, search, calculation, codebase inspection, memory,
  specialist routing, clarification, escalation, or refusal.
- Produce compact intermediate state when useful.
- Preserve provenance links through the final answer.
- Stop once the task is handled.

### Negative And Correction Corpus

Small models improve disproportionately from seeing what not to do. Generate
paired examples:

- Answered from memory when a tool was required.
- Searched when local context was sufficient.
- Asked a clarification when there was a reasonable default.
- Claimed current facts without checking time-sensitive sources.
- Cited evidence that did not support the claim.
- Produced a valid-looking but invalid JSON/tool schema.
- Continued after the user asked to stop.
- Refused a safe request.
- Complied with an unsafe request.
- Updated the wrong memory rows.

Each negative sample should include:

- The bad output.
- The failure label.
- The corrected output.
- The smallest evidence needed to justify the correction.

### Specialist And Tool Corpus

VECL-QB's natural role is to generate specialist traces. Useful domains include:

- Codebase inspection and patch planning.
- Python/Rust build and test loops.
- Symbolic math with SymPy-style verification.
- Chess with Stockfish-style move validation.
- Bioinformatics with BLAST-style provenance.
- Time-series forecasting with model and horizon metadata.
- Infrastructure with Terraform/Kubernetes validation.
- Hardware/EDA with Yosys/OpenROAD-style tool outputs.
- Web/search tasks with query quality and citation labels.

For each domain, capture both the action and the validation result. The model
should learn that tool output is evidence, not decoration.

### Memory And SMFT Corpus

The Heirloom advantage is that memory layers are native. We should train and
evaluate memory behavior explicitly rather than hope it emerges.

Memory trace schema:

```json
{
  "example_id": "memtrace_000001",
  "model_config": {
    "memory_layers": [8, 16, 24],
    "memory_slots": 1048576,
    "lookup": "product_key",
    "top_k": 32
  },
  "input_ref": "qbtrace_000001",
  "selected_rows": {
    "layer_8": [1201, 99120, 421004],
    "layer_16": [42, 51000, 100332],
    "layer_24": [8844, 900100, 900104]
  },
  "access_counts_ref": "artifact://memory/access_counts/step_001000.bin",
  "smft_mask_ref": "artifact://memory/masks/step_001000.bin",
  "expected_trainable_rows": {
    "layer_8": [1201, 99120],
    "layer_16": [42],
    "layer_24": [900100, 900104]
  },
  "update_policy": "smft_masked_sparse_rows",
  "labels": {
    "foreground_rows_present": true,
    "background_rows_preserved": true,
    "row_budget_respected": true
  }
}
```

Training and evaluation goals:

- Reuse memory for repeated entities, tools, projects, and long-running threads.
- Separate foreground task memory from background stable knowledge.
- Respect row budgets and SMFT masks.
- Track per-row provenance for sparse updates.
- Detect product-key collisions and lookup instability.

## Training Mixture

Initial pretraining mixture:

- 60%-70% open foundation text from vetted Dolma/OLMo/Nemotron-style sources.
- 10%-15% code, API docs, schemas, logs, tables, and config data.
- 5%-10% math, symbolic traces, proofs, and verifiable problem solutions.
- 5%-10% structured operational traces, including early QB traces.

QB-native continued training / anneal:

- 30%-40% QB traces.
- 15%-25% tool and specialist traces.
- 10%-15% temporal awareness and long-thread continuation traces.
- 10%-15% evidence, citation, provenance, and summary-update traces.
- 10%-15% negative/correction traces.
- 5%-10% memory/SMFT traces.
- Remainder: high-quality foundation/code/math refresh to prevent collapse.

Post-training:

- SFT for exact schemas, action selection, and final-answer discipline.
- Preference data for concise, evidence-faithful, non-chatty behavior.
- Verifiable reward tasks for tool calls, code tests, math, JSON validity, search
  citation support, and refusal/escalation correctness.
- Environment rollouts where success is measured by completed action, not prose
  plausibility.

## Data Artifacts

### Token Dataset Manifest V2

The implemented v2 token-dataset manifest is the training hard path. It records
tokenizer fingerprint metadata, repeated source inputs, split token counts,
train/valid shard metadata paths, token hashes, and `storage="binary_shards"`.
v1 JSON manifests remain supported for compatibility/debug runs.

### Corpus Blend Manifest

The next layer is a corpus-level blend manifest that points at licensed source
collections and derived token-dataset manifests:

```json
{
  "format": "heirloom.corpus_manifest",
  "version": 2,
  "corpus_id": "qb_native_foundation_v0",
  "created_at": "2026-06-10T00:00:00Z",
  "tokenizer": {
    "id": "heirloom-bpe-32768-v0",
    "path": "gs://vecl-qb-artifacts/tokenizers/heirloom-bpe-32768-v0.json",
    "sha256": "..."
  },
  "sources": [
    {
      "source_id": "dolma_style_web",
      "license": "source-specific",
      "provenance_uri": "https://arxiv.org/abs/2402.00159",
      "filters": ["dedupe", "quality", "pii_scan"],
      "blend_weight": 0.45
    }
  ],
  "splits": {
    "train": ["shards/train-000000.tokens.bin"],
    "valid": ["shards/valid-000000.tokens.bin"],
    "test": ["shards/test-000000.tokens.bin"]
  }
}
```

### Binary Token Shards

The implemented shard payload is intentionally simple:

- `*.tokens.bin`: contiguous token payload, `u16` if vocab <= 65536, otherwise
  `u32`.
- document/sample offset indexes remain future work for better packing and eval
  provenance.
- `*.json`: shard header with magic, version, dtype, tokenizer id, split, counts,
  hashes, and source blend summary.

Required properties:

- Sequential read throughput is measurable.
- Random sample access is possible from offsets.
- DDP rank sharding is deterministic.
- Document boundaries are preserved for packing and eval.
- Per-shard hashes make GCS artifact corruption detectable.
- The format can be mmap-backed locally and streamed from prepared local disks on
  Vertex.

### QB JSONL Shards

Keep structured traces separate from token shards:

- Raw trace JSONL is the source of truth.
- Tokenized binary shards are derived artifacts.
- The manifest records the transform code version and tokenizer fingerprint.
- Failed-schema examples are retained in a separate negative split when useful.

This keeps us from losing provenance and labels during tokenization.

## Tokenizer Plan

The target tokenizer is 32K for the first serious QB-native runs.

Requirements:

- Handles code, JSON, logs, markdown, and prose without pathological fertility.
- Preserves byte fallback behavior.
- Stable BOS/EOS/PAD and future control tokens.
- Trains from streaming/sharded samples, not a single in-memory string.
- Emits a fingerprint used in every manifest and checkpoint.
- Uses native Heirloom training and runtime code, with no `tokenizers` or
  SentencePiece dependency.

Implemented v2 path:

- `heirloom tokenizer train-corpus` emits tokenizer artifact v2 with reserved
  tokens, source/sample hashes, training config, and validation metadata.
- `heirloom tokenizer validate` checks stable special IDs, byte fallback,
  reserved-token atomics, merge graph integrity, and v1 compatibility.
- `heirloom tokenizer fertility` reports tokens/byte, byte-token share, and
  reserved-token hits by source.
- `scripts/run_qb_tokenizer_hardpath.sh` runs tokenizer train/validate/fertility,
  governed materialization to manifest v2 binary shards, and memory
  train/resume/eval/generation.

## Evaluation Gates

Data gates:

- License and provenance completeness.
- Duplicate and near-duplicate rates.
- PII, secret, and credential scan.
- Train/valid/test split integrity.
- Benchmark contamination checks.
- Tokenization fertility by source type.
- Long-context packing waste.
- JSON/schema validity.
- Tool-call validation rate.
- Evidence support and citation fidelity.
- Temporal reasoning accuracy.
- Refusal/escalation correctness.
- Memory-row telemetry completeness.
- Sharded read throughput and dataloader wait time.

Runtime gates:

- Tokens/sec per GPU.
- Active parameter estimate per token.
- Estimated FLOPs/token.
- Dense-core MFU and end-to-end MFU estimates, with the target path aiming
  toward roughly 40% MFU on 4x A100.
- CUDA-event bucket timing for GPU-resident forward/backward, optimizer, and
  NCCL all-reduce paths when running on CUDA.
- Tensor Core usage for dense paths.
- Hand-rolled PTX kernel family used for each dense and memory hot path.
- Memory kernel counters for lookup/top-k/weighted values.
- Product-key candidate quality.
- Sparse row update counts.
- SMFT mask hit rates.
- Row-union all-reduce size in DDP.
- Dataloader stall time and CUDA launch/sync overhead.
- Checkpoint compatibility with model family and memory config metadata.

## Roadmap

### D0: Freeze Current Baseline

- Document v1 prepared-data limitations and v2 binary-shard hard-path evidence.
- Keep TinyStories JSON-token path available as a small correctness fixture.
- Add a scale-readiness checklist to the memory-transformer docs.

### D1: Tokenizer V2

- Implemented native artifact v2, reserved-token registry, tokenizer report
  blocks in train/eval/generation reports, and fertility reports.
- Implemented governed materialization from `heirloom.corpus_blend` into text
  shards plus manifest v2 binary token shards.
- Remaining hard gate: run `full` mode with materialized approved Dolma,
  Nemotron-CC, Dolmino 100B, Nemotron-CC-Math, and VECL-QB v1-hard source
  samples.

### D2: Binary Shards

- Done for the first hard path: binary token shard writer/reader, manifest v2,
  deterministic DDP sampling over shards, streaming train/eval loaders, and
  host plus CUDA-event timing/MFU report fields.
- Remaining: document/sample offset indexes, richer source-blend metadata, and
  loader-throughput stress tests at larger shard counts.

### D3: VECL-QB Trace Export

- Export existing VECL-QB tool-use examples and provenance traces into QB JSONL.
- Add temporal envelopes.
- Add positive, negative, and corrected variants.
- Preserve artifact references and source ids through tokenization.

### D4: Open Corpus Intake

- Build a first small governed blend from Dolma/OLMo/Nemotron-style sources.
- Store only private/internal raw source slices in the project GCS bucket, with
  attribution, license status, source paths, and hashes captured in inventory.
- Record source licenses, filters, and hashes in manifest v2.

### D5: Memory/SMFT Supervision

- Generate synthetic memory traces at small scale.
- Add product-key collision and recall stress sets.
- Track selected rows, masks, foreground/background counts, and update budgets.

### D6: 4x A100 Scale Gate

- Done for the tiny dense QB data hard path on 4x A100.
- Done for the memory-transformer QB data hard path on manifest v2 binary
  shards with `HEIRLOOM_QB_DATA_HARDPATH_MODEL=memory`.
- Next: run the tokenizer/materializer hard path in `full` mode once approved
  source slices are available.
- Report loss, validation loss, tokens/sec, MFU, dataloader wait, memory kernel
  counters, sparse row update counts, and checkpoint metadata.
- Treat these as data/runtime gates, not model-quality claims.

### P0: MFU Optimization Track

- Add a stable FLOPs/token estimator for dense backbone, attention, memory lookup,
  sparse row updates, and optimizer work.
- Add per-step timing buckets for dataloader, forward, backward, optimizer,
  all-reduce, checkpoint/reporting, and generation/eval escapes. CUDA training
  reports should include both host elapsed buckets and CUDA event buckets for
  GPU work.
- Add kernel-family counters to distinguish staged CTA GEMM, wide/swizzled GEMM,
  attention Tensor Core kernels, memory lookup/top-k kernels, and sparse row
  update kernels.
- Build A100 microbenchmarks for the hot PTX kernels without introducing cuBLAS
  as a runtime dependency.
- Tune toward ~40% MFU only after the measurements are honest enough to make the
  number meaningful.

## Immediate Tickets

1. Acquire approved bounded Dolma/Nemotron/OLMo/QB slices and run
   `data materialize-blend --target-tokens 20000000000 --mode full`.
2. Add a VECL-QB trace exporter for `STATE -> DELTA -> ACTION -> EVIDENCE -> ANSWER`.
3. Add temporal envelope fields to every chat/tool trace generator.
4. Add product-key lookup hardening tests with large-slot synthetic data.
5. Add memory/SMFT trace export from memory-transformer training runs.
6. Run the first full tokenizer/materializer/memory-transformer gate on 4x A100.
7. Add document/sample offset indexes to token shards for packing and eval
   provenance.
8. Expand A100 performance reporting from estimates into kernel-family
   throughput and roofline views for the hand-rolled PTX path.

The most important sequencing principle: make the data path scale-ready before
we broaden model behavior. A small model with a clean substrate and sharp data
will teach us more than a wider behavioral sweep running through JSON token
arrays and an undersized tokenizer.
