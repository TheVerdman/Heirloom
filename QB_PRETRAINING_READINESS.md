# QB-Native Pretraining Readiness

This phase turns the proven memory-transformer hard path into a pretraining
system with real source governance, a production tokenizer, learning sanity
checks, and performance baselines.

## Current Position

- Manifest v2 binary token shards are validated on CPU and 4x A100.
- Dense and memory-transformer paths train, resume, eval, and generate from the
  v2 shard loader.
- The full memory hard path passed on 4x A100 with NCCL, AMP BF16, sparse-row
  memory updates, SMFT, Tensor Core gates, CUDA-event timing, eval, generation,
  and artifact validation.
- The latest memory run was a systems gate. Its train loss was flat, so quality
  work should start with tokenizer/data/learning sanity rather than kernel
  tuning alone.

## Tokenizer Decision

Use a Heirloom-owned production 32K byte-level BPE tokenizer.

This means:

- Keep byte fallback and deterministic decode round trips.
- Force ASCII digit isolation for production v2 tokenizers so digits remain
  byte tokens and learned BPE merges cannot absorb digits into inconsistent
  math/code/schema fragments.
- Preserve stable special IDs for PAD/BOS/EOS.
- Add a reserved-token registry for chat, tool, memory, trace, document, schema,
  governance, and control tokens before full 32K training.
- Store training config, source manifest hash, tokenizer hash, corpus sample
  sizes, byte coverage, token length histograms, and special-token registry in
  the tokenizer artifact.
- Train from a corpus blend manifest rather than from ad hoc text files.
- Keep the runtime dependency-free and compatible with existing shard manifests.

The first native implementation is tokenizer artifact v2:

- `heirloom tokenizer train-corpus` trains from a corpus-blend manifest.
- `heirloom tokenizer validate` checks artifact invariants.
- `heirloom tokenizer fertility` reports per-source tokenization behavior.
- `heirloom data materialize-blend` turns the governed corpus blend into
  curated text shards plus a manifest v2 binary-shard token dataset for
  training.
- `scripts/run_qb_tokenizer_hardpath.sh` validates the path through manifest v2
  binary shards and memory-transformer train/resume/eval/generation.

The remaining production requirement is acquiring/materializing approved
full-mode Dolma/Nemotron/OLMo/QB source files and running the materializer at
the target scale. Smoke mode proves the machinery but is not a production
corpus gate.

Production slices should be stored in
`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/`.
The full tokenizer hard path can receive those slices as local files, flat
local directories, single-object `gs://` URIs, or flat `gs://` prefixes ending
in `/`. `gs://` inputs are staged onto the Vertex worker's local disk for the
run, and the generated blend manifest records the original GCS URI. A developer
laptop should only hold fixtures and small inspection samples.

Hugging Face artifacts are sliced with `scripts/slice_hf_dataset.py`, which
streams selected dataset shards into uncompressed private JSONL slices or shard
directories, writes a redacted `heirloom.hf_source_slice_report`, and
optionally uploads the data plus report to GCS. Use `--shard-output-bytes` for
production-sized slices so each source is a flat directory/prefix of
deterministic `*-part-NNNNN.jsonl` files with a file-set hash. The script can
read `HF_TOKEN` from the environment or from the VECL-QB
`.env` file when network auth is needed, but token values are never emitted.
The current OLMo/Dolma 3 slot uses the mixed-down
`allenai/dolma3_dolmino_mix-100B-1125` artifact rather than the raw Dolma 3
pool. Use the slicer's `--preset dolmino-qb` for the first rehearsal slice so
the path-level adult-content category is excluded before the materializer's own
record filters run.

Currently staged source slices:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/dolma-v1_7/dolma-v1_7-100m.jsonl
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/dolma-v1_7/dolma-v1_7-100m.report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/nemotron-cc-high-actual/nemotron-cc-high-actual-sample.jsonl
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/nemotron-cc-high-actual/nemotron-cc-high-actual-sample.report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/dolma3-dolmino-mix-100b-1125/dolmino-100m.jsonl
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/dolma3-dolmino-mix-100b-1125/dolmino-100m.report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/nemotron-cc-math/nemotron-cc-math-sample.jsonl
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/nemotron-cc-math/nemotron-cc-math-sample.report.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/vecl-qb-v1-hard/corpus.jsonl
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/vecl-qb-v1-hard/metadata.json
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices/source-slice-inventory.json
```

The first external rehearsal slices are staged:

- Dolma v1.7 100M: `159,470` selected records, `400,001,277` selected text
  bytes, and `100,000,319` estimated tokens.
- Dolmino 100M: `72,909` selected records, `400,004,346` selected text bytes,
  and `100,001,086` estimated tokens.
- Nemotron-CC High-Quality sample: `765` selected records and `612,867`
  estimated tokens.
- Nemotron-CC-MATH sample: `954` selected records and `795,753` estimated
  tokens.

The refreshed inventory has all `5` planned sources available and `0` pending
sources.

The first larger sharded source-slice rehearsal is staged under:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/qb-native-pretraining-v1/source-slices-1b-rehearsal/
```

It records all `5` planned sources available, `0` pending sources,
`4,954,749,509` bytes, and `1,189,942` records. Dolma v1.7 contributes
`5` shard files / `796,588` records / `500,002,478` estimated tokens, and
Dolma 3 Dolmino contributes `5` shard files / `366,635` records /
`500,000,380` estimated tokens. Nemotron sample-artifact slices and VECL-QB
`v1-hard` are carried forward unchanged.

The first all-source local rehearsal completed in
`runs/qb-tokenizer-rehearsal-100m-512-doccap/` using a bounded 512-vocab
tokenizer smoke over the five staged sources. It produced manifest v2 binary
shards with `9,388` train tokens and `1,629` valid tokens, reported
`loader.kind="binary_shard_streaming"` and `tokens_materialized=false`, and
completed memory train/resume/eval/generation.

The stronger 4K local release rehearsal completed in
`runs/qb-tokenizer-rehearsal-100m-4096-release/` after moving tokenizer v2 merge
training to the incremental native trainer
`heirloom.byte_bpe.native_incremental.v2`. It trained vocab `4096` from the
five staged sources, produced tokenizer hash `03363a37b7e06036`, materialized
manifest v2 binary shards from `11` selected docs / `10,786` selected tokens,
and completed memory train/resume/eval/generation with the streaming loader.

The exact-32K bounded local rehearsal completed in
`runs/qb-tokenizer-rehearsal-100m-32768-16m-heap-rebuild/` after adding
deterministic heap compaction to tokenizer v2 training and a cached
rank-ordered merge encoder for runtime tokenization. It trained vocab `32768`
with `--require-exact-vocab` from the five staged sources, produced tokenizer
hash `2996feb60e39fa2e`, materialized manifest v2 binary shards from `11`
selected docs / `10,736` selected tokens, and completed memory
train/resume/eval/generation with the streaming loader. The remaining
production scale-readiness work is larger tokenizer samples, spillable
trainer/materializer work shards if needed, and full target-budget
materialization.

Follow-on 64 MiB and 256 MiB exact-32K requests completed in
`runs/qb-tokenizer-rehearsal-100m-32768-64m-heap-rebuild/` and
`runs/qb-tokenizer-rehearsal-100m-32768-256m-heap-rebuild/`. They both exhausted
the current staged local source slices at `44,159,551` physical sampled bytes,
which means the next scale gate needs larger approved source slices rather than
only a higher `--sample-bytes` value. The 256 MiB-request tokenizer produced
hash `ff87d08a04792bec` and completed memory train/resume/eval/generation
through manifest v2 binary shards with the streaming loader.

The source path is now directory-ready. `data materialize-blend`,
`tokenizer train-corpus`, the tokenizer hard-path wrapper, the Hugging Face
slicer, and the source staging inventory all support flat source directories.
A directory-backed smoke in
`runs/qb-tokenizer-hardpath-directory-smoke-2/` used a sharded source directory,
trained a tokenizer, emitted manifest v2 binary shards, and completed memory
train/resume/eval/generation with
`loader.kind="binary_shard_streaming"` and `tokens_materialized=false`.
This clears the operational path for larger GCS prefix slices.

The corrected 1B-source local tokenizer rehearsal completed in
`runs/qb-tokenizer-rehearsal-1b-32768-512m-textfix/`. It fixed a tokenizer
sampling renderer bug where general JSONL records with a `text` field were
being treated like QB prompt/target traces. The corrected run trained exact
vocab `32768` from a requested `512 MiB` sample, with `318,117,790` physical
sampled bytes and `554,132,722` weighted training bytes. Big Dolma/Dolmino
sources hit their byte quotas without exhaustion; the smaller Nemotron/QB
sources exhausted and were represented with effective weights. Tokenizer
training took `514,447 ms` total, including `19,295 ms` sample materialization
and `495,010 ms` native tokenizer training. The tokenizer report hash is
`1574298cf19369db` and saved artifact hash is `50291e31a9676a4b`.

The same tokenizer completed a capped local memory-transformer smoke using
manifest v2 binary shards from `137` selected documents / `135,410` tokens.
The memory train loss moved `10.178035 -> 9.795135`, resume loss was
`9.971405`, eval loss was `9.823355` with perplexity `18459.874713`, and the
loader reported `binary_shard_streaming` with `tokens_materialized=false`.
The unbounded local materializer scan was stopped after proving it was active
but too slow for a quick CPU smoke; production-scale materialization now needs
progress reporting, curation throughput work, and/or bounded staged passes.

A follow-up tokenizer audit confirmed Claude's digit-segmentation concern on
the previous artifact: learned tokens included digit-bearing fragments such as
`20`, `31`, `32`, `3)`, `2)`, `0.`, source slugs, and JSON confidence
fragments. The tokenizer runtime now records `digit_isolation=true` for new
production v2 artifacts, splits digits before BPE training/encoding, validates
that learned merge payloads contain no ASCII digits, and expands the default
reserved registry from `34` to `128` tokens. The digit-isolated exact-32K
rehearsal in `runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/` produced
tokenizer hash `10079cec3ace7a4b`, artifact hash `92be858136043b4f`, and
`learned_digit_token_count=0`. Its hard-path summary passed with manifest v2
binary shards, `loader.kind="binary_shard_streaming"`, and
`tokens_materialized=false`.

Source governance verdicts are tracked in `QB_SOURCE_GOVERNANCE.md`. The
current rule is conservative: Codex/Heirloom performs first-pass evidence
review, Andrew/project owner approves sources for this research build, and
formal legal review is reserved for commercial/company policy or ambiguous
redistribution terms.

## Corpus Blend V1

The first blend metadata contract is `heirloom.corpus_blend`, generated with:

```bash
cargo run --bin heirloom -- data corpus-blend \
  --qb-root /Users/andrewverdiramo/Desktop/VECL-QB/data \
  --out /tmp/heirloom-qb-native-corpus-blend.json \
  --blend-id qb-native-pretraining-v1 \
  --tokenizer-vocab-size 32768
```

The current source plan is:

| Source | Role | Initial Weight | Status |
| --- | --- | ---: | --- |
| `allenai.dolma.v1_7` | general language backbone | 0.35 | planned |
| `nvidia.nemotron_cc.high_actual` | high-quality web backbone | 0.25 | planned |
| `allenai.dolma3_dolmino_mix-100B-1125` | OLMo 3 Dolmino 100B mix | 0.20 | planned |
| `nvidia.nemotron_cc_math` | math/science/code reasoning | 0.10 | planned |
| `vecl_qb.synthetic.v1-hard` | tool routing and memory trace supervision | 0.10 | available |
| `vecl_qb.synthetic.v0-full` | regression/dev source | 0.00 | available |
| `vecl_qb.synthetic.v0-small` | fast test source | 0.00 | available |

TinyStories is intentionally not a production blend member. It stays useful for
smoke tests and tiny runtime gates, but it should not drive the production
language distribution.

## Source Governance

The blend manifest separates source-level governance from prepared token shards.
Each source records:

- source ID and role,
- planned or available status,
- path or source URL,
- data format,
- license status,
- provenance tags,
- sampling weight,
- tokenizer/pretraining/memory-trace inclusion flags,
- local byte and record counts when available,
- content and metadata hashes when available,
- split/domain/category/task-kind summaries for QB corpora.

External source licenses must be rechecked before production download. Raw
external slices are private/internal training inputs, not redistributable
artifacts. Dolma v1.7 is recorded as ODC-BY plus original source terms and is
usable for this project when attribution and exact slice hashes are preserved.
Nemotron-CC and Nemotron-CC-Math slices from
`nvidia/Nemotron-Pretraining-Dataset-sample` are recorded as internal-training
only under NVIDIA's Data Agreement; full-size NVIDIA objects outside that sample
artifact still require separate evidence.

`heirloom data materialize-blend` is the bridge from source governance to the
training loader. It rejects unapproved license statuses by default, requires
local uncompressed source files or flat source directories, applies
deterministic token quotas from source weights, filters
malformed/duplicate/secret-like/repetitive/high-fertility records, writes
curation reports, and emits a manifest v2 binary-shard dataset.
The materializer now records load/scan/selection/write/total timings,
aggregate throughput, per-source source-file counts, scan and write throughput,
write shard counts, and cap flags such as `limited_by_max_source_bytes`.
Long runs can use `--progress-every-records` and `--progress-every-bytes`; the
tokenizer hard-path wrapper defaults those heartbeats on in `full` mode.
Candidate retention can be bounded with
`--candidate-retention-token-multiplier`; with a multiplier of at least `1.0`,
the retained best-score prefix still contains enough tokens to satisfy each
source quota while pruning lower-score candidates during scan. Full tokenizer
hard-path mode defaults this to `1.0` with periodic pruning. Full mode also
defaults to `--candidate-text-mode rescan`, which drops rendered text from scan
candidates and later re-reads sources with a hash prefilter so only selected
documents are tokenized and written.
The materializer also supports scan-phase checkpointing with
`--checkpoint-dir`, `--checkpoint-every-records`, `--checkpoint-every-bytes`,
and `--resume-checkpoint`. Source checkpoints store the scan cursor, curation
stats, retained candidates, and accepted document hashes so an interrupted run
can resume the expensive candidate scan before writing fresh text/token shards.
Reports include a `checkpoint` block plus a `sizing` block with selected text
bytes, estimated `u16`/`u32` token payload bytes, train/valid token estimates,
and text/token shard estimates. Full tokenizer hard-path mode defaults
checkpointing to the run directory, with resume still explicit.
The production target can be set with `--target-tokens 20000000000`; `--mode
full` requires at least 95% of that target to be selected, while `--mode sample`
keeps local gates bounded.

Tokenizer encode throughput is now directly measurable with `heirloom tokenizer
bench-encode`. The native runtime uses a reserved-token first-byte lookup and a
priority-queue/linked-neighbor BPE merge encoder that preserves the sequential
merge reference. On the digit-isolated exact-32K tokenizer
(`10079cec3ace7a4b`) and a 16 MiB mixed Dolma/Dolmino line sample, the local
debug-build bench improved from about `532k` to `606k` tokens/sec after the
fast chunk encoder. The capped metadata-only rescan materializer rehearsal
improved from `61,779 ms` total / `58,463 ms` tokenizer encode to `20,325 ms`
total / `17,250 ms` tokenizer encode, with report throughput rising from about
`146.6k` to `496.9k` tokenizer-encoded tokens/sec. These are CPU debug-build
baselines for bottleneck tracking, not production speed claims.

## Learning Sanity Ladder

Before a larger quality run, prove:

1. Tiny overfit on a fixed shard with dense LM.
2. Tiny overfit on the same shard with memory layers disabled.
3. Tiny overfit with exact memory lookup and SMFT disabled.
4. Tiny overfit with exact memory lookup and SMFT enabled.
5. Product-key memory lookup parity on the same fixture.
6. LR and grad-accumulation sweep on 4x A100.
7. Longer run with the 32K tokenizer and blend manifest.

The first goal is not benchmark quality. It is monotonic learning under the
same loader, checkpoint, resume, eval, and generation contracts as the hard
path.

Learning sanity evidence is now validated with a native Rust readiness command:

```bash
cargo run --bin heirloom -- readiness validate-learning-sanity \
  --manifest learning-sanity-ladder.json \
  --report learning-sanity-validation.json
```

The smallest local producer for fresh evidence is:

```bash
bash scripts/run_learning_sanity_ladder.sh
```

It builds a deterministic fixed text shard, trains a tiny dense LM, a tiny
memory LM with memory layers explicitly disabled, and a tiny exact-memory LM
with SMFT disabled/enabled plus a product-key memory LM through the manifest v2
binary-shard loader, writes a `learning-sanity-ladder.json` manifest, and
validates it with the Rust readiness command. This is a local partial ladder
for the first five stages, not the full quality ladder.

The manifest is a small local contract that points at existing train reports:

```json
{
  "format": "heirloom.learning_sanity_ladder",
  "version": 0,
  "stages": [
    {
      "stage_id": "dense_fixed_shard",
      "report": "dense-train-report.json",
      "expected_command": "train-lm",
      "expected_model_family": "tiny_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "min_loss_reduction": 0.0
    },
    {
      "stage_id": "memory_layers_disabled",
      "report": "memory-layers-disabled-report.json",
      "expected_command": "train-memory-lm",
      "expected_model_family": "memory_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "expected_memory_lookup": "exact",
      "expected_smft_mode": "disabled",
      "min_loss_reduction": 0.0
    },
    {
      "stage_id": "memory_exact_smft_disabled",
      "report": "memory-train-report.json",
      "expected_command": "train-memory-lm",
      "expected_model_family": "memory_transformer",
      "expected_memory_lookup": "exact",
      "expected_smft_mode": "disabled",
      "min_loss_reduction": 0.0
    },
    {
      "stage_id": "memory_exact_smft_enabled",
      "report": "memory-exact-smft-enabled-report.json",
      "expected_command": "train-memory-lm",
      "expected_model_family": "memory_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "expected_memory_lookup": "exact",
      "expected_memory_update_policy": "sparse_rows",
      "expected_smft_mode": "masked_memory_rows",
      "min_loss_reduction": 0.0
    },
    {
      "stage_id": "product_key_parity",
      "report": "memory-product-key-report.json",
      "expected_command": "train-memory-lm",
      "expected_model_family": "memory_transformer",
      "expected_loader_kind": "binary_shard_streaming",
      "expected_memory_lookup": "product_key",
      "expected_memory_update_policy": "full",
      "expected_smft_mode": "disabled",
      "min_loss_reduction": 0.0
    }
  ]
}
```

Accepted `stage_id` values mirror the ladder:
`dense_fixed_shard`, `memory_layers_disabled`,
`memory_exact_smft_disabled`, `memory_exact_smft_enabled`,
`product_key_parity`, `lr_grad_accumulation_sweep`, and
`longer_32k_blend`. The validator checks finite `initial_loss`/`final_loss`,
computed `loss_reduction`, optional command/model/loader/memory expectations,
and the per-stage minimum loss reduction. For `memory_exact_smft_enabled`, it
also requires sparse-row memory updates, a row mask attached to sparse updates,
online refresh evidence, generated/active mask summaries, and nonzero SMFT
access counts. For `product_key_parity`, it requires square memory slots, even
key dimension, product-key memory selection, product-key memory-table
parameters, and an executed product-key lookup kernel path.

For `lr_grad_accumulation_sweep`, the stage report points to a nested sweep
manifest with format `heirloom.learning_sanity_lr_grad_accumulation_sweep`.
Each sweep run points at a normal `train-lm` report and may pin
`expected_learning_rate` plus `expected_grad_accumulation_steps`. The Rust
validator requires, by default, at least four runs, at least two distinct
learning rates, at least two distinct grad-accumulation values, 4-way NCCL
distributed evidence, AMP BF16 precision, CUDA device labels, nonzero
all-reduce counters, positive per-rank loss reductions, consistent
global-effective-batch math, and `performance.tokens_seen > 0`. The checked
fixture under `tests/fixtures/learning_sanity/lr-grad-sweep-valid.json`
validates this contract with synthetic reports only; it is not a substitute for
the paid 4x A100 sweep.

For `longer_32k_blend`, the stage report points at the
`scripts/run_qb_tokenizer_hardpath.sh` summary artifact. The Rust validator
walks that summary's child artifact paths and requires a passed tokenizer
hard-path bundle: tokenizer v2 with vocab `32768`, at least `128` reserved
tokens, digit isolation, non-empty source-blend/sample/registry hashes, a
governed `heirloom.corpus_blend` manifest, no TinyStories source, materializer
curation evidence, manifest v2 binary shards,
`loader.kind="binary_shard_streaming"`, `tokens_materialized=false`, combined
train-plus-resume loss improvement, a resume report that advances the step,
eval metrics, and a generation report. Stage fields such as
`min_selected_tokens`, `min_selected_docs`, `min_blend_sources`,
`min_final_step`, and `require_production_gate` make the same validator usable
for local rehearsals and stricter full-mode evidence. The tokenizer hard-path
wrapper writes `learning-sanity-ladder.json` and
`learning-sanity-validation.json` automatically for 32K runs unless
`HEIRLOOM_QB_TOKENIZER_HARDPATH_LEARNING_SANITY=0`. The checked fixture under
`tests/fixtures/learning_sanity/longer-32k-blend-valid.json` validates this
contract with synthetic artifacts only; it is not a substitute for the longer
4x A100 production-data run.

The existing digit-isolated local 32K rehearsal now carries a validated
learning-sanity manifest at
`runs/qb-tokenizer-rehearsal-1b-32768-64m-digitiso/learning-sanity-ladder.json`.
It passes the 32K/blend artifact contract with train-plus-resume loss moving
`10.049116 -> 9.851956` (`loss_reduction=0.0196196`) over `35,705`
materialized tokens from `34` selected documents. This remains bounded local
evidence, not the final paid 4x A100 production-data run.

This does not replace the actual ladder runs; it makes their evidence
machine-checkable before larger quality claims.

## Performance Readiness

MFU work should proceed as measurement first:

- GEMM microbench report for every PTX Linear kernel family.
- Attention microbench report for ragged and tile-aligned cases.
- Memory lookup/update microbench report for exact and product-key paths.
- DDP report for dense gradients, row-union masks, and compact sparse-gradient
  transport.
- End-to-end report that clearly separates dense-core MFU from sparse/memory
  work.

The first CUDA throughput harness is `heirloom gpu tensor-core-microbench`.
It reports BF16 Tensor Core GEMM and current materialized attention
timing/correctness against BF16-rounded CPU references, plus explicit
experimental request/fallback counters for the future `ldmatrix` and
`cp.async` instruction tiers. The GEMM `ldmatrix` request tier now launches a
guarded ldmatrix-A shared-memory MMA kernel, and the GEMM `cp.async` request
tier launches a guarded double-buffered async-copy pipeline feeding the same
ldmatrix-A MMA path. GEMM-only Vertex job `9203551123560988672`
A100 microbench-validated both instruction-path GEMM tiers with no fallback
counters, but they remain guarded throughput gates rather than default
training-path semantics. That job did not validate Tensor Core flash
attention. Attention ldmatrix/cp.async toggles still report fallback for the
materialized attention matmuls.

The same microbench now compares current materialized BF16 causal attention
with a guarded forward-only scalar-streaming flash-like BF16 path that avoids
materializing `[B,H,T,T]`. That path reports requested/executed/fallback,
scalar-streaming QK/AV tile, ragged, causal, timing, and avoided-byte counters,
but it deliberately leaves Tensor Core flash/MMA counters at zero because the
scalar-streaming implementation does not execute MMA instructions. A separate
experimental Tensor Core tiled flash forward candidate is available only when
`HEIRLOOM_CUDA_FLASH_BF16_ATTENTION=1` and
`HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE=1` are both set; it uses
MMA-backed tiled QK/PV with online softmax and compact row max/denom state.
Grad-required calls still fall back to the current materialized attention path
unless the additional experimental
`HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD=1` guard is set. With that guard,
the exact-tile sm80 path saves compact row stats and runs tiled Tensor Core
backward kernels for QK recompute, dP, dQ, dK, and dV without materializing
`[B,H,T,T]`. The forward-only artifact exists as Vertex job
`6050468434448220160` for `batch=4, heads=16, time=512, head_dim=64`. The
target training-path artifact exists as Vertex job `894480179806601216`: 4x
A100 DDP, per-rank `batch=4`, `time=512`, `heads=16`, `head_dim=64`,
positive flash forward/backward MMA counters, positive CUDA-event timing,
positive MFU fields, NCCL all-reduce evidence, zero checksum drift, and zero
flash fallback/scalar fallback/hard-require/materialized-reference counters.
`scripts/gcp/submit_vertex_qb_memory_flash_gate.sh` now applies the same guarded
exact-tile route to the QB manifest v2 memory-transformer hard path, with
binary-shard streaming, `train-memory-lm`, sparse-row memory updates, SMFT row
mask evidence, flash forward/backward hard-require checks, and `cp.async` GEMM
hard-require checks. Vertex job `6192402191454568448` passed that QB-memory
flash gate on 4x A100 with manifest v2 binary-shard streaming,
train/resume/eval/generation completion, `loader.kind="binary_shard_streaming"`,
`tokens_materialized=false`, positive NCCL and sparse row-union all-reduce
evidence, zero checksum drift, Tensor Core flash forward/backward counters on
train and resume, positive flash CUDA-event timing, zero flash fallback, zero
scalar backward tiles, zero hard-require failures, zero materialized-reference
BF16 attention, and positive `cp.async` GEMM execution with zero fallback/hard
failures.
The scale/tuning wrapper is
`scripts/gcp/submit_vertex_qb_memory_flash_scale_gate.sh`. It keeps the same
hard flash and `cp.async` requirements but raises the QB memory hard-path shape
to `block=1024`, `d_model=1024`, `heads=16`, `head_dim=64`, and grad
accumulation `2`. `scripts/run_qb_data_hardpath.sh` now records `shape_gate`,
`flash_attention_metrics`, and `cp_async_gemm_metrics` in `summary.json`, and
`scripts/validate_qb_data_hardpath_artifacts.py` can enforce minimum
block/model/head/token/throughput/MFU thresholds for scale comparisons.
Vertex job `660398552299601920` passed that scale gate: `block=1024`,
`head_dim=64`, grad accumulation `2`, `65536` train tokens seen,
`5892.465383923754` tokens/sec, `dense_core_mfu_estimate=0.0015504495422534528`,
flash forward/backward executed `64/64`, total flash `9.16802978515625` us/token,
`cp.async` GEMM executed `1008`, positive NCCL and sparse row-union evidence,
zero checksum drift, and zero flash/cp.async/materialized-reference fallbacks.
The shape-coverage wrapper is
`scripts/gcp/submit_vertex_qb_memory_flash_head_dim128_gate.sh`, which keeps
`block=1024`, `d_model=1024`, `ff_hidden=4096`, and grad accumulation `2`, but
sets `heads=8` so the exact-tile flash path runs `head_dim=128`. Vertex job
`7673373985324662784` passed this gate with `65536` tokens seen,
`5715.182698177378` tokens/sec, `dense_core_mfu_estimate=0.0014982224574761746`,
flash forward/backward executed `64/64`, total flash `14.773788452148438`
us/token, `cp.async` GEMM executed `1008`, positive NCCL and sparse row-union
evidence, zero checksum drift, and zero flash/cp.async/materialized-reference
fallbacks.
The dedicated measured-throughput wrapper is
`scripts/gcp/submit_vertex_qb_memory_throughput.sh`. It keeps the known-good
`block=1024`, `d_model=1024`, `heads=16`, `head_dim=64`, `ff_hidden=4096` QB
memory hard-path shape, raises grad accumulation to `4`, runs 16 warmup steps,
then treats the 128-step resumed train report as the measured window. It builds
the measured hard-path binary with `HEIRLOOM_QB_DATA_HARDPATH_CARGO_PROFILE=release`,
skips eval/generation, and validates with
`scripts/validate_qb_memory_throughput_artifacts.py`, which reports the measured
tokens/sec, MFU estimates, flash share, NCCL share, optimizer share, and
host-to-device share for bottleneck work. It can also require optional dense GEMM
elapsed timing from diagnostic runs with `--require-cp-async-gemm-timing`.
Vertex job `1689902075711848448` passed the release-profile measured lane:
`2097152` measured tokens, `6078.807167660886` tokens/sec,
`dense_core_mfu_estimate=0.0015686277322282277`,
`end_to_end_mfu_estimate=0.0015645241103662447`, flash `9.052537441253662`
us/token and only `0.05517296518618946` of forward/backward CUDA time,
`cp.async` GEMM executed `32256`, NCCL all-reduce calls/bytes `1024/13385728`,
and zero flash/cp.async/materialized-reference fallbacks. The prior dev-profile
baseline, job `2438906988738904064`, recorded `6035.629795488428` tokens/sec and
`dense_core_mfu_estimate=0.0015616325048544738`, so release mode was necessary
for clean measurement but not the bottleneck. This says the next MFU work should
target dense forward/backward runtime structure before flash attention.
Diagnostic Vertex job `7447965305537560576` then enabled
`HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING=1` for a short `2 + 8` step run and
validated with `--require-cp-async-gemm-timing`: timed `cp.async` GEMM was only
`416223` us across `2016` calls, or `0.019202696445407154` of forward/backward
CUDA time, while flash was `0.05486458994569623`. Rank 0 still recorded
`19942` kernel launches and `3232` syncs over the measured window, so the next
production-readiness target is launch-family attribution and fusion/reduction of
the dominant small elementwise/layout/autograd kernels rather than deeper
`cp.async` GEMM work.
Launch-family attribution is now reportable and gateable. Diagnostic Vertex job
`2782376829070082048` passed the same short release-profile `2 + 8` measured
lane with `--require-kernel-launch-families`: `131072` measured tokens,
`5951.59605866594` tokens/sec,
`dense_core_mfu_estimate=0.001554337522768353`,
`end_to_end_mfu_estimate=0.001531783337112599`, and
`forward_backward_cuda_elapsed_ms=21703.435668945312`. The top rank-0 launch
families were `vector_elementwise` `9270/19942` calls (`46.48%`),
`bias_2d` `3936/19942` calls (`19.74%`), `tensor_core_gemm_cp_async`
`2016/19942` calls (`10.11%`), `materialize_matrix_layout` `1344/19942`
calls (`6.74%`), `layer_norm` `864/19942` calls (`4.33%`),
`matmul_strided_f32_reference` `640/19942` calls (`3.21%`), and
`scaled_vector_elementwise` `592/19942` calls (`2.97%`). Together this confirms
that the next MFU implementation pass should reduce launch count and layout
traffic in the dense block runtime before retuning flash attention or
`cp.async` GEMM.
The first fused launch-reduction slice is complete. `bf16_activation_roundtrip`
now uses a single CUDA `f32 -> bf16 -> f32` roundtrip kernel for CUDA f32
activations, preserving the previous BF16 round-to-nearest-even numerics and
cast-like gradient behavior while replacing two cast launches with one. Vertex
job `7398284972148129792` passed the same short release-profile `2 + 8`
measured lane with `--require-kernel-launch-families`: total rank-0 launches
fell `19942 -> 19302`, `vector_elementwise` fell `9270 -> 7990`, and the new
`bf16_roundtrip` family recorded `640` calls. Forward/backward CUDA elapsed
time moved `21703.435668945312 -> 21630.9130859375` ms and dense-core MFU moved
`0.001554337522768353 -> 0.0015595487947830517`, but end-to-end tokens/sec in
the short lane moved `5951.59605866594 -> 5842.299977713395`, so this should be
treated as a correct launch-count cleanup rather than a proven throughput win.
The second launch-reduction slice fused Linear bias into the exact-tile
double-buffered `cp.async` Tensor Core GEMM store path and added a fused
matmul+bias autograd node so bias gradients still flow through the training
graph. Vertex job `5917310979254779904` passed the same short release-profile
`2 + 8` measured lane with `--require-kernel-launch-families`: total measured
rank-0 launches moved `19302 -> 18630`, `bias_2d` moved `3936 -> 3264`, and
launched elements moved `19912896596 -> 19006926932`. Measured resume
tokens/sec moved `5842.299977713395 -> 5887.701015182823`, but dense-core MFU
stayed effectively flat (`0.0015595487947830517 -> 0.0015533740387336967`), so
this is another correct launch-count cleanup rather than a proven throughput
win. The third launch-reduction slice skips BF16 layout materialization for
already-contiguous row-major operands in CUDA matmul backward. Vertex job
`863075928694063104` passed the same short release-profile `2 + 8` measured
lane with `--require-kernel-launch-families`: total measured rank-0 launches
moved `18630 -> 17958`, `materialize_matrix_layout` moved `1344 -> 672`, and
launched elements moved `19006926932 -> 18100957268`. Measured resume
tokens/sec moved `5887.701015182823 -> 5977.653121722078`, but dense-core MFU
again stayed effectively flat (`0.0015533740387336967 ->
0.001552803693424789`), so this should be treated as another launch/layout
cleanup rather than a proven MFU win. The fourth launch-reduction slice pairs
the two BF16 transposes used to prepare Linear weight-gradient Tensor Core
GEMM. Vertex job `5016520685036503040` passed the same short release-profile
`2 + 8` lane: total measured rank-0 launches moved `17958 -> 17286`,
`bias_2d` moved `3264 -> 1920`, and the new `transpose2d_pair_bf16` family
recorded `672` calls. Forward/backward CUDA elapsed moved
`21724.873901367188 -> 21655.704345703125` ms, but end-to-end tokens/sec moved
`5977.653121722078 -> 5932.470353942246`, so this should also be treated as a
launch-count cleanup rather than a proven MFU win. The next experiment removed
the remaining Linear input-gradient transposed layout materialization with an
exact-tile normal-RHS Tensor Core GEMM. Vertex job `4641877491034619904` passed
the same short release-profile `2 + 8` lane, but it is kept experimental and
off by default behind `HEIRLOOM_CUDA_TENSOR_CORE_NORMAL_RHS_GEMM=1`: total
measured rank-0 launches moved `17286 -> 16614`, `materialize_matrix_layout`
moved `672 -> 0`, `cp.async` GEMM moved `2016 -> 1344`, and the new
`tensor_core_gemm_normal_rhs_staged` family recorded `672` calls, but
forward/backward CUDA elapsed worsened `21538.968505859375 ->
21819.369567871094` ms on rank 0 and aggregate tokens/sec moved
`5932.470353942246 -> 5418.437370814386`. The next MFU slice should turn that
normal/strided RHS case into an `ldmatrix`/`cp.async` instruction-path GEMM,
reduce the remaining `matmul_strided_f32_reference` traffic, or attack the
remaining high-count elementwise/layernorm families.
This is a guarded target-shape gate, not default production routing; ragged
flash backward, broader shape coverage, and tuning remain pending.

The 40% MFU target should be treated as an optimization program after the
training shape, tokenizer, and blend are stable.

## Exit Criteria

This phase is complete when:

- A 32K tokenizer is trained from a governed blend manifest.
- The tokenizer artifact records reserved tokens, source manifest hash, training
  config, and validation metrics.
- The governed materializer emits the production manifest v2 binary-shard
  dataset from approved source slices.
- VECL-QB v1-hard is ingested from `/Users/andrewverdiramo/Desktop/VECL-QB/data`
  with source metadata intact.
- Product-key lookup has parity/stress tests.
- A small learning sanity ladder shows loss improvement.
- Microbench reports exist for GEMM, attention, memory lookup/update, and DDP.
- A longer 4x A100 memory-transformer run improves loss under the production
  data/tokenizer path.
