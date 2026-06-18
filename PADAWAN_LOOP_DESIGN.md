# Padawan Loop Design

Status: draft sidecar design
Date: 2026-06-11

This document begins the Padawan Loop design without changing the synchronous
QB-native pretraining readiness track. It is intentionally additive: no
production corpus weights, tokenizer defaults, materializer gates, or
`train-memory-lm` behavior are changed by this proposal.

## Purpose

Padawan Loop is a post-training and continual-learning system for the
QB-native memory model.

The base pretraining run teaches the model language, code, document structure,
QB trace grammar, tool syntax, evidence markers, and memory/SMFT vocabulary.
Padawan Loop teaches the already-pretrained model how to complete tasks:

- understand the request,
- resolve constraints,
- choose tools,
- act in an environment,
- inspect failures,
- repair the artifact,
- validate the result,
- stop when done,
- update sparse memory safely when the task distribution changes.

The short form:

```text
teacher guides the environment
Padawan acts
verifier grades artifacts
only verified Padawan behavior becomes training signal
```

Teacher output is provenance and context, not a target answer.

## Non-Interference Contract

Padawan Loop must not step on the synchronous pretraining readiness work.

Until the 20B-token production materializer and base learning gates are stable,
Padawan work is limited to:

- sidecar design documents,
- raw JSONL schema design,
- local validators,
- small synthetic fixtures,
- verifier harnesses,
- held-out evaluation definitions,
- post-training recipes that consume finished checkpoints.

Padawan Loop must not:

- alter `QB_PRETRAINING_READINESS.md` exit criteria,
- change the production `0.35/0.25/0.20/0.10/0.10` blend,
- inject Padawan traces into the 20B pretraining target,
- change the default reserved-token registry before tokenizer freeze,
- consume the same A100 run slot as the synchronous readiness gate,
- mutate a base checkpoint in place,
- mix teacher-generated final answers into any training target.

The first implementation should live under sidecar artifact roots such as:

```text
runs/padawan-loop/
gs://vecl-qb-artifacts/padawan-loop/
```

Derived token shards, if added later, should be separate from the production
pretraining manifest and clearly marked as post-training artifacts.

## Existing Substrate

The current Heirloom/QB substrate already contains most of the vocabulary needed
for Padawan traces:

- tokenizer v2 reserved tokens cover chat, tools, tool results, memory, SMFT,
  traces, evidence, citations, documents, temporal fields, and QB references;
- `QB_NATIVE_DATA_STRATEGY.md` already defines the
  `STATE -> DELTA -> ACTION -> EVIDENCE -> ANSWER` thesis;
- `train-memory-lm` reports memory selection, SMFT access counts, sparse-row
  update metadata, and CUDA memory-kernel counters;
- `data materialize-blend` separates raw source governance from derived token
  shards.

The initial Padawan format should reuse existing reserved tokens instead of
adding new ones. If the first pilots prove that Padawan-specific markers are
worth atomic tokens, add them through an explicit tokenizer v3 discussion after
the 32K readiness path is no longer at risk.

## Core Roles

### Curriculum Service

Chooses tasks at the edge of the Padawan checkpoint's current capability.

Inputs:

- target skill,
- checkpoint identity,
- recent failure classes,
- verifier coverage,
- source governance limits,
- compute budget.

Outputs:

- task seed,
- domain,
- expected artifact type,
- verifier contract,
- guidance budget.

### Teacher

The teacher may provide:

- mission briefing,
- constraints,
- rubric,
- pitfalls,
- allowed tools,
- hidden-test intent,
- hints.

The teacher must not provide:

- final code,
- final proof,
- final answer,
- a complete plan that is effectively the artifact,
- private hidden-test contents.

Teacher tokens are stored for provenance and analysis, but masked from target
loss. Later filtering should down-rank episodes whose success depends on
teacher scaffolding.

### Guidance Levels

Guidance should correlate with task difficulty and the task's estimated distance
from the current Padawan checkpoint's learned parameter space.

Parameter-space distance is not directly observable, so the loop should use
operational proxies:

- recent pass rate for the task family,
- novelty of schemas, tools, APIs, or domains,
- distance from prior task clusters in an embedding or retrieval index,
- verifier failure diversity,
- number of required tool/environment steps,
- memory pressure and row novelty,
- low confidence or low likelihood under the current checkpoint, if measured.

Initial guidance levels:

| Level | Name | Teacher may provide | Default training use |
| --- | --- | --- | --- |
| 0 | Teacherless | no teacher context | highest-weight SFT/RLVR data |
| 1 | Rubric | constraints, rubric, grading criteria | high-weight if verified |
| 2 | Hints | pitfalls, scoped hints, missing checks | medium-weight unless replayed |
| 3 | Decomposition | subgoal outline without artifact details | low-weight; replay required for high-weight SFT |
| 4 | Rescue | post-failure diagnosis or narrow unblocker | repair/curriculum data, not positive SFT by default |

A high-difficulty, high-distance task can legitimately receive more guidance.
An episode is over-guided when the teacher guidance is stronger than the
computed difficulty/distance band, contains artifact-level details, or succeeds
only while the guidance is present. Over-guided successes should be routed to
repair, preference, or curriculum analysis unless a reduced-guidance replay
also passes.

### Padawan Worker

The Padawan is the model being trained or evaluated. It receives a task and
possibly a teacher briefing, then produces a structured work trace plus an
artifact.

The work trace should be structured and inspectable, not unconstrained private
chain-of-thought. It should include both structured process fields and a
compressed natural-language rationale field. The rationale is a short
post-action explanation of why the Padawan chose its approach, not a long hidden
scratchpad.

Preferred trace fields:

- task understanding,
- assumptions,
- compressed rationale,
- subgoals,
- tool calls,
- observations,
- failed checks,
- repairs,
- final validation,
- final artifact reference.

### Environment And Verifier

The verifier grades artifacts, not prose plausibility.

Examples:

- code compiles and tests pass,
- JSON validates against schema,
- tool call executes with expected result,
- cited claim is supported by cited evidence,
- memory-row budget and SMFT mask are respected,
- no unrelated files changed,
- no secret or unsafe output was produced.

### Replay Scheduler

After a guided success, the scheduler creates a reduced-guidance replay:

```text
same task family
similar difficulty
less teacher context
same verifier class
```

Successful teacherless or minimally guided attempts become the highest-value
training data.

## Episode Lifecycle

```text
1. sample task seed
2. construct environment and verifier
3. request teacher briefing, if guidance_level > 0
4. run Padawan worker in an isolated workspace
5. collect tool calls, observations, artifact hashes, and memory telemetry
6. run verifier
7. classify failures
8. schedule repair or replay
9. store raw episode JSONL plus artifact bundle
10. filter into SFT, preference, repair, RLVR, and SMFT datasets
```

The strongest episode sequence is:

```text
teacher-guided attempt
verified artifact
teacherless replay
verified artifact
training selection
```

This prevents the model from becoming dependent on frontier-model scaffolding.

## Episode Schema

Raw Padawan episodes should remain JSONL source records. Tokenized records are
derived artifacts.

Draft record:

```json
{
  "episode_id": "padawan_000001",
  "source_id": "heirloom.padawan.v0.guided",
  "created_at": "2026-06-11T00:00:00Z",
  "checkpoint_ref": "artifact://checkpoints/qb-base-step-...",
  "task": {
    "domain": "code",
    "task_kind": "heirloom_runtime_patch",
    "prompt": "Add a schema validator for Padawan episode JSONL.",
    "artifact_type": "git_diff",
    "difficulty": 0.42
  },
  "teacher": {
    "model": "frontier-teacher",
    "guidance_level": 2,
    "brief_ref": "artifact://padawan/briefs/padawan_000001.txt",
    "teacher_tokens_masked_from_loss": true,
    "forbidden_outputs_checked": true
  },
  "padawan": {
    "model": "qb-memory-1b-candidate",
    "trace_ref": "artifact://padawan/traces/padawan_000001.json",
    "compressed_rationale_ref": "artifact://padawan/rationales/padawan_000001.txt",
    "artifact_ref": "artifact://padawan/artifacts/padawan_000001.diff",
    "final_response_ref": "artifact://padawan/finals/padawan_000001.txt"
  },
  "verifier": {
    "verifier_id": "heirloom.code.patch.v0",
    "tests_passed": true,
    "schema_valid": true,
    "hidden_tests_passed": true,
    "unrelated_changes": 0,
    "reward": 1.0,
    "failure_class": null
  },
  "memory": {
    "selection_report_ref": "artifact://padawan/memory/padawan_000001-selection.json",
    "smft_access_counts_ref": "artifact://padawan/memory/padawan_000001-counts.json",
    "smft_mask_ref": null
  },
  "selection": {
    "eligible_for_sft": true,
    "eligible_for_preference": true,
    "eligible_for_rlvr_replay": true,
    "eligible_for_smft": true,
    "requires_teacherless_replay": true
  }
}
```

Teacher text is retained by reference so audits can detect leakage, but the
training target is the Padawan side only.

## Trace Rendering

Initial rendering should use existing reserved tokens:

```text
<|source_id|>heirloom.padawan.v0.teacherless
<|trace|>heirloom_runtime_patch
<|time_now|>2026-06-11T00:00:00Z
<|user|>...
<|message_end|>
<|state|>...
<|action|>...
<|tool_call|>...
<|tool_result|>...
<|evidence|>...
<|assistant|>...
<|message_end|>
<|record_end|>
```

Rendering modes:

- `guided_context`: includes teacher brief as masked context only.
- `padawan_target`: includes Padawan structured trace and final artifact
  summary as target.
- `repair_target`: includes failed attempt and verifier error as context, with
  corrected Padawan action as target.
- `preference_pair`: stores chosen/rejected trajectories separately for DPO.
- `rlvr_prompt`: stores task and environment contract for fresh online attempts.

Do not flatten artifact bundles into prose when the artifact has its own
canonical format. Store diffs, JSON, logs, reports, and memory telemetry by
content-addressed reference.

## Training Products

One Padawan run can feed several post-training products.

### Positive SFT

Use successful Padawan trajectories:

```text
task + optional masked teacher context -> structured trace + artifact summary
```

Best examples are successful teacherless replays. At the start, teacherless
replay should be required for high-weight positive SFT, not for every positive
example. Guided successes may still enter lower-weight positive SFT when the
artifact is verified and the teacher did not provide artifact-level content.

### Repair SFT

Use failure correction:

```text
task + failed attempt + verifier error -> corrected attempt
```

This is likely high value for a small operational model because it teaches
inspection and recovery.

### Preference Data

Pair successful or robust trajectories against failed, brittle, or over-guided
ones:

```text
chosen: verified, minimal, evidence-faithful
rejected: failed tests, invalid schema, unsupported citation, unrelated edits
```

### RLVR

Run fresh Padawan attempts against verifiers:

```text
reward = artifact correctness + schema validity + evidence support + minimality
```

RLVR should start with small bounded tasks, then expand only after verifiers
prove difficult to game.

First RLVR verifier families should be the ones with the hardest objective
checks:

1. JSON/schema and tool-call formatting tasks with exact validators.
2. Deterministic data transformation tasks with golden outputs.
3. Small code patch tasks in isolated fixtures with unit and hidden tests.
4. Arithmetic and symbolic tasks checked by exact evaluators.
5. Local evidence/citation tasks where claims must map to exact source spans.
6. Temporal reasoning tasks with a fixed clock and exact expected dates.
7. Memory/SMFT fixtures that verify telemetry, row budgets, and mask behavior.

Soft research summaries, live web tasks, broad UI work, and subjective writing
should not be first-wave RLVR domains. They can enter SFT or preference data
earlier, but RLVR should begin where the reward is hard to spoof.

### SMFT And Memory Updates

For memory checkpoints, successful episodes can produce foreground access
counts. Compare those counts against background counts to derive sparse memory
row masks.

Initial policy:

- offline masks only for distributed runs,
- single-rank online refresh only in small pilots,
- no background-count generation inside the production pretraining job,
- no mutation of base checkpoints in place.

SMFT eligibility should require:

- verified task success,
- memory telemetry present for every configured memory layer,
- no forbidden row updates,
- foreground rows with lift over background access,
- replay or repeated-task stability for the top accessed rows,
- row use within a strict task-family budget.

Initial Padawan budgets should be stricter than generic runtime smoke-test
defaults:

```text
per-episode trainable row cap per memory table = max(32, min(4096, 0.005 * memory_slots))
per-task-family rolling mask cap per memory table = max(128, min(16384, 0.02 * memory_slots))
foreground/background lift target = at least 4x for promoted rows
top-row replay stability target = at least 0.30 Jaccard overlap across successful replays
```

These are calibration defaults, not final constants. If early runs show that the
model spreads useful task state across many rows, the budget should rise slowly
and only with held-out retention checks.

## Reward Design

Reward must be domain-specific and verifier-backed.

Code:

```text
+1.0 tests pass
+0.2 hidden tests pass
+0.1 lint/format clean
+0.1 minimal diff
-0.5 unrelated file edits
-1.0 hardcoded test hack
-1.0 unsafe command or secret exposure
```

JSON/tool calls:

```text
+1.0 schema valid
+0.5 fields semantically correct
+0.2 no extra prose
-0.5 invalid tool call
-0.5 unnecessary tool call
```

Evidence tasks:

```text
+1.0 all claims supported
+0.5 complete evidence coverage
-1.0 unsupported claim
-0.5 citation mismatch
-0.5 stale temporal assumption
```

Memory/SMFT:

```text
+1.0 correct task result
+0.5 foreground rows captured
+0.3 row budget respected
+0.3 background rows preserved
-1.0 forbidden row update
-0.5 product-key collision instability
```

The reward should be recorded as components, not only as a scalar.

## Curriculum

Start with domains where verifiers are strong:

- Heirloom runtime/codebase tasks,
- Rust/Python schema validation,
- JSON and tool-call formatting,
- local document extraction with exact citations,
- arithmetic and symbolic checks,
- memory access and SMFT mask fixtures,
- long-thread temporal continuation,
- QB routing and specialist handoff.

Avoid early domains where grading is mostly subjective. Padawan Loop should earn
trust through hard verifiers before moving into softer workflows.

## Quality Gates

An episode is not trainable until it passes the relevant gates:

- raw JSONL schema valid,
- all artifact references resolvable,
- artifact bundle hashes recorded,
- teacher output checked for forbidden final artifacts,
- teacher tokens masked from loss,
- verifier result reproducible,
- hidden tests or alternate checks available where possible,
- no unrelated workspace edits,
- no leaked secrets or credentials,
- source license/provenance recorded,
- train/valid/eval split assigned before tokenization,
- benchmark contamination check applied for known evals,
- memory telemetry present when `eligible_for_smft=true`.

Failure data is valuable, but it must be explicitly labeled as failure or
repair context. Failed trajectories should never be silently mixed into positive
SFT.

### Artifact And Hidden-Test Audit Storage

Training context and audit context should be separated.

Training-visible episode records may contain:

- artifact content hashes,
- artifact sizes and media types,
- public verifier ID and version,
- scalar and component rewards,
- opaque hidden-test result summaries.

Training-visible episode records must not contain:

- hidden-test source,
- hidden expected outputs,
- hidden-test prompts,
- private judge rubrics that reveal the answer.

The audit bundle should store a restricted verifier manifest:

```json
{
  "verifier_pack_id": "heirloom.code.patch.v0.20260611",
  "verifier_pack_hash": "sha256:...",
  "public_verifier_hash": "sha256:...",
  "hidden_verifier_hash": "sha256:...",
  "hidden_case_ids_hmac": ["hmac-sha256:..."],
  "artifact_hashes": {
    "diff": "sha256:...",
    "stdout": "sha256:...",
    "report": "sha256:..."
  },
  "result_digest": "sha256:...",
  "restricted_bundle_ref": "artifact://restricted/verifiers/..."
}
```

Use canonical serialization before hashing. Hidden case identifiers should be
HMACs with an audit key, not raw names, so auditors can reproduce identity
checks without making the hidden tests learnable from training data.

## Relationship To Pretraining Readiness

Padawan Loop is downstream of the synchronous readiness track.

### Now

Safe work now:

- finalize this design,
- define JSONL schemas,
- write validators,
- define verifier contracts,
- create tiny synthetic fixtures,
- design eval suites.

Unsafe work now:

- changing production corpus weights,
- adding Padawan traces to the 20B target,
- changing tokenizer reserved tokens,
- using scarce A100 readiness windows for Padawan pilots.

### After Learning Sanity Ladder

Safe next step:

- run tiny local Padawan fixtures against a small checkpoint,
- generate repair examples and verifier reports,
- inspect memory access telemetry,
- keep all artifacts under `runs/padawan-loop/`.

### After Base 20B Checkpoint

Safe next step:

- run teacher-guided and teacherless Padawan episodes,
- build post-training SFT/repair/preference datasets,
- run small SFT or LoRA-like adaptation if available,
- run SMFT memory-only pilots from verified foreground access counts.

### After First Post-Training Gate

Safe next step:

- schedule continual Padawan rollouts,
- promote stable task families into RLVR,
- compare dense SFT, memory-only SMFT, and hybrid updates,
- add only the best stable Padawan source families to future governed blends.

Padawan datasets may flow into future pretraining sources if they prove
fruitful, but not into the current 20B-token readiness target. Promotion into a
future pretraining blend requires normal source governance: provenance,
dedupe/contamination checks, verifier quality reports, license status, and a
clear split between raw source JSONL and derived token shards.

## Initial Milestones

P0 design sidecar:

- `PADAWAN_LOOP_DESIGN.md` exists and does not modify readiness work.

P1 schema:

- done for the first sidecar slice:
  - `padawan/schemas/padawan_episode_v0.schema.json`,
  - `padawan/schemas/padawan_teacher_brief_v0.schema.json`,
  - `padawan/schemas/padawan_verifier_result_v0.schema.json`,
  - `padawan/schemas/padawan_artifact_bundle_v0.schema.json`.

P2 validators:

- done for the first sidecar slice:
  - `heirloom padawan validate`,
  - implemented in `src/bin/heirloom.rs`,
  - validates raw episode JSONL,
  - verifies artifact references and content hashes,
  - checks teacher masking and forbidden-output flags,
  - checks high-weight guided replay tracking,
  - checks SMFT eligibility evidence.

Run:

```bash
cargo run --bin heirloom -- padawan validate \
  --episodes padawan/fixtures/episode_valid.jsonl \
  --artifact-root padawan/fixtures
```

P3 local verifier harness:

- done for the first local sidecar slice:
  - `heirloom padawan verify`,
  - implemented in `src/bin/heirloom.rs`,
  - code patch verifier,
  - JSON/tool-call verifier,
  - evidence/citation verifier,
  - memory/SMFT telemetry verifier.

Run:

```bash
cargo run --bin heirloom -- padawan verify \
  --episodes padawan/fixtures/episode_valid.jsonl \
  --artifact-root padawan/fixtures
```

P4 tiny pilot:

- run a small local Padawan episode set after the learning sanity ladder,
- no production corpus changes,
- no base checkpoint mutation.

P5 post-base pilot:

- run guided and teacherless episodes from the first base checkpoint,
- export SFT, repair, preference, RLVR, and SMFT candidate datasets,
- evaluate against held-out task families.

## Resolved Decisions

- Padawan traces include structured process fields and a compressed
  natural-language rationale field.
- Guidance level correlates with task difficulty and estimated distance from
  the current Padawan checkpoint's learned space.
- Successful teacherless replay is required for high-weight positive SFT at the
  start, not for every positive example.
- First RLVR domains should use hard deterministic verifiers.
- Hidden-test contents stay out of training records; audit reproducibility comes
  from canonical hashes, restricted verifier bundles, and HMAC-hidden case IDs.
- SMFT eligibility starts with strict row budgets and foreground/background lift
  requirements.
- Fruitful Padawan datasets may be promoted into future governed pretraining
  sources, after the current readiness path is complete and after normal source
  governance.

## Calibration Questions

- What empirical guidance-level thresholds best predict teacherless replay
  success?
- How much compressed rationale improves task transfer before it becomes noisy
  or theatrical?
- Which verifier families remain robust after the Padawan sees thousands of
  similar episodes?
- What SMFT row budgets preserve old skills while adapting quickly to new task
  families?

## Design Rule

Do not train the model to imitate the teacher. Train it to complete verified
tasks with its own actions, artifacts, repairs, and memory updates.
