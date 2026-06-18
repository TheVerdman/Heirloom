export const meta = {
  name: 'heirloom-full-review',
  description: 'Full read-only code review of the Heirloom Rust/Python tensor-runtime codebase, fanned out by component with adversarial verification of high-severity findings',
  whenToUse: 'Comprehensive review of the Heirloom codebase: strengths + issues per component, verified.',
  phases: [
    { title: 'Review', detail: 'one reviewer per component, static read-only review' },
    { title: 'Verify', detail: 'adversarially verify each critical/high finding against the code' },
  ],
}

const ROOT = '/Users/andrewverdiramo/Desktop/Heirloom'

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['component', 'summary', 'strengths', 'issues'],
  properties: {
    component: { type: 'string' },
    summary: { type: 'string', description: '3-5 sentence overall assessment of this component: maturity, correctness posture, biggest risk.' },
    strengths: {
      type: 'array',
      description: 'Concrete things done well, with file references where applicable.',
      items: { type: 'string' },
    },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'severity', 'category', 'location', 'detail', 'recommendation'],
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category: { type: 'string', enum: ['correctness', 'safety', 'security', 'performance', 'maintainability', 'testing', 'docs-drift', 'reproducibility'] },
          location: { type: 'string', description: 'file:line or file path' },
          detail: { type: 'string', description: 'What is wrong and why it matters. Quote the relevant code.' },
          recommendation: { type: 'string', description: 'Concrete fix.' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['isReal', 'confidence', 'correctedSeverity', 'verdict'],
  properties: {
    isReal: { type: 'boolean', description: 'True only if you confirmed the issue against the actual code.' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    correctedSeverity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'not-an-issue'] },
    verdict: { type: 'string', description: 'Evidence: what you read, exact file:line, why the claim holds or fails.' },
  },
}

const COMMON = `You are reviewing the Heirloom codebase (a serious, correctness-first Rust prototype of a PyTorch-like tensor + reverse-mode autograd runtime, plus an opt-in CUDA path, a CLI, Python bindings, and a data pipeline). Repo root: ${ROOT}.

RULES:
- READ-ONLY. Do NOT edit, write, or create any file. Do NOT run \`cargo build/test/clippy/run\` (a separate process holds the build lock; parallel cargo would deadlock on it). You MAY use Read, Grep/rg, and read-only shell (wc, sed -n, head, find).
- Be specific and evidence-based: every issue and every strength must cite file:line and quote the relevant code. No generic advice.
- Judge against what the code CLAIMS to be: a correctness-first prototype that explicitly documents its gaps. Do NOT file "not production-ready" or "missing feature X" as an issue when the docs already disclaim X — instead check whether the docs honestly match the code (doc-drift / overclaiming IS a valid issue).
- Severity: critical = silent wrong results / memory unsafety / data loss; high = correctness bug on a real path, or a safety/security hole; medium = latent bug, fragile invariant, or significant maintainability problem; low = polish.
- Report BOTH strengths and issues. The user explicitly wants "what is done well and what needs improvement."
- If a file is huge, read it in full via multiple Read calls — do not skim. Coverage matters more than speed.`

const COMPONENTS = [
  {
    key: 'tensor-autograd',
    label: 'core tensor + autograd',
    focus: `Core tensor model and reverse-mode autograd.
Files: src/tensor.rs (~5000 lines), src/tensor/autograd.rs (~2800), src/shape.rs, src/storage.rs, src/grad_mode.rs, src/rng.rs, src/error.rs.
Scrutinize: Rc<RefCell> tensor handle and aliasing/overlap hazards (expand zero-strides, as_strided escape hatch); strided-view correctness (transpose/permute/narrow/expand/view/reshape/contiguous); storage version counters and saved-tensor version checks; in-place mutation rejection through overlapping views; autograd graph recording, graph release/retain, retain_grad, grad hooks; correctness of every backward formula vs the forward op; broadcasting in backward (sum-to-shape); no_grad thread-local behavior; RNG determinism/reproducibility. Look for: gradient formulas that are subtly wrong, missing version checks that would let a mutated tensor produce stale grads, panics that should be Results, integer overflow in stride/offset math.`,
  },
  {
    key: 'nn-dispatch-amp',
    label: 'nn + dispatch + amp + extension',
    focus: `Modules, optimizers, dispatch, mixed precision, extension API.
Files: src/nn.rs (~2900), src/dispatch.rs (~925), src/amp.rs (~377), src/extension.rs (~178).
Scrutinize: AdamW and SGD update math (bias correction, weight decay decoupling, epsilon placement, state init/restore); Linear/Embedding/LayerNorm/GELU/CausalSelfAttention/TransformerBlock forward+param wiring; cross-entropy and MSE loss correctness incl. numerical stability (logsumexp, label handling); BF16 activation-rounding AMP path correctness and where rounding is applied; dispatch dtype/device resolution rules and the "unsupported CUDA math returns clear error, never silent CPU fallback" claim; custom unary op forward/backward registry seams. Look for: optimizer state desync on resume, LayerNorm/softmax numerical issues, dispatch rules that mis-resolve dtype.`,
  },
  {
    key: 'memory-transformer',
    label: 'memory transformer (project core)',
    focus: `The memory-augmented transformer — this is the project's central research goal (a QB-native memory LM: small dense backbone + large memory pool).
Files: src/memory_transformer.rs (~2128). Cross-reference tests/memory_transformer.rs and MEMORY_TRANSFORMER_DESIGN.md for intended semantics.
Scrutinize: memory read/write/selection mechanism correctness; how memory slots are addressed, updated, and gradient-tracked; whether the implementation matches the design doc; differentiability of the memory path; block/segment handling; any state that persists across steps and whether it is correctly detached/retained; off-by-one in indexing memory blocks. Look for: places where the memory mechanism is a no-op or leaks gradients incorrectly, mismatch between design doc and code.`,
  },
  {
    key: 'data-tokenizer',
    label: 'data pipeline + tokenizer',
    focus: `Tokenizer and prepared-data pipeline.
Files: src/tokenizer.rs (~1653), src/data.rs (~2066).
Scrutinize: BPE-like training and encode/decode correctness and round-trip; versioning of the tokenizer format; digit isolation; special-token handling; determinism of training (tie-breaking in merge selection — does it produce stable output?); prepared token-data manifest format, train/valid split, batching and dataset state/shuffle determinism; bounds checks on block_size vs sequence length; UTF-8 boundary handling on decode. Look for: non-deterministic merge ordering, panics on empty/short input, silent data corruption in sharding.`,
  },
  {
    key: 'io-checkpoint',
    label: 'npy + checkpoint serialization',
    focus: `Serialization and checkpointing.
Files: src/npy.rs (~285), src/checkpoint.rs (~258).
Scrutinize: .npy header parsing/writing for all supported dtypes (f32/bf16/f64/i64/bool), endianness, fortran_order, shape parsing robustness against malformed headers; state_dict / LM checkpoint save+load round-trip fidelity; resume correctness (optimizer + RNG + step counter restored); version/compat fields. Look for: truncated/oversized buffer reads, dtype confusion, resume that silently drops optimizer state or step count, header injection via untrusted shape string.`,
  },
  {
    key: 'cli',
    label: 'heirloom CLI (12k lines)',
    focus: `The main CLI binary.
Files: src/bin/heirloom.rs (~12000 lines / 487KB — READ IT FULLY across multiple Read calls). Also src/bin/{train,classify,custom}.rs and examples/microgpt_heirloom.rs.
Scrutinize: subcommand surface (tokenizer/data/train-lm/eval-lm/generate and any qb-* / materializer commands); argument validation and error messages; report JSON generation; sampling logic in generate (temperature/top-k/top-p/repetition/frequency/presence penalties, seeding); resume flag handling; the sheer size of this file as a maintainability issue (should it be split into modules?); duplicated logic; unwrap/expect on user-facing paths that should be graceful errors. Look for: panics on bad CLI input, inconsistent flag semantics, copy-pasted blocks that have drifted.`,
  },
  {
    key: 'cuda-kernels',
    label: 'CUDA kernels (unsafe/FFI)',
    focus: `The unsafe CUDA layer — HIGHEST memory-safety risk in the repo.
Files: heirloom-kernels/src/cuda.rs (~19258 lines — READ IT FULLY across many Read calls), heirloom-kernels/src/lib.rs.
Scrutinize: every unsafe block and Driver API FFI call; CudaBuffer ownership/Drop (double-free, leak, use-after-free); pointer/length arithmetic passed to kernels; bounds and dtype assumptions on device buffers; error-code checking after every cu* call (are failures propagated or ignored?); the claim that ALL unsafe is confined here so the main crate can be #![forbid(unsafe_code)] — verify no unsafe leaks across the boundary and that the safe wrapper cannot be driven into UB from safe code; raw PTX/kernel-launch parameter packing; stream/sync correctness; the BF16 Tensor Core / mma / cp.async / ldmatrix paths. Look for: unchecked CUDA error returns, integer overflow in byte-size computations, missing synchronization, soundness holes where a safe API call triggers UB.`,
  },
  {
    key: 'python-bindings',
    label: 'PyO3 bindings + parity harness',
    focus: `Python bindings and the PyTorch parity test harness.
Files: heirloom-python/src/lib.rs (~462), python/tests/test_heirloom_py_parity.py (~226), tools/generate_pytorch_fixtures.py (~214), tests/fixtures/pytorch_parity.json, tests/pytorch_fixture_parity.rs, tests/parity.rs.
Scrutinize: PyO3 binding soundness (panics crossing the FFI boundary, GIL handling, buffer lifetime); whether the parity harness actually compares against PyTorch with meaningful tolerances and covers the ops it claims (constructors, grads, backward, core/view ops, layer norm, embedding, cross-entropy, causal attention); whether the checked fixture can drift from the live PyTorch run (staleness risk); coverage gaps in parity. Look for: parity tests that assert trivially / with tolerances so loose they can't catch bugs, ops claimed-covered but untested.`,
  },
  {
    key: 'python-scripts',
    label: 'QB data + validation Python scripts',
    focus: `The QB-native data-staging and artifact-validation Python scripts.
Files: scripts/slice_hf_dataset.py (~864), scripts/stage_qb_source_slices.py (~523), scripts/validate_qb_data_hardpath_artifacts.py (~548), scripts/validate_qb_memory_throughput_artifacts.py (~477), scripts/validate_memory_fixture_artifacts.py (~433), scripts/validate_padawan_loop.py (~345). Cross-ref QB_NATIVE_DATA_STRATEGY.md, QB_SOURCE_GOVERNANCE.md, QB_PRETRAINING_READINESS.md.
Scrutinize: correctness of the slicing/staging logic; reproducibility/seeding; governance/licensing checks claimed in QB_SOURCE_GOVERNANCE; error handling on missing/malformed inputs; whether validators actually validate (real assertions vs rubber-stamp); hashing/dedup logic; path handling and injection. Look for: validators that pass on bad artifacts, non-deterministic sampling, silent skips.`,
  },
  {
    key: 'padawan',
    label: 'padawan agent-loop design',
    focus: `The "padawan" self-improvement / teacher-loop design and its schemas.
Files: padawan/README.md, padawan/schemas/*.schema.json (4 schemas), padawan/fixtures/* (one full fixture set), scripts/validate_padawan_loop.py (~345), PADAWAN_LOOP_DESIGN.md, DISPATCH.md, OPERATORS.md.
Scrutinize: JSON Schema correctness (do the schemas actually constrain the fixtures? required fields, additionalProperties, types); whether the fixtures validate against the schemas; coherence between the design doc and the schemas/validator; the validator's rigor. Look for: schemas that under-constrain, fixtures that wouldn't validate, design described but not represented in any schema.`,
  },
  {
    key: 'tests-quality',
    label: 'Rust test suite quality',
    focus: `The Rust test suite — assess whether it actually protects the invariants the project cares about.
Files: tests/*.rs (~9360 lines total; notably tests/cuda_storage.rs ~2880, tests/memory_transformer.rs ~2364, tests/transformer_runtime.rs ~1695, plus gradcheck.rs, properties.rs, aliasing_invariants.rs, graph_lifecycle.rs, dtype_dispatch.rs, hard_path.rs, distributed_cli.rs, custom_ops.rs, operator_registry.rs, io_rng_state.rs, pytorch_semantics.rs, tensor_ops.rs, nn_training.rs).
Scrutinize: does gradcheck use finite-difference vs analytic grads with sane tolerances? do property tests (proptest) cover stride/view/broadcast invariants meaningfully? are aliasing/version-counter invariants actually asserted? coverage gaps (which src/ modules have weak/no test coverage — e.g. is the CUDA path tested only behind a feature gate that CI never runs?); tests that are smoke-only (run but assert little); flakiness/determinism. Look for: assert!(true)-style tests, tolerances too loose, large test files that exercise little.`,
  },
  {
    key: 'build-ci-docs',
    label: 'build / CI / docs / hygiene',
    focus: `Project hygiene, reproducibility, and doc/code fidelity.
Files: Cargo.toml, heirloom-kernels/Cargo.toml, heirloom-python/Cargo.toml, pyproject.toml, Cargo.lock, .github/workflows/ci.yml, scripts/validate.sh, scripts/*.sh (all shell scripts), scripts/gcp/*.sh, Dockerfile, .dockerignore, .gitignore. Plus a fidelity check of the major docs (README.md, ARCHITECTURE.md, HARD_MODE.md, ALIASING.md, EXTENDING.md) against the actual code surface.
Scrutinize: does CI actually run the full validate.sh gate or only a subset? is the CUDA path / python-parity ever exercised in CI? dependency pinning and the very recent toolchain (edition 2021, pyo3 0.27); shell-script robustness (set -euo pipefail, quoting, the gcp submit scripts); Dockerfile reproducibility; the fact that the project is NOT in a git repo (no .git) despite heavy reliance on resume/checkpoints and a large uncommitted surface — call out the operational risk; doc claims that overstate implemented status. Look for: CI that would pass while real breakage exists, docs that claim features the code lacks, scripts that fail silently.`,
  },
]

function reviewPrompt(c) {
  return `${COMMON}

## Your component: ${c.label}

${c.focus}

Read every listed file thoroughly (use multiple Read calls for large files — do not skim or sample). Then return findings via the structured schema: a 'summary', a list of concrete 'strengths' (with file refs), and a list of 'issues' (each with severity, category, file:line location, detailed evidence quoting code, and a concrete recommendation). It is fine to return many strengths and few issues, or vice-versa — report what the code actually shows. Do not invent issues to pad the list; do not omit real ones.`
}

function verifyPrompt(c, iss) {
  return `${COMMON}

You are an ADVERSARIAL VERIFIER. A reviewer of the "${c.label}" component filed this finding. Your job is to REFUTE it. Open the actual code at the cited location and the surrounding context, and determine whether the claim is true. Default to "not real" unless the code clearly confirms it. Watch for: misread control flow, a guard/check that exists elsewhere, behavior the docs explicitly disclaim (which makes it not-a-bug), or a severity that is overstated.

FINDING UNDER REVIEW:
- Title: ${iss.title}
- Severity claimed: ${iss.severity}
- Category: ${iss.category}
- Location: ${iss.location}
- Detail: ${iss.detail}
- Recommendation: ${iss.recommendation}

Read the cited file(s) and return your verdict via the schema. In 'verdict', cite the exact file:line you read and quote the decisive code. Set correctedSeverity to 'not-an-issue' if you refute it, otherwise the severity you believe is accurate.`
}

phase('Review')
log(`Reviewing ${COMPONENTS.length} components, each verifies its own critical/high findings as it finishes...`)

const results = await pipeline(
  COMPONENTS,
  (c) => agent(reviewPrompt(c), { label: `review:${c.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }),
  (review, c) => {
    if (!review || !Array.isArray(review.issues)) return { component: c.label, summary: '(reviewer produced no output)', strengths: [], issues: [], verifiedIssues: [] }
    const toVerify = review.issues.filter((i) => i.severity === 'critical' || i.severity === 'high')
    return parallel(
      toVerify.map((iss) => () =>
        agent(verifyPrompt(c, iss), { label: `verify:${c.key}`, phase: 'Verify', schema: VERDICT_SCHEMA })
          .then((v) => ({ ...iss, verdict: v }))
          .catch(() => ({ ...iss, verdict: { isReal: null, confidence: 'low', correctedSeverity: iss.severity, verdict: '(verification failed to run)' } }))
      )
    ).then((verified) => ({ ...review, verifiedIssues: verified }))
  }
)

const clean = results.filter(Boolean)
log(`Review complete: ${clean.length} components. Total issues: ${clean.reduce((n, r) => n + (r.issues ? r.issues.length : 0), 0)}.`)
return clean
