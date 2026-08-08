# Experimental GCP validation launchers

These scripts preserve the project’s GPU validation infrastructure. They are
**experimental supporting tooling**, not part of the five-minute review path.
Running a launcher can create billable Vertex AI work; none of these commands
is part of the default validation gate.

The launchers contain no personal project or bucket defaults. Supply an
existing project and artifact bucket explicitly:

```bash
export PROJECT_ID="your-gcp-project"
export BUCKET="gs://your-existing-artifact-bucket"
export REGION="us-central1"
```

The general launcher packages the current checkout, submits a custom Vertex AI
job, and uploads its reports beneath the supplied bucket:

```bash
HEIRLOOM_VERTEX_VALIDATE_MODE=quick \
  zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

The launcher refuses a dirty worktree, packages the exact `HEAD` tree with
`git archive`, and records that revision in both success and failure summaries.
For a bounded current-commit CUDA/NCCL run with the existing microbenchmarks:

```bash
HEIRLOOM_VERTEX_VALIDATE_MODE=quick \
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=1 \
HEIRLOOM_RUN_NCCL_PROBE=1 \
HEIRLOOM_RUN_TENSOR_CORE_MICROBENCH=1 \
  zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

This uses one replica and the launcher’s two-hour timeout. With the default
accelerator it requests one `NVIDIA_A100_80GB` on `a2-ultragpu-1g`. The NCCL
option runs both the launcher probe and the advertised ignored-test lane.

Specialized `submit_vertex_*.sh` wrappers configure the same launcher for
historical CUDA, NCCL, tokenizer, memory-transformer, and throughput gates.
They intentionally retain their research-sized defaults, so inspect the
requested accelerator count and workload before running one.

Corpus workflows additionally require a caller-owned, governed source prefix:

```bash
export HEIRLOOM_QB1_SOURCE_ROOT="gs://your-existing-artifact-bucket/heirloom/source-slices"
bash scripts/gcp/run_local_qb1_tokenizer.sh
```

Set `HEIRLOOM_QB1_TOKENIZER_UPLOAD_PREFIX` only when output upload is desired;
an unset value keeps outputs local under `runs/`.

## Evidence and provenance

The concise public GPU evidence is in
[`docs/evidence/gpu-validation.md`](../../docs/evidence/gpu-validation.md).
The dated infrastructure notes, including historical private object names, are
preserved as provenance in
[`docs/history/gcp/VALIDATION_LOG.md`](../../docs/history/gcp/VALIDATION_LOG.md).
Those historical identifiers are not reusable configuration and do not imply
that reviewers have access to the underlying objects.

For current validation commands and hardware-test classification, see
[`TESTING.md`](../../TESTING.md).
