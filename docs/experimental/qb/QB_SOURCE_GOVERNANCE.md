# QB Source Governance

Status: active
Date: 2026-06-11

This note records the first-pass source review posture for the QB-native
pretraining slices. It is intentionally conservative: source slices can be
stored in GCS only after we can point to the governing source/license evidence
and make an explicit approval decision.

## Operating Policy

Raw external source slices are private/internal training inputs for this
project. We do not publish, sublicense, redistribute, or make the raw slices
available outside the project. Where a source requires attribution, the slice
inventory and model/data cards must preserve that attribution trail. Generated
token shards, model checkpoints, reports, and summaries stay separated from raw
source redistribution.

## Review Ownership

- Codex/Heirloom can perform the first-pass evidence review:
  - dataset card or artifact page,
  - license field,
  - upstream terms of use,
  - redistribution/storage constraints,
  - attribution/removal/opt-out requirements,
  - source URL and object/hash evidence in the slice inventory.
- The project owner is the approval authority for this research build.
- Formal legal review is only needed if this becomes commercial/company policy
  or if a source has ambiguous redistribution terms we still want to use.

## Current Verdicts

| Source | Current verdict | Why |
| --- | --- | --- |
| `vecl_qb.synthetic.v1-hard` | approved internal synthetic | Project-internal synthetic corpus with local metadata, hashes, and no production user data. |
| `allenai.dolma.v1_7` | conditionally approved for internal training with attribution | Dolma is ODC-BY, not Apache-2.0. We can use private/internal slices when attribution is recorded, raw slices are not redistributed, and exact selected objects/sub-sources are hashed in the inventory. |
| `nvidia.nemotron_cc.high_actual` | approved for internal training under NVIDIA Data Agreement | The NVIDIA Data Agreement for Model Training permits "internal training of Company AI Solutions with facts and ideas, including patterns and correlations," requires no attribution, and prohibits redistribution — which this build already satisfies (private slices, never republished). It defines "Datasets, or any portions thereof, that NVIDIA may share," so full-size Nemotron-CC pulled from the NVIDIA / Common Crawl mirror is covered by the same Agreement; record the source URL and object hashes for the copy actually pulled. Not Apache-2.0; no public redistribution. |
| `allenai.dolma3_dolmino_mix-100B-1125` | conditionally approved for internal training with attribution | The OLMo 3 stage-2 annealing pool is published as `allenai/dolma3_dolmino_mix-100B-1125` with license `odc-by`. We can use private/internal slices when attribution is recorded, raw slices are not redistributed, and exact selected objects/sub-sources are hashed in the inventory. |
| `nvidia.nemotron_cc_math` | approved for internal training under NVIDIA Data Agreement | Same NVIDIA Data Agreement as `nemotron_cc.high_actual`: internal training permitted, no attribution required, no redistribution. Full-size Nemotron-CC-Math is covered by the same Agreement rather than needing separate license review; record the source URL and object hashes for the pulled copy. If a specific mirror attaches a different license, record that instead. |

## Evidence Pointers

- Dolma: `https://huggingface.co/datasets/allenai/dolma`
- Related raw Dolma 3 pool: `https://huggingface.co/datasets/allenai/dolma3_pool`
- Dolma 3 Dolmino 100B / OLMo 3 stage-2 annealing mix: `https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125`
- ODC-BY: `https://opendatacommons.org/licenses/by/1-0/`
- Nemotron-CC index: `https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html`
- NVIDIA Nemotron pretraining sample: `https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample`
- NVIDIA Data Agreement for Model Training: `https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/blob/main/LICENSE.md`
- Common Crawl terms: `https://commoncrawl.org/terms-of-use`
- OLMo 3 paper: `https://arxiv.org/abs/2512.13961`
- Nemotron-CC-Math paper: `https://arxiv.org/abs/2508.15096`

## Approval Rule

Do not stage external source slices with `--upload` unless the source is passed
with an explicit approved license status, such as `--dolma-license-status
odc_by_internal_attribution`, `--dolma-license-status odc_by_verified`, or
`--nemotron-cc-license-status nvidia_data_agreement_internal_training`, and the
inventory records the evidence behind that decision.

ODC-BY sources are attribution sources. For this project, they may be staged
into the private GCS artifact bucket for internal model training when the
inventory records the source URL, selected object/path evidence, hashes, and the
fact that raw slices are not redistributed.

NVIDIA Data Agreement sources are internal-training-only sources for this
project. They may be staged into the private GCS artifact bucket for model
training, but raw slices must not be made public, sublicensed, redistributed, or
treated as open-source training data. The agreement also leaves responsibility
for underlying third-party copyrighted material with the user/company.

The staging inventory format records both pending and available sources, so a
blocked source is still visible to the runbook without becoming eligible for a
production hard-path gate.
