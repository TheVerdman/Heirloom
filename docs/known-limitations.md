# Known limitations and evidence scope

These limitations were checked against source at `92c0905f45b95fef8b8908b74edbfcdf73379e26` on September 24, 2026. This documentation pass made no runtime fixes and ran no CPU, GPU, or training tests. The findings apply to the paths described below; they do not invalidate every framework path.

## Sparse gradient accumulation

In the inspected memory-training path, each forward replaces `last_selected_rows`. The sparse optimizer descriptors use the latest selection after all accumulated microbatches finish. A row selected only by an earlier microbatch can therefore have an accumulated gradient but receive no update. This affects exact and product-key lookup; shared memory combines the latest selections across layers, not across microbatches. Treat sparse memory updates with `grad_accumulation_steps > 1` as unsupported.

Source: [selection assignment](../src/memory_transformer.rs#L756), [shared and per-layer descriptors](../src/memory_transformer.rs#L1293-L1423), [microbatch loop and subsequent step](../src/bin/heirloom/runtime.rs#L2201-L2259), and [sparse optimizer dispatch](../src/bin/heirloom/runtime.rs#L2416-L2435). Single-step sparse tests do not establish accumulation correctness.

## Policy changes on resume

Memory-model resume constructs the model from saved `memory_config` and restores the saved optimizer. CLI `--memory-update-policy` and `--smft-mode` select policy only for fresh initialization; a conflicting request on resume is neither merged nor explicitly rejected. For example, resuming a saved `full` policy with a `sparse-rows` request can continue full updates. Treat policy transitions through resume as unsupported.

Source: [checkpoint construction](../src/checkpoint.rs#L239-L247), [resume branch](../src/bin/heirloom/runtime.rs#L2087-L2102), and [fresh-model policy assignment](../src/bin/heirloom/runtime.rs#L2126-L2127).

## Nonzero optimizer weight decay

The inspected dense and sparse CPU paths and dense/sparse PTX kernels form `gradient * clip_scale + weight_decay * parameter` before updating both Adam moments. This is coupled L2-style decay. Standard [decoupled AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) keeps decay out of the moments, so the implementation's `AdamW` name does not establish those semantics for nonzero decay.

Source: [dense CPU](../src/nn.rs#L2444-L2453), [sparse CPU](../src/nn.rs#L2524-L2535), [dense PTX](../heirloom-kernels/src/ptx/kernels.ptx#L419-L453), [sparse PTX](../heirloom-kernels/src/ptx/kernels.ptx#L639-L651), and [compact sparse PTX](../heirloom-kernels/src/ptx/kernels.ptx#L779-L791).

The [CPU/CUDA comparison](../tests/cuda_storage.rs#L2651-L2694) uses Heirloom's CPU implementation as its oracle, so it can pass with the same semantics error on both sides. The inspected [unmasked](../tests/memory_transformer.rs#L1280-L1295) and [masked](../tests/memory_transformer.rs#L1356-L1372) sparse tests use zero decay. Those tests do not establish nonzero-decay AdamW parity.

## Evidence scope

Heirloom implements tensor/autograd, CPU/CUDA, training, checkpoint, and sparse-memory research infrastructure. Improved adaptation, retention, or training cost versus a matched dense baseline remains a research objective, not a demonstrated result.

| Evidence | What it supports and what it does not |
| --- | --- |
| September 24, 2026 release audit at the source revision above | The audit recorded 65 selected CPU contract tests passing offline and locked with Rust 1.95.0. That was bounded tensor/autograd/dtype/IO/fixture-parity coverage, with no fresh GPU or training run. Those tests were not rerun in this documentation pass. |
| [Committed learning-sanity reports](../tests/fixtures/learning_sanity/) | Synthetic validator fixtures, not measured learning gains. The [readiness notes](experimental/qb/QB_PRETRAINING_READINESS.md#L443-L445) explicitly distinguish the fixtures from live experiments. |
| [August 8 GPU record](evidence/gpu-validation.md) | Records 61 CUDA tests and one single-rank NCCL test on one A100 at `74307a195a9bba0ad53117efd160df77801445da`. The raw artifacts remain private and were not retrieved or independently revalidated in this pass. Its “current-code” labels refer to that recorded snapshot; the separate older four-A100 record is historical evidence. |
| [June 9 32-block memory-transformer run](history/milestones/MILESTONE_32_BLOCK_MEMORY_TINYSTORIES.md#L23-L66) | A recorded runtime gate at `d_model=64`; training loss increased from about 6.8154 to 7.0453. Successful execution and the separately reported resume segment do not turn this into a learning-improvement result. |

Historical reports, results, source hashes, and artifact hashes are preserved. This pass checked source and documentation only; it did not reproduce private artifacts or run new research experiments.
