# Memory Transformer Design

Status: implemented research prototype; see the main README and `HARD_MODE.md`
for current validation evidence and limitations. Historical workspace notes live under
`docs/history/`.

## Source Summary

Primary sources:

- `Memory Layers at Scale`, arXiv 2412.09764, https://arxiv.org/abs/2412.09764
- `Continual Learning via Sparse Memory Finetuning`, arXiv 2510.15103, https://arxiv.org/abs/2510.15103

`Memory Layers at Scale` replaces selected transformer FFNs with trainable key-value memory layers. For each token, the hidden state is projected to a query, the top-k memory keys are selected, selected scores are normalized with softmax, and selected values are combined. The paper writes the core operation as selecting top-k indices from `Kq`, computing `Softmax(K_I q)`, and returning the weighted value sum `s V_I`.

The paper scales lookup with product keys: the query is split into two halves, two smaller half-key tables are searched, and full key scores are formed from pairwise half-key scores. It also uses shared memory across multiple memory layers, with ablations showing that a small number of centered memory layers improves performance while replacing too many dense FFNs degrades performance. The Memory+ variant adds an input-dependent gate/nonlinearity and output projection around the memory value aggregation. The implementation notes emphasize that memory layers are bandwidth-bound, require custom EmbeddingBag-like CUDA kernels, and need careful backward accumulation into repeated memory rows.

`Sparse Memory Finetuning` uses the sparse activation pattern of memory layers to reduce catastrophic forgetting. Instead of updating all model parameters, it updates memory slots that are highly activated on new data relative to background/pretraining usage. The method ranks memory indices with a TF-IDF-like access score, masks trainable slots, and freezes the rest of the model or most memory rows depending on the finetuning mode. The key practical requirement for Heirloom is that memory access reports must expose selected rows and row usage so an SMFT policy can later choose which rows are trainable.

## Paper Requirements Mapped To Heirloom

Memory placement:

- Paper behavior: replace one or more FFNs, typically centered in depth and not every layer.
- Heirloom config: `memory_layer_indices` explicitly lists which blocks use memory FFNs. A 32-block model can select centered layers such as `[8, 16, 24]` or any explicit set.

Key/value tables:

- Paper behavior: trainable memory keys and values, with values often same dimension as model hidden size and key dimension commonly smaller.
- Heirloom config: `memory_slots`, `memory_key_dim`, `memory_value_dim`, and `memory_heads`. Exact lookup uses one flat table of `[memory_slots, memory_key_dim]` keys and `[memory_slots, memory_value_dim]` values. Product-key lookup uses separate trainable left/right half-key tables plus the same slot value table.

Top-k lookup:

- Paper behavior: sparse top-k retrieval, product-key lookup for large memories.
- Implemented now: `memory_lookup=exact` uses exact top-k over the full key table for small research fixtures.
- Implemented now: `memory_lookup=product-key` adds CPU and CUDA product-key-style candidate generators with separate trainable `product_key_left` and `product_key_right` tables. Square memory rows map to `(left, right)` pairs, left/right side scores are ranked independently, candidate pairs are formed, and selected scores backpropagate into the query and both half-key tables.
- Approximated: this is still not a production product-key memory index. It uses small deterministic candidate beams and scalar kernels, not a tuned large-memory retrieval engine.

Shared memory:

- Paper behavior: selected memory layers share one memory pool.
- Implemented now: `shared_memory` controls whether memory layers share the same key/value tensors. Shared tables are named once under `shared_memory.*` to avoid duplicate state-dict entries.

Gating / Memory+:

- Paper behavior: value aggregation is gated by an input-dependent projection with a SiLU-style nonlinearity and projected back to model width.
- Implemented now: `memory_plus` enables a gated path using an input projection and GELU because Heirloom does not currently expose SiLU or sigmoid. This is an explicit approximation. Output projection is implemented.

Sparse updates and SMFT:

- Paper behavior: finetuning freezes dense model weights and updates only selected memory rows ranked by access relative to a background corpus.
- Implemented now: config carries `memory_update_policy` and `smft_mode`; forward exposes selected rows for access accounting.
- Implemented now: CUDA sparse-row AdamW updates only selected exact key/value table rows for `sparse-rows` and SMFT modes. Product-key sparse-row updates route selected slot rows to value rows and selected `slot / side` / `slot % side` rows for the trainable half-key tables.
- Implemented now: `MemoryAccessCounts` and `SmftRowMask` derive TF-IDF-like trainable row masks from foreground/background selected-row counts. CUDA selected-row tensors use a device-side `i64 -> u64 counts` kernel before the compact count vector is inspected for SMFT artifacts.
- Implemented now: `SparseAdamWRowsUpdate` can carry a Bool row mask, and the CUDA sparse-row AdamW kernel skips selected rows whose mask entry is false.
- Implemented now: `train-memory-lm --smft-row-mask path.json` accepts a persisted `SmftRowMask` and routes sparse memory updates through masked update descriptors. `--smft-access-counts-out`, `--smft-background-counts`, and `--smft-mask-out` persist foreground counts and derived masks for offline SMFT workflows.
- Implemented now: product-key SMFT masks use a conservative projection. Value rows preserve the exact full-slot mask; a left or right half-key row is trainable only if every full slot sharing that half row is trainable. Reports include the projection policy and half-row counts.
- Implemented now: `train-memory-lm --smft-refresh-every N` can refresh the active sparse-row mask during training from accumulated foreground access counts and optional background counts, then apply that refreshed mask before sparse-row AdamW. Reports include refresh cadence, count, last step, active mask source, and active mask size.
- Not implemented yet: true sparse row-gradient buffers, long-running background corpus collection/windowing, distributed online mask refresh synchronization, and exact non-conservative SMFT full-slot mask parity for shared product-key half rows.

## Implemented Now

First implementation target:

- New module `src/memory_transformer.rs`.
- New public structs:
  - `MemoryTransformerConfig`
  - `MemoryLayerConfig`
  - `MemoryFeedForward`
  - `MemoryTransformerBlock`
  - `MemoryTransformerLm`
- CPU memory lookup:
  - query projection,
  - exact top-k over memory keys, or product-key-style candidate top-k over trainable half-key tables for square memories,
  - selected-key scoring through `Tensor::memory_selected_scores` for exact lookup and product-key selected scoring through `Tensor::memory_product_key_selected_scores`,
  - softmax over selected scores,
  - weighted value aggregation,
  - Memory+ style gated value path,
  - output projection,
  - dense FFN fallback for non-memory layers.
- CUDA exact memory routing, first narrow path:
  - query-key scoring uses existing CUDA matmul,
  - exact device-side top-k uses `heirloom_memory_topk_f32`,
  - selected-key score recomputation uses fused `heirloom_memory_selected_scores_forward_f32_i64`,
  - selected-key/query backward uses fused CUDA kernels including atomic selected-row scatter-add into the key table gradient,
  - selected-score softmax uses a rank-2 CUDA dim-1 softmax/backward path,
  - value aggregation uses fused `heirloom_memory_weighted_value_forward_f32_i64`,
  - value/weight backward uses fused CUDA kernels including atomic selected-row scatter-add into the value table gradient,
  - CUDA product-key candidate lookup uses trainable half-key side scores, a device-side candidate-combine kernel, and product-key selected-score forward/backward kernels that keep half-key gradients on device.
- Stable parameter naming:
  - shared memory is named once,
  - per-block memory projections remain block-local,
  - dense blocks use stable dense FFN names.
- `train-memory-lm` reports:
  - selected-row events and unique selected rows per memory layer,
  - foreground SMFT access counts and top accessed rows,
  - memory optimizer path, dense-gradient accumulation status, compact selected-gradient-row gather status, and compressed-transport status,
  - memory table parameter checksums,
  - implemented CUDA memory-kernel surface and counters,
  - a DDP report contract and memory-rank synchronization fields for `Full`, `MemoryOnly`, and row-union `SparseRows` distributed runs.
- Baseline `TinyTransformerLm` remains unchanged.

## Approximations

- Product-key lookup now uses separate trainable half-key tables on CPU and CUDA. It remains approximate because candidate selection is scalar and beam-based, not a tuned large-scale product-key retrieval system.
- GELU approximates Memory+ SiLU gating until a SiLU primitive exists.
- Memory keys and values are dense Tensor parameters. Exact lookup exposes dense `key`/`value`; product-key lookup exposes `product_key_left`/`product_key_right`/`value`. `train-memory-lm --memory-update-policy memory-only` updates only memory tables, `sparse-rows` routes exact rows or product-key half rows through the CUDA sparse-row AdamW bridge when running on CUDA, and `Full` uses dense all-parameter AdamW. Single-rank sparse AdamW gathers compact selected gradient rows on CUDA before the row update. Distributed `sparse-rows` now all-reduces per-table row-union masks with NCCL so every rank applies the same selected memory rows; this is synchronized but still not production sparse-gradient DDP because gradients are all-reduced densely first and row-union mode iterates all rows as sparse-update candidates after building the union mask.
- `--smft-row-mask path.json` attaches a persisted row mask to single-rank sparse updates. Distributed sparse-row DDP can now load the same offline mask on every rank and intersect it with the CUDA row-union mask before sparse AdamW. Distributed background-count mask generation and online refresh remain rejected until refresh decisions can be synchronized across ranks.
- For product-key SMFT, `--smft-row-mask` attaches the exact full-slot mask to the value table and conservative derived masks to the half-key tables. This prevents updating a half-key row that would necessarily alter frozen slots, at the cost of sometimes freezing more half-key parameters than the full-slot mask selected.
- CPU top-k routing materializes CPU query/key data. CUDA tensors use device-side exact top-k or product-key half-key candidate top-k and do not use the CPU fallback.
- CUDA memory backward currently emits dense gradient buffers after selected-row scatter-add. It is CUDA-resident, but it is not yet a true sparse row-gradient buffer pipeline.
- AMP BF16 is only valid for dense Linear/attention paths that already use Heirloom Tensor Core gates. Memory lookup/top-k/selected-score/softmax/weighted-value kernels remain FP32.

## CUDA Primitive Plan

The CUDA path must be added under `heirloom-kernels` with safe wrappers exposed to the main crate. Required kernels:

1. `memory_query_key_scores_f32`
   - Input: queries `[tokens, key_dim]`, keys `[slots, key_dim]`.
   - Output: scores `[tokens, slots]` or a tiled streaming top-k workspace.
   - Purpose: exact baseline lookup for small/medium memory tables.

2. `memory_product_key_scores_f32`
   - Input: query halves, key-half tables.
   - Output: candidate pairs and scores.
   - Purpose: paper-aligned scalable product-key lookup.
   - Implemented now as CPU and CUDA candidate generators over trainable half-key tables, plus product-key selected-score forward/backward for half-key gradients. The remaining gap is production-scale indexing and tuned kernels.

3. `memory_topk_f32`
   - Input: scores or streamed tile candidates.
   - Output: top-k indices and scores per token.
   - Requirement: deterministic tie-breaking by lower slot id.
   - Implemented now: exact scores-to-indices kernel `heirloom_memory_topk_f32` for rank-2 score matrices.

4. `memory_softmax_topk_f32`
   - Input: top-k scores `[tokens, k]`.
   - Output: normalized weights `[tokens, k]`.
   - Implemented now for rank-2 dim-1 softmax through `heirloom_softmax_dim1_f32` and `heirloom_softmax_dim1_backward_f32`.

5. `memory_weighted_value_forward_f32`
   - Input: value table `[slots, value_dim]`, indices `[tokens, k]`, weights `[tokens, k]`.
   - Output: aggregated values `[tokens, value_dim]`.
   - This is the EmbeddingBag-style bandwidth-bound kernel.
   - Implemented now as `heirloom_memory_weighted_value_forward_f32_i64`.

6. `memory_weighted_value_backward_f32`
   - Input: output gradients, selected values, indices, weights.
   - Output: gradients for weights and sparse gradients for selected value rows.
   - Strategies to test: atomics, row locks, and reverse-index grouping.
   - Implemented now as separate weight-gradient and value-gradient kernels. The value-gradient path uses atomic scatter-add into a dense value-table gradient buffer.

7. `memory_selected_key_backward_f32`
   - Input: query gradients from selected scores and selected key rows.
   - Output: query gradients and sparse key-row gradients.
   - Implemented now as `heirloom_memory_selected_scores_backward_query_f32_i64` and `heirloom_memory_selected_scores_backward_keys_f32_i64`. The key-gradient path uses atomic scatter-add into a dense key-table gradient buffer.

8. `memory_scatter_add_rows_f32`
   - Input: row ids and row gradients.
   - Output: accumulated key/value gradients without CPU materialization.
   - Partially implemented for memory keys and values through dense CUDA gradient buffers with selected-row atomic scatter-add. Dedicated sparse row-gradient buffers are not implemented.

9. `memory_gather_selected_rows_f32_i64`
   - Input: a dense CUDA f32 table or dense CUDA f32 gradient buffer plus selected row ids.
   - Output: compact `[selected_rows, row_dim]` CUDA f32 row payload.
   - Purpose: prerequisite for compressed sparse row-gradient transport and row-local optimizer paths.
   - Implemented now as `heirloom_memory_gather_selected_rows_f32_i64`, with `Tensor::cuda_memory_gather_selected_rows_f32_i64` and `Tensor::cuda_memory_gather_selected_grad_rows_f32_i64`. `SparseRows` DDP now uses it after row-union compaction to gather compact memory-gradient payloads before NCCL all-reduce.

10. `memory_sparse_adamw_rows_f32`
    - Input: parameter table, selected row ids, row gradients, first/second moments.
    - Output: updated selected rows and moment rows only.
    - Purpose: SMFT and sparse memory optimizer path.
   - Implemented now as `heirloom_memory_sparse_adamw_compact_rows_f32_i64`: the Tensor optimizer path gathers compact selected gradient rows on CUDA for single-rank sparse updates, and the DDP sparse path can consume externally averaged compact gradient rows. Duplicate selected rows are deduplicated by first occurrence. An optional Bool row mask skips non-trainable selected rows on device. The older dense-gradient-table kernel `heirloom_memory_sparse_adamw_rows_f32_i64` remains available as a reference/fallback surface. `MemoryTransformerLm::memory_sparse_adamw_updates` aggregates latest selected rows without CPU staging for CUDA tensors, `memory_sparse_adamw_updates_with_mask` attaches an `SmftRowMask`, product-key mode conservatively projects full-slot masks to half-key rows, and `train-memory-lm --smft-row-mask` loads a persisted JSON mask for sparse memory modes. Autograd still accumulates dense CUDA gradient buffers inside each rank, but `SparseRows` DDP no longer all-reduces dense memory-table gradients before sparse AdamW.

11. `memory_access_count_rows_i64_u64`
   - Input: selected row ids.
   - Output: row usage counters for SMFT ranking/reporting.
   - Implemented now as `heirloom_memory_access_count_rows_i64_u64`, an atomic CUDA count kernel for CUDA selected-row tensors, plus CPU `MemoryAccessCounts` validation/merge for persisted artifacts and optional online mask refresh in `train-memory-lm`.

12. `memory_selected_rows_to_f32_mask`, `f32_mask_to_bool`, `bool_mask_to_i64_indices`, and `i64_arange`
   - Input: per-rank selected memory rows.
   - Output: a CUDA Bool row-union mask after NCCL f32 sum all-reduce, plus compact CUDA i64 candidate rows for synchronized sparse AdamW.
   - Purpose: distributed sparse-row memory updates without rank drift.
   - Implemented now for `train-memory-lm --distributed nccl --memory-update-policy sparse-rows`. Each rank builds a local row mask, NCCL-sums masks across ranks, converts nonzero counts to Bool, optionally intersects an offline SMFT Bool mask, compacts the final Bool mask to CUDA i64 row ids, gathers local compact gradient rows for those ids, contributes CUDA zero rows when a rank did not touch a unioned row locally, NCCL-sums the compact gradient payload, scales it by `1/world_size` on device, and applies sparse-row AdamW over those compact rows. This preserves synchronization and avoids dense gradient transport for sparse memory-table DDP updates.

Instrumentation counters:

- memory lookup calls,
- selected tokens,
- selected rows,
- top-k calls,
- weighted aggregation forward/backward calls,
- key/value sparse scatter-add calls,
- compact selected-row gather calls,
- compact sparse-gradient all-reduce calls/bytes in DDP rank reports,
- Bool mask to compact i64 row-index calls,
- sparse optimizer row-update calls,
- CPU fallback attempts rejected,
- bytes read/written for key/value tables.

## AMP BF16 Compatibility

Policy:

- Dense attention and Linear projections may use the existing `amp-bf16` Tensor Core path.
- Memory lookup, top-k, softmax over selected scores, and sparse row updates are FP32 until dedicated BF16-safe kernels exist.
- AMP reports must include memory op decisions with `tensor_core=false` and explicit `kernel_path`.
- `HEIRLOOM_REQUIRE_TENSOR_CORES=1` must not force memory lookup to claim Tensor Core use.
- CUDA memory lookup must not materialize tensors on CPU during training.

## Checkpoint Compatibility

Current LM checkpoints assume `TinyTransformerLm`. Memory checkpoints need additive model-family metadata:

```json
{
  "model_family": "memory_transformer",
  "memory_config": { "...": "..." }
}
```

Loading must dispatch by `model_family`; old checkpoints without the field remain TinyTransformer checkpoints for compatibility. This first CPU module can use `save_state_dict`/`load_state_dict` in tests before full checkpoint-family support is wired into `src/checkpoint.rs`.

## DDP Rank Report Compatibility

Memory DDP support must report:

- model family,
- memory layer indices,
- shared memory flag,
- memory table parameter indices included in gradient sync,
- selected row counts per rank,
- sparse row update counts,
- memory kernel counters,
- AMP validation per rank,
- checksum drift including memory tables.

Implemented now:

- `train-memory-lm --devices cuda:... --distributed nccl` launches hidden `train-memory-lm-rank` workers through the shared distributed launcher.
- `Full` and `MemoryOnly` memory update policies are supported; gradients are NCCL averaged before identical local optimizer steps.
- `SparseRows` is supported through a row-union protocol: local selected rows become a CUDA f32 mask, NCCL all-reduce sums that mask, nonzero rows become a CUDA Bool row mask, optional offline SMFT Bool masks are intersected with that union on CUDA, the final mask is compacted to CUDA i64 row ids, and sparse-row AdamW uses those compact candidate rows.
- Rank reports include model family, memory config, memory table parameter indices/counts, memory-gradient presence count, row-union all-reduce calls/bytes/candidate rows, offline SMFT row-mask metadata when present, memory kernel counters, memory table checksums, per-step memory checksum samples, AMP validation, and checkpoint metadata.
- The parent report fails if no NCCL gradient all-reduces are recorded, if sparse-row runs do not record row-union all-reduces, if any rank lacks memory-table gradients, or if memory-table checksum drift exceeds tolerance.

Still deferred:

- Distributed SMFT background-count mask generation and online refresh. Offline row-mask intersection exists, but generated/refresh masks still need synchronized cross-rank decisions before they can run safely.
- Distributed online SMFT mask refresh synchronization.
- Live 2/4-rank A100 memory DDP validation.

## Validation Plan

Always-on CPU tests:

- 32-layer construction with selected memory layers.
- Forward output shape `[batch, time, vocab]`.
- Finite loss.
- Stable memory parameter names.
- State-dict save/load round trip.
- Small fixed-batch training decreases loss.

Gated CUDA tests:

- memory layer forward stays on `cuda:0`;
- backward gradients stay CUDA-resident;
- optimizer step updates selected memory rows;
- one memory-transformer LM CUDA step has finite before/after loss;
- checkpoint reload to CUDA works;
- memory kernel counters prove CUDA kernels ran.

Gated distributed tests:

- 2-rank then 4-rank NCCL fixture after single-GPU CUDA memory kernels pass;
- rank reports prove memory gradients participated in sync;
- memory parameter checksum drift remains zero.
