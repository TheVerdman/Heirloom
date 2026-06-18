# 4x A100 32-Block Memory Transformer TinyStories-Valid Milestone

This milestone records the June 9, 2026 4x A100 validation run for Heirloom's memory-augmented decoder-only transformer. It is a runtime, shape, resume, CUDA-kernel, AMP, NCCL, and sparse-memory optimizer gate. It is not a learning-quality or model-quality claim.

## Result

- Vertex job id: `6080483452619063296`
- Display name: `heirloom-validate-quick-20260609-003752`
- Final state: `JOB_STATE_SUCCEEDED`
- Accelerator shape: `4x NVIDIA A100 80GB`
- Project: `project-49b1b523-d248-434f-bd4`
- Region: `us-central1`
- Source package: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/packages/heirloom-source-20260609-003752.tar.gz`
- Artifact prefix: `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/heirloom/reference-runs/heirloom-validate-quick-20260609-003752`

## Configuration

- Data source: `tinystories-valid`
- Source bytes: `19,447,282`
- Prepared train tokens: `7,547,004`
- Prepared valid tokens: `1,886,751`
- Model family: `memory_transformer`
- Layers: `32`
- Memory layers: `[8, 16, 24]`
- `d_model=64`
- `n_heads=4`
- `ff_hidden=256`
- `block_size=64`
- Vocab size: `1024`
- Memory slots: `1024`
- Memory key/value dims: `32/64`
- Memory top-k: `4`
- Memory heads: `1`
- Memory lookup: `exact`
- Shared memory: `true`
- Memory+: `true`
- Memory update policy: `sparse-rows`
- SMFT mode: `masked-memory-rows`
- Precision: `amp-bf16`
- Distributed mode: `nccl`
- World size: `4`
- Per-rank batch size: `1`
- Global batch size: `4`
- Train steps: `100`
- Resume steps: `5`
- Learning rate: `0.0003`

## Metrics

- NCCL preflight: `max_abs_error=0.0`
- Train status: `passed`
- Train step range: `0 -> 100`
- Train loss: `6.8154191970825195 -> 7.045341491699219`
- Train loss reduction: `-0.033735605685872146`
- Resume status: `passed`
- Resume step range: `100 -> 105`
- Resume loss: `7.017897129058838 -> 6.9734978675842285`
- Resume loss reduction: `0.006326576274645922`
- Memory table checksum drift: `0.0`
- AMP validation: `passed`
- Unexpected host staging: `0`
- Train all-reduce calls: `800`
- Train compact sparse-gradient all-reduce calls: `800`
- Train compact sparse-gradient all-reduce bytes: `7,011,840`
- Train row-union all-reduce calls: `800`
- Train row-union candidate rows: `36,520`

## Kernel Evidence

Rank-0 Tensor Core counters during training:

- `bf16_tensor_core_matmul_calls=132900`
- `bf16_tensor_core_matmul_forward_calls=44300`
- `bf16_tensor_core_matmul_backward_calls=88600`
- `bf16_scalar_matmul_fallback_calls=0`
- `bf16_tensor_core_attention_forward_calls=3200`
- `bf16_tensor_core_attention_backward_calls=3200`
- QK, AV, score-gradient, dQ, dK, and dV Tensor Core attention counters were all positive.

Rank-0 CUDA memory kernel counters during training:

- `topk_calls=300`
- `weighted_value_forward_calls=300`
- `weighted_value_backward_calls=300`
- `selected_key_backward_calls=300`
- `scatter_add_rows_calls=600`
- `gather_selected_rows_calls=200`
- `bool_mask_to_indices_calls=200`
- `sparse_adamw_compact_rows_calls=200`
- `selected_rows=76800`

The artifact validator passed with:

```text
memory_fixture_artifacts status=passed distributed=True sparse_rows=True train_compact_gradient_all_reduce_calls=800
```

## Reproduction

Use the documented paid 4x A100 memory fixture command in `scripts/gcp/README.md`. The important gate settings are:

```bash
HEIRLOOM_VERTEX_ACCELERATOR_COUNT=4
HEIRLOOM_RUN_NCCL_PROBE=1
HEIRLOOM_NCCL_PROBE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
HEIRLOOM_RUN_CUDA_TRAIN_MEMORY_LM_FIXTURE=1
HEIRLOOM_CUDA_MEMORY_FIXTURE_DATA_SOURCE=tinystories-valid
HEIRLOOM_CUDA_MEMORY_FIXTURE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
HEIRLOOM_CUDA_MEMORY_FIXTURE_DISTRIBUTED=nccl
HEIRLOOM_CUDA_MEMORY_FIXTURE_PRECISION=amp-bf16
HEIRLOOM_CUDA_MEMORY_FIXTURE_N_LAYERS=32
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_LAYER_INDICES=8,16,24
HEIRLOOM_CUDA_MEMORY_FIXTURE_MEMORY_UPDATE_POLICY=sparse-rows
HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_MODE=masked-memory-rows
HEIRLOOM_CUDA_MEMORY_FIXTURE_SMFT_ROW_MASK=auto
HEIRLOOM_CUDA_MEMORY_FIXTURE_STEPS=100
HEIRLOOM_CUDA_MEMORY_FIXTURE_RESUME_STEPS=5
HEIRLOOM_CUDA_MEMORY_FIXTURE_MIN_REDUCTION=-1.0
zsh scripts/gcp/submit_vertex_heirloom_validate.sh
```

## Limitations

- The loss did not meaningfully improve. The run consumed only about `25,600` training tokens, roughly `0.34%` of the prepared train split, and was intentionally gated as a runtime proof rather than a convergence proof.
- The reported first/final losses are noisy rank-local training losses, not a full heldout evaluation curve.
- The tokenizer is Heirloom's homegrown byte-level BPE-like tokenizer, not a modern production tokenizer.
- The distributed target is single-node 4x A100 only.
- The CUDA kernels are correctness/instrumentation-first and are not cuBLAS/CUTLASS/NCCL-performance parity claims.
- This milestone does not claim PyTorch parity or production model quality.
