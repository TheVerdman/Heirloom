# Public API documentation policy

Heirloom documents contracts in review order, not declaration-count order. The
current priority surface is:

1. tensor layout, storage/device movement, and autograd;
2. modules, optimizers, and checkpoint compatibility;
3. checked CPU/CUDA kernel boundaries;
4. memory-transformer construction, routing evidence, and sparse updates.

The crate roots warn on broken intra-doc links, and the primary tensor/autograd
example is compiled as a doctest. `warn(missing_docs)` is intentionally not yet
enabled across the whole workspace: doing so today would generate hundreds of
low-information comments on experimental reporting/configuration fields.

The staged path is to enable `warn(missing_docs)` module-by-module after the
core contracts above have useful type and method documentation, beginning with
`heirloom-kernels`, then tensor/storage, `nn`, checkpointing, and the stable
memory-transformer entry points. Experimental corpus, CLI reporting, and skill
tracks should remain outside that gate until their public status is resolved.
