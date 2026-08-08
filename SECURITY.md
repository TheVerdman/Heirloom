# Security

Please report suspected vulnerabilities privately through GitHub's security-advisory interface rather than a public issue.

The most security-sensitive code is the checked boundary around unsafe matrix, CUDA Driver, dynamic-library, and NCCL calls in `heirloom-kernels`. Reports involving integer overflow, invalid shapes or strides, allocation-size errors, aliasing, lifetime violations, FFI signatures, or unchecked device pointers are especially useful.

`cargo audit` runs in CI. Heirloom is a research prototype and is not hardened for untrusted model files, untrusted PTX, multi-tenant execution, or production deployment.
