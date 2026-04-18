# mamba2_triton scope freeze

## In scope
- Standalone Mamba2 extracted under `mamba2_triton/`
- Target input layout: `B x H x W x C`
- Prefill-style forward inference
- Backward training support

## Out of scope
- Decode mode and token-by-token `step`
- Inference cache allocation/state management
- Varlen (`cu_seqlens`) path
- Distributed/tensor-parallel plumbing
- HuggingFace model mixins and LM wrappers

## Step status
- Step-0/Step-1: complete (scope freeze + extraction)
- Step-2: complete (BHWC input and prefill forward path over flattened spatial tokens)
- Step-3: complete (DW 2D conv on xBC branch, scan core still flattened 1D)
- Step-4: complete (2D split+conv+scan glue helper wired into `Mamba2_2D`)
- Step-5: complete (prefill-only public surface tightened; non-runtime selective-scan shim removed)
- Step-7: complete (standalone README + runnable demo script added)

## Packaging intent
- Keep a minimal set of Triton kernels and utility files inside this package tree
- Avoid external compiled dependency on `causal_conv1d` in the standalone forward path
