# Step-1 extraction map

This directory is a WIP standalone package and currently includes copied kernel/util modules from `mamba_ssm`:

- `ops/triton/ssd_combined.py` (prefill runtime uses `mamba_split_conv2d_scan_combined`)
- `ops/triton/ssd_bmm.py`
- `ops/triton/ssd_chunk_state.py`
- `ops/triton/ssd_state_passing.py`
- `ops/triton/ssd_chunk_scan.py`
- `ops/triton/layernorm_gated.py`
- `ops/triton/k_activations.py`
- `ops/triton/softplus.py`
- `utils/torch.py`
- `utils/determinism.py`

All internal imports were rewritten from `mamba_ssm.*` to `mamba2_triton.*`.
Reference-only selective scan shim has been removed in Step-5.

> Note: Step-2/3/4 functional BHWC prefill path is implemented; Step-5 trims non-runtime paths.
