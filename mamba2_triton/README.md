# mamba2_triton

Standalone Mamba2 runtime (prefill + training backward) for **BHWC** input.

## Scope
- ✅ Input layout: `B x H x W x C`
- ✅ Prefill-style forward
- ✅ Backward training (autograd path)
- ✅ Depthwise 2D conv in xBC branch
- ❌ Decode / KV cache / varlen / distributed paths

## Quick start

```python
import torch
from mamba2_triton import Mamba2_2D, Mamba2_2DConfig

cfg = Mamba2_2DConfig(
    d_model=128,
    d_state=64,
    expand=2,
    headdim=64,
    ngroups=1,
    kernel_size=(3, 3),
    chunk_size=128,
)
model = Mamba2_2D(cfg).cuda()

x = torch.randn(2, 32, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
y = model(x)
loss = y.float().mean()
loss.backward()
print(y.shape)
```

## Demo script

Run:

```bash
python mamba2_triton/demo_prefill_train.py
```

The demo performs:
1. model construction
2. BHWC forward pass
3. scalar loss backward
4. gradient sanity prints

## Notes
- Runtime path is built around `mamba_split_conv2d_scan_combined`.
- Internally, scan core is still flattened 1D over `H*W` tokens.
