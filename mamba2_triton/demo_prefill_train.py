"""Minimal standalone demo for BHWC prefill + backward training.

Run:
    python mamba2_triton/demo_prefill_train.py
"""

from __future__ import annotations

import sys


try:
    import torch
except Exception as exc:  # pragma: no cover - environment-dependent
    print("[demo] torch is not available in this environment.")
    print(f"[demo] import error: {exc}")
    sys.exit(0)

from mamba2_triton import Mamba2_2D, Mamba2_2DConfig


def main() -> int:
    if not torch.cuda.is_available():
        print("[demo] CUDA is not available; demo requires CUDA for Triton kernels.")
        return 0

    device = "cuda"
    dtype = torch.float16

    cfg = Mamba2_2DConfig(
        d_model=128,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=1,
        kernel_size=(3, 3),
        chunk_size=128,
    )
    model = Mamba2_2D(cfg, device=device, dtype=dtype)

    x = torch.randn(2, 16, 16, 128, device=device, dtype=dtype, requires_grad=True)
    y = model(x)
    loss = y.float().mean()
    loss.backward()

    print("[demo] output shape:", tuple(y.shape))
    print("[demo] loss:", float(loss.detach().cpu()))
    print("[demo] grad(x) abs-mean:", float(x.grad.detach().abs().mean().cpu()))
    print("[demo] grad(in_proj.weight) abs-mean:", float(model.in_proj.weight.grad.detach().abs().mean().cpu()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
