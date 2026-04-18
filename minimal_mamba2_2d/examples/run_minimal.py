"""Minimal forward + backward demo.

Run from repo root:
    python minimal_mamba2_2d/examples/run_minimal.py
"""

import os
import sys

import torch

# Allow running directly from this folder without installation.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from minimal_mamba2_2d import Mamba2_2D


def main() -> None:
    torch.manual_seed(0)

    model = Mamba2_2D(dim=32, expansion=2, kernel_size=3)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)

    y = model(x)
    loss = y.square().mean()
    loss.backward()

    print(f"forward ok: y.shape={tuple(y.shape)}")
    print(f"backward ok: loss={loss.item():.6f}, grad_norm={x.grad.norm().item():.6f}")


if __name__ == "__main__":
    main()
