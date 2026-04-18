import torch
import torch.nn as nn


class Mamba2_2D(nn.Module):
    """A minimal 2D block with depthwise mixing + gated projection.

    This is a lightweight standalone implementation for quick experiments.
    Input / output: (B, C, H, W)
    """

    def __init__(self, dim: int, expansion: int = 2, kernel_size: int = 3):
        super().__init__()
        hidden = dim * expansion
        padding = kernel_size // 2

        self.norm = nn.GroupNorm(1, dim)
        self.in_proj = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gate, value = self.in_proj(x).chunk(2, dim=1)
        value = self.dwconv(value)
        x = self.act(gate) * value
        x = self.out_proj(x)
        return x + residual
