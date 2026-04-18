"""Standalone Mamba2 module with 2D input layout and 2D DW-conv front-end."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math

import torch
import torch.nn as nn

from mamba2_triton.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba2_triton.ops.triton.ssd_combined import mamba_split_conv2d_scan_combined


@dataclass(frozen=True)
class Mamba2_2DConfig:
    """Configuration for standalone 2D-layout Mamba2.

    Input layout is fixed to (B, H, W, C).
    """

    d_model: int
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    d_ssm: Optional[int] = None
    ngroups: int = 1
    kernel_size: Union[int, Tuple[int, int]] = (3, 3)
    conv_init: Optional[float] = None
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: Tuple[float, float] = (0.0, float("inf"))
    A_init_range: Tuple[float, float] = (1.0, 16.0)
    D_has_hdim: bool = False
    bias: bool = False
    conv_bias: bool = True
    chunk_size: int = 256


class Mamba2_2D(nn.Module):
    """Prefill+training standalone Mamba2 for BHWC input.

    Implementation notes for Step-3:
    - xBC branch now uses depthwise Conv2d with configurable kernel_size.
    - To avoid explicit memory reorders/copies, tensor layout changes use `permute`
      views only (no `.contiguous()` calls).
    """

    def __init__(self, config: Mamba2_2DConfig, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = config.d_model
        self.d_state = config.d_state
        self.expand = config.expand
        self.d_inner = self.expand * self.d_model
        self.headdim = config.headdim
        self.d_ssm = self.d_inner if config.d_ssm is None else config.d_ssm
        self.ngroups = config.ngroups
        self.D_has_hdim = config.D_has_hdim
        self.rmsnorm = config.rmsnorm
        self.norm_before_gate = config.norm_before_gate
        self.dt_limit = config.dt_limit
        self.chunk_size = config.chunk_size

        assert self.d_ssm % self.headdim == 0, "d_ssm must be divisible by headdim"
        assert self.ngroups > 0 and self.d_ssm > 0
        self.nheads = self.d_ssm // self.headdim
        assert self.nheads % self.ngroups == 0, "nheads must be divisible by ngroups"

        self.kernel_size = self._resolve_kernel_size(config.kernel_size)

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=config.bias, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.dwconv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=config.conv_bias,
            kernel_size=self.kernel_size,
            groups=conv_dim,
            padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2),
            **factory_kwargs,
        )
        if config.conv_init is not None:
            nn.init.uniform_(self.dwconv2d.weight, -config.conv_init, config.conv_init)

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        dt = torch.clamp(dt, min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        assert config.A_init_range[0] > 0 and config.A_init_range[1] >= config.A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*config.A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // self.ngroups,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias, **factory_kwargs)

    @staticmethod
    def _resolve_kernel_size(kernel_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(kernel_size, int):
            k = kernel_size
            if k <= 0:
                raise ValueError(f"kernel_size must be positive, got {kernel_size}")
            return (k, k)
        if len(kernel_size) != 2:
            raise ValueError(f"kernel_size must be int or 2-tuple, got {kernel_size}")
        kh, kw = kernel_size
        if kh <= 0 or kw <= 0:
            raise ValueError(f"kernel_size values must be positive, got {kernel_size}")
        return (kh, kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prefill forward.

        Args:
            x: input tensor with shape (B, H, W, C).

        Returns:
            Tensor with shape (B, H, W, C).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input rank 4 (B,H,W,C), got shape={tuple(x.shape)}")
        b, h, w, c = x.shape
        if c != self.d_model:
            raise ValueError(f"Input channel C={c} does not match d_model={self.d_model}")

        # Linear projection directly on BHWC last-dim.
        zxbcdt = self.in_proj(x)
        A = -torch.exp(self.A_log.float())
        y = mamba_split_conv2d_scan_combined(
            zxbcdt,
            self.dwconv2d.weight,
            self.dwconv2d.bias,
            self.dt_bias,
            A,
            self.D.view(self.nheads, self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            dt_limit=self.dt_limit,
            activation="silu",
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
        )
        return y.reshape(b, h, w, -1)
