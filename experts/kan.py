"""Kolmogorov-Arnold layer — the paper's fusion head, which this repo never had.

TARKAN specifies `KAN 2 layers x width 256 / grid 5 / spline order 3` as the fusion head
over `[t_a ; v_a ; g_a]`. Chapter A implemented it and measured the fusion family
(KAN vs MLP vs gated) as **flat**; Chapters B/C then moved to a log-average ensemble of
specialist towers, which has no single fusion head, so KAN dropped out entirely.

Standard efficient-KAN formulation: each edge carries a learnable univariate function
represented as `w_base * SiLU(x) + w_spline . B(x)`, with `B` the B-spline basis on a fixed
uniform grid. The base branch is what keeps it trainable — a pure spline layer optimises
badly at this width.

    from experts.kan import KAN
    head = KAN([768, 256, 3])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5,
                 spline_order: int = 3, grid_range=(-1.0, 1.0), scale_noise: float = 0.1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * step
                + grid_range[0])
        # [in, G + 2k + 1] — fixed (not learned); the paper does not specify grid updating
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        with torch.no_grad():
            # small noise, not zeros: a zero-init spline branch plus a zero scale is the
            # dead-gradient trap §D.22 hit in PACS (dL/da = fb = 0 and dL/dfb = a = 0)
            self.spline_weight.normal_(0.0, scale_noise / (grid_size + spline_order))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor recursion. x [B, in] -> bases [B, in, G + k]."""
        grid = self.grid                                   # [in, G + 2k + 1]
        x = x.unsqueeze(-1)                                # [B, in, 1]
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(F.silu(x), self.base_weight)
        # clamp into the grid support; outside it every basis is 0 and the edge goes dead
        xs = x.clamp(self.grid[0, self.spline_order].item(),
                     self.grid[0, -self.spline_order - 1].item())
        spline = F.linear(self.b_splines(xs).view(x.size(0), -1),
                          self.spline_weight.view(self.out_features, -1))
        return base + spline


class KAN(nn.Module):
    """Stack of KANLayers. `[768, 256, 3]` = the paper's 2-layer width-256 head."""

    def __init__(self, widths, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            KANLayer(a, b, grid_size, spline_order)
            for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x
