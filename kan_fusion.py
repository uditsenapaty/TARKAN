"""KAN-based multimodal fusion (paper Eqs. 18-20) + all Table-10 fusion strategies.

    u_k = [t_k ; v_tilde_k ; g_tilde_k]                                  (Eq. 18)
    z_{l+1,j} = sum_i psi^(l)_ij(z_{l,i})   (learnable univariate edges)  (Eq. 19)
    z_k = KAN(u_k)                                                        (Eq. 20)

KAN backend is selectable (config.kan_backend):
  - 'efficient_kan' (default): B-spline edges (closest to Eq. 19), pip git install.
  - 'fastkan'      : Gaussian-RBF KAN (pip fastkan), faster.
  - 'rkan'         : rational-KAN activations (paper ref [41]).
A self-contained RBF-KAN (RBFKAN) is vendored as a zero-dependency fallback so the
fusion ALWAYS runs (it is a genuine KAN: learnable univariate edge functions).

Every fusion in FUSION_REGISTRY implements forward(t, v, g) -> z, mapping three
[K, d] modality tensors to a fused [K, d] (Table 10).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


# --------------------------------------------------------------------------- #
# Vendored fallback KAN (Gaussian-RBF edges) — always available.
# --------------------------------------------------------------------------- #
class RBFKANLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, num_grids: int = 8, grid_min: float = -2.0, grid_max: float = 2.0):
        super().__init__()
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_grids))
        self.h = (grid_max - grid_min) / max(num_grids - 1, 1)
        self.coeff = nn.Parameter(torch.randn(out_f, in_f, num_grids) * 0.1)
        self.base = nn.Linear(in_f, out_f)
        self.norm = nn.LayerNorm(in_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        rbf = torch.exp(-(((xn.unsqueeze(-1) - self.grid) / self.h) ** 2))  # [*, in, G]
        spline = torch.einsum("...ig,oig->...o", rbf, self.coeff)            # [*, out]
        return spline + self.base(F.silu(xn))


class RBFKAN(nn.Module):
    def __init__(self, layers_hidden: List[int], num_grids: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [RBFKANLayer(layers_hidden[i], layers_hidden[i + 1], num_grids) for i in range(len(layers_hidden) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _build_kan(backend: str, widths: List[int], grid_size: int, spline_order: int) -> nn.Module:
    backend = (backend or "efficient_kan").lower()
    if backend == "efficient_kan":
        try:
            from efficient_kan import KAN  # B-spline KAN

            return KAN(layers_hidden=widths, grid_size=grid_size, spline_order=spline_order)
        except Exception:
            pass  # fall through to RBF fallback
    if backend == "fastkan":
        try:
            from fastkan import FastKAN

            return FastKAN(layers_hidden=widths, num_grids=max(grid_size + 3, 8))
        except Exception:
            pass
    if backend == "rkan":
        try:
            return _RationalKAN(widths)
        except Exception:
            pass
    # universal fallback (genuine RBF-KAN)
    return RBFKAN(widths, num_grids=max(grid_size + 3, 8))


class _RationalKAN(nn.Module):
    """Linear layers with rational-KAN activations (paper ref [41])."""

    def __init__(self, widths: List[int]):
        super().__init__()
        from rkan.torch import JacobiRKAN  # type: ignore

        blocks = []
        for i in range(len(widths) - 1):
            blocks.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                blocks.append(JacobiRKAN(3))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Fusion strategies (Table 10). All: forward(t, v, g) -> z  ([K, d] each).
# --------------------------------------------------------------------------- #
class KANFusion(nn.Module):
    def __init__(self, d: int = None, hidden=None, backend: str = None, grid_size: int = None,
                 spline_order: int = None, dropout: float = None, in_extra: int = 0):
        super().__init__()
        d = d or CONFIG.hidden_dim
        hidden = list(hidden if hidden is not None else CONFIG.kan_hidden)
        # in_extra: appended confidence features (A9) widen the input beyond [t; v; g]
        widths = [3 * d + in_extra, *hidden, d]
        self.in_extra = in_extra
        self.net = _build_kan(
            backend or CONFIG.kan_backend, widths,
            grid_size or CONFIG.kan_grid_size, spline_order or CONFIG.kan_spline_order,
        )

    def forward(self, t, v, g, conf=None):
        parts = [t, v, g]
        if self.in_extra:
            if conf is None:
                conf = t.new_zeros((*t.shape[:-1], self.in_extra))
            parts.append(conf)
        return self.net(torch.cat(parts, dim=-1))  # Eq. 18 -> Eq. 20 (+A9 confidence features)


class ConcatLinear(nn.Module):
    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        self.lin = nn.Linear(3 * d, d)

    def forward(self, t, v, g):
        return self.lin(torch.cat([t, v, g], dim=-1))


class ConcatMLP(nn.Module):
    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.net = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, t, v, g):
        return self.net(torch.cat([t, v, g], dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        self.proj = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.gate = nn.Linear(3 * d, 3)

    def forward(self, t, v, g):
        mods = [self.proj[i](x) for i, x in enumerate([t, v, g])]  # each [K, d]
        w = torch.softmax(self.gate(torch.cat([t, v, g], dim=-1)), dim=-1)  # [K, 3]
        stacked = torch.stack(mods, dim=1)                          # [K, 3, d]
        return (w.unsqueeze(-1) * stacked).sum(dim=1)               # [K, d]


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, d: int = None, heads: int = 8, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, t, v, g):
        tokens = torch.stack([t, v, g], dim=1)        # [K, 3, d]
        out, _ = self.attn(tokens, tokens, tokens)    # [K, 3, d]
        return self.norm(out.mean(dim=1))             # [K, d]


class BilinearFusion(nn.Module):
    """Low-rank bilinear pooling (MLB): W_o (W_t t (*) W_v v).

    A full `nn.Bilinear(d, d, d)` is a d^3 tensor — 453 M parameters at d=768, two orders
    of magnitude past every other row of Table 14 and not what "bilinear fusion" means in
    the MABSA literature. The standard low-rank factorisation is used instead, so the
    comparison is between fusion *mechanisms* and not between parameter budgets.
    """

    def __init__(self, d: int = None, dropout: float = None, rank: int = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        rank = rank or d
        dropout = CONFIG.dropout if dropout is None else dropout
        self.pt = nn.Linear(d, rank)
        self.pv = nn.Linear(d, rank)
        self.pg = nn.Linear(d, rank)
        self.out = nn.Linear(rank, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, t, v, g):
        z = self.pt(t) * self.pv(v) + self.pt(t) * self.pg(g)
        return self.out(self.drop(z))


class TensorFusion(nn.Module):
    """Low-rank tensor fusion (LMF; Liu et al. 2018) — tractable outer-product fusion."""

    def __init__(self, d: int = None, rank: int = 4, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        self.rank = rank
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.randn(rank, d + 1, d) * 0.1) for _ in range(3)]
        )

    def forward(self, t, v, g):
        zs = []
        for x, W in zip([t, v, g], self.factors):
            ones = x.new_ones((x.size(0), 1))
            xb = torch.cat([x, ones], dim=-1)               # [K, d+1]
            zs.append(torch.einsum("kd,rdo->kro", xb, W))   # [K, rank, d]
        fused = zs[0] * zs[1] * zs[2]                        # [K, rank, d]
        return fused.sum(dim=1)                              # [K, d]


class InteractionKANFusion(nn.Module):
    """NOVEL (not in the paper) — interaction-KAN.

    Eq. 18 hands the KAN one blunt concatenation `[t; v~; g~]` and asks it to discover
    cross-modal structure inside it. Here each stream is projected to `dproj` with its own
    LayerNorm (so the KAN does not spend capacity learning that text is an order of
    magnitude larger than the evidence streams) and the interaction variables are handed
    over EXPLICITLY:

        x = [t', v', g', t'(*)v', t'(*)g', |t' - v'|]   ->  KAN  ->  d

    scaled by a learnable alpha initialised **0.1 and not zero**: a zero branch times a
    zero scale is an identically-dead gradient. models.py already adds the result
    residually to h^t, so the text path is the residual baseline.

    Zero-preservation matters here. Tokens outside any aspect carry v = g = 0, and
    `evidence_dropout` zeroes evidence for whole instances to teach text-only extraction.
    LayerNorm(0) is the bias, not 0, so the streams are re-masked after normalisation —
    without that, "no evidence" would silently become "a constant evidence vector".
    """

    def __init__(self, d: int = None, dropout: float = None, dproj: int = None,
                 hidden=None, backend: str = None, grid_size: int = None,
                 spline_order: int = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dproj = dproj or getattr(CONFIG, "ikan_dproj", 192)
        hidden = list(hidden if hidden is not None else CONFIG.kan_hidden)
        self.pt, self.pv, self.pg = (nn.Linear(d, dproj) for _ in range(3))
        self.nt, self.nv, self.ng = (nn.LayerNorm(dproj) for _ in range(3))
        self.net = _build_kan(
            backend or CONFIG.kan_backend, [6 * dproj, *hidden, d],
            grid_size or CONFIG.kan_grid_size, spline_order or CONFIG.kan_spline_order,
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # RESIDUAL form: z = z_text + alpha * KAN(interactions). Without this the fused
        # aspect representation is ONLY the correction — the paper's ASC head (Eq. 25)
        # then reads a small delta with no text baseline under it, which is why a strong
        # text signal could not survive into the polarity path.
        self.residual = bool(getattr(CONFIG, "ikan_residual", True))
        self.z_text = nn.Linear(d, d) if self.residual else None

    def forward(self, t, v, g):
        mv = (v.abs().sum(-1, keepdim=True) > 0).to(v.dtype)
        mg = (g.abs().sum(-1, keepdim=True) > 0).to(g.dtype)
        t_ = self.nt(self.pt(t))
        v_ = self.nv(self.pv(v)) * mv
        g_ = self.ng(self.pg(g)) * mg
        x = torch.cat([t_, v_, g_, t_ * v_, t_ * g_, (t_ - v_).abs() * mv], dim=-1)
        dz = self.alpha * self.net(x)
        return (self.z_text(t) + dz) if self.residual else dz


class HierarchicalAttentionFusion(nn.Module):
    """Paper Table 14 row "Hierarchical Attention" — the strongest non-KAN alternative.

    Two levels, as the name implies:
      level 1  text attends to each evidence stream separately -> c_v, c_g
      level 2  a second attention pools {t, c_v, c_g} into the fused vector
    """

    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.q1 = nn.Linear(d, d)
        self.k1 = nn.Linear(d, d)
        self.q2 = nn.Linear(d, d)
        self.k2 = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.scale = d ** -0.5

    def forward(self, t, v, g):
        q = self.q1(t)
        # level 1: one scalar gate per evidence stream, conditioned on the text
        c_v = v * torch.sigmoid((q * self.k1(v)).sum(-1, keepdim=True) * self.scale)
        c_g = g * torch.sigmoid((q * self.k1(g)).sum(-1, keepdim=True) * self.scale)
        # level 2: attention over the three streams
        S = torch.stack([t, c_v, c_g], dim=1)                    # [N, 3, d]
        a = torch.softmax((self.q2(t).unsqueeze(1) * self.k2(S)).sum(-1) * self.scale, -1)
        return self.out(self.drop((a.unsqueeze(-1) * S).sum(1)))


FUSION_REGISTRY = {
    "kan": KANFusion,
    "ikan": InteractionKANFusion,   # NOVEL — see class docstring
    "hierarchical_attention": HierarchicalAttentionFusion,
    "concat_linear": ConcatLinear,
    "concat_mlp": ConcatMLP,        # also the "w/o KAN, MLP fusion" ablation (Table 6)
    "gated": GatedFusion,
    "cross_modal_attention": CrossModalAttentionFusion,
    "bilinear": BilinearFusion,
    "tensor": TensorFusion,
}


def build_fusion(name: str = None, d: int = None, dropout: float = None, in_extra: int = 0) -> nn.Module:
    name = name or CONFIG.fusion
    if name not in FUSION_REGISTRY:
        raise ValueError(f"unknown fusion '{name}'. options: {list(FUSION_REGISTRY)}")
    if name == "kan":
        return KANFusion(d=d, dropout=dropout, in_extra=in_extra)
    return FUSION_REGISTRY[name](d=d, dropout=dropout)  # A9 conf-append is KAN-only
