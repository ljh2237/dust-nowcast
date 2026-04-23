from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B*N, T, F]
        h = self.proj(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=True)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h[:, -1, :], attn_w


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, D], adj: [N, N]
        b, n, _ = x.shape
        h = self.w(x)
        h_i = h.unsqueeze(2).repeat(1, 1, n, 1)
        h_j = h.unsqueeze(1).repeat(1, n, 1, 1)
        e = F.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1), negative_slope=0.2)

        mask = (adj > 0).unsqueeze(0).expand(b, -1, -1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.matmul(alpha, h)
        return out, alpha


class DustRiskFormer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        static_dim: int,
        hidden_dim: int,
        num_heads: int,
        horizons: int,
        num_risk_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.temporal = TemporalEncoder(in_dim, hidden_dim, num_heads, dropout)
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gat = GraphAttentionLayer(hidden_dim * 2, hidden_dim, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.reg_head = nn.Linear(hidden_dim, horizons)
        self.risk_head = nn.Linear(hidden_dim, horizons * num_risk_classes)
        self.warn_head = nn.Linear(hidden_dim, horizons)

    def forward(self, x: torch.Tensor, x_static: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, N, F], x_static: [N, S], adj: [N, N]
        b, t, n, f = x.shape
        x_reshape = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        t_repr, t_attn = self.temporal(x_reshape)
        t_repr = t_repr.reshape(b, n, -1)

        s_repr = self.static_proj(x_static).unsqueeze(0).expand(b, -1, -1)
        node_repr = torch.cat([t_repr, s_repr], dim=-1)

        g_repr, g_attn = self.gat(node_repr, adj)
        fused = self.fuse(torch.cat([node_repr, g_repr], dim=-1))

        reg = self.reg_head(fused)
        risk_logits = self.risk_head(fused).reshape(b, n, self.horizons, -1)
        warn_logit = self.warn_head(fused)

        return {
            "wind": reg,
            "risk_logits": risk_logits,
            "warn_logit": warn_logit,
            "temporal_attention": t_attn,
            "graph_attention": g_attn,
        }


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    y_wind: torch.Tensor,
    y_risk: torch.Tensor,
    y_warn: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> Dict[str, torch.Tensor]:
    loss_reg = F.smooth_l1_loss(outputs["wind"], y_wind)
    b, n, h, c = outputs["risk_logits"].shape
    loss_risk = F.cross_entropy(outputs["risk_logits"].reshape(b * n * h, c), y_risk.reshape(-1))
    loss_warn = F.binary_cross_entropy_with_logits(outputs["warn_logit"], y_warn)
    total = alpha * loss_reg + beta * loss_risk + gamma * loss_warn
    return {
        "total": total,
        "reg": loss_reg,
        "risk": loss_risk,
        "warn": loss_warn,
    }
