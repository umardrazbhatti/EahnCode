"""
models/cross_attention.py — Cross-Attention Fusion that produces intrinsic
explanation maps M_t ∈ (B, T, h, w).

The attention matrix is the mechanism that both fuses temporal and spatial
streams AND generates the explanation — the explanation and the detection
pathway are the same computation (no post-hoc attribution needed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        Q: torch.Tensor,   # (B, T, N, d_model) — temporal queries
        S: torch.Tensor,   # (B, T, N, d_model) — spatial keys/values
    ):
        """
        Returns:
            M_t      : (B, T, h, w)  intrinsic explanation maps, normalised to [0,1]
            attn_out : None           (downstream classification uses CLS, not this)
        """
        B, T, N, d = Q.shape
        h = w = int(N ** 0.5)

        Q_flat = Q.reshape(B * T, N, d)    # (B*T, N, d)
        S_flat = S.reshape(B * T, N, d)

        # Scaled dot-product attention
        attn_scores  = torch.bmm(Q_flat, S_flat.transpose(1, 2)) / self.scale  # (B*T, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)                           # (B*T, N, N)

        # Aggregate over query positions → per-key importance
        M_flat = attn_weights.mean(dim=1)   # (B*T, N)
        M_t    = M_flat.reshape(B, T, h, w)

        # Per-frame min-max normalisation to [0,1]
        M_min = M_t.reshape(B, T, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        M_max = M_t.reshape(B, T, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        M_t   = (M_t - M_min) / (M_max - M_min + 1e-8)

        return M_t, None
