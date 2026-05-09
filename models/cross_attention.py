"""
models/cross_attention.py — Cross-Attention Fusion with learnable temperature.

Returns (M_t, attn_pool):
  M_t      : (B, T, h, w)  intrinsic explanation maps, max-normalised to [0,1]
  attn_pool : (B, d_model)  attention-weighted spatial pooling for classifier gradient path

The attn_pool → classifier residual path ensures that L_cls gradients flow back
through the attention weights into M_t (fixes the attention-collapse bug).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = math.sqrt(self.head_dim)

        # Learnable temperature: τ = exp(log_temp), initialised to 4.0.
        # Higher τ sharpens the softmax distribution, breaking uniform collapse.
        self.log_temp = nn.Parameter(torch.tensor(math.log(4.0)))

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        Q: torch.Tensor,   # (B, T, L, d_model)  temporal queries
        S: torch.Tensor,   # (B, T, L, d_model)  spatial keys/values
    ):
        B, T, L, d = Q.shape
        h = w = int(math.sqrt(L))   # h=w=7 for 224px input, stride-32 backbone

        Q_flat = Q.reshape(B * T, L, d)
        S_flat = S.reshape(B * T, L, d)

        Qp = self.q_proj(Q_flat)    # (B·T, L, d)
        Kp = self.k_proj(S_flat)
        Vp = self.v_proj(S_flat)

        # Temperature-scaled attention
        tau    = torch.exp(self.log_temp).clamp(min=1.0, max=16.0)
        scores = torch.bmm(Qp, Kp.transpose(-2, -1)) / (self.scale * tau)  # (B·T, L, L)
        A      = F.softmax(scores, dim=-1)  # softmax over key dimension

        # Attended values (creates gradient path from classifier back to A)
        attended = self.out_proj(torch.bmm(A, Vp))  # (B·T, L, d)

        # Explanation map: mean over query positions → each key location's total weight
        M_flat = A.mean(dim=-2)          # (B·T, L)
        M_t    = M_flat.reshape(B, T, h, w)

        # Max-norm per frame — preserves absolute importance; gradient still flows
        # when map is flat (unlike min-max which collapses flat maps to a constant)
        M_max = M_t.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        M_t   = M_t / M_max

        # Attention-weighted spatial pooling for classifier gradient path.
        # CRITICAL: grad(L_cls) → attn_pool → M_flat → A → Q, K projections → M_t
        M_weights = M_flat.unsqueeze(-1)                                      # (B·T, L, 1)
        S_pool    = (M_weights * Vp).sum(dim=1) / (M_weights.sum(dim=1) + 1e-8)  # (B·T, d)
        attn_pool = S_pool.reshape(B, T, d).mean(dim=1)                       # (B, d)

        return M_t, attn_pool
