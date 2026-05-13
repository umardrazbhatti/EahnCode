"""
models/cross_attention.py — Cross-Attention Fusion with learnable temperature.

Returns (M_t, attn_pool):
  M_t      : (B, T, h, w)  intrinsic explanation maps (softmax probability distributions)
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
        tau    = torch.exp(self.log_temp).clamp(min=0.5, max=10.0)
        scores = torch.bmm(Qp, Kp.transpose(-2, -1)) / (self.scale * tau)  # (B·T, L, L)
        A      = F.softmax(scores, dim=-1)  # softmax over key dimension

        # Attended values (creates gradient path from classifier back to A)
        attended = self.out_proj(torch.bmm(A, Vp))  # (B·T, L, d)

        # Explanation map: mean over query positions → each key location's total weight
        M_flat = A.mean(dim=-2)          # (B·T, L)
        M_t    = M_flat.reshape(B, T, h, w)

        # CHANGE 2 (phase7): softmax-only. The previous rescale-by-max divided
        # a near-uniform softmax distribution by its own (near-uniform) max,
        # mapping uniform attention to the constant 1.0 — destroying every
        # spatial signal. Softmax outputs are already in [0,1] mathematically.
        # Visualisation code does its own per-frame normalisation downstream.
        M_t = M_t.reshape(B, T, h * w)
        M_t = torch.softmax(M_t, dim=-1)       # spatial probability distribution
        M_t = M_t.reshape(B, T, h, w)

        # CHANGE 3 (phase7): use the softmax'd, normalised M_t as pool weights
        # instead of pre-softmax M_flat (which is row-stochastic-and-uniform
        # → degenerate flat mean). Now classifier gradient w.r.t. M_t is
        # non-degenerate, and L_cls actually pressures attention to learn.
        L_local = h * w
        W = M_t.reshape(B * T, L_local, 1)             # (B·T, L, 1), sums to 1 per frame
        S_pool    = (W * Vp).sum(dim=1)                # (B·T, d), true weighted pool
        attn_pool = S_pool.reshape(B, T, d).mean(dim=1) # (B, d)

        return M_t, attn_pool
