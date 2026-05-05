"""
losses/temporal.py — Gated Temporal Consistency loss L_temp.

w_t = exp(-γ · ||φ(f_t) − φ(f_{t+1})||₂)
L_temp = Σ_t w_t · ||M_t − M_{t+1}||²_F  (mean over batch)
"""

import torch
import torch.nn as nn


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        M_t:      torch.Tensor,   # (B, T, h, w)
        low_level: torch.Tensor,  # (B, T, C_low, Hl, Wl)
    ) -> torch.Tensor:
        B, T = M_t.shape[:2]
        if T < 2:
            return M_t.new_zeros(1).squeeze()

        total = M_t.new_zeros(1).squeeze()
        for t in range(T - 1):
            # Gate based on low-level feature distance
            feat_t    = low_level[:, t].reshape(B, -1).float()
            feat_next = low_level[:, t + 1].reshape(B, -1).float()
            dist = (feat_t - feat_next).pow(2).sum(dim=1).add(1e-8).sqrt()  # (B,)
            w    = torch.exp(-self.gamma * dist)                              # (B,)

            # Squared difference of explanation maps
            diff = (M_t[:, t] - M_t[:, t + 1]).pow(2).reshape(B, -1).mean(dim=1)  # (B,)
            total = total + (w * diff).mean()

        return total / (T - 1)
