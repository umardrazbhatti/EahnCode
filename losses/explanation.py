"""
losses/explanation.py — L_exp:
  * Supervised (has pixel mask): MSE(M_t_avg, gt_mask)
  * Weak supervision (no mask):  α·Entropy(M_t) + β·TV(M_t) + diversity_weight·l_div

The diversity term penalizes pairwise cosine similarity between heatmaps from
different samples in the batch.  The earlier per-sample peakedness formulation
(relu(0.05 - std)) was replaced because it cannot distinguish "all samples
produce a sharp peak at the same location" (collapse) from "all samples produce
sharp peaks at different locations" (healthy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExplanationLossOutput:
    loss:            torch.Tensor
    l_h:             float   # entropy term
    l_tv:            float   # total-variation term
    l_div:           float   # inter-sample diversity term
    inter_sample_sim: float  # mean pairwise cosine similarity (diagnostic)


class ExplanationLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, beta: float = 0.5,
                 diversity_weight: float = 2.5):
        super().__init__()
        self.alpha            = alpha
        self.beta             = beta
        self.diversity_weight = diversity_weight

    def forward(
        self,
        M_t:     torch.Tensor,   # (B, T, h, w)  normalised to [0,1]
        masks:   torch.Tensor,   # (B, h, w)      ground-truth (or zeros)
        has_mask: torch.Tensor,  # (B,) bool
    ) -> ExplanationLossOutput:
        B, T, h, w = M_t.shape
        loss = M_t.new_zeros(1).squeeze()

        l_h_acc   = 0.0
        l_tv_acc  = 0.0

        for i in range(B):
            m_avg = M_t[i].mean(0)   # (h, w)

            if has_mask[i]:
                gt = masks[i]
                if gt.shape != (h, w):
                    gt = F.interpolate(
                        gt.unsqueeze(0).unsqueeze(0).float(),
                        size=(h, w), mode='bilinear', align_corners=False
                    ).squeeze()
                loss = loss + F.mse_loss(m_avg, gt)
            else:
                # Sparsity via entropy
                m_flat  = m_avg.clamp(1e-8, 1 - 1e-8).flatten()
                entropy = -(m_flat * m_flat.log()).sum()

                # Smoothness via total variation
                tv_h = (M_t[i, :, :, 1:] - M_t[i, :, :, :-1]).abs().mean()
                tv_w = (M_t[i, :, 1:, :] - M_t[i, :, :-1, :]).abs().mean()
                tv   = tv_h + tv_w

                loss      = loss + (self.alpha * entropy + self.beta * tv)
                l_h_acc  += entropy.item()
                l_tv_acc += tv.item()

        loss = loss / B

        # Inter-sample diversity — Jensen-Shannon divergence (phase8).
        # Cosine on probability vectors has a high structural floor (~0.5+) on
        # the simplex: two distributions sharing the uniform component 1/L
        # cannot have cos < ~0.5 without becoming near-one-hot. JS divergence
        # is the correct metric for distributional difference; its range is
        # [0, log(2)], with 0 meaning identical and log(2) meaning disjoint.
        import math as _math
        N = B * T
        eye = torch.eye(N, dtype=torch.bool, device=M_t.device)
        n_pairs = N * (N - 1)

        eps = 1e-8
        P = M_t.reshape(N, h * w) + eps             # (N, L), smooth for log safety
        P = P / P.sum(dim=-1, keepdim=True)          # renormalise to valid distributions

        log_P = P.log()                              # (N, L)
        P_i = P.unsqueeze(1)                         # (N, 1, L)
        P_j = P.unsqueeze(0)                         # (1, N, L)
        M_mix = 0.5 * (P_i + P_j)                   # (N, N, L)
        log_M = M_mix.log()                          # (N, N, L)
        log_P_i = log_P.unsqueeze(1)                 # (N, 1, L)
        log_P_j = log_P.unsqueeze(0)                 # (1, N, L)
        kl_im = (P_i * (log_P_i - log_M)).sum(dim=-1)  # (N, N)
        kl_jm = (P_j * (log_P_j - log_M)).sum(dim=-1)  # (N, N)
        js_matrix = 0.5 * (kl_im + kl_jm)           # (N, N), values in [0, log 2]

        js_off = js_matrix.masked_fill(eye, 0.0)
        mean_js_tensor = js_off.sum() / max(n_pairs, 1)
        log2 = _math.log(2.0)
        l_div_tensor = (log2 - mean_js_tensor).clamp_min(0.0)
        loss = loss + self.diversity_weight * l_div_tensor

        # Cosine similarity kept as diagnostic for backward-compat with metric curves.
        flat = M_t.reshape(N, h * w)
        flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        cos_matrix = flat @ flat.T
        inter_sample_sim = float(
            cos_matrix.masked_fill(eye, 0.0).sum().item() / (n_pairs + 1e-8)
        )

        return ExplanationLossOutput(
            loss=loss,
            l_h=l_h_acc / max(B, 1),
            l_tv=l_tv_acc / max(B, 1),
            l_div=float(l_div_tensor.item()),
            inter_sample_sim=inter_sample_sim,
        )
