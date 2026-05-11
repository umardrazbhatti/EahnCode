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

        # Inter-sample diversity — penalise when different samples produce similar heatmaps.
        # A one-hot map at the same location across all samples gives inter_sim ≈ 1.0.
        flat = M_t.reshape(B * T, h * w)
        flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = flat @ flat.T                              # (BT, BT)
        eye = torch.eye(B * T, dtype=torch.bool, device=M_t.device)
        n_pairs = B * T * (B * T - 1)
        inter_sample_sim = float(
            sim_matrix.masked_fill(eye, 0.0).sum().item() / (n_pairs + 1e-8)
        )
        l_div_tensor = F.relu(
            sim_matrix.masked_fill(eye, 0.0).sum() / (n_pairs + 1e-8) - 0.5
        )
        loss = loss + self.diversity_weight * l_div_tensor

        return ExplanationLossOutput(
            loss=loss,
            l_h=l_h_acc / max(B, 1),
            l_tv=l_tv_acc / max(B, 1),
            l_div=float(l_div_tensor.item()),
            inter_sample_sim=inter_sample_sim,
        )
