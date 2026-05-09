"""
losses/explanation.py — L_exp:
  * Supervised (has pixel mask): MSE(M_t_avg, gt_mask)
  * Weak supervision (no mask):  α·Entropy(M_t) + β·TV(M_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplanationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 diversity_weight: float = 0.5):
        super().__init__()
        self.alpha           = alpha
        self.beta            = beta
        self.diversity_weight = diversity_weight

    def forward(
        self,
        M_t:     torch.Tensor,   # (B, T, h, w)  normalised to [0,1]
        masks:   torch.Tensor,   # (B, h, w)      ground-truth (or zeros)
        has_mask: torch.Tensor,  # (B,) bool
    ) -> torch.Tensor:
        B = M_t.shape[0]
        loss = M_t.new_zeros(1).squeeze()

        h, w = M_t.shape[2], M_t.shape[3]

        for i in range(B):
            m_avg = M_t[i].mean(0)   # (h, w) — average over frames

            if has_mask[i]:
                # Resize ground-truth mask to match M_t spatial resolution
                gt = masks[i]                        # (mh, mw)
                if gt.shape != (h, w):
                    gt = F.interpolate(
                        gt.unsqueeze(0).unsqueeze(0).float(),
                        size=(h, w), mode='bilinear', align_corners=False
                    ).squeeze()
                loss = loss + F.mse_loss(m_avg, gt)
            else:
                # Sparsity via entropy
                m_flat = m_avg.clamp(1e-8, 1 - 1e-8).flatten()
                entropy = -(m_flat * m_flat.log()).sum()

                # Smoothness via total variation
                tv_h = (M_t[i, :, :, 1:] - M_t[i, :, :, :-1]).abs().mean()
                tv_w = (M_t[i, :, 1:, :] - M_t[i, :, :-1, :]).abs().mean()
                tv   = tv_h + tv_w

                # Diversity penalty: push std(M_t) above 0.3 per frame
                # relu(0.3 - std) is > 0 only when map is too uniform (std < 0.3)
                spatial_std   = M_t[i].std(dim=(-1, -2))          # (T,)
                diversity_loss = F.relu(0.3 - spatial_std).mean()

                loss = loss + (self.alpha * entropy + self.beta * tv
                               + self.diversity_weight * diversity_loss)

        return loss / B
