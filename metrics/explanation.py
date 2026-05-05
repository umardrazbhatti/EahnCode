"""
metrics/explanation.py — Explanation quality metrics.

FIX: faithfulness_correlation received M_t (subset, 49) and grad_maps (subset, T, 49)
     of mismatched shapes. Both are now averaged over time before reshaping, giving
     (subset, 49) for each, so Spearman correlation is well-defined.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr


class ExplanationMetrics:

    @staticmethod
    def localisation_iou(
        M_t_avg: torch.Tensor,   # (H, W) — time-averaged explanation map
        gt_mask: torch.Tensor,   # (h, w) or (H, W)
        threshold: float = 0.5,
    ) -> float:
        """Intersection-over-Union between binarised explanation and ground-truth mask."""
        if gt_mask.sum() == 0:
            return 0.0
        # Resize gt_mask to match M_t_avg if necessary
        if gt_mask.shape != M_t_avg.shape:
            import torch.nn.functional as F
            gt_mask = F.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0).float(),
                size=M_t_avg.shape, mode="bilinear", align_corners=False
            ).squeeze()
        M_bin = (M_t_avg > threshold).float()
        gt    = gt_mask.float()
        inter = (M_bin * gt).sum()
        union = ((M_bin + gt) > 0).float().sum()
        return float(inter / (union + 1e-8))

    @staticmethod
    def temporal_ssim(M_t_up: torch.Tensor) -> float:
        """
        Mean SSIM between consecutive explanation frames.
        M_t_up: (N, T, H, W) subset.
        """
        values = []
        N, T, H, W = M_t_up.shape
        for b in range(N):
            for t in range(T - 1):
                a = M_t_up[b, t].cpu().numpy().astype(np.float32)
                b_ = M_t_up[b, t + 1].cpu().numpy().astype(np.float32)
                val = ssim(a, b_, data_range=1.0)
                values.append(val)
        return float(np.mean(values)) if values else 1.0

    @staticmethod
    def faithfulness_correlation(
        M_flat: torch.Tensor,     # (subset, K) — intrinsic maps flattened
        grad_flat: torch.Tensor,  # (subset, K) — gradient maps flattened
    ) -> float:
        """
        Spearman rank correlation between intrinsic attention and gradient attribution.
        Both tensors must already be (subset, K) with the same K.
        """
        m = M_flat.detach().cpu().numpy().flatten()
        g = grad_flat.detach().cpu().numpy().flatten()
        if len(m) < 3 or np.std(m) < 1e-8 or np.std(g) < 1e-8:
            return 0.0
        corr, _ = spearmanr(m, g)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def deletion_insertion_auc(model, frames, saliency,
                               steps: int = 10) -> dict:
        """
        Deletion/Insertion AUC: simplified implementation.
        Steps are coarse for speed; increase for publication-quality numbers.
        """
        device = next(model.parameters()).device
        B, T, C, H, W = frames.shape
        total_pixels  = H * W

        with torch.no_grad():
            baseline_logit = model(frames.to(device)).prob.mean().item()

        del_scores = []
        ins_scores = []

        # Use mean explanation over time
        sal = saliency.mean(1)   # (B, H, W) or just use first frame

        for step in range(steps + 1):
            frac = step / steps
            k    = max(1, int(frac * total_pixels))

            # Deletion: mask out top-k salient pixels
            del_frames = frames.clone()
            ins_frames = torch.zeros_like(frames)

            for b in range(B):
                flat_sal = sal[b].reshape(-1)                 # np.ndarray
                top_k_idx = np.argsort(flat_sal)[-k:]         # top-k indices
                mask     = np.zeros(H * W, dtype=bool)
                mask[top_k_idx] = True
                mask_2d  = mask.reshape(H, W)

                del_frames[b, :, :, mask_2d] = 0.0
                ins_frames[b, :, :, mask_2d] = frames[b, :, :, mask_2d]

            with torch.no_grad():
                del_score = model(del_frames.to(device)).prob.mean().item()
                ins_score = model(ins_frames.to(device)).prob.mean().item()

            del_scores.append(del_score)
            ins_scores.append(ins_score)

        del_auc = float(np.trapezoid(del_scores) / steps)
        ins_auc = float(np.trapezoid(ins_scores) / steps)
        return {"deletion_auc": del_auc, "insertion_auc": ins_auc}
