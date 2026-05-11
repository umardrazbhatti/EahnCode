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
from typing import Dict


class ExplanationMetrics:

    @staticmethod
    def localisation_iou(
        M_t_avg: torch.Tensor,       # (B, H, W) or (H, W) — time-averaged maps
        gt_masks: torch.Tensor,      # (B, h, w) or (h, w) — ground-truth masks
        has_mask_flags,              # list[bool] or BoolTensor of length B
        threshold: float = 0.5,
    ):
        """
        Intersection-over-Union between binarised explanations and GT masks.

        Only samples where has_mask_flags[i] is True are included.
        Returns the mean IoU over valid samples, or None if no valid samples exist.
        """
        import torch.nn.functional as F

        # Normalise to batch form
        if M_t_avg.dim() == 2:
            M_t_avg   = M_t_avg.unsqueeze(0)
            gt_masks  = gt_masks.unsqueeze(0)
            has_mask_flags = [has_mask_flags] \
                if not hasattr(has_mask_flags, "__len__") else list(has_mask_flags)

        B = M_t_avg.shape[0]
        valid_ious = []

        for i in range(B):
            if not bool(has_mask_flags[i]):
                continue
            gt = gt_masks[i]
            if gt.sum() == 0:
                continue
            m = M_t_avg[i]
            if gt.shape != m.shape:
                gt = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0).float(),
                    size=m.shape, mode="bilinear", align_corners=False,
                ).squeeze()
            M_bin = (m > threshold).float()
            gt_f  = gt.float()
            inter = (M_bin * gt_f).sum()
            union = ((M_bin + gt_f) > 0).float().sum()
            valid_ious.append(float(inter / (union + 1e-8)))

        if len(valid_ious) == 0:
            return None
        return float(np.mean(valid_ious))

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

        _trapz = getattr(np, "trapezoid", np.trapz)
        del_auc = float(_trapz(del_scores) / steps)
        ins_auc = float(_trapz(ins_scores) / steps)
        return {"deletion_auc": del_auc, "insertion_auc": ins_auc}

    @staticmethod
    def collapse_diagnostics(all_M_t: torch.Tensor) -> Dict[str, float]:
        """
        Compute three collapse diagnostic metrics on the full test-set M_t tensor.

        Parameters
        ----------
        all_M_t : (N, T, H, W)  — explanation maps for all test samples

        Returns
        -------
        dict with keys:
            inter_sample_cosine_mean  — mean pairwise cosine sim; < 0.5 healthy
            peak_mode_share           — fraction of samples whose argmax lands at
                                        the most common (row, col); < 0.2 healthy
            m_t_std_mean              — mean M_t std across samples; > 0.13 = one-hot
            m_t_std_max               — max  M_t std across samples
        """
        N, T, H, W = all_M_t.shape

        # --- inter-sample cosine similarity ---
        flat = all_M_t.mean(dim=1).reshape(N, H * W).float()   # (N, H*W) — time-averaged
        flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = flat_norm @ flat_norm.T                    # (N, N)
        eye = torch.eye(N, dtype=torch.bool, device=all_M_t.device)
        n_pairs = N * (N - 1)
        inter_cosine = float(
            sim_matrix.masked_fill(eye, 0.0).sum().item() / max(n_pairs, 1)
        )

        # --- peak-coordinate mode share ---
        mean_maps = all_M_t.mean(dim=1)                         # (N, H, W)
        peak_indices = mean_maps.reshape(N, -1).argmax(dim=-1)  # (N,)
        peak_rc = [(int(idx) // W, int(idx) % W) for idx in peak_indices.tolist()]
        from collections import Counter
        most_common_count = Counter(peak_rc).most_common(1)[0][1]
        peak_mode_share = float(most_common_count) / N

        # --- M_t std (per-sample, time-and-space) ---
        stds = all_M_t.std(dim=(-1, -2)).mean(dim=-1)           # (N,) mean over T
        m_t_std_mean = float(stds.mean().item())
        m_t_std_max  = float(stds.max().item())

        return {
            "inter_sample_cosine_mean": inter_cosine,
            "peak_mode_share":          peak_mode_share,
            "m_t_std_mean":             m_t_std_mean,
            "m_t_std_max":              m_t_std_max,
        }
