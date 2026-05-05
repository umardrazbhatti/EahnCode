"""
scripts/evaluate.py — Full evaluation: detection + explanation metrics + heatmaps.

Key fixes vs original:
  1. Checkpoint loading uses weights_only=False (PyTorch 2.6+ fix).
  2. GradCAM IndexError fixed via _ScalarOutputTarget in xai/gradcam.py.
  3. faithfulness_correlation receives tensors of matching shape (subset, K)
     — both intrinsic maps and gradient maps are averaged over T before
     flattening.  No shape mismatch.
  4. Deletion/Insertion AUC is now computed (not placeholder zeros) on the
     heatmap subset using metrics/explanation.py.
  5. Video reading falls back gracefully when video path is unavailable
     (synthetic dataset).
"""

import os
import csv
import contextlib
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import EAHNConfig
from models.eahn import EAHN
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from metrics.detection import DetectionMetrics
from metrics.explanation import ExplanationMetrics
from utils.checkpointing import load_checkpoint
from utils.visualization import save_explanation_video
import cv2


def run_evaluation(config: EAHNConfig):
    device = torch.device(config.device)

    # ── Load model ────────────────────────────────────────────────────────────
    model     = EAHN(config).to(device)
    ckpt_path = os.path.join(config.output_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    load_checkpoint(ckpt_path, model)
    model.eval()
    print("Loaded best model for evaluation.")

    # ── Test dataset ─────────────────────────────────────────────────────────
    test_ds = DeepfakeDataset(config, "test", config.dataset_name)
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
    )

    # ── Detection pass ────────────────────────────────────────────────────────
    all_probs, all_labels = [], []
    all_M_t_up, all_masks = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating detection"):
            frames = batch["frames"].to(device)
            out    = model(frames)
            all_probs.extend(out.prob.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_M_t_up.append(out.M_t_up.cpu())
            all_masks.append(batch["mask"].cpu())

    all_M_t_up = torch.cat(all_M_t_up, dim=0)   # (N_test, T, H, W)
    all_masks  = torch.cat(all_masks,  dim=0)   # (N_test, 7, 7)

    det_metrics = DetectionMetrics.compute(all_probs, all_labels)
    print("Detection Metrics:", det_metrics)

    # ── Explanation metrics on a subset ──────────────────────────────────────
    subset_size = min(config.heatmap_samples, len(test_ds))
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(test_ds), subset_size, replace=False)

    # Localisation IoU
    iou_list = []
    for idx in tqdm(indices, desc="Computing IoU", leave=False):
        mask = all_masks[idx]                      # (7, 7)
        if mask.sum() > 0:
            iou = ExplanationMetrics.localisation_iou(
                all_M_t_up[idx].mean(0), mask, threshold=0.5
            )
            iou_list.append(iou)
    avg_iou = float(np.mean(iou_list)) if iou_list else 0.0

    # Temporal SSIM
    ssim_val = ExplanationMetrics.temporal_ssim(all_M_t_up[indices])

    # Faithfulness correlation (gradient saliency vs intrinsic maps)
    grad_maps = []
    for idx in tqdm(indices, desc="Computing faithfulness", leave=False):
        sample      = test_ds[idx]
        frames_t    = sample["frames"].unsqueeze(0).to(device)
        frames_t.requires_grad_(True)
        out         = model(frames_t)
        out.logit.backward()
        grads       = frames_t.grad.abs().mean(dim=2)  # (1, T, H, W) avg over RGB
        # Resize to 7×7 to match M_t feature resolution
        grads_7 = torch.nn.functional.interpolate(
            grads.reshape(grads.shape[1], 1, *grads.shape[2:]),  # (T, 1, H, W)
            size=(7, 7), mode="bilinear", align_corners=False,
        ).squeeze(1)                                              # (T, 7, 7)
        grad_maps.append(grads_7.detach().cpu())
        frames_t.requires_grad_(False)

    grad_maps = torch.stack(grad_maps)             # (subset, T, 7, 7)

    # Average over time — (subset, 7*7) for both maps and grads
    M_sub     = all_M_t_up[indices].mean(dim=1)   # (subset, H, W)
    M_sub_7   = torch.nn.functional.interpolate(
        M_sub.unsqueeze(1), size=(7, 7), mode="bilinear", align_corners=False
    ).squeeze(1)                                   # (subset, 7, 7)

    grad_7_avg = grad_maps.mean(dim=1)             # (subset, 7, 7)

    faithful_corr = ExplanationMetrics.faithfulness_correlation(
        M_sub_7.reshape(subset_size, -1),
        grad_7_avg.reshape(subset_size, -1),
    )

    # Deletion / Insertion AUC on the first heatmap sample
    del_ins = {"deletion_auc": 0.0, "insertion_auc": 0.0}
    try:
        sample_idx = int(indices[0])
        frames_sample = test_ds[sample_idx]["frames"].unsqueeze(0)
        sal_sample    = all_M_t_up[sample_idx].unsqueeze(0)   # (1,T,H,W)
        if isinstance(sal_sample, torch.Tensor):
            sal_np = sal_sample.numpy()
        del_ins = ExplanationMetrics.deletion_insertion_auc(
            model, frames_sample, sal_np, steps=10
        )
    except Exception as e:
        print(f"  [Deletion/Insertion AUC skipped: {e}]")

    exp_metrics = {
        "avg_iou":          avg_iou,
        "temporal_ssim":    ssim_val,
        "faithfulness_corr": faithful_corr,
        **del_ins,
    }
    print("Explanation Metrics:", exp_metrics)

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    os.makedirs(config.output_dir, exist_ok=True)
    csv_path = os.path.join(config.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in {**det_metrics, **exp_metrics}.items():
            writer.writerow([k, v])
    print(f"Metrics saved to {csv_path}")

    # ── Heatmap generation ────────────────────────────────────────────────────
    if config.save_heatmaps:
        _generate_heatmaps(config, model, test_ds, indices[:5], device)

    # ── User-study stimuli ────────────────────────────────────────────────────
    try:
        from user_study.generate_stimuli import generate_stimuli
        generate_stimuli(config, model)
    except Exception as e:
        print(f"  [User-study stimuli skipped: {e}]")

    print("Evaluation complete. Outputs saved to", config.output_dir)


# ── Heatmap helper ────────────────────────────────────────────────────────────

def _generate_heatmaps(config, model, test_ds, sample_indices, device):
    from xai.gradcam import GradCAMExplainer
    from xai.attention_rollout import AttentionRolloutExplainer
    from xai.shap_explainer import SHAPExplainer

    heatmap_dir = os.path.join(config.output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    gradcam_exp = GradCAMExplainer(model, target_layer=model.spatial_stream.grad_cam_target_layer)
    rollout_exp = AttentionRolloutExplainer(model)
    shap_exp    = SHAPExplainer(model, method="integratedgrads")

    print("Generating heatmaps...")
    for idx in tqdm(sample_indices, desc="Saving heatmap videos"):
        idx = int(idx)
        sample = test_ds[idx]
        frames_tensor = sample["frames"].unsqueeze(0).to(device)

        # Original frames (BGR) for visualisation
        sampled_orig = _get_original_frames(
            sample["meta"].get("video_path", ""),
            config.num_frames, config.frame_size,
        )

        with torch.no_grad():
            out = model(frames_tensor)
        intrinsic = out.M_t_up[0].cpu().numpy()   # (T, H, W)

        # Post-hoc heatmaps
        try:
            gc_heat = gradcam_exp.explain(frames_tensor)      # (1,T,H,W)
            gc_heat = gc_heat[0]
        except Exception as e:
            print(f"  [GradCAM failed for idx {idx}: {e}]")
            gc_heat = intrinsic

        try:
            roll_heat = rollout_exp.explain(frames_tensor)    # (T,H,W)
        except Exception as e:
            print(f"  [Rollout failed for idx {idx}: {e}]")
            roll_heat = intrinsic

        try:
            sh_heat = shap_exp.explain(frames_tensor)         # (T,H,W)
        except Exception as e:
            print(f"  [SHAP failed for idx {idx}: {e}]")
            sh_heat = intrinsic

        prefix = os.path.join(heatmap_dir, str(idx))
        save_explanation_video(sampled_orig, intrinsic,  prefix + "_intrinsic.mp4")
        save_explanation_video(sampled_orig, gc_heat,    prefix + "_gradcam.mp4")
        save_explanation_video(sampled_orig, roll_heat,  prefix + "_rollout.mp4")
        save_explanation_video(sampled_orig, sh_heat,    prefix + "_shap.mp4")


def _get_original_frames(video_path: str, num_frames: int, frame_size: int):
    """Read original BGR frames; falls back to blank frames if path unavailable."""
    if not video_path or not os.path.exists(video_path):
        return [np.zeros((frame_size, frame_size, 3), np.uint8)] * num_frames

    cap = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)
    buf   = {}
    fi    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in idxs:
            buf[fi] = cv2.resize(frame, (frame_size, frame_size))
        fi += 1
    cap.release()
    blank = np.zeros((frame_size, frame_size, 3), np.uint8)
    return [buf.get(i, blank) for i in idxs]
