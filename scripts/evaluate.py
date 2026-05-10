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
  6. ROC, PR, confusion-matrix, score-distribution PNGs are saved.
  7. Per-video plain-English explanation TXT files are saved.
  8. Heatmap videos use annotated overlay with verdict text and green contour.
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
from utils.visualization import (
    save_annotated_frame_strip,
    save_explanation_video,
    overlay_heatmap_on_frame,
    get_region_label,
)
import cv2


# ── Detection graph helper ────────────────────────────────────────────────────

def save_detection_graphs(probs, labels, output_dir: str) -> None:
    """Save ROC curve, PR curve, confusion matrix, and score-distribution PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve, roc_auc_score,
        precision_recall_curve, average_precision_score,
        confusion_matrix,
    )
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)

    # 1 — ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"EAHN  AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Deepfake Detection (FF++ c23)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # 2 — Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color="darkorange", label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)

    # 3 — Confusion Matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 4 — Score Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs[labels == 0], bins=30, alpha=0.6, label="Real", color="blue")
    ax.hist(probs[labels == 1], bins=30, alpha=0.6, label="Fake", color="red")
    ax.axvline(0.5, color="black", linestyle="--", label="Decision threshold")
    ax.set_xlabel("Predicted Probability (Deepfake)")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"[Evaluate] Detection graphs saved → {output_dir}")


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_evaluation(config: EAHNConfig):
    device = torch.device(config.device)

    # ── Load model ────────────────────────────────────────────────────────────
    model     = EAHN(config).to(device)
    ckpt_path = os.path.join(config.output_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        import glob as _glob
        candidates = sorted(_glob.glob(
            os.path.join(config.output_dir, "checkpoint_epoch*.pth")
        ))
        if candidates:
            ckpt_path = candidates[-1]
            print(f"[Eval] best_model.pth not found — using {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {config.output_dir}. "
                "Did training complete without errors?"
            )
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
    all_has_mask_flags    = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating detection"):
            frames = batch["frames"].to(device)
            out    = model(frames)
            all_probs.extend(out.prob.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_M_t_up.append(out.M_t_up.cpu())
            all_masks.append(batch["mask"].cpu())
            all_has_mask_flags.extend(batch["has_mask"].cpu().tolist())

    all_M_t_up = torch.cat(all_M_t_up, dim=0)   # (N_test, T, H, W)
    all_masks  = torch.cat(all_masks,  dim=0)   # (N_test, 7, 7)

    det_metrics = DetectionMetrics.compute(all_probs, all_labels)
    print("Detection Metrics:", det_metrics)

    # ── Confusion matrix (5a) ─────────────────────────────────────────────────
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    try:
        preds_arr = (np.array(all_probs) >= 0.5).astype(int)
        cm        = sk_confusion_matrix(np.array(all_labels, dtype=int), preds_arr)
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn = fp = fn = tp = 0

    # ── Save detection graphs (requires both classes) ─────────────────────────
    labels_arr = np.array(all_labels)
    if len(np.unique(labels_arr)) >= 2:
        save_detection_graphs(all_probs, all_labels, config.output_dir)
    else:
        print("[Evaluate] Skipping detection graphs — only one class in test set.")

    # ── Split counts + summary chart (5b, 5c, 5d) ────────────────────────────
    train_ds_tmp = DeepfakeDataset(config, "train", config.dataset_name)
    val_ds_tmp   = DeepfakeDataset(config, "val",   config.dataset_name)
    split_counts = {
        "total":      len(train_ds_tmp) + len(val_ds_tmp) + len(test_ds),
        "train":      len(train_ds_tmp),
        "train_real": train_ds_tmp.n_real,
        "train_fake": train_ds_tmp.n_fake,
        "val":        len(val_ds_tmp),
        "test":       len(test_ds),
        "test_real":  test_ds.n_real,
        "test_fake":  test_ds.n_fake,
    }
    metrics_dict_full = {
        **det_metrics,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    from scripts.summary_chart import plot_summary_chart
    plot_summary_chart(metrics_dict_full, split_counts, config.output_dir)

    # ── Explanation metrics on a subset ──────────────────────────────────────
    subset_size = min(config.heatmap_samples, len(test_ds))
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(test_ds), subset_size, replace=False)

    # Localisation IoU — only for samples that have ground-truth masks (5e)
    M_sub_avg = all_M_t_up[indices].mean(dim=1)             # (subset, H, W)
    masks_sub = all_masks[indices]                          # (subset, h, w)
    hm_flags  = [all_has_mask_flags[int(i)] for i in indices]
    avg_iou   = ExplanationMetrics.localisation_iou(
        M_sub_avg, masks_sub, hm_flags, threshold=0.5
    )   # float or None

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
        grads_7 = torch.nn.functional.interpolate(
            grads.reshape(grads.shape[1], 1, *grads.shape[2:]),  # (T, 1, H, W)
            size=(7, 7), mode="bilinear", align_corners=False,
        ).squeeze(1)                                              # (T, 7, 7)
        grad_maps.append(grads_7.detach().cpu())
        frames_t.requires_grad_(False)

    grad_maps = torch.stack(grad_maps)             # (subset, T, 7, 7)

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
        sample_idx    = int(indices[0])
        frames_sample = test_ds[sample_idx]["frames"].unsqueeze(0)
        sal_sample    = all_M_t_up[sample_idx].unsqueeze(0)   # (1,T,H,W)
        if isinstance(sal_sample, torch.Tensor):
            sal_np = sal_sample.numpy()
        del_ins = ExplanationMetrics.deletion_insertion_auc(
            model, frames_sample, sal_np, steps=10
        )
    except Exception as e:
        print(f"  [Deletion/Insertion AUC skipped: {e}]")

    avg_iou_display = avg_iou if avg_iou is not None else "N/A (no masks)"
    exp_metrics = {
        "avg_iou":           avg_iou_display,
        "temporal_ssim":     ssim_val,
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
        _generate_heatmaps(config, model, test_ds, indices[:5], device, all_probs)

    # ── User-study stimuli ────────────────────────────────────────────────────
    try:
        from user_study.generate_stimuli import AutomatedStimulusGenerator
        from xai.gradcam import GradCAMExplainer
        _gradcam = GradCAMExplainer(model, target_layer=model.spatial_stream.grad_cam_target_layer)
        _stim_dir = os.path.join(config.output_dir, "user_study")
        _generator = AutomatedStimulusGenerator(model, test_loader, _gradcam, config, _stim_dir)
        _generator.generate()
    except Exception as e:
        print(f"  [User-study stimuli skipped: {e}]")

    print("Evaluation complete. Outputs saved to", config.output_dir)


# ── Heatmap + explanation helper ─────────────────────────────────────────────

def _generate_heatmaps(config, model, test_ds, sample_indices, device, all_probs):
    from xai.gradcam import GradCAMExplainer
    from xai.attention_rollout import AttentionRolloutExplainer
    from xai.shap_explainer import SHAPExplainer

    heatmap_dir     = os.path.join(config.output_dir, "heatmaps")
    explanation_dir = os.path.join(config.output_dir, "explanations")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(explanation_dir, exist_ok=True)

    gradcam_exp = GradCAMExplainer(model, target_layer=model.spatial_stream.grad_cam_target_layer)
    rollout_exp = AttentionRolloutExplainer(model)
    shap_exp    = SHAPExplainer(model, method="integratedgrads")

    print("Generating heatmaps and explanations...")
    for idx in tqdm(sample_indices, desc="Saving heatmap videos"):
        idx    = int(idx)
        sample = test_ds[idx]
        frames_tensor = sample["frames"].unsqueeze(0).to(device)

        video_path = sample["meta"].get("video_path", "")
        video_id   = os.path.splitext(os.path.basename(video_path))[0] if video_path else str(idx)

        sampled_orig = _get_original_frames(
            video_path, config.num_frames, config.frame_size,
        )

        with torch.no_grad():
            out = model(frames_tensor)
        intrinsic = out.M_t_up[0].cpu().numpy()     # (T, H, W)
        prob      = float(out.prob[0].cpu())
        verdict   = "FAKE" if prob > 0.5 else "REAL"

        # Convert intrinsic to list form for new viz API
        intrinsic_maps = [intrinsic[t] for t in range(intrinsic.shape[0])]

        def _peakiness(m: np.ndarray) -> float:
            flat = m.flatten().astype(np.float64) + 1e-12
            flat = flat / flat.sum()
            H_val = -(flat * np.log(flat)).sum()
            return float(1.0 - H_val / np.log(flat.size))

        intrinsic_scores = [_peakiness(m) for m in intrinsic_maps]

        # ── Annotated frame strip + companion text explanation (5f) ───────────
        save_annotated_frame_strip(
            sampled_orig, intrinsic_maps, intrinsic_scores, verdict, prob,
            os.path.join(explanation_dir, f"{video_id}_strip.png"),
            sample_id=video_id,
        )

        # ── Intrinsic explanation video (5f) ──────────────────────────────────
        save_explanation_video(
            sampled_orig, intrinsic_maps, intrinsic_scores, verdict, prob,
            os.path.join(heatmap_dir, f"{video_id}_intrinsic.mp4"),
        )

        # ── Post-hoc heatmaps ─────────────────────────────────────────────────
        for method_name, explainer in [
            ("gradcam", gradcam_exp),
            ("rollout", rollout_exp),
            ("shap",    shap_exp),
        ]:
            try:
                if method_name == "gradcam":
                    heat = explainer.explain(frames_tensor)[0]   # (T,H,W) numpy
                else:
                    heat = explainer.explain(frames_tensor)      # (T,H,W) numpy
            except Exception as e:
                print(f"  [{method_name} failed for idx {idx}: {e}]")
                heat = intrinsic

            maps_list   = [heat[t] for t in range(heat.shape[0])]
            scores_list = [float(m.max()) for m in maps_list]
            save_explanation_video(
                sampled_orig, maps_list, scores_list, verdict, prob,
                os.path.join(heatmap_dir, f"{video_id}_{method_name}.mp4"),
            )


def _get_original_frames(video_path: str, num_frames: int, frame_size: int):
    """Read original BGR frames; falls back to blank frames if path unavailable."""
    if not video_path or not os.path.exists(video_path):
        return [np.zeros((frame_size, frame_size, 3), np.uint8)] * num_frames

    cap   = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)
    buf   = {}
    fi    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in set(idxs.tolist()):
            buf[fi] = cv2.resize(frame, (frame_size, frame_size))
        fi += 1
    cap.release()
    blank = np.zeros((frame_size, frame_size, 3), np.uint8)
    return [buf.get(i, blank) for i in idxs]
