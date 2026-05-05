"""
user_study/generate_stimuli.py — Generates three-condition video stimuli for
the controlled user study described in the thesis (Chapter 3).

Condition A: Original frames + label text only (no heatmap) — black-box
Condition B: Original frames + Grad-CAM overlay + label — post-hoc XAI
Condition C: Original frames + intrinsic cross-attention overlay + label

Outputs 60 balanced clips (30 real + 30 fake) per condition to
{output_dir}/user_study/stimuli/{A,B,C}/XXXX.mp4
"""

import os
import cv2
import torch
import numpy as np

from config import EAHNConfig
from data.datasets import DeepfakeDataset
from utils.visualization import save_explanation_video, overlay_explanation


def generate_stimuli(config: EAHNConfig, model):
    device = torch.device(config.device)
    model.eval()

    test_ds = DeepfakeDataset(config, "test", config.dataset_name)
    real_idx = [i for i, s in enumerate(test_ds.samples) if s.get("label", 0) == 0]
    fake_idx = [i for i, s in enumerate(test_ds.samples) if s.get("label", 0) == 1]

    rng = np.random.default_rng(0)
    n_per_class = min(30, len(real_idx), len(fake_idx))
    selected = (
        rng.choice(real_idx, n_per_class, replace=False).tolist() +
        rng.choice(fake_idx, n_per_class, replace=False).tolist()
    )

    out_root = os.path.join(config.output_dir, "user_study", "stimuli")
    for cond in ["A", "B", "C"]:
        os.makedirs(os.path.join(out_root, cond), exist_ok=True)

    for idx in selected:
        sample      = test_ds[idx]
        label_int   = int(sample["label"].item()) if torch.is_tensor(sample["label"]) else int(sample["label"])
        label_text  = "FAKE" if label_int == 1 else "REAL"
        color       = (0, 0, 255) if label_int == 1 else (0, 200, 0)

        frames_tensor = sample["frames"].unsqueeze(0).to(device)

        # ── Original BGR frames ───────────────────────────────────────────────
        video_path  = sample["meta"].get("video_path", "")
        orig_frames = _read_original_frames(
            video_path, config.num_frames, config.frame_size
        )

        # Add label text to all frames
        orig_labelled = []
        for f in orig_frames:
            f = f.copy()
            cv2.putText(f, label_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            orig_labelled.append(f)

        # ── Condition A: black-box (label only, no heatmap) ───────────────────
        zero_maps = np.zeros((config.num_frames,
                              config.frame_size, config.frame_size), np.float32)
        save_explanation_video(
            orig_labelled, zero_maps,
            os.path.join(out_root, "A", f"{idx:04d}.mp4"),
            fps=5, alpha=0.0,
        )

        # ── Condition B: post-hoc Grad-CAM overlay ────────────────────────────
        try:
            from xai.gradcam import GradCAMExplainer
            gc_exp = GradCAMExplainer(model, model.spatial_stream.grad_cam_target_layer)
            gc_heat = gc_exp.explain(frames_tensor)[0]   # (T, H, W)
        except Exception:
            gc_heat = zero_maps

        save_explanation_video(
            orig_labelled, gc_heat,
            os.path.join(out_root, "B", f"{idx:04d}.mp4"),
            fps=5, alpha=0.4,
        )

        # ── Condition C: intrinsic cross-attention overlay ────────────────────
        with torch.no_grad():
            out = model(frames_tensor)
        intrinsic = out.M_t_up[0].cpu().numpy()   # (T, H, W)

        save_explanation_video(
            orig_labelled, intrinsic,
            os.path.join(out_root, "C", f"{idx:04d}.mp4"),
            fps=5, alpha=0.4,
        )

    print(f"User-study stimuli generated: {out_root}")


def _read_original_frames(video_path: str, num_frames: int,
                           frame_size: int) -> list:
    blank = np.zeros((frame_size, frame_size, 3), np.uint8)
    if not video_path or not os.path.exists(video_path):
        return [blank.copy() for _ in range(num_frames)]

    cap    = cv2.VideoCapture(video_path)
    total  = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs   = np.linspace(0, total - 1, num_frames, dtype=int)
    idx_set = set(idxs.tolist())
    buf    = {}
    fi     = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in idx_set:
            buf[fi] = cv2.resize(frame, (frame_size, frame_size))
        fi += 1
    cap.release()
    return [buf.get(i, blank.copy()) for i in idxs]
