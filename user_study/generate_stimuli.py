"""
Automated stimulus generator. Replace test samples with study protocol materials
and deploy via web interface for live human study administration.

This module generates three-condition comparison PNGs for the first 20 test
samples without any randomness or human input.  Outputs are saved to
user_study/stimuli/ and can be used directly as appendix evidence or wired
into a web-based study interface.
"""

import os
import cv2
import numpy as np
import torch

from config import EAHNConfig
from utils.visualization import (
    overlay_explanation,
    save_annotated_frame_strip,
    overlay_heatmap_on_frame,
)


class AutomatedStimulusGenerator:
    """
    Generate three-condition comparison images for 20 test samples.

    Conditions:
        A (black-box):  original frames + verdict text only
        B (post-hoc):   frames + GradCAM overlay + verdict text
        C (intrinsic):  frames + M_t overlay + verdict text + per-frame bar

    Outputs per sample:
        stimuli/A/{id}_bb.png
        stimuli/B/{id}_posthoc.png
        stimuli/C/{id}_intrinsic.png
        stimuli/comparison_{id}.png  — all three side by side
    """

    N_SAMPLES    = 20
    FRAME_SIZE   = 224
    STRIP_FRAMES = 8

    def __init__(self, model, test_loader, gradcam_explainer,
                 config: EAHNConfig, output_dir: str):
        self.model           = model
        self.test_loader     = test_loader
        self.gradcam         = gradcam_explainer
        self.config          = config
        self.device          = torch.device(config.device)
        self.output_dir      = output_dir

        for cond in ("A", "B", "C"):
            os.makedirs(os.path.join(output_dir, "stimuli", cond), exist_ok=True)

    # ── public entry point ─────────────────────────────────────────────────────

    def generate(self):
        """Process the first N_SAMPLES batches from test_loader; no randomness."""
        self.model.eval()
        sample_id = 0

        for batch_idx, batch in enumerate(self.test_loader):
            if batch_idx >= self.N_SAMPLES:
                break

            frames    = batch["frames"].to(self.device)   # (B, T, 3, H, W)
            labels    = batch["label"]
            B         = frames.shape[0]

            with torch.no_grad():
                output = self.model(frames)

            # GradCAM maps
            try:
                gradcam_maps = self.gradcam.explain(frames)   # (B, T, H, W) or similar
            except Exception:
                gradcam_maps = None

            for b in range(B):
                if sample_id >= self.N_SAMPLES:
                    break

                sid       = f"{sample_id:04d}"
                label_int = int(labels[b].item()) if torch.is_tensor(labels[b]) else int(labels[b])
                verdict   = "FAKE" if label_int == 1 else "REAL"
                prob_val  = output.prob[b].item()

                # Raw frames as BGR numpy arrays (T, H, W, 3)
                frames_bgr = self._tensor_to_bgr_list(frames[b])  # list of T arrays

                # Intrinsic maps: M_t_up (T, H, W) numpy
                M_t_up = output.M_t_up[b].cpu().numpy()

                # attention_scores for bar chart: per-frame M_t mean
                M_t_sample = output.M_t[b].cpu()  # (T, h, w)
                attn_scores = [M_t_sample[t].mean().item() for t in range(M_t_sample.shape[0])]

                # GradCAM maps for this sample
                if gradcam_maps is not None:
                    try:
                        gc_maps = gradcam_maps[b].cpu().numpy()  # (T, H, W)
                    except Exception:
                        gc_maps = np.zeros_like(M_t_up)
                else:
                    gc_maps = np.zeros_like(M_t_up)

                # Build the three condition images
                path_a = self._make_condition_a(frames_bgr, verdict, prob_val, sid)
                path_b = self._make_condition_b(frames_bgr, gc_maps, verdict, prob_val, sid)
                path_c = self._make_condition_c(
                    frames_bgr, M_t_up, attn_scores, verdict, prob_val, sid
                )

                self._make_comparison(path_a, path_b, path_c, sid)
                sample_id += 1

        self._write_readme()
        print(f"[AutomatedStimulusGenerator] Done — {sample_id} samples saved to {self.output_dir}/stimuli/")

    # ── condition builders ─────────────────────────────────────────────────────

    def _make_condition_a(self, frames_bgr, verdict, prob, sid):
        """Black-box: original frames + verdict text only."""
        n  = min(self.STRIP_FRAMES, len(frames_bgr))
        sel = np.linspace(0, len(frames_bgr) - 1, n, dtype=int)
        labelled = []
        color = (80, 80, 255) if verdict == "FAKE" else (80, 255, 80)
        for idx in sel:
            f = cv2.resize(frames_bgr[idx], (self.FRAME_SIZE, self.FRAME_SIZE))
            cv2.putText(f, verdict, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(f, f"p={prob:.2f}", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            labelled.append(f)
        path = os.path.join(self.output_dir, "stimuli", "A", f"{sid}_bb.png")
        cv2.imwrite(path, np.hstack(labelled))
        return path

    def _make_condition_b(self, frames_bgr, gc_maps, verdict, prob, sid):
        """Post-hoc: GradCAM overlay + verdict text."""
        n   = min(self.STRIP_FRAMES, len(frames_bgr))
        sel = np.linspace(0, len(frames_bgr) - 1, n, dtype=int)
        color = (80, 80, 255) if verdict == "FAKE" else (80, 255, 80)
        overlaid = []
        for t_pos, idx in enumerate(sel):
            f = cv2.resize(frames_bgr[idx], (self.FRAME_SIZE, self.FRAME_SIZE))
            if t_pos < gc_maps.shape[0]:
                f = overlay_explanation(f, gc_maps[t_pos], alpha=0.4)
            cv2.putText(f, verdict, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            overlaid.append(f)
        path = os.path.join(self.output_dir, "stimuli", "B", f"{sid}_posthoc.png")
        cv2.imwrite(path, np.hstack(overlaid))
        return path

    def _make_condition_c(self, frames_bgr, M_t_up, attn_scores, verdict, prob, sid):
        """Intrinsic: M_t overlay + verdict text + per-frame bar."""
        path = os.path.join(self.output_dir, "stimuli", "C", f"{sid}_intrinsic.png")
        attn_maps_list = [M_t_up[t] for t in range(M_t_up.shape[0])]
        save_annotated_frame_strip(
            frames_bgr=frames_bgr,
            attention_maps=attn_maps_list,
            attention_scores=attn_scores,
            verdict=verdict,
            prob=prob,
            output_path=path,
            sample_id=sid,
        )
        return path

    def _make_comparison(self, path_a, path_b, path_c, sid):
        """Horizontal concatenation of all three condition images."""
        imgs = []
        for p in (path_a, path_b, path_c):
            img = cv2.imread(p)
            if img is not None:
                imgs.append(img)
        if not imgs:
            return

        target_h = max(i.shape[0] for i in imgs)
        padded = []
        for img in imgs:
            if img.shape[0] < target_h:
                pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)

        comparison = np.hstack(padded)
        out_path   = os.path.join(self.output_dir, "stimuli", f"comparison_{sid}.png")
        cv2.imwrite(out_path, comparison)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _tensor_to_bgr_list(self, frames_tensor):
        """Convert (T, 3, H, W) float tensor in [0,1] to list of BGR uint8 arrays."""
        bgr_list = []
        T = frames_tensor.shape[0]
        for t in range(T):
            frame = frames_tensor[t].cpu().numpy()          # (3, H, W)
            frame = np.transpose(frame, (1, 2, 0))          # (H, W, 3)
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bgr_list.append(frame)
        return bgr_list

    def _write_readme(self):
        readme = os.path.join(self.output_dir, "stimuli", "README.txt")
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "EAHN User Study Stimuli\n"
                "=======================\n\n"
                "This directory contains automated comparison images for 20 test samples.\n\n"
                "Condition A (stimuli/A/):  Original frames + verdict only (black-box baseline)\n"
                "Condition B (stimuli/B/):  Frames + GradCAM post-hoc explanation overlay\n"
                "Condition C (stimuli/C/):  Frames + intrinsic cross-attention M_t overlay\n"
                "comparison_*.png:          All three conditions side by side for one sample\n\n"
                "To conduct a real human study:\n"
                "  1. Replace the 20 test samples with your study protocol materials.\n"
                "  2. Randomise presentation order and condition assignment.\n"
                "  3. Administer via a web interface (e.g., jsPsych or PsychoPy online).\n"
                "  4. Collect participant judgements (real/fake) and response times.\n"
            )


# Alias for backward compatibility with evaluate.py import
generate_stimuli = AutomatedStimulusGenerator
