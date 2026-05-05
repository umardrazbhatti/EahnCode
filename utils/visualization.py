"""
utils/visualization.py — Overlay explanation maps on frames; save video clips.
"""

import cv2
import numpy as np


def overlay_explanation(
    frame: np.ndarray,      # (H, W, 3) BGR uint8
    M_t: np.ndarray,        # (H, W) float [0, 1]
    colormap=cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a heatmap over a BGR frame."""
    heatmap   = cv2.applyColorMap(np.uint8(255 * np.clip(M_t, 0, 1)), colormap)
    overlayed = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return overlayed


def save_explanation_video(
    frames_np: list,         # list of (H, W, 3) BGR arrays
    M_t_seq,                 # (T, H, W) np array  OR  list of (H, W) arrays
    path: str,
    fps: int = 5,
    colormap=cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> None:
    """Save an explanation-overlay video to disk."""
    if len(frames_np) == 0:
        return

    H, W = frames_np[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(path, fourcc, fps, (W, H))

    if isinstance(M_t_seq, np.ndarray):
        # (T, H, W) — iterate over T
        maps = [M_t_seq[i] for i in range(M_t_seq.shape[0])]
    else:
        maps = list(M_t_seq)

    # Pad or truncate maps to match frames length
    n = len(frames_np)
    maps = (maps * ((n // len(maps)) + 1))[:n]

    for frame, m in zip(frames_np, maps):
        # Resize frame and map if needed
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H))
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H))

        if alpha > 0 and m.max() > 0:
            overlayed = overlay_explanation(frame, m, colormap, alpha)
        else:
            overlayed = frame

        out.write(overlayed)

    out.release()
