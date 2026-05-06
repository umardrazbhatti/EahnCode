"""
utils/visualization.py — Overlay explanation maps on frames; save video clips.
"""

import cv2
import numpy as np


def overlay_explanation(
    frame_bgr: np.ndarray,   # H×W×3 uint8
    M_t: np.ndarray,         # H×W float32 in [0,1]
    prob: float,             # model's deepfake probability
    frame_idx: int,
    alpha: float = 0.45,
    threshold: float = 0.5,
) -> np.ndarray:
    """Blend a JET heatmap over a BGR frame with contour, verdict, and frame annotation."""
    H, W = frame_bgr.shape[:2]
    M_resized = cv2.resize(M_t.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    M_resized = np.clip(M_resized, 0, 1)

    heatmap_u8    = (M_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    blended       = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_color, alpha, 0)

    binary = (M_resized >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)

    verdict = "DEEPFAKE" if prob >= 0.5 else "REAL"
    color   = (0, 0, 255) if prob >= 0.5 else (0, 200, 0)
    cv2.putText(blended, f"{verdict}  conf={prob:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(blended, f"Frame {frame_idx + 1} | attn={M_t.max():.2f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return blended


def generate_text_explanation(
    prob: float,
    M_t_seq: list,           # list of H×W numpy arrays (one per frame)
    video_path: str = "",
) -> str:
    """Produce a plain-English per-video explanation from attention maps and confidence."""
    verdict  = "likely a DEEPFAKE" if prob >= 0.5 else "likely AUTHENTIC"
    conf_pct = int(max(prob, 1 - prob) * 100)

    peak_scores = [m.max() for m in M_t_seq]
    top_frames  = sorted(range(len(peak_scores)),
                         key=lambda i: peak_scores[i], reverse=True)[:3]

    top_map = M_t_seq[top_frames[0]]
    H, W    = top_map.shape
    yx      = np.unravel_index(np.argmax(top_map), top_map.shape)
    y_loc   = "upper" if yx[0] < H // 2 else "lower"
    x_loc   = "left"  if yx[1] < W // 2 else "right"
    region  = f"{y_loc}-{x_loc} facial region"

    lines = [
        f"VERDICT: This video is {verdict} (confidence: {conf_pct}%).",
        f"",
        f"EXPLANATION:",
        f"  • The model's attention was highest in frames "
        f"{', '.join(str(f + 1) for f in top_frames)}.",
        f"  • The primary area of concern is the {region}.",
        f"  • High attention in this area may indicate blending boundary",
        f"    artifacts, unnatural skin texture, or identity inconsistencies",
        f"    commonly introduced by face-swap deepfake methods.",
        f"",
        f"ATTENTION SCORES PER FRAME:",
    ]
    for i, score in enumerate(peak_scores):
        bar = "█" * int(score * 20)
        lines.append(f"  Frame {i + 1:02d}: {bar:<20} {score:.3f}")

    return "\n".join(lines)


def save_explanation_video(
    frames_np: list,         # list of (H, W, 3) BGR arrays
    M_t_seq,                 # (T, H, W) np array  OR  list of (H, W) arrays
    path: str,
    prob: float = 0.5,
    fps: int = 5,
    alpha: float = 0.45,
) -> None:
    """Save an explanation-overlay video using the annotated overlay_explanation renderer."""
    if len(frames_np) == 0:
        return

    H, W = frames_np[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(path, fourcc, fps, (W, H))

    if isinstance(M_t_seq, np.ndarray):
        maps = [M_t_seq[i] for i in range(M_t_seq.shape[0])]
    else:
        maps = list(M_t_seq)

    n = len(frames_np)
    maps = (maps * ((n // max(len(maps), 1)) + 1))[:n]

    for fi, (frame, m) in enumerate(zip(frames_np, maps)):
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H))
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H))
        overlayed = overlay_explanation(frame, m, prob=prob, frame_idx=fi, alpha=alpha)
        out.write(overlayed)

    out.release()
