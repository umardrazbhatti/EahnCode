"""
scripts/verify_dataset.py — Pre-training dataset sanity check.

Usage:
    python scripts/verify_dataset.py --data_root /kaggle/input/.../ffpp_data

Exits with code 0 if all checks pass, code 1 if any check fails.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


def check_directory(label: str, path: str) -> tuple:
    """Return (exists, file_count, sample_filename)."""
    if not os.path.isdir(path):
        return False, 0, ""
    files = glob.glob(os.path.join(path, "*.mp4"))
    sample = os.path.basename(files[0]) if files else ""
    return True, len(files), sample


def main():
    parser = argparse.ArgumentParser(description="Verify FF++ dataset layout")
    parser.add_argument("--data_root", required=True,
                        help="Root directory of the FF++ dataset")
    args = parser.parse_args()

    data_root   = args.data_root
    compression = "c23"
    MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    failures = []

    # ── 1. Directory table ────────────────────────────────────────────────────
    print("\n=== Directory Check ===")
    header = f"{'Directory':<60} {'Exists':<8} {'Count':<8} {'Sample'}"
    print(header)
    print("-" * len(header))

    real_dir = os.path.join(data_root, "original_sequences", "youtube", compression, "videos")
    exists, count, sample = check_directory("real", real_dir)
    print(f"{real_dir[-60:]:<60} {'YES' if exists else 'NO':<8} {count:<8} {sample}")
    if not exists or count == 0:
        failures.append(f"Real video directory missing or empty: {real_dir}")

    total_fake = 0
    for method in MANIPULATIONS:
        vdir = os.path.join(data_root, "manipulated_sequences", method, compression, "videos")
        exists, count, sample = check_directory(method, vdir)
        print(f"{vdir[-60:]:<60} {'YES' if exists else 'NO':<8} {count:<8} {sample}")
        total_fake += count

    if total_fake == 0:
        failures.append("Zero fake videos found across all manipulation methods.")

    # ── 2. Dataset loading check ──────────────────────────────────────────────
    print("\n=== Dataset Loading Check ===")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import EAHNConfig
        from data.datasets import DeepfakeDataset
        from data.collate import deepfake_collate_fn

        config = EAHNConfig()
        config.data_root   = data_root
        config.num_frames  = 4   # fast check
        config.frame_size  = 224
        config.train_split = 0.8
        config.val_split   = 0.1
        config.cache_dir   = os.path.join(data_root, ".face_cache_verify")
        config.device      = "cpu"

        ds = DeepfakeDataset(config, "train", "ff++")
        loader = DataLoader(ds, batch_size=2, num_workers=0,
                            collate_fn=deepfake_collate_fn)
        batch = next(iter(loader))

        labels = batch["label"].tolist()
        print(f"  batch['label']    : {labels}")
        if not (0.0 in labels and 1.0 in labels):
            failures.append(
                f"Batch contains only one class: {labels}. "
                "Class balancing failed."
            )

        print(f"  batch['frames'].shape : {tuple(batch['frames'].shape)}")
        print(f"  batch['mask'].shape   : {tuple(batch['mask'].shape)}")
        print(f"  batch['mask'] sum     : {batch['mask'].sum().item():.4f}  "
              f"(expected 0.0 — no masks in this dataset)")
        print(f"  batch['has_mask']     : {batch['has_mask'].tolist()}  "
              f"(expected all False)")

        if batch["mask"].sum().item() != 0.0:
            failures.append("Mask tensor is non-zero — expected all-zero for this dataset.")
        if any(batch["has_mask"].tolist()):
            failures.append("has_mask is True for some samples — expected False.")

    except Exception as exc:
        failures.append(f"Dataset loading raised: {exc}")
        print(f"  ERROR: {exc}")

    # ── 3. Forward-pass check ─────────────────────────────────────────────────
    print("\n=== Model Forward-Pass Check ===")
    try:
        from models.eahn import EAHN

        model = EAHN(config).to("cpu")
        model.eval()

        frames_t = batch["frames"]
        with torch.no_grad():
            out = model(frames_t)

        probs = out.prob.tolist()
        m_min = out.M_t_up.min().item()
        m_max = out.M_t_up.max().item()
        m_mean = out.M_t_up.mean().item()

        print(f"  out.prob          : {probs}")
        print(f"  out.M_t_up min    : {m_min:.4f}")
        print(f"  out.M_t_up max    : {m_max:.4f}")
        print(f"  out.M_t_up mean   : {m_mean:.4f}")

        if m_max == 0.0:
            print("  WARNING: M_t_up is all zeros (model not trained yet — expected before training).")

    except Exception as exc:
        failures.append(f"Forward pass raised: {exc}")
        print(f"  ERROR: {exc}")

    # ── Result ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED — {len(failures)} issue(s) found:")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED — dataset is ready for training.")
        sys.exit(0)


if __name__ == "__main__":
    main()
