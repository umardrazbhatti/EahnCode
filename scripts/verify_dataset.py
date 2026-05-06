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
    batch = None
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
        config.cache_dir   = "/kaggle/working/.face_cache_verify"
        config.device      = "cpu"

        ds = DeepfakeDataset(config, "train", "ff++")
        loader = DataLoader(
            ds,
            batch_size=min(4, len(ds)),
            collate_fn=deepfake_collate_fn,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))

        labels_in_batch = batch["label"].tolist()
        print(f"  Labels in batch : {labels_in_batch}")
        print(f"  Frames shape    : {tuple(batch['frames'].shape)}")
        print(f"  Mask shape      : {tuple(batch['mask'].shape)}")
        print(f"  has_mask        : {batch['has_mask'].tolist()}")
        if len(set(labels_in_batch)) < 2:
            failures.append(
                "Batch contains only one class — check class balance "
                "and shuffle/sampler settings."
            )
    except Exception as exc:
        failures.append(f"Dataset loading raised: {exc}")
        print(f"  ERROR: {exc}")
        import traceback; traceback.print_exc()

    # ── 3. Forward-pass check ─────────────────────────────────────────────────
    print("\n=== Model Forward-Pass Check ===")
    if batch is not None:
        try:
            from models.eahn import EAHN

            model = EAHN(config).to("cpu")
            model.eval()
            with torch.no_grad():
                out = model(batch["frames"])

            print(f"  prob values : {[f'{p:.3f}' for p in out.prob.cpu().tolist()]}")
            mt = out.M_t
            print(f"  M_t shape   : {tuple(mt.shape)}")
            print(f"  M_t min/max : {mt.min():.4f} / {mt.max():.4f}")
            if mt.min() == mt.max():
                failures.append(
                    "M_t is constant (all same value). Model is not "
                    "producing spatial attention. Check cross-attention module."
                )
        except Exception as exc:
            failures.append(f"Forward pass raised: {exc}")
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
    else:
        failures.append(
            "Forward pass skipped — batch was not loaded. "
            "Fix the dataset loading error above first."
        )

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
