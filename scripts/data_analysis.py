"""
scripts/data_analysis.py — Dataset statistics and class distribution analysis.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from config import EAHNConfig
from data.datasets import DeepfakeDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    default="/kaggle/input/")
    parser.add_argument("--dataset_name", default="ff++")
    parser.add_argument("--output_dir",   default="outputs/")
    args = parser.parse_args()

    cfg = EAHNConfig(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        ds = DeepfakeDataset(cfg, split, args.dataset_name)
        labels = [s.get("label", s) if isinstance(s, dict) else s
                  for s in [ds.samples[i] for i in range(len(ds.samples))]]
        labels = [s["label"] if isinstance(s, dict) else s for s in ds.samples]
        counts = pd.Series(labels).value_counts().rename({0: "real", 1: "fake"})
        print(f"{split}: {counts.to_dict()}")

    print("Data analysis complete.")


if __name__ == "__main__":
    main()
