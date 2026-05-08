"""
data/collate.py — custom collate function for deepfake batches with optional masks.
"""

import torch


def deepfake_collate_fn(batch):
    frames   = torch.stack([item["frames"] for item in batch])               # (B,T,3,H,W)
    labels   = torch.tensor([item["label"]    for item in batch],
                             dtype=torch.float32)                            # (B,)
    masks    = torch.stack([item["mask"]   for item in batch])               # (B,h,w)
    has_mask = torch.tensor([item["has_mask"] for item in batch],
                             dtype=torch.bool)                               # (B,)
    meta     = [item["meta"] for item in batch]
    return {
        "frames":   frames,
        "label":    labels,
        "mask":     masks,
        "has_mask": has_mask,
        "meta":     meta,
    }
