"""
config.py — single source of truth for all EAHN hyperparameters.
CLI overrides via argparse; no hardcoded paths anywhere else.
"""

import argparse
import warnings
import torch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EAHNConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: str = "/kaggle/input/"
    output_dir: str = "/kaggle/working/outputs/"
    cache_dir: str = "/kaggle/working/.face_cache/"
    resume_checkpoint: str = ""

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: Literal["synthetic", "ff++", "celeb_df", "dfdc"] = "ff++"
    dataset_compression: str = "c23"
    num_frames: int = 16
    frame_size: int = 224
    train_split: float = 0.8
    val_split: float = 0.1

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = "efficientnet_b4"
    backbone_pretrained: bool = True
    transformer_layers: int = 4
    transformer_heads: int = 8
    d_model: int = 256
    dropout: float = 0.1

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda1: float = 1.0   # L_exp weight
    lambda2: float = 0.1   # L_temp weight (was 0.5 — reduced to avoid freezing heatmaps)
    alpha: float = 0.2     # entropy weight in weak supervision (was 0.5 — reduced to avoid one-hot collapse)
    beta: float = 0.5      # TV weight in weak supervision
    gamma: float = 0.1     # gate decay rate in L_temp (was 10.0 — caused exp→0)
    attn_temp_init: float = 1.386   # log(4.0) — learnable cross-attention temperature init
    attn_diversity_weight: float = 2.5  # weight for attention diversity penalty in L_exp
    cls_dropout_p: float = 0.3    # probability of dropping CLS_out to force gradient through attn branch

    # ── Classification loss ───────────────────────────────────────────────────
    cls_loss_type: str = "bce"   # "bce" | "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 8
    grad_accum_steps: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-2
    mixed_precision: bool = True
    num_workers: int = 0   # 0 = safe for Kaggle CUDA; increase locally if desired

    # ── Evaluation / Visualisation ────────────────────────────────────────────
    eval_after_train: bool = True
    save_heatmaps: bool = True
    heatmap_samples: int = 20

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                warnings.warn("No GPU found. Switching to CPU with reduced settings.")
                self._apply_cpu_safe_overrides()

    def _apply_cpu_safe_overrides(self):
        self.num_frames = 4
        self.transformer_layers = 2
        self.transformer_heads = 2
        self.batch_size = 2
        self.mixed_precision = False
        self.num_workers = 0
        if "efficientnet_b4" in self.backbone:
            self.backbone = "efficientnet_b0"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EAHNConfig":
        cfg = cls()
        for key, val in vars(args).items():
            if hasattr(cfg, key) and val is not None:
                setattr(cfg, key, val)
        return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAHN Training and Evaluation")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None,
                        choices=["synthetic", "ff++", "celeb_df", "dfdc"])
    parser.add_argument("--dataset_compression", type=str, default=None,
                        help="FF++ compression level, e.g. c23 (default) or c40")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader worker processes. Use 0 on Kaggle to avoid fork errors.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--heatmap_samples", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--eval_after_train", action="store_true", default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--attn_temp_init", type=float, default=None)
    parser.add_argument("--attn_diversity_weight", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--cls_dropout_p", type=float, default=None)
    parser.add_argument("--cls_loss_type", type=str, default=None,
                        choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    return parser.parse_args()
