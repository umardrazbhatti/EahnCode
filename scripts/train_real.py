"""
scripts/train_real.py — Phase 2 GPU training on FF++/Celeb-DF/DFDC.

Fixes vs original:
  - autocast used correctly (no-op if not mixed precision / CPU)
  - Scheduler (CosineAnnealingLR) wired in, as specified in the thesis
  - grad_accum_steps respected
  - eval_after_train uses run_evaluation from evaluate.py
"""

import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast

from config import EAHNConfig, parse_args
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from models.eahn import EAHN
from losses.classification import ClassificationLoss
from losses.explanation import ExplanationLoss
from losses.temporal import TemporalConsistencyLoss
from metrics.detection import DetectionMetrics
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.logging_utils import Logger


def main(config: EAHNConfig):
    device = torch.device(config.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        cap  = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
        print(f"[Device] {name} | CUDA capability sm_{cap[0]}{cap[1]}")
        if cap[0] < 7:
            print(
                f"[WARNING] sm_{cap[0]}{cap[1]} is below PyTorch minimum "
                f"(sm_70). Switch Kaggle accelerator to T4. "
                f"Falling back to CPU for MTCNN. AMP disabled."
            )
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(config, "train", config.dataset_name)
    val_ds   = DeepfakeDataset(config, "val",   config.dataset_name)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_labels   = [s["label"] for s in train_ds.samples]
    class_counts   = [train_labels.count(0), train_labels.count(1)]
    class_weights  = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = EAHN(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Mixed precision — requires CUDA capability sm_70+ (Volta and above)
    use_amp = (
        config.mixed_precision
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 7
    )
    scaler  = GradScaler("cuda") if use_amp else None

    logger  = Logger(config.output_dir)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_auc    = -1.0
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        ckpt = load_checkpoint(config.resume_checkpoint, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {start_epoch}, best AUC {best_auc:.4f}")

    # ── Losses ────────────────────────────────────────────────────────────────
    cls_loss_fn  = ClassificationLoss()
    exp_loss_fn  = ExplanationLoss(alpha=config.alpha, beta=config.beta)
    temp_loss_fn = TemporalConsistencyLoss(gamma=config.gamma)

    ckpt_path = os.path.join(config.output_dir, "best_model.pth")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch_idx, batch in pbar:
            frames   = batch["frames"].to(device)
            labels   = batch["label"].to(device)
            masks    = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device)

            import contextlib
            ctx = autocast("cuda") if use_amp else contextlib.nullcontext()

            with ctx:
                out    = model(frames)
                l_cls  = cls_loss_fn(out.logit, labels)
                l_exp  = exp_loss_fn(out.M_t, masks, has_mask)
                l_temp = temp_loss_fn(out.M_t, out.low_level)
                loss   = l_cls + config.lambda1 * l_exp + config.lambda2 * l_temp
                loss   = loss / config.grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * config.grad_accum_steps
            pbar.set_postfix({"loss": f"{loss.item() * config.grad_accum_steps:.4f}"})

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                frames = batch["frames"].to(device)
                out    = model(frames)
                probs_list.extend(out.prob.cpu().tolist())
                labels_list.extend(batch["label"].cpu().tolist())

        metrics = DetectionMetrics.compute(probs_list, labels_list)
        logger.log_scalars("val", metrics, epoch)
        avg_loss = running_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val AUC-ROC: {metrics['auc_roc']:.3f} | "
            f"F1: {metrics['f1']:.3f}"
        )

        # Save best
        val_auc = metrics.get("auc_roc", float("nan"))
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc,
                            config, ckpt_path)
            print(f"  --> Best model saved (AUC-ROC: {best_auc:.4f})")

        # Always save a per-epoch fallback checkpoint
        last_ckpt = os.path.join(config.output_dir, f"checkpoint_epoch{epoch:03d}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, val_auc, config, last_ckpt)

    logger.close()
    print(f"\nTraining complete. Best AUC-ROC: {best_auc:.4f}")

    if config.eval_after_train:
        from scripts.evaluate import run_evaluation
        print("\n--- Starting evaluation ---")
        run_evaluation(config)


if __name__ == "__main__":
    args   = parse_args()
    config = EAHNConfig.from_args(args)
    main(config)
