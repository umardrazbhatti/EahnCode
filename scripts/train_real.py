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
import contextlib
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast

from config import EAHNConfig, parse_args
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from models.eahn import EAHN
from losses.classification import build_classification_loss, FocalLoss
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

    labels_arr    = np.array([s["label"] for s in train_ds.samples], dtype=int)
    class_counts  = np.bincount(labels_arr, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(
        f"[Sampler] class_counts={class_counts.tolist()} "
        f"class_weights={class_weights.tolist()}"
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

    # ── First-batch class-balance smoke check ─────────────────────────────────
    _sb = next(iter(train_loader))
    _bl = _sb["label"].cpu().numpy().astype(int)
    _n_real, _n_fake = int((_bl == 0).sum()), int((_bl == 1).sum())
    print(f"[Smoke] First batch: real={_n_real} fake={_n_fake}")
    assert _n_real > 0 and _n_fake > 0, (
        f"First batch is single-class (real={_n_real}, fake={_n_fake}). "
        "Sampler or split is broken — check DeepfakeDataset._split()."
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = EAHN(config).to(device)

    if config.grad_checkpoint:
        model.enable_gradient_checkpointing()
        print("[GradCkpt] Gradient checkpointing enabled on TemporalStream.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # AMP — FP16 on T4 (sm_75). BF16 only on Ampere+. Disable on CPU.
    _use_amp = (
        config.use_amp
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 7
    )
    _amp_dtype = torch.float16 if config.amp_dtype == "fp16" else torch.bfloat16
    _dev_str   = device.type   # "cuda" or "cpu"
    scaler     = GradScaler(_dev_str, enabled=_use_amp)
    print(f"[AMP] use_amp={_use_amp}  dtype={config.amp_dtype}")

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
    if config.cls_loss_type == "focal":
        cls_loss_fn = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )
        print(f"[ClsLoss] FocalLoss(alpha={config.focal_alpha}, gamma={config.focal_gamma})")
    elif config.cls_loss_type == "bce":
        cls_loss_fn = torch.nn.BCEWithLogitsLoss()
        print("[ClsLoss] BCEWithLogitsLoss")
    else:
        raise ValueError(f"Unknown cls_loss_type: {config.cls_loss_type}")
    exp_loss_fn  = ExplanationLoss(alpha=config.alpha, beta=config.beta, diversity_weight=config.attn_diversity_weight)
    temp_loss_fn = TemporalConsistencyLoss(gamma=config.gamma)

    ckpt_path = os.path.join(config.output_dir, "best_model.pth")

    # ── Training loop ─────────────────────────────────────────────────────────
    total_batches = len(train_loader)
    epoch_w = len(str(config.epochs))
    batch_w = len(str(total_batches))

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            frames   = batch["frames"].to(device, non_blocking=True)
            labels   = batch["label"].to(device, non_blocking=True)
            masks    = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)

            with autocast(_dev_str, enabled=_use_amp, dtype=_amp_dtype):
                out      = model(frames)
                l_cls    = cls_loss_fn(out.logit, labels)
                exp_out  = exp_loss_fn(out.M_t, masks, has_mask)
                l_exp    = exp_out.loss
                l_temp   = temp_loss_fn(out.M_t, out.low_level)
                _global_step = epoch * len(train_loader) + batch_idx
                _lambda1_eff = config.lambda1 * min(1.0, _global_step / 200.0)
                l_total = l_cls + _lambda1_eff * l_exp + config.lambda2 * l_temp
                loss    = l_total / config.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if epoch == 0 and batch_idx == 0:
                print(f"[DIAG] M_t mean={out.M_t.mean():.4f} std={out.M_t.std():.4f}")
                print(f"[DIAG] L_cls={l_cls.item():.6f} L_exp={l_exp.item():.6f} L_temp={l_temp.item():.6f}")
                print(f"[DIAG] attn_temp=exp({model.cross_attention.log_temp.item():.3f})={torch.exp(model.cross_attention.log_temp).item():.3f}")

            if batch_idx % 20 == 0:
                _live_std = out.M_t.std().item()
                _live_tau = model.cross_attention.log_temp.exp().item()
                print(
                    f"[LIVE E{epoch+1} B{batch_idx:03d}] "
                    f"M_t_std={_live_std:.3f}  "
                    f"tau={_live_tau:.2f}  "
                    f"L_cls={l_cls.item():.6f}  "
                    f"L_H={exp_out.l_h:.6f}  "
                    f"L_TV={exp_out.l_tv:.6f}  "
                    f"L_div={exp_out.l_div:.6f}  "
                    f"L_temp={l_temp.item():.6f}  "
                    f"sample_sim={exp_out.inter_sample_sim:.2f}"
                )

            if (batch_idx + 1) % 50 == 0:
                bl = batch["label"].detach().cpu().numpy().astype(int)
                n_real, n_fake = int((bl == 0).sum()), int((bl == 1).sum())
                print(f"[BatchBalance] step={batch_idx+1} real={n_real} fake={n_fake}")

            running_loss += l_total.item()
            print(
                f"Epoch {epoch + 1:>{epoch_w}}/{config.epochs} | "
                f"Batch {batch_idx + 1:>{batch_w}}/{total_batches} | "
                f"Loss: {l_total.item():.6f} | "
                f"Cls: {l_cls.item():.6f} | "
                f"Exp: {l_exp.item():.6f} | "
                f"Temp: {l_temp.item():.6f} | "
                f"sim: {exp_out.inter_sample_sim:.2f}"
            )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                out    = model(frames)
                probs_list.extend(out.prob.cpu().tolist())
                labels_list.extend(batch["label"].cpu().tolist())

        metrics = DetectionMetrics.compute(probs_list, labels_list)
        logger.log_scalars("val", metrics, epoch)
        print(
            f"Epoch {epoch + 1:>{epoch_w}}/{config.epochs} | "
            f"Val AUC-ROC: {metrics['auc_roc']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        # Save best
        val_auc = metrics.get("auc_roc", float("nan"))
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc,
                            config, ckpt_path)
            print(f"--> Best model saved (AUC-ROC: {best_auc:.4f})")

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
