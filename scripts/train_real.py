"""
scripts/train_real.py — Phase 6 GPU training on FF++/Celeb-DF/DFDC.

Phase 6 changes vs phase 5d:
  - --max_per_class flag for balanced 1k/1k subsampling  (CHANGE 1)
  - WeightedRandomSampler safety net rebuild              (CHANGE 2)
  - 100-batch rolling log (not per-step)                 (CHANGE 3)
  - Per-epoch attention-diversity diagnostic              (CHANGE 4)
  - label_smoothing wired through build_classification_loss (CHANGE 6)
  - loss_curves.png + metric_curves.png +
    training_history.csv emitted at end of training       (CHANGE 12)
"""

import os
import csv as _csv
import math
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import random

from config import EAHNConfig, parse_args
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from models.eahn import EAHN
from losses.classification import build_classification_loss
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

    # ── CHANGE 1: optional per-class subsampling (balanced 1k/1k) ─────────────
    max_per_class = int(getattr(config, "max_per_class", 0) or 0)
    if max_per_class > 0:
        random.seed(42)
        buckets = defaultdict(list)
        for s in train_ds.samples:
            buckets[s["label"]].append(s)
        new_samples = []
        for lbl, lst in sorted(buckets.items()):
            random.shuffle(lst)
            kept = lst[:max_per_class]
            new_samples.extend(kept)
            print(f"[balance] class={lbl}: kept {len(kept)} of {len(lst)} samples")
        random.shuffle(new_samples)
        train_ds.samples = new_samples
        print(f"[balance] train set is now {len(train_ds.samples)} samples total")

    # ── CHANGE 2: WeightedRandomSampler safety net ────────────────────────────
    train_labels  = [s["label"] for s in train_ds.samples]
    class_counts  = np.bincount(train_labels, minlength=2)
    print(f"[sampler] class counts: real={class_counts[0]}, fake={class_counts[1]}")
    class_weights  = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [float(class_weights[l]) for l in train_labels]
    train_sampler  = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=deepfake_collate_fn,
        pin_memory=(config.device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(config.device == "cuda"),
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
    model = EAHN(config).to(device)

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
    _dev_str   = device.type
    scaler     = GradScaler(_dev_str, enabled=_use_amp)
    print(f"[AMP] use_amp={_use_amp}  dtype={config.amp_dtype}")

    logger = Logger(config.output_dir)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_auc    = -1.0
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        ckpt = load_checkpoint(config.resume_checkpoint, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {start_epoch}, best AUC {best_auc:.4f}")

    # ── Losses ────────────────────────────────────────────────────────────────
    # CHANGE 6: label_smoothing read from config by build_classification_loss
    cls_loss_fn = build_classification_loss(config)
    print(
        f"[ClsLoss] {cls_loss_fn.__class__.__name__}  "
        f"label_smoothing={getattr(config, 'label_smoothing', 0.0)}"
    )
    exp_loss_fn  = ExplanationLoss(
        alpha=config.alpha,
        beta=config.beta,
        diversity_weight=config.attn_diversity_weight,
    )
    temp_loss_fn = TemporalConsistencyLoss(gamma=config.gamma)

    ckpt_path = os.path.join(config.output_dir, "best_model.pth")

    # ── CHANGE 12a: epoch-level training history ───────────────────────────────
    history = {
        "epoch":               [],
        "train_total":         [], "train_cls":  [],
        "train_exp":           [], "train_temp": [],
        "val_auc_roc":         [], "val_balanced_acc":      [],
        "val_real_acc":        [], "val_fake_acc":          [],
        "val_inter_sample_cos": [], "val_mt_std":           [],
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    total_batches = len(train_loader)
    epoch_w = len(str(config.epochs))

    for epoch in range(start_epoch, config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # ── CHANGE 12b: per-epoch loss accumulator ────────────────────────────
        epoch_acc = {"total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0, "n": 0}

        # ── CHANGE 3: 100-batch rolling log accumulator ───────────────────────
        LOG_EVERY = 100
        run = {"total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0, "n": 0}

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
                l_total  = l_cls + _lambda1_eff * l_exp + config.lambda2 * l_temp
                loss     = l_total / config.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # ── First-batch diagnostics (epoch 0 only) ────────────────────────
            if epoch == 0 and batch_idx == 0:
                print(f"[DIAG] M_t mean={out.M_t.mean():.4f} std={out.M_t.std():.4f}")
                print(f"[DIAG] L_cls={l_cls.item():.6f} L_exp={l_exp.item():.6f} "
                      f"L_temp={l_temp.item():.6f}")
                print(f"[DIAG] attn_temp=exp({model.cross_attention.log_temp.item():.3f})"
                      f"={torch.exp(model.cross_attention.log_temp).item():.3f}")

            # ── Batch balance check every 50 steps ────────────────────────────
            if (batch_idx + 1) % 50 == 0:
                bl = batch["label"].detach().cpu().numpy().astype(int)
                n_real, n_fake = int((bl == 0).sum()), int((bl == 1).sum())
                print(f"[BatchBalance] step={batch_idx+1} real={n_real} fake={n_fake}")

            # ── Accumulate losses ─────────────────────────────────────────────
            _lt = l_total.item()
            _lc = l_cls.item()
            _le = l_exp.item()
            _lp = l_temp.item()

            run["total"] += _lt;  run["cls"] += _lc
            run["exp"]   += _le;  run["temp"] += _lp;  run["n"] += 1

            epoch_acc["total"] += _lt;  epoch_acc["cls"] += _lc
            epoch_acc["exp"]   += _le;  epoch_acc["temp"] += _lp;  epoch_acc["n"] += 1

            # ── CHANGE 3: rolling 100-batch log ───────────────────────────────
            if (batch_idx + 1) % LOG_EVERY == 0 or (batch_idx + 1) == total_batches:
                n = max(run["n"], 1)
                _tau = model.cross_attention.log_temp.exp().item()
                print(
                    f"[E{epoch+1:>{epoch_w}} {batch_idx+1:4d}/{total_batches}] "
                    f"total={run['total']/n:.4f}  cls={run['cls']/n:.4f}  "
                    f"exp={run['exp']/n:.4f}  temp={run['temp']/n:.4f}  "
                    f"tau={_tau:.2f}  sim={exp_out.inter_sample_sim:.2f}"
                )
                run = {"total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0, "n": 0}

        scheduler.step()

        # ── CHANGE 12b cont.: store epoch-average train losses ─────────────────
        n = max(epoch_acc["n"], 1)
        history["epoch"].append(epoch)
        history["train_total"].append(epoch_acc["total"] / n)
        history["train_cls"].append(epoch_acc["cls"]   / n)
        history["train_exp"].append(epoch_acc["exp"]   / n)
        history["train_temp"].append(epoch_acc["temp"] / n)

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
            f"F1: {metrics['f1_at_0.5']:.4f}"
        )

        _val_real_acc = float(metrics.get("real_accuracy",    0.0))
        _val_fake_acc = float(metrics.get("fake_accuracy",    0.0))
        _val_bal_acc  = float(metrics.get("balanced_accuracy", 0.0))
        print(
            f"[ValMetrics] epoch={epoch + 1} "
            f"real_acc={_val_real_acc:.3f} "
            f"fake_acc={_val_fake_acc:.3f} "
            f"balanced_acc={_val_bal_acc:.3f}"
        )

        # ── CHANGE 4: attention-diversity diagnostic on first val batch ────────
        with torch.no_grad():
            diag_batch  = next(iter(val_loader))
            diag_frames = diag_batch["frames"].to(device)
            diag_out    = model(diag_frames)
            mt          = diag_out.M_t.mean(dim=1)          # (B, h, w)
            mt_flat     = mt.reshape(mt.size(0), -1)        # (B, hw)
            mt_norm     = torch.nn.functional.normalize(mt_flat, dim=1)
            cos_mat     = mt_norm @ mt_norm.t()
            B_d         = cos_mat.size(0)
            off_mask    = ~torch.eye(B_d, dtype=torch.bool, device=cos_mat.device)
            off         = cos_mat[off_mask]
            diag_cosine = float(off.mean()) if off.numel() > 0 else 0.0
            diag_std    = float(mt_flat.std(dim=1).mean())
        model.train()
        print(
            f"[Diag] epoch={epoch+1} "
            f"inter_sample_cos={diag_cosine:.3f}  mt_std={diag_std:.4f}"
        )

        # ── CHANGE 12c: val metrics history ───────────────────────────────────
        history["val_auc_roc"].append(float(metrics.get("auc_roc", float("nan"))))
        history["val_balanced_acc"].append(_val_bal_acc)
        history["val_real_acc"].append(_val_real_acc)
        history["val_fake_acc"].append(_val_fake_acc)
        history["val_inter_sample_cos"].append(diag_cosine)
        history["val_mt_std"].append(diag_std)

        # ── Checkpoint ────────────────────────────────────────────────────────
        val_auc = metrics.get("auc_roc", float("nan"))
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc,
                            config, ckpt_path)
            print(f"--> Best model saved (AUC-ROC: {best_auc:.4f})")

        last_ckpt = os.path.join(
            config.output_dir, f"checkpoint_epoch{epoch:03d}.pth"
        )
        save_checkpoint(model, optimizer, scheduler, epoch, val_auc, config, last_ckpt)

    logger.close()
    print(f"\nTraining complete. Best AUC-ROC: {best_auc:.4f}")

    # ── CHANGE 12d: end-of-run plots and CSV ──────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save raw history to CSV
        csv_hist = out_path / "training_history.csv"
        with open(csv_hist, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(list(history.keys()))
            w.writerows(zip(*history.values()))
        print(f"[plot] saved {csv_hist}")

        # Plot 1: training loss convergence (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (key, title) in zip(axes.flat, [
            ("train_total", "Total Loss"),
            ("train_cls",   "Classification Loss"),
            ("train_exp",   "Explanation Loss"),
            ("train_temp",  "Temporal Consistency Loss"),
        ]):
            ax.plot(history["epoch"], history[key], marker="o", linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
        fig.suptitle("EAHN — Training Loss Convergence", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path / "loss_curves.png", dpi=120)
        plt.close(fig)
        print(f"[plot] saved {out_path / 'loss_curves.png'}")

        # Plot 2: validation metric trajectories (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (keys, title) in zip(axes.flat, [
            (["val_auc_roc"],                         "Val AUC-ROC"),
            (["val_real_acc", "val_fake_acc"],        "Per-class Val Accuracy"),
            (["val_balanced_acc"],                    "Val Balanced Accuracy"),
            (["val_inter_sample_cos", "val_mt_std"],  "Attention Diversity"),
        ]):
            for k in keys:
                ax.plot(history["epoch"], history[k],
                        marker="o", linewidth=2, label=k)
            if "AUC" in title or "Balanced" in title:
                ax.axhline(0.5, color="grey", linestyle="--",
                           alpha=0.5, label="random")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("EAHN — Validation Metric Trajectories", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path / "metric_curves.png", dpi=120)
        plt.close(fig)
        print(f"[plot] saved {out_path / 'metric_curves.png'}")

    except Exception as _plot_err:
        print(f"[plot] Warning: could not generate training plots: {_plot_err}")

    if config.eval_after_train:
        from scripts.evaluate import run_evaluation
        print("\n--- Starting evaluation ---")
        run_evaluation(config)


if __name__ == "__main__":
    args   = parse_args()
    config = EAHNConfig.from_args(args)
    main(config)
