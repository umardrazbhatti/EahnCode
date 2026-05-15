"""
data/datasets.py  —  EAHN DeepfakeDataset
==========================================
Locked to FF++ c23 custom-layout dataset.

Verified folder layout on Kaggle
(umardrazbhatti/ffpp-c23-custom-layout/ffpp_data/):

  Real:  original_sequences/youtube/c23/videos/*.mp4          (1 000 videos, label=0)
  Fake:  manipulated_sequences/{Method}/c23/videos/*.mp4      (1 000 per method × 5, label=1)
         Methods: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures

No pixel-level manipulation masks exist in this snapshot → weakly-supervised
L_exp (entropy + TV + inter-sample diversity) throughout.

Class imbalance is handled at the DataLoader level via WeightedRandomSampler
(see scripts/train_real.py).  heavy_aug is set automatically when the class
ratio exceeds 3:1 — on FF++ c23 this triggers heavy augmentation for the 1 000
real videos (the minority class).

Other dataset types:
  synthetic — generated in RAM for unit tests / smoke tests
  dfdc      — DFDC metadata.json layout
  celeb_df  — DEFERRED to future work (raises NotImplementedError)
"""

import os
import json
import random
import warnings
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    warnings.warn(
        "decord not available; falling back to OpenCV for video reading. "
        "Install with: pip install decord"
    )

from data.face_align import FaceAligner
from data.transforms import get_transforms, get_heavy_transforms
from data.synthetic_generator import SyntheticDataGenerator

# ---------------------------------------------------------------------------
# FF++ manipulation methods in this dataset snapshot
# ---------------------------------------------------------------------------
FF_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]


class DeepfakeDataset(Dataset):
    """
    Unified dataset loader for FF++, DFDC and synthetic data.

    Each __getitem__ returns a dict:
        frames    : Tensor (T, 3, H, W)  — normalised face-aligned frames
        label     : int                  — 0 = real, 1 = fake
        mask      : Tensor (h, w)        — manipulation mask or zeros
        has_mask  : bool                 — whether mask is a real GT mask
        meta      : dict                 — video_path, frame_indices
    """

    def __init__(
        self,
        config,
        mode: Literal["train", "val", "test"],
        dataset_type: Literal["synthetic", "ff++", "celeb_df", "dfdc"],
    ):
        self.config       = config
        self.mode         = mode
        self.dataset_type = dataset_type
        self.transform    = get_transforms(mode, config.frame_size)
        self.has_masks: bool = False
        self.heavy_aug: bool = False
        self.minority_class: int = 1
        self.samples: list[dict] = []

        # Face aligner — shared across all dataset types
        self.face_aligner = FaceAligner(
            margin=0.30,
            cache_dir=getattr(config, "cache_dir", None),
            device=config.device,
        )

        # ── Build sample list ────────────────────────────────────────────
        if dataset_type == "synthetic":
            self._build_synthetic()
        elif dataset_type == "ff++":
            self._build_ffpp()
        elif dataset_type == "celeb_df":
            self._build_celeb_df()
        elif dataset_type == "dfdc":
            self._build_dfdc()
        else:
            raise ValueError(f"Unknown dataset_type: '{dataset_type}'")

        # ── Dataset-agnostic imbalance detection ─────────────────────────
        if self.samples:
            _labels = np.array([s["label"] for s in self.samples], dtype=int)
            _counts = np.bincount(_labels, minlength=2)
            _ratio  = _counts.max() / max(_counts.min(), 1)
            self.heavy_aug      = bool(_ratio > 3.0)
            self.minority_class = int(_counts.argmin())
            print(
                f"[Imbalance] real={_counts[0]} fake={_counts[1]} "
                f"ratio={_ratio:.2f} heavy_aug={self.heavy_aug} "
                f"minority_class={self.minority_class}"
            )

        # ── Stratified train / val / test split ──────────────────────────
        self.samples = self._split(
            self.samples, mode, config.train_split, config.val_split
        )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[DeepfakeDataset] No samples found for "
                f"dataset='{dataset_type}', mode='{mode}'. "
                f"Check config.data_root='{config.data_root}'."
            )

        # ── Post-split class distribution ────────────────────────────────
        self.n_real = sum(1 for s in self.samples if s["label"] == 0)
        self.n_fake = sum(1 for s in self.samples if s["label"] == 1)
        _split_ratio = self.n_fake / max(self.n_real, 1)
        from collections import Counter as _Counter
        _fake_types = _Counter(
            s["manipulation"] for s in self.samples if s["label"] == 1
        )
        _per_type = (
            f"DF={_fake_types.get('Deepfakes', 0)} "
            f"F2F={_fake_types.get('Face2Face', 0)} "
            f"FShift={_fake_types.get('FaceShifter', 0)} "
            f"FSwap={_fake_types.get('FaceSwap', 0)} "
            f"NT={_fake_types.get('NeuralTextures', 0)}"
        )
        print(
            f"[DeepfakeDataset | {dataset_type} / {mode}] "
            f"total={len(self.samples)}  real={self.n_real}  fake={self.n_fake}  "
            f"ratio={_split_ratio:.1f}:1 (per_fake_type: {_per_type})"
        )

    # ====================================================================
    # Dataset builders
    # ====================================================================

    def _build_ffpp(self):
        """
        FF++ c23 custom-layout builder.

        Relative to config.data_root (= .../ffpp_data/):
            Real: original_sequences/youtube/c23/videos/*.mp4
            Fake: manipulated_sequences/{Method}/c23/videos/*.mp4

        When config.max_per_class > 0 (Regime A), the cap is applied per
        binary class BEFORE the train/val/test split:
            Real: up to max_per_class real videos (random, seed=42)
            Fake: up to max_per_class // 5 per manipulation type (seed=42),
                  remainder videos distributed in FF_METHODS order.
        This guarantees type-balanced fake coverage in every split.
        """
        root      = Path(self.config.data_root)
        real_dir  = root / "original_sequences" / "youtube" / "c23" / "videos"
        fake_root = root / "manipulated_sequences"

        real_videos: list[Path] = list(sorted(real_dir.glob("*.mp4")))

        # Build per-type fake video lists
        fake_by_type: dict[str, list[Path]] = {}
        for method in FF_METHODS:
            d = fake_root / method / "c23" / "videos"
            if d.exists():
                fake_by_type[method] = list(sorted(d.glob("*.mp4")))
            else:
                warnings.warn(f"[FF++] Method directory not found: {d}")
                fake_by_type[method] = []

        assert len(real_videos) > 0, (
            f"No real videos found at {real_dir}. "
            "Check config.data_root points to ffpp_data/."
        )
        assert sum(len(v) for v in fake_by_type.values()) > 0, (
            f"No fake videos found under {fake_root}. "
            "Check that manipulated_sequences/ subdirectories exist."
        )

        # Apply max_per_class cap per binary class, not per directory.
        # Fake side distributes the budget evenly across the 5 types so that
        # train/val/test splits all contain proportional type coverage.
        max_n = int(getattr(self.config, "max_per_class", 0) or 0)
        if max_n > 0:
            _rng = random.Random(42)

            # Real side: sample up to max_n
            if len(real_videos) > max_n:
                _rng.shuffle(real_videos)
                real_videos = real_videos[:max_n]

            # Fake side: max_n // 5 per type, remainder → first types in order
            n_types   = len(FF_METHODS)
            base_cap  = max_n // n_types
            remainder = max_n % n_types
            type_counts: dict[str, int] = {}
            capped_fake: list[Path] = []
            for j, method in enumerate(FF_METHODS):
                cap_this = base_cap + (1 if j < remainder else 0)
                vids = fake_by_type[method][:]
                if len(vids) > cap_this:
                    _rng.shuffle(vids)
                    vids = vids[:cap_this]
                capped_fake.extend(vids)
                type_counts[method] = len(vids)

            _type_str = " ".join(f"{m}={type_counts[m]}" for m in FF_METHODS)
            print(
                f"[MaxPerClass cap={max_n}] real={len(real_videos)} fake={len(capped_fake)} "
                f"(per_type: {_type_str})"
            )
            fake_videos_all: list[Path] = capped_fake
        else:
            fake_videos_all = []
            for method in FF_METHODS:
                fake_videos_all.extend(fake_by_type[method])

        # Mask detection — forward-compatible
        possible_mask_root = fake_root / "FaceSwap" / "c23" / "masks"
        self.has_masks = possible_mask_root.is_dir()
        if self.has_masks:
            print("[FF++] Pixel-level masks found → supervised L_exp will be used.")
        else:
            print("[FF++] No pixel-level masks → weakly-supervised L_exp (entropy+TV+diversity).")

        print(
            f"[FF++ c23] real={len(real_videos)} fake={len(fake_videos_all)} "
            f"total={len(real_videos) + len(fake_videos_all)}"
        )

        for v in real_videos:
            self.samples.append(
                {"video_path": str(v), "label": 0,
                 "mask_path": None, "manipulation": "original"}
            )
        for v in fake_videos_all:
            self.samples.append(
                {"video_path": str(v), "label": 1,
                 "mask_path": None,
                 "manipulation": v.parent.parent.parent.name}
            )

    def _build_celeb_df(self):
        """Celeb-DF v2 is deferred to future work."""
        raise NotImplementedError(
            "Celeb-DF v2 is deferred to future work. "
            "Use dataset_name='ff++' for the current pipeline."
        )

    def _build_dfdc(self):
        """
        DFDC layout relative to config.data_root:
            dfdc_train_part_*/videos/*.mp4
            dfdc_train_part_*/metadata.json
        """
        root = Path(self.config.data_root)
        for part in sorted(root.iterdir()):
            if not part.name.startswith("dfdc_train_part") or not part.is_dir():
                continue
            meta_path = part / "metadata.json"
            if not meta_path.exists():
                warnings.warn(f"[DFDC] metadata.json not found in: {part}")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            for fname, info in meta.items():
                vpath = part / "videos" / fname
                if not vpath.exists():
                    continue
                label = 1 if info.get("label") == "FAKE" else 0
                self.samples.append(
                    {"video_path": str(vpath), "label": label,
                     "mask_path": None, "manipulation": "dfdc"}
                )
        self.has_masks = False
        n_real = sum(1 for s in self.samples if s["label"] == 0)
        n_fake = sum(1 for s in self.samples if s["label"] == 1)
        print(
            f"[DFDC] {n_real} real + {n_fake} fake = {len(self.samples)} total "
            f"(ratio {n_fake/max(n_real,1):.1f}:1)"
        )

    def _build_synthetic(self):
        """Synthetic data — generated entirely in RAM, no disk I/O."""
        self.has_masks = True
        self.generator = SyntheticDataGenerator()
        n_total = 200  # 100 real + 100 fake
        for i in range(n_total):
            label = i % 2
            self.samples.append(
                {"video_path": f"synthetic_{i}", "label": label,
                 "mask_path": None, "manipulation": "synthetic"}
            )

    # ====================================================================
    # Helpers
    # ====================================================================

    @staticmethod
    def _glob_mp4(directory: str) -> list[str]:
        """Returns sorted list of .mp4 paths in a directory."""
        if not os.path.isdir(directory):
            return []
        return sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".mp4")
        )

    @staticmethod
    def _split(
        samples: list,
        mode: str,
        train_frac: float,
        val_frac: float,
    ) -> list:
        """
        Stratified train/val/test split via sklearn.model_selection.train_test_split.

        Stratification key: "{label}_{manipulation}" — finer than binary label.
        For FF++ this preserves per-type fake coverage in every split (Regime A).
        For DFDC/synthetic it is equivalent to binary-label stratification.
        A class-presence assertion fires at construction time so single-class
        splits are caught immediately.
        """
        if len(samples) == 0:
            return []

        # Composite key: "{label}_{manipulation}" gives per-type balance on FF++
        # while remaining correct for DFDC ("1_dfdc") and synthetic ("1_synthetic").
        strat_keys = [f"{s['label']}_{s['manipulation']}" for s in samples]

        # Need at least 2 samples per stratum to stratify
        from collections import Counter
        key_counts = Counter(strat_keys)
        if min(key_counts.values()) < 2:
            warnings.warn(
                f"[_split] Too few samples in a stratum "
                f"({key_counts}) — falling back to non-stratified split."
            )
            data = samples[:]
            random.Random(0).shuffle(data)
            n = len(data)
            n_train = int(n * train_frac)
            n_val   = int(n * val_frac)
            if mode == "train":
                return data[:n_train]
            elif mode == "val":
                return data[n_train: n_train + n_val]
            else:
                return data[n_train + n_val:]

        test_frac = 1.0 - train_frac - val_frac
        train_val, test = train_test_split(
            samples, test_size=test_frac, stratify=strat_keys, random_state=42
        )
        tv_keys      = [f"{s['label']}_{s['manipulation']}" for s in train_val]
        val_relative = val_frac / (train_frac + val_frac)
        train, val   = train_test_split(
            train_val, test_size=val_relative, stratify=tv_keys, random_state=42
        )

        # Class-presence assertion — catches broken splits at construction time
        for name, split in [("train", train), ("val", val), ("test", test)]:
            present = {s["label"] for s in split}
            assert present == {0, 1}, (
                f"[DeepfakeDataset] {name} split is missing a class: {present}. "
                "Stratification failed — check that both classes exist in the data."
            )
            print(f"[Split] {name}: n={len(split)} classes={sorted(present)}")

        if mode == "train":
            return train
        elif mode == "val":
            return val
        else:
            return test

    # ====================================================================
    # Core Dataset interface
    # ====================================================================

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── Synthetic: generate on-the-fly ──────────────────────────────
        if self.dataset_type == "synthetic":
            seed = int(sample["video_path"].split("_")[1])
            frames, label, mask_full = self.generator.generate_sequence(
                num_frames=self.config.num_frames,
                frame_size=(self.config.frame_size, self.config.frame_size),
                seed=seed,
            )
            h = w = self.config.frame_size // 32
            mask_np = mask_full.numpy() if hasattr(mask_full, "numpy") else mask_full
            mask_small = torch.tensor(
                cv2.resize(mask_np.astype(np.float32), (w, h)),
                dtype=torch.float32,
            )
            return {
                "frames":   frames,
                "label":    label,
                "mask":     mask_small,
                "has_mask": True,
                "meta":     {"video_path": sample["video_path"], "frame_indices": []},
            }

        # ── Real datasets: read from disk ────────────────────────────────
        frames_np = self._read_frames(sample["video_path"])

        # Face alignment (uses cache if cache_dir is set)
        video_id  = os.path.splitext(os.path.basename(sample["video_path"]))[0]
        frames_np = self.face_aligner.align_frames(frames_np, video_id)

        # Heavy augmentation for minority-class samples when ratio > 3:1
        label = sample["label"]
        if (
            self.mode == "train"
            and self.heavy_aug
            and label == self.minority_class
        ):
            aug = get_heavy_transforms(self.config.frame_size)
        else:
            aug = self.transform

        frames_tensor = torch.stack(
            [aug(Image.fromarray(f)) for f in frames_np]
        )  # (T, 3, H, W)

        # ── Mask ─────────────────────────────────────────────────────────
        h = w = self.config.frame_size // 32  # 7 for 224px at stride 32
        if (
            self.has_masks
            and sample.get("mask_path")
            and os.path.exists(sample["mask_path"])
        ):
            mask_img = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (w, h)).astype(np.float32) / 255.0
            mask     = torch.tensor(mask_img, dtype=torch.float32)
            has_mask = True
        else:
            mask     = torch.zeros(h, w, dtype=torch.float32)
            has_mask = False

        return {
            "frames":   frames_tensor,
            "label":    label,
            "mask":     mask,
            "has_mask": has_mask,
            "meta": {
                "video_path":    sample["video_path"],
                "frame_indices": [],
            },
        }

    def _read_frames(self, video_path: str) -> list[np.ndarray]:
        """
        Uniformly samples config.num_frames frames from a video file.
        Tries decord first; falls back to OpenCV.
        Returns list of (H, W, 3) uint8 RGB arrays.
        """
        T = self.config.num_frames

        if DECORD_AVAILABLE:
            try:
                vr      = VideoReader(video_path, ctx=decord_cpu(0))
                total   = len(vr)
                indices = np.linspace(0, total - 1, T, dtype=int).tolist()
                batch   = vr.get_batch(indices).asnumpy()
                return [batch[i] for i in range(T)]
            except Exception as exc:
                warnings.warn(
                    f"decord failed on '{video_path}': {exc}. Using OpenCV fallback."
                )

        # OpenCV fallback
        cap    = cv2.VideoCapture(video_path)
        total  = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        target = set(np.linspace(0, total - 1, T, dtype=int).tolist())
        frames: list[np.ndarray] = []
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in target:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fi += 1
            if len(frames) == T:
                break
        cap.release()

        if not frames:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        while len(frames) < T:
            frames.append(frames[-1].copy())
        return frames
