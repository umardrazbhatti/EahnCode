"""
data/datasets.py  —  EAHN DeepfakeDataset
==========================================
Corrected path conventions for the actual Kaggle FF++ c23 layout:

  Fake: {data_root}/manipulated_sequences/{Method}/c23/videos/*.mp4   (label=1)
  Real: {data_root}/original_sequences/youtube/c23/videos/*.mp4       (label=0)

No pixel-level manipulation masks exist in this dataset snapshot,
so has_masks=False and weakly-supervised L_exp is applied throughout.

Class balance: each class is capped at exactly 1000 samples (random.sample with
a fixed seed=42) giving at most 2000 balanced samples before splitting.
self.n_real and self.n_fake are set after the split so they reflect the count
in that particular mode (train / val / test).
"""

import os
import json
import random
import warnings
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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
from data.transforms import get_transforms
from data.synthetic_generator import SyntheticDataGenerator

# ---------------------------------------------------------------------------
# FF++ manipulation methods present in this dataset snapshot
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
    Unified dataset loader for FF++, Celeb-DF, DFDC and synthetic data.

    Each __getitem__ returns a dict with keys:
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
        self.config = config
        self.mode = mode
        self.dataset_type = dataset_type
        self.transform = get_transforms(mode, config.frame_size)
        self.has_masks: bool = False
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

        # ── Train / val / test split ────────────────────────────────────
        self.samples = self._split(
            self.samples, mode, config.train_split, config.val_split
        )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[DeepfakeDataset] No samples found for "
                f"dataset='{dataset_type}', mode='{mode}'. "
                f"Check config.data_root='{config.data_root}'."
            )

        # ── Store and log class distribution (post-split counts) ────────
        self.n_real = sum(1 for s in self.samples if s["label"] == 0)
        self.n_fake = sum(1 for s in self.samples if s["label"] == 1)
        print(
            f"[DeepfakeDataset] Balanced to {self.n_real} real + "
            f"{self.n_fake} fake = {len(self.samples)} total"
        )
        print(
            f"[DeepfakeDataset | {dataset_type} / {mode}] "
            f"total={len(self.samples)}  real={self.n_real}  fake={self.n_fake}"
        )

    # ====================================================================
    # Dataset builders
    # ====================================================================

    def _build_ffpp(self):
        """
        Scans the actual FF++ c23 directory layout.

        Relative to config.data_root (= .../ffpp_data/):
            Fake: manipulated_sequences/{Method}/c23/videos/*.mp4
            Real: original_sequences/youtube/c23/videos/*.mp4
        """
        root = self.config.data_root

        # ── Real videos (label = 0) ──────────────────────────────────────
        real_dir = os.path.join(
            root, "original_sequences", "youtube", "c23", "videos"
        )
        real_paths = self._glob_mp4(real_dir)
        if not real_paths:
            warnings.warn(
                f"[FF++] No real videos found at: {real_dir}\n"
                "Check that config.data_root points to the ffpp_data/ folder."
            )
        for p in real_paths:
            self.samples.append(
                {"video_path": p, "label": 0, "mask_path": None}
            )

        # ── Fake videos (label = 1) ──────────────────────────────────────
        for method in FF_METHODS:
            fake_dir = os.path.join(
                root, "manipulated_sequences", method, "c23", "videos"
            )
            fake_paths = self._glob_mp4(fake_dir)
            if not fake_paths:
                warnings.warn(
                    f"[FF++] No fake videos found for method '{method}' at: {fake_dir}"
                )
                continue
            for p in fake_paths:
                self.samples.append(
                    {"video_path": p, "label": 1, "mask_path": None}
                )

        # ── Mask detection ───────────────────────────────────────────────
        # This dataset snapshot contains no mask directories.
        # Check anyway so the code is forward-compatible if masks are added.
        possible_mask_root = os.path.join(
            root, "manipulated_sequences", "FaceSwap", "c23", "masks"
        )
        self.has_masks = os.path.isdir(possible_mask_root)
        if self.has_masks:
            print("[FF++] Pixel-level masks found → supervised L_exp will be used.")
        else:
            print("[FF++] No pixel-level masks found → weakly-supervised L_exp (entropy + TV).")

        # ── Balance classes ──────────────────────────────────────────────
        # 5 methods × ~1000 fakes = ~5000 fakes vs ~1000 reals → 5:1 ratio.
        # Cap fakes at 2× the number of reals before splitting.
        self._balance_classes()

    def _build_celeb_df(self):
        """
        Expected layout relative to config.data_root:
            videos/real/*.mp4
            videos/synthesis/*.mp4
        No pixel-level masks available.
        """
        root = self.config.data_root
        real_dir = os.path.join(root, "videos", "real")
        fake_dir = os.path.join(root, "videos", "synthesis")
        for p in self._glob_mp4(real_dir):
            self.samples.append({"video_path": p, "label": 0, "mask_path": None})
        for p in self._glob_mp4(fake_dir):
            self.samples.append({"video_path": p, "label": 1, "mask_path": None})
        self.has_masks = False
        self._balance_classes()

    def _build_dfdc(self):
        """
        Expected layout relative to config.data_root:
            dfdc_train_part_*/videos/*.mp4
            dfdc_train_part_*/metadata.json
        """
        root = self.config.data_root
        for part in sorted(os.listdir(root)):
            if not part.startswith("dfdc_train_part"):
                continue
            part_dir = os.path.join(root, part)
            if not os.path.isdir(part_dir):
                continue
            meta_path = os.path.join(part_dir, "metadata.json")
            if not os.path.exists(meta_path):
                warnings.warn(f"[DFDC] metadata.json not found in: {part_dir}")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            for fname, info in meta.items():
                vpath = os.path.join(part_dir, "videos", fname)
                if not os.path.exists(vpath):
                    continue
                label = 1 if info.get("label") == "FAKE" else 0
                self.samples.append(
                    {"video_path": vpath, "label": label, "mask_path": None}
                )
        self.has_masks = False
        self._balance_classes()

    def _build_synthetic(self):
        """
        Synthetic data: generated entirely in RAM, no disk I/O.
        Always produces masks, enabling supervised L_exp testing.
        """
        self.has_masks = True
        self.generator = SyntheticDataGenerator()
        n_total = 200  # 100 real + 100 fake
        for i in range(n_total):
            label = i % 2
            self.samples.append(
                {"video_path": f"synthetic_{i}", "label": label, "mask_path": None}
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

    def _balance_classes(self):
        """
        Cap each class to exactly 1000 samples using a fixed seed for
        reproducibility.  After balancing the dataset contains at most
        2000 samples (1000 real + 1000 fake) before the train/val/test split.
        """
        real = [s for s in self.samples if s["label"] == 0]
        fake = [s for s in self.samples if s["label"] == 1]

        if not real or not fake:
            warnings.warn(
                "[DeepfakeDataset] One class is missing entirely before balancing. "
                f"real={len(real)}, fake={len(fake)}. "
                "Check that video files exist at the expected paths."
            )
            return

        rng  = random.Random(42)
        real = rng.sample(real, min(len(real), 1000))
        fake = rng.sample(fake, min(len(fake), 1000))

        combined = real + fake
        random.Random(42).shuffle(combined)
        self.samples = combined

    @staticmethod
    def _split(
        samples: list,
        mode: str,
        train_frac: float,
        val_frac: float,
    ) -> list:
        """
        Deterministic 80/10/10 split (or as configured).
        Shuffled with seed=0 before splitting to ensure reproducibility.
        """
        data = samples[:]
        random.Random(0).shuffle(data)
        n = len(data)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        if mode == "train":
            return data[:n_train]
        elif mode == "val":
            return data[n_train: n_train + n_val]
        else:  # test
            return data[n_train + n_val:]

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
            # Downsample mask to backbone stride (7×7 for 224px input at stride 32)
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
        video_id = os.path.splitext(os.path.basename(sample["video_path"]))[0]
        frames_np = self.face_aligner.align_frames(frames_np, video_id)

        # Convert to tensor via transform
        frames_tensor = torch.stack(
            [self.transform(Image.fromarray(f)) for f in frames_np]
        )  # shape: (T, 3, H, W)

        # ── Mask ─────────────────────────────────────────────────────────
        h = w = self.config.frame_size // 32  # 7 for 224px at stride 32
        if (
            self.has_masks
            and sample["mask_path"]
            and os.path.exists(sample["mask_path"])
        ):
            mask_img = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (w, h)).astype(np.float32) / 255.0
            mask = torch.tensor(mask_img, dtype=torch.float32)
            has_mask = True
        else:
            mask = torch.zeros(h, w, dtype=torch.float32)
            has_mask = False

        return {
            "frames":   frames_tensor,
            "label":    sample["label"],
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
        Tries decord first (faster); falls back to OpenCV.
        Returns a list of (H, W, 3) uint8 RGB arrays.
        """
        T = self.config.num_frames

        if DECORD_AVAILABLE:
            try:
                vr = VideoReader(video_path, ctx=decord_cpu(0))
                total = len(vr)
                indices = np.linspace(0, total - 1, T, dtype=int).tolist()
                batch = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) RGB
                return [batch[i] for i in range(T)]
            except Exception as exc:
                warnings.warn(
                    f"decord failed on '{video_path}': {exc}. Using OpenCV fallback."
                )

        # ── OpenCV fallback ──────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        target_indices = set(np.linspace(0, total - 1, T, dtype=int).tolist())
        frames: list[np.ndarray] = []
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in target_indices:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fi += 1
            if len(frames) == T:
                break
        cap.release()

        # Pad with the last frame if the video was shorter than expected
        if not frames:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        while len(frames) < T:
            frames.append(frames[-1].copy())

        return frames
