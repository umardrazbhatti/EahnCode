"""
data/datasets.py — DeepfakeDataset supporting FF++, Celeb-DF, DFDC and Synthetic modes.

Key fixes vs original:
  - Mask resize target is (7,7) which matches the feature map h×w for EfficientNet-B4/B0,
    averaged over time → (7,7) ground-truth mask used in L_exp.
  - Augmentation is applied before normalisation; applied per-frame on (C,H,W) tensor.
  - Mask shape returned is (7,7) always — consistent with M_t grid size in EAHN.
  - has_mask is returned as a bool scalar tensor.
  - Synthetic mode returns the mask resized to match the spatial grid.
"""

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os, glob, json
from typing import Literal, List, Dict, Any
import cv2

from config import EAHNConfig
from data.face_align import FaceAligner
from data.transforms import get_augmentation_transforms, get_normalization_transform
from data.synthetic_generator import SyntheticDataGenerator

# Feature map resolution for EfficientNet-B4/B0 on 224×224 input → 7×7
_MASK_GRID = 7


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        config: EAHNConfig,
        mode: Literal["train", "val", "test"],
        dataset_type: Literal["synthetic", "ff++", "celeb_df", "dfdc"],
    ):
        self.config = config
        self.mode = mode
        self.dataset_type = dataset_type
        self.face_aligner = FaceAligner(
            margin=0.3, cache_dir=config.cache_dir, device=config.device
        )
        self.aug_transform  = get_augmentation_transforms() if mode == "train" else None
        self.norm           = get_normalization_transform()
        self.has_masks      = False
        self.samples: List[Dict[str, Any]] = []

        if dataset_type == "synthetic":
            self._init_synthetic()
        elif dataset_type == "ff++":
            self._init_ffpp()
        elif dataset_type == "celeb_df":
            self._init_celebdf()
        elif dataset_type == "dfdc":
            self._init_dfdc()

        if dataset_type != "synthetic":
            self._apply_split()

    # ── dataset initialisers ───────────────────────────────────────────────────

    def _init_synthetic(self):
        gen = SyntheticDataGenerator(seed=42)
        all_samples = []
        for i in range(200):
            frames, label, mask = gen.generate_sequence(
                num_frames=self.config.num_frames,
                frame_size=(self.config.frame_size, self.config.frame_size),
            )
            all_samples.append({
                "frames": frames, "label": label,
                "mask": mask, "has_mask": True, "meta": {"id": i},
            })
        splits = {"train": all_samples[:160], "val": all_samples[160:180],
                  "test": all_samples[180:]}
        self.samples = splits[self.mode]
        self.has_masks = True

    def _init_ffpp(self):
        root = self.config.data_root
        manip_root = os.path.join(root, "manipulated_sequences")
        methods = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
        if os.path.isdir(manip_root):
            for method in methods:
                vdir = os.path.join(manip_root, method, "c23", "videos")
                if not os.path.isdir(vdir):
                    continue
                for vpath in sorted(glob.glob(os.path.join(vdir, "*.mp4"))):
                    mask_dir = os.path.join(
                        root, "masks", method,
                        os.path.splitext(os.path.basename(vpath))[0]
                    )
                    self.samples.append({
                        "video_path": vpath,
                        "label": 1,
                        "mask_dir": mask_dir if os.path.isdir(mask_dir) else None,
                    })
        real_dir = os.path.join(root, "original_sequences", "youtube", "c23", "videos")
        if os.path.isdir(real_dir):
            for vpath in sorted(glob.glob(os.path.join(real_dir, "*.mp4"))):
                self.samples.append({"video_path": vpath, "label": 0, "mask_dir": None})
        self.has_masks = any(s.get("mask_dir") for s in self.samples)

    def _init_celebdf(self):
        root = os.path.join(self.config.data_root, "celeb_df")
        for vpath in sorted(glob.glob(os.path.join(root, "videos", "real", "*.mp4"))):
            self.samples.append({"video_path": vpath, "label": 0, "mask_dir": None})
        for vpath in sorted(glob.glob(os.path.join(root, "videos", "synthesis", "*.mp4"))):
            self.samples.append({"video_path": vpath, "label": 1, "mask_dir": None})
        self.has_masks = False

    def _init_dfdc(self):
        root = os.path.join(self.config.data_root, "dfdc")
        meta_path = os.path.join(root, "metadata.json")
        if os.path.exists(meta_path):
            metadata = json.load(open(meta_path))
            for video_name, info in metadata.items():
                self.samples.append({
                    "video_path": os.path.join(root, "videos", video_name),
                    "label": 1 if info["label"] == "FAKE" else 0,
                    "mask_dir": None,
                })
        self.has_masks = False

    def _apply_split(self):
        n = len(self.samples)
        train_end = int(self.config.train_split * n)
        val_end   = int((self.config.train_split + self.config.val_split) * n)
        if self.mode == "train":
            self.samples = self.samples[:train_end]
        elif self.mode == "val":
            self.samples = self.samples[train_end:val_end]
        else:
            self.samples = self.samples[val_end:]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        if self.dataset_type == "synthetic":
            return self._getitem_synthetic(sample)
        return self._getitem_video(sample)

    # ── synthetic item ────────────────────────────────────────────────────────

    def _getitem_synthetic(self, sample: dict) -> dict:
        frames = sample["frames"]          # (T, 3, H, W) float [0,1], unnormalised
        label  = sample["label"]
        mask   = sample["mask"]            # (H, W) float

        # Augment each frame
        if self.aug_transform is not None:
            frames = torch.stack([self.aug_transform(frames[t]) for t in range(frames.shape[0])])

        frames = torch.stack([self.norm(frames[t]) for t in range(frames.shape[0])])

        # Resize mask to spatial grid used by EAHN
        mask_grid = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(_MASK_GRID, _MASK_GRID), mode="area"
        ).squeeze()  # (_MASK_GRID, _MASK_GRID)

        return {
            "frames":   frames,
            "label":    torch.tensor(label, dtype=torch.float),
            "mask":     mask_grid,
            "has_mask": torch.tensor(True, dtype=torch.bool),
            "meta":     sample.get("meta", {}),
        }

    # ── video item ────────────────────────────────────────────────────────────

    def _getitem_video(self, sample: dict) -> dict:
        video_path = sample["video_path"]
        label      = sample["label"]
        T          = self.config.num_frames
        H = W      = self.config.frame_size

        # ── Read frames ───────────────────────────────────────────────────────
        raw_frames, indices = self._read_video(video_path, T)

        # ── Face alignment ────────────────────────────────────────────────────
        aligned = self.face_aligner.align_frames(
            raw_frames, video_id=os.path.basename(video_path), output_size=H
        )

        # ── To tensor [0,1] → augment → normalise ────────────────────────────
        frames = torch.from_numpy(
            np.stack(aligned).astype(np.float32)
        ).permute(0, 3, 1, 2) / 255.0     # (T, 3, H, W)

        if self.aug_transform is not None:
            frames = torch.stack([self.aug_transform(frames[t]) for t in range(T)])
        frames = torch.stack([self.norm(frames[t]) for t in range(T)])

        # ── Load mask ─────────────────────────────────────────────────────────
        has_mask  = False
        mask_grid = torch.zeros(_MASK_GRID, _MASK_GRID, dtype=torch.float)

        if sample.get("mask_dir"):
            mask_grid, has_mask = self._load_mask(
                sample["mask_dir"], indices, H, _MASK_GRID
            )

        return {
            "frames":   frames,
            "label":    torch.tensor(label, dtype=torch.float),
            "mask":     mask_grid,
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "meta":     {"video_path": video_path},
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _read_video(video_path: str, num_frames: int):
        """Read T frames uniformly sampled from the video. Returns list of np arrays."""
        try:
            import decord
            vr = decord.VideoReader(video_path)
            total = len(vr)
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = [vr[int(i)].asnumpy() for i in indices]
        except Exception:
            cap = cv2.VideoCapture(video_path)
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            idx_set = set(indices.tolist())
            frames = []
            buf = {}
            fi = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fi in idx_set:
                    buf[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fi += 1
            cap.release()
            frames = [buf.get(i, np.zeros((224, 224, 3), np.uint8)) for i in indices]

        return frames, indices

    @staticmethod
    def _load_mask(mask_dir: str, frame_indices, frame_size: int,
                   grid_size: int):
        """Load per-frame masks, average over frames, return (grid_size,grid_size) tensor."""
        mask_frames = []
        for fi in frame_indices:
            mpath = os.path.join(mask_dir, f"{fi:04d}.png")
            if os.path.exists(mpath):
                mimg = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if mimg is None:
                    mimg = np.zeros((frame_size, frame_size), np.uint8)
                mimg = cv2.resize(mimg, (frame_size, frame_size))
            else:
                mimg = np.zeros((frame_size, frame_size), np.uint8)
            mask_frames.append(mimg)

        mask_t = torch.from_numpy(
            np.stack(mask_frames).astype(np.float32)
        ) / 255.0                                           # (T, H, W)
        mask_grid = F.interpolate(
            mask_t.unsqueeze(1), size=(grid_size, grid_size), mode="area"
        ).squeeze(1).mean(0)                                # (grid_size, grid_size)

        has_mask = mask_grid.sum() > 0
        return mask_grid, bool(has_mask)
