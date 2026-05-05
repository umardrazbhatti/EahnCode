"""
data/transforms.py — augmentation and normalisation pipelines.
Augmentations operate on [0,1] float tensors (C,H,W).
"""

import torchvision.transforms as T


def get_augmentation_transforms() -> T.Compose:
    """Random augmentations applied per-frame during training."""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
    ])


def get_normalization_transform() -> T.Normalize:
    """ImageNet normalisation for [0,1] tensors."""
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
