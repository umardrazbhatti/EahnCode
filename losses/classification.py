import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.bce(logit, label)


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced binary classification.

    Use when WeightedRandomSampler is turned off (e.g., Celeb-DF where
    sampler may cause overfitting on the 890 real samples).
    With WeightedRandomSampler active, default to BCE.

    alpha : down-weights the easy-majority-class loss
    gamma : focusing parameter — higher = more focus on hard examples
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # prob: sigmoid output in [0,1], shape (B,)
        prob = torch.sigmoid(logit)
        bce  = F.binary_cross_entropy(prob, target.float(), reduction='none')
        pt   = torch.where(target.bool(), prob, 1 - prob)
        focal = self.alpha * (1 - pt).pow(self.gamma) * bce
        return focal.mean()


def build_classification_loss(config) -> nn.Module:
    """Factory: returns FocalLoss or ClassificationLoss based on config.cls_loss_type."""
    if getattr(config, "cls_loss_type", "bce") == "focal":
        return FocalLoss(
            alpha=getattr(config, "focal_alpha", 0.25),
            gamma=getattr(config, "focal_gamma", 2.0),
        )
    return ClassificationLoss()
