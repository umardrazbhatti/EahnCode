import torch.nn as nn
import torch


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.bce(logit, label)
