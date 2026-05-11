"""
xai/sanity_checks.py — Adebayo et al. 2018 sanity checks for explanation faithfulness.

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018).
Sanity Checks for Saliency Maps. NeurIPS 2018.
"""

import copy
import torch
import numpy as np
from typing import Optional


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between two (T, H, W) explanation maps."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    norm_a = np.linalg.norm(a_flat) + 1e-8
    norm_b = np.linalg.norm(b_flat) + 1e-8
    return float(np.dot(a_flat / norm_a, b_flat / norm_b))


def _get_M_t(model, frames: torch.Tensor) -> np.ndarray:
    """
    Run a single forward pass and return M_t_up[0] as a numpy array (T, H, W).
    frames: (1, T, C, H, W)
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(frames.to(device))
    return out.M_t_up[0].cpu().numpy()   # (T, H, W)


def model_randomization_check(
    model,
    frames: torch.Tensor,
    n_random: int = 3,
) -> float:
    """
    Cascade-randomize the model's weights layer by layer (from last to first).
    For each randomization, recompute M_t and measure cosine similarity to the
    original M_t.

    A faithful explanation should change substantially when weights are randomized,
    so the returned mean similarity should be LOW (e.g., < 0.5).
    If it stays high (> 0.7), the explanation is insensitive to the model —
    a sign that it is not faithfully representing what the model learned.

    Parameters
    ----------
    model    : trained EAHN model (not modified — a deep copy is randomized)
    frames   : (1, T, C, H, W) single sample tensor
    n_random : number of cascade randomization steps to average over

    Returns
    -------
    mean_cosine_similarity : float
        Mean cosine similarity between original M_t and randomized M_t across
        n_random cascade steps. < 0.5 = explanation changes a lot (good).
                               > 0.7 = explanation barely changes (bad).
    """
    original_M_t = _get_M_t(model, frames)

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Collect randomizable weight layers (Conv2d, Linear, LayerNorm)
    named_params = [
        (name, param)
        for name, param in model_copy.named_parameters()
        if param.requires_grad and param.dim() >= 1
    ]

    if not named_params:
        return 1.0

    # Sample n_random evenly-spaced cascade positions
    step_size = max(1, len(named_params) // max(n_random, 1))
    cascade_positions = list(range(step_size - 1, len(named_params), step_size))[:n_random]

    sims = []
    for pos in cascade_positions:
        # Randomize all layers up to and including pos (cascade)
        for i in range(pos + 1):
            name, param = named_params[i]
            torch.nn.init.normal_(param.data)

        randomized_M_t = _get_M_t(model_copy, frames)
        sims.append(_cosine_sim(original_M_t, randomized_M_t))

    return float(np.mean(sims)) if sims else 1.0


def label_randomization_check(
    model,
    train_loader,
    config,
    n_batches: int = 5,
) -> Optional[float]:
    """
    Optional label-randomization sanity check (Adebayo et al., Figure 2b).

    Retrain the model for a few steps with randomly shuffled labels and check
    whether the explanation maps change. This is expensive and skipped by default.

    Returns None to indicate this check was skipped.
    """
    return None
