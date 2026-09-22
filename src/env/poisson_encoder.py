"""Poisson rate encoder — mirrors SNAP-V hardware Coding Unit.

Pixel intensity in [0,1] -> spike probability per timestep.
Output: (T, *input_shape) binary spikes. Numpy-only.
"""
import numpy as np


def poisson_encode(x: np.ndarray, T: int = 50, seed: int | None = None,
                     gain: float = 1.0) -> np.ndarray:
    """Rate-code flat or spatial obs to (T, *shape) spikes.

    gain>1 compensates dark/low-contrast frames that otherwise silence the
    SNN (robustness dark split dropped to ~59% agreement). Prefer gain in
    the encoder over retraining when lighting shifts at deploy time.
    """
    x = np.clip(x.astype(np.float32) * gain, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    probs = np.broadcast_to(x, (T,) + x.shape)
    return (rng.random((T,) + x.shape) < probs).astype(np.float32)


def auto_gain(x: np.ndarray, target_mean: float = 0.25) -> float:
    """Suggest a gain that brings mean intensity toward target_mean.

    Clamped to [1.0, 2.0] so bright frames are untouched and dark frames
    get at most 2x boost. Numpy-only, safe to call on host CPU per batch.
    """
    m = float(np.clip(x.astype(np.float32), 0.0, 1.0).mean())
    if m <= 1e-6:
        return 1.0
    return float(min(2.0, max(1.0, target_mean / m)))


def spike_counts(spikes: np.ndarray) -> np.ndarray:
    """Sum over time: (*input_shape,) firing counts."""
    return spikes.sum(axis=0)


def firing_rate(spikes: np.ndarray) -> np.ndarray:
    return spikes.mean(axis=0)
