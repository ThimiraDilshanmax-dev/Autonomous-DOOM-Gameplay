"""Poisson rate encoder — mirrors SNAP-V hardware Coding Unit.

Pixel intensity in [0,1] -> spike probability per timestep.
Output: (T, *input_shape) binary spikes. Numpy-only.
"""
import numpy as np


def poisson_encode(x: np.ndarray, T: int = 50, seed: int | None = None) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    probs = np.broadcast_to(x, (T,) + x.shape)
    return (rng.random((T,) + x.shape) < probs).astype(np.float32)


def spike_counts(spikes: np.ndarray) -> np.ndarray:
    """Sum over time: (*input_shape,) firing counts."""
    return spikes.sum(axis=0)


def firing_rate(spikes: np.ndarray) -> np.ndarray:
    return spikes.mean(axis=0)
