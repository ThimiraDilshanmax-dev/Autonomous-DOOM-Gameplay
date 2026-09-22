"""Frame preprocessing: grayscale -> resize -> normalize -> stack.

Numpy/PIL only so tools/smoke_test.py runs without vizdoom/torch/cv2.
Matches SNAP-V MainCore role: CPU handles pixels, SNN handles decisions.
"""
import numpy as np
from PIL import Image


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """HWC RGB uint8 -> HW float32 in [0,1]. Accepts HW grayscale too."""
    if frame.ndim == 2:
        return frame.astype(np.float32) / 255.0
    # ITU-R BT.601 luma
    gray = (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2])
    return gray.astype(np.float32) / 255.0


def resize(gray: np.ndarray, size: int = 28) -> np.ndarray:
    img = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def preprocess(frame: np.ndarray, size: int = 28) -> np.ndarray:
    return resize(to_grayscale(frame), size)


class FrameStack:
    """Stack of k recent encoded frames, shape (k, size, size)."""

    def __init__(self, k: int = 4, size: int = 28):
        self.k = k
        self.size = size
        self.frames: list[np.ndarray] = []

    def reset(self, frame: np.ndarray) -> np.ndarray:
        enc = preprocess(frame, self.size)
        self.frames = [enc] * self.k
        return self.get()

    def step(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(preprocess(frame, self.size))
        self.frames = self.frames[-self.k:]
        return self.get()

    def get(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)


def obs_dim(size: int, stack_frames: int, obs_mode: str = "mean") -> int:
    """Flat MLP input dim for an obs_mode.

    - "mean": average the k frames -> size*size (legacy, basic/defend compat).
    - "stack": preserve all k frames -> k*size*size (required for navigation
      like health_gathering where mean destroys motion cues).
    """
    if obs_mode == "stack":
        return stack_frames * size * size
    return size * size


def stack_to_obs(stack: np.ndarray, obs_mode: str = "mean") -> np.ndarray:
    """Convert (k, size, size) stack to flat vector for MLP/SNN.

    Keeps a single entry point so train/convert/eval can't drift apart.
    """
    if obs_mode == "stack":
        return stack.reshape(-1).astype(np.float32)
    return stack.mean(axis=0).reshape(-1).astype(np.float32)


def stack_to_spike_shape(obs: np.ndarray, size: int, stack_frames: int,
                         obs_mode: str = "mean") -> np.ndarray:
    """Reshape flat obs back to spatial form for Poisson encoding diagnostics.

    Returns (..., size, size) for mean or (..., stack_frames, size, size)
    for stack. Pure reshape helper, no copy semantics change.
    """
    if obs_mode == "stack":
        return obs.reshape(-1, stack_frames, size, size)
    return obs.reshape(-1, size, size)
