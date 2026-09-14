"""Offline smoke test — no vizdoom, no GPU needed.

Exercises: preprocessing (numpy/PIL) -> Poisson encoder -> ANN forward
(if torch present) -> SNN forward (fallback LIF) -> Cerebra export check.
Run:  python tools/smoke_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.env.preprocessing import preprocess, FrameStack
from src.env.poisson_encoder import poisson_encode


def main():
    # 1. preprocessing on a fake RGB frame
    fake = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
    enc = preprocess(fake, size=28)
    assert enc.shape == (28, 28) and enc.min() >= 0 and enc.max() <= 1
    stk = FrameStack(k=4, size=28)
    s = stk.reset(fake)
    assert s.shape == (4, 28, 28)
    print(f"[1/4] preprocessing OK {s.shape}")

    # 2. Poisson encoder rate ~ intensity
    x = np.full((28, 28), 0.8, dtype=np.float32)
    spikes = poisson_encode(x, T=50, seed=0)
    rate = spikes.mean()
    assert spikes.shape == (50, 28, 28) and 0.6 < rate < 0.95, rate
    print(f"[2/4] poisson OK mean_rate={rate:.3f}")

    # 3-4. torch models (skip gracefully if torch missing)
    try:
        import torch
    except ImportError:
        print("[3/4] torch MISSING — skipping ANN/SNN forward (install per requirements.txt)")
        return
    from src.models.ann_mlp import AnnMlp
    from src.models.snn_mlp import SnnMlp
    ann = AnnMlp(784, 128, 3)
    q = ann(torch.randn(2, 784))
    assert q.shape == (2, 3)
    print(f"[3/4] ann OK {tuple(q.shape)}")

    snn = SnnMlp(784, 32, 3, T=4, beta=0.5)  # tiny T for speed
    rate_in = torch.rand(4, 2, 784)
    out = snn(rate_in)
    assert out.shape == (2, 3) and torch.isfinite(out).all()
    assert snn.neuron_count() <= 1024
    print(f"[4/4] snn OK {tuple(out.shape)} neurons={snn.neuron_count()}")

    from src.export.cerebra_export import export_snn
    import tempfile
    cfg = export_snn(snn, out_dir=tempfile.mkdtemp(), beta=0.5, T=4)
    print(f"export OK total_neurons={cfg['total_neurons']}")
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
