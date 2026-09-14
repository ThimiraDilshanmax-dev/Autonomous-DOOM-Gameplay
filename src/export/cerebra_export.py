"""Cerebra-H export bundle (Week 6): weights + thresholds + cluster map.

Fits 32 neurons/cluster packing; keeps dense layers inside one cluster
group so traffic stays on the L1 router (proposal Sec. 4).
"""
from __future__ import annotations
import json
import numpy as np

CLUSTER_SIZE = 32


def cluster_map(n_neurons: int, cluster_size: int = CLUSTER_SIZE) -> list[int]:
    return [i // cluster_size for i in range(n_neurons)]


def export_snn(model, out_dir: str = "artifacts", beta: float = 0.5,
               threshold: float = 1.0, T: int = 50) -> dict:
    import os
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
             if "lif" not in k.lower()}
    np.savez(os.path.join(out_dir, "weights.npz"), **state)
    n_hidden = getattr(model, "hidden", 128)
    n_out = model.fc3.out_features if hasattr(model, "fc3") else 3
    assert 2 * n_hidden + n_out <= 1024, "exceeds Cerebra-H 1024-neuron budget"
    cfg = {
        "beta": beta, "threshold": threshold, "T": T,
        "reset_mode": "subtractive",
        "hidden": n_hidden, "n_actions": n_out,
        "total_neurons": 2 * n_hidden + n_out,
        "cluster_map_hidden1": cluster_map(n_hidden),
        "cluster_map_hidden2": cluster_map(n_hidden),
    }
    with open(os.path.join(out_dir, "cerebra_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg
