"""ANN baseline MLP: flattened frames -> hidden -> Q-values/logits.

Two geometries (see preprocessing.obs_dim):
- mean: 28*28=784 (legacy basic/defend checkpoints)
- stack: 4*28*28=3136 (navigation/health — preserves motion cues)
Hidden 128 keeps total neurons 2*H+A <= 1024 and synapses < ~524k for
Cerebra-H (stack-128 ≈ 418k synapses, mean-128 ≈ 117k).
"""
import torch
import torch.nn as nn


class AnnMlp(nn.Module):
    def __init__(self, input_dim: int = 784, hidden: int = 128, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))
