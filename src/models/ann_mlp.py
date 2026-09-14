"""ANN baseline MLP: flattened frame-stack -> hidden -> Q-values/logits.

Small on purpose: input 4*28*28=3136 (or single 28*28=784) -> H=128 -> A.
Total params ~400k — fits easily; SNN twin keeps same sizes (proposal Table 1).
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
