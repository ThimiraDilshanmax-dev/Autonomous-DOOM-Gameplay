"""Feedforward LIF MLP — Cerebra-H compatible.

Constraints enforced (proposal Table 1):
- LIF neurons only, decay beta in {0.125, 0.25, 0.5, 0.75} (shift-only HW)
- Feedforward MLP, hidden <= 256, total neurons <= 1024
- Rate-coded input (T, B, input_dim); readout = output membrane (no spikes
  needed at head, same trick as SpikingJelly NonSpikingLIF)

Uses snnTorch if installed, else a minimal pure-torch LIF fallback so the
file imports everywhere (Colab CPU included).
"""
import torch
import torch.nn as nn

ALLOWED_BETAS = (0.125, 0.25, 0.5, 0.75)

try:
    import snntorch as snn
    _HAS_SNNTORCH = True
except ImportError:
    _HAS_SNNTORCH = False


class _FallbackLIF(nn.Module):
    """Minimal LIF: mem = beta*mem + x - thr*spike; spike = mem>thr."""

    def __init__(self, beta: float = 0.5, threshold: float = 1.0):
        super().__init__()
        assert beta in ALLOWED_BETAS, f"beta {beta} not HW-compatible {ALLOWED_BETAS}"
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        mem = self.beta * mem + x
        spike = (mem >= self.threshold).float()
        mem = mem - spike * self.threshold  # subtractive reset (HW mode)
        return spike, mem


class SnnMlp(nn.Module):
    def __init__(self, input_dim: int = 784, hidden: int = 128,
                 n_actions: int = 3, T: int = 50, beta: float = 0.5,
                 threshold: float = 1.0):
        super().__init__()
        assert hidden <= 256, "hidden must be <=256 to fit Cerebra-H budget"
        assert beta in ALLOWED_BETAS, f"beta must be one of {ALLOWED_BETAS}"
        self.T, self.hidden = T, hidden
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)
        if _HAS_SNNTORCH:
            self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="subtract")
            self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="subtract")
        else:
            self.lif1 = _FallbackLIF(beta, threshold)
            self.lif2 = _FallbackLIF(beta, threshold)
        self.threshold = threshold

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (T, B, input_dim) rate-coded spikes -> (B, n_actions) membranes."""
        B = x_seq.shape[1]
        dev = x_seq.device
        if _HAS_SNNTORCH:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem_out = torch.zeros(B, self.fc3.out_features, device=dev)
            for t in range(self.T):
                s1 = self.lif1(self.fc1(x_seq[t]), mem1)
                # snnTorch Leaky returns (spike, mem)
                spk1, mem1 = (s1, mem1) if not isinstance(s1, tuple) else s1
                s2 = self.lif2(self.fc2(spk1), mem2)
                spk2, mem2 = (s2, mem2) if not isinstance(s2, tuple) else s2
                mem_out = mem_out + self.fc3(spk2)  # integrate output current
            return mem_out / self.T
        else:
            mem1 = torch.zeros(B, self.hidden, device=dev)
            mem2 = torch.zeros(B, self.hidden, device=dev)
            mem_out = torch.zeros(B, self.fc3.out_features, device=dev)
            for t in range(self.T):
                spk1, mem1 = self.lif1(self.fc1(x_seq[t]), mem1)
                spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
                mem_out = mem_out + self.fc3(spk2)
            return mem_out / self.T

    def neuron_count(self) -> int:
        return 2 * self.hidden + self.fc3.out_features
