# Autonomous DOOM Gameplay

Autonomous reinforcement learning agents playing ViZDoom using Spiking Neural Networks (SNN) and Artificial Neural Networks (ANN).

## Overview

This repository implements reinforcement learning pipelines for ViZDoom scenarios (e.g. `basic`, `defend_the_center`, `deadly_corridor`) with neuromorphic computing in mind:
- **Frame Preprocessing & Stacking**: Grayscale conversion, downsampling (28x28 / 16x16), and multi-frame stacking.
- **Poisson Spike Encoding**: Rate-based Poisson encoding converting visual frames into spike trains over temporal windows ($T$).
- **Architectures**:
  - **ANN MLP**: Baseline deep Q-network policy.
  - **SNN MLP**: Multi-layer Leaky Integrate-and-Fire (LIF) network (via `snnTorch` / SpikingJelly with local fallback support).
- **Hardware Deployment**: Export pipeline targeting Cerebra neuromorphic architecture.

## Repository Structure

```
├── configs/
│   ├── basic.yaml            # Configuration for basic ViZDoom scenario
│   └── defend.yaml           # Configuration for defend_the_center scenario
├── src/
│   ├── env/
│   │   ├── poisson_encoder.py # Rate-based Poisson spike train generator
│   │   ├── preprocessing.py   # Frame resizing, normalization, and stacking
│   │   └── vizdoom_wrapper.py # Gym-compatible ViZDoom environment wrapper
│   ├── eval/
│   │   └── evaluate.py        # Evaluation loops and metrics
│   ├── export/
│   │   └── cerebra_export.py  # Weights and topology export for Cerebra
│   ├── models/
│   │   ├── ann_mlp.py         # Standard ANN DQN model
│   │   └── snn_mlp.py         # Spiking LIF neural network model
│   └── train/
│       └── train_ann_basic.py # DQN training pipeline
├── tools/
│   ├── check_env.py          # ViZDoom, PyTorch, and CUDA environment check
│   └── smoke_test.py         # Offline smoke test (no GPU or ViZDoom needed)
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note for ViZDoom**: Native Windows wheels for ViZDoom can be finicky. For training with live ViZDoom instances, running inside WSL2 or Linux is recommended.

### 2. Environment & Smoke Tests

Verify dependencies and run offline verification without requiring ViZDoom or a GPU:
```bash
python tools/check_env.py
python tools/smoke_test.py
```

### 3. Training

Run the baseline training pipeline:
```bash
python src/train/train_ann_basic.py
```
