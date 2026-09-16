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
│   ├── defend.yaml           # Configuration for defend_the_center scenario
│   ├── corridor.yaml         # Configuration for deadly_corridor scenario
│   └── health.yaml           # Configuration for health_gathering scenario
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
│       ├── train_ann_basic.py # DQN training pipeline (checkpoints/video/TensorBoard)
│       └── convert_ann_to_snn.py # ANN->SNN conversion + threshold calibration
├── tools/
│   ├── check_env.py          # ViZDoom, PyTorch, and CUDA environment check
│   ├── smoke_test.py         # Offline smoke test (no GPU or ViZDoom needed)
│   ├── wsl_verify_basic.py   # WSL: ViZDoom random-agent loop check
│   ├── wsl_full_verify.py    # WSL: ANN->SNN->export + short DQN run
│   ├── wsl_eval_basic.py     # WSL: greedy evaluation of trained checkpoints
│   ├── wsl_logstat.py        # WSL: kill-rate stats from a training log
│   └── wsl_robustness.py     # WSL: noise/occlusion/lighting robustness
├── docs/
│   └── fpga_porting_note.md  # Cerebra-H mapping (clusters, packets, checklist)
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Installation

Windows (smoke test only — no ViZDoom, no GPU needed):

```bash
pip install numpy pillow pyyaml torch --index-url https://download.pytorch.org/whl/cpu
python tools/check_env.py
python tools/smoke_test.py
```

WSL2 Ubuntu (full pipeline — ViZDoom + training). One-time setup:

```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv python3-dev cmake build-essential libboost-all-dev libopenal-dev libsdl2-dev libjpeg-dev
python3 -m venv ~/snn-venv
~/snn-venv/bin/pip install numpy pillow pyyaml tqdm opencv-python-headless gymnasium vizdoom
~/snn-venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~/snn-venv/bin/pip install snntorch spikingjelly stable-baselines3 tensorboard
```

> GPU note (RTX 2050): the CPU torch build above is enough for all
> verification. For overnight training, either upgrade WSL torch to
> CUDA (`~/snn-venv/bin/pip install torch torchvision --index-url
> https://download.pytorch.org/whl/cu121`) or use Colab/Kaggle GPUs.
> Native Windows ViZDoom wheels are unreliable — always train in WSL2/Linux.

### 2. Environment & Smoke Tests

Verify dependencies and run offline verification without requiring ViZDoom or a GPU (Windows-safe):

```bash
python tools/check_env.py
python tools/smoke_test.py
```

Full WSL2 verification (ViZDoom loop + ANN->SNN + export + 2-episode DQN):

```bash
~/snn-venv/bin/python tools/wsl_verify_basic.py
~/snn-venv/bin/python tools/wsl_full_verify.py
```

### 3. Training (WSL2)

Train the ANN baseline with checkpoints, gameplay videos, TensorBoard logs
and a score table (`checkpoints/`, `videos/`, `runs/`, `scores.json`):

```bash
~/snn-venv/bin/python src/train/train_ann_basic.py --config configs/basic.yaml --episodes 800 --save-every 50 --out-dir training_basic1000
```

Evaluate trained checkpoints greedily (kill rate over 50 episodes):

```bash
~/snn-venv/bin/python tools/wsl_eval_basic.py
```

### 4. ANN-to-SNN conversion (WSL2)

Convert a trained ANN to a spiking network with percentile threshold
calibration and a T-sweep (agreement report + Cerebra export bundle):

```bash
~/snn-venv/bin/python src/train/convert_ann_to_snn.py --checkpoint training_basic1000/checkpoints/best.pth --config configs/basic.yaml --out-dir conversion_basic --timesteps 50 25
```

Surrogate-gradient fine-tuning (distillation from the ANN, BPTT through
LIF neurons) to close the remaining ANN/SNN gap, refreshing the export
bundle:

```bash
~/snn-venv/bin/python src/train/finetune_snn.py --checkpoint training_basic4000/checkpoints/final.pth --config configs/basic.yaml --out-dir finetune_basic --epochs 8
```

8-bit quantization simulation (fidelity vs float SNN, <3% exit bar):

```bash
~/snn-venv/bin/python src/export/quantize_sim.py --snn-weights finetune_basic/snn_finetuned.pth --ann-checkpoint training_basic4000/checkpoints/final.pth --config configs/basic.yaml --out-dir quant_basic --t 50
```

Robustness checks (noise / occlusion / lighting shifts, ANN stability +
SNN agreement):

```bash
~/snn-venv/bin/python tools/wsl_robustness.py --config configs/basic.yaml --ann-checkpoint training_basic4000/checkpoints/final.pth --snn-weights finetune_basic/snn_finetuned.pth --out-dir robust_basic --n 256 --t 25
```
