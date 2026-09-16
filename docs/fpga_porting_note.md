# FPGA Porting Note — Basic DOOM SNN → Cerebra-H

Target: PeraMorphIQ Cerebra-H neuromorphic accelerator on the SNAP-V SoC
(32 clusters × 32 neurons = 1,024 neurons; ~524k synapses; LIF neurons with
shift-only decay ∈ {0.125, 0.25, 0.5, 0.75}; 11-bit spike packets; L1/L2
routers; inference-only on chip, offline training on host).

Source artifacts (this repo):
- `finetune_basic/snn_finetuned.pth` — fine-tuned float weights
- `finetune_basic/cerebra_bundle/weights.npz` — exported arrays
- `finetune_basic/cerebra_bundle/cerebra_config.json` — thresholds, decay, T
- `quant_basic/quant_report.json` — int8 fidelity (2.93% deviation, PASS)
- `conversion_basic/conversion_report.json` — ANN/SNN agreement 82.4% @ T=50

## 1. Network inventory (basic scenario)

| Layer | Shape (out × in) | Synapses | On-chip neurons |
|---|---|---|---|
| fc1 (input → hidden1) | 128 × 784 | 100,352 | 128 (hidden1) |
| fc2 (hidden1 → hidden2) | 128 × 128 | 16,384 | 128 (hidden2) |
| fc3 (hidden2 → output) | 3 × 128 | 384 | 3 (readout) |
| **Total** | | **117,120** | **259** |

- Input (784) is **not** on chip: the 28×28 frame is Poisson-encoded on the
  host CPU (MainCore + Coding Hardware role), exactly as in SNAP-V.
- Neuron budget: 259 / 1,024 (25%). Synapse budget: 117,120 / ~524k (22%).
- LIF config: `beta = 0.5` (shift-only compatible), subtractive reset
  (hardware mode), thresholds per layer in `cerebra_config.json`
  (hidden1 ≈ 0.327 after ×0.7 scale — see conversion report for the run).

## 2. Cluster packing (32 neurons / cluster)

| Layer | Neurons | Clusters | IDs |
|---|---|---|---|
| hidden1 | 128 | 0–3 (full) | `cluster_map_hidden1` in config json |
| hidden2 | 128 | 4–7 (full) | `cluster_map_hidden2` in config json |
| output (readout) | 3 | 8 (3 used, 29 spare) | appended after hidden2 |

Each dense projection stays inside one 4-cluster group, so all spike
traffic uses the **L1 router only**; L2 is untouched. Spare capacity
(clusters 9–31, 743 neurons) is free for the defend/corridor heads later
(corridor head: 7 outputs → same cluster 8 has room).

## 3. Spike packet mapping (11-bit)

- 1,024 neurons ⇒ 10-bit source ID + payload bits = 11-bit packet.
- Poisson input spikes stream from the host encoder; hidden/output spikes
  route cluster-to-cluster over L1.
- Inference = T timesteps (validated T = 50, path to 25 at 76.6%
  agreement). Decision = spike-count vote (argmax) on the host.

## 4. Weight precision

- Training: float32. Export: simulated symmetric per-tensor int8
  (`quant_report.json`: scales ≈ 0.006–0.010, float-vs-int8 agreement
  93.75%, relative Q deviation 2.93% < 3% bar, membranes fit int16).
- Burn the dequantized int8 values (or int8 + per-tensor scale) into weight
  memory, which dominates chip power (~96%) — hence the small dense model.

## 5. Porting checklist

1. Load `weights.npz` (fc1/fc2/fc3) + `cerebra_config.json` (beta,
   thresholds, T) onto host.
2. Program clusters 0–8 per §2; set LIF decay 0.5, subtractive reset.
3. Stream Poisson spikes from the Python encoder (`src/env/poisson_encoder.py`)
   for one frame-stack observation; assert spike-count vote matches the
   software SNN on the 256-vector validation set (agreement ≥ 80% @ T=50).
4. Close the loop with ViZDoom (`src/env/vizdoom_wrapper.py`, basic.cfg) and
   record the gameplay video.

## 6. Open issues (honest)

- **Biases**: `weights.npz` includes fc biases. Map each to a constant
  current injection per neuron, or fold into thresholds — confirm against
  the Cerebra-H programming manual on hardware access.
- **Exact 11-bit packet field layout**: assumed 10-bit ID + control; verify
  against RTL/docs when available.
- **Quantization margin**: 2.93% vs the 3% bar is thin; per-layer scales
  (vs per-tensor) buy headroom if the hardware loop disagrees.
- **Robustness**: SNN agreement drops to ~59% under darkened inputs
  (`robust_basic/`); prefer stable lighting or add gain control in the
  encoder for deployment.
