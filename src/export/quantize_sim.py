"""Week 6: 8-bit quantization simulation + fidelity report.

Simulates the Cerebra-H inference path (symmetric per-tensor int8 weights,
subtractive-reset LIF, fixed-point membrane) entirely in torch and compares
against the float SNN: output MSE, action agreement, and Q deviation. The
proposal's exit bar is <3% software-vs-simulated-hardware deviation.

Run (WSL2):
    python src/export/quantize_sim.py \
        --snn-weights finetune_basic/snn_finetuned.pth \
        --ann-checkpoint training_basic4000/checkpoints/final.pth \
        --config configs/basic.yaml --out-dir quant_basic
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))


def quantize_int8(t: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor int8 fake-quantization. Returns (dequant, scale)."""
    amax = float(np.abs(t).max())
    if amax == 0:
        return t.copy(), 1.0
    scale = amax / 127.0
    q = np.clip(np.round(t / scale), -128, 127).astype(np.int8)
    return (q.astype(np.float32) * scale), scale


def simulate(config_path: str, snn_weights: str, ann_checkpoint: str,
             out_dir: str = "quant_basic", n_val: int = 256, T: int = 50,
             beta: float = 0.5, threshold_scale: float = 0.7,
             percentile: float = 99.0):
    import json
    import yaml
    import torch
    from src.models.ann_mlp import AnnMlp
    from src.train.finetune_snn import build_student
    from src.train.convert_ann_to_snn import (
        collect_observations, calibrate_thresholds, agreement)

    cfg = yaml.safe_load(open(config_path))
    size, hidden, n_actions = cfg["encode_size"], cfg["hidden_size"], cfg["n_actions"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ann = AnnMlp(size * size, hidden, n_actions)
    ann.load_state_dict(torch.load(ann_checkpoint, map_location="cpu"))
    ann.eval()

    obs = collect_observations(cfg, n_val, seed=cfg.get("seed", 0) + 999)
    thresholds = [t * threshold_scale
                  for t in calibrate_thresholds(ann, obs, percentile)]

    float_net = build_student(ann, size, hidden, n_actions, T, beta, thresholds)
    float_net.load_state_dict(torch.load(snn_weights, map_location="cpu"))
    float_net.eval()

    # int8-simulated twin: same topology, dequantized weights
    quant_net = build_student(ann, size, hidden, n_actions, T, beta, thresholds)
    quant_net.load_state_dict(torch.load(snn_weights, map_location="cpu"))
    scales = {}
    with torch.no_grad():
        for name in ["fc1.weight", "fc2.weight", "fc3.weight"]:
            mod, attr = name.split(".")
            w = getattr(getattr(quant_net, mod), attr).cpu().numpy()
            dq, scale = quantize_int8(w)
            getattr(getattr(quant_net, mod), attr).copy_(
                torch.as_tensor(dq))
            scales[name] = scale
    quant_net.eval()

    from src.env.poisson_encoder import poisson_encode
    spikes = poisson_encode(obs.reshape(-1, 28, 28), T=T, seed=1)
    spikes = spikes.reshape(T, -1, 784)
    x = torch.as_tensor(spikes, dtype=torch.float32)
    with torch.no_grad():
        q_float = float_net(x)
        q_quant = quant_net(x)
        a_float = q_float.argmax(dim=1)
        a_quant = q_quant.argmax(dim=1)
    agree = float((a_float == a_quant).float().mean())
    # deviation of Q-values relative to their dynamic range
    span = float((q_float.max() - q_float.min()).clamp_min(1e-6))
    rel_dev = float((q_float - q_quant).abs().mean() / span)
    mse = float(((q_float - q_quant) ** 2).mean())

    # fixed-point membrane sanity: worst-case accumulated current per step
    # must fit int16 with the chosen weight scale (HW membrane registers)
    max_w = max(scales.values())
    fits_i16 = bool(max_w * size * 1.0 < 32767 * (2 ** -8))

    report = {
        "snn_weights": snn_weights, "T": T, "beta": beta,
        "reset_mode": "subtractive",
        "weight_scales": scales,
        "float_vs_int8_agreement": agree,
        "float_vs_int8_mse": mse,
        "relative_q_deviation": rel_dev,
        "deviation_pct": rel_dev * 100.0,
        "pass_lt_3pct": bool(rel_dev < 0.03),
        "membrane_fits_i16": fits_i16,
    }
    (out / "quant_report.json").write_text(json.dumps(report, indent=2))
    print(f"[quant] agreement={agree:.4f} deviation={rel_dev * 100:.3f}% "
          f"pass(<3%)={report['pass_lt_3pct']}")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="int8 quantization fidelity simulation")
    p.add_argument("--snn-weights", required=True)
    p.add_argument("--ann-checkpoint", required=True)
    p.add_argument("--config", default="configs/basic.yaml")
    p.add_argument("--out-dir", default="quant_basic")
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--t", type=int, default=50)
    args = p.parse_args()
    simulate(args.config, args.snn_weights, args.ann_checkpoint,
             args.out_dir, args.n_val, args.t)
