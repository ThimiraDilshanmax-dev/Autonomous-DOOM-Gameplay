"""Week 7: robustness checks — noise, occlusion, brightness shifts.

Measures action stability of the ANN baseline and the fine-tuned SNN on
perturbed ViZDoom observations (offline proxy for unseen textures/lighting).
Writes robustness_report.json.

Run (WSL2):
    python tools/wsl_robustness.py --config configs/basic.yaml \
        --ann-checkpoint training_basic4000/checkpoints/final.pth \
        --snn-weights finetune_basic/snn_finetuned.pth --out-dir robust_basic
"""
import sys
from pathlib import Path as _P
for _cand in ("/mnt/c/Users/Admin/Desktop/SNN", "C:/Users/Admin/Desktop/SNN",
              str(_P(__file__).resolve().parents[1])):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)
import numpy as np


def perturb(obs: np.ndarray, kind: str, seed: int = 0, size: int = 28) -> np.ndarray:
    """Perturb flat obs (mean or stack) in spatial form.

    Reshapes to (-1, k, size, size) when stacked so occlusion hits each
    frame coherently instead of assuming a single 28x28 frame.
    """
    rng = np.random.default_rng(seed)
    was_1d = (obs.ndim == 1)
    b = obs.reshape(1, -1) if was_1d else obs
    d = b.shape[1]
    if d % (size * size) != 0:
        raise ValueError(f"obs dim {d} not divisible by {size}x{size}")
    k = d // (size * size)
    x = b.reshape(-1, k, size, size).copy()
    if kind == "gauss01":
        x = np.clip(x + rng.normal(0, 0.1, x.shape), 0, 1)
    elif kind == "gauss02":
        x = np.clip(x + rng.normal(0, 0.2, x.shape), 0, 1)
    elif kind == "occlude":
        for i in range(len(x)):
            for f in range(k):
                h, w = rng.integers(0, 20, 2)
                x[i, f, h:h + 8, w:w + 8] = 0.0
    elif kind == "bright":
        x = np.clip(x * 1.3, 0, 1)
    elif kind == "dark":
        x = np.clip(x * 0.7, 0, 1)
    out = x.reshape(b.shape).astype(np.float32)
    return out.reshape(obs.shape).astype(np.float32)


def main():
    import argparse
    import json
    import torch
    from pathlib import Path
    from src.models.ann_mlp import AnnMlp
    from src.train.finetune_snn import build_student
    from src.train.convert_ann_to_snn import (
        collect_observations, calibrate_thresholds)
    from src.env.poisson_encoder import poisson_encode
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/basic.yaml")
    p.add_argument("--ann-checkpoint", required=True)
    p.add_argument("--snn-weights", required=True)
    p.add_argument("--out-dir", default="robust_basic")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--t", type=int, default=50)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    size, hidden, n_actions = cfg["encode_size"], cfg["hidden_size"], cfg["n_actions"]
    from src.env.preprocessing import obs_dim
    input_dim = obs_dim(size, int(cfg.get("stack_frames", 4)),
                        cfg.get("obs_mode", "mean"))
    ann = AnnMlp(input_dim, hidden, n_actions)
    ann.load_state_dict(torch.load(args.ann_checkpoint, map_location="cpu"))
    ann.eval()
    try:
        calib_obs = collect_observations(cfg, 256, seed=7)
    except Exception as e:
        print(f"[robust] vizdoom unavailable ({e}) — synthetic calibration")
        import numpy as _np
        calib_obs = _np.random.default_rng(7).random((256, input_dim), dtype=_np.float32)
    thresholds = [t * 0.7 for t in calibrate_thresholds(ann, calib_obs)]
    snn = build_student(ann, size, hidden, n_actions, args.t, 0.5, thresholds)
    snn.load_state_dict(torch.load(args.snn_weights, map_location="cpu"), strict=False)
    snn.eval()

    try:
        obs = collect_observations(cfg, args.n, seed=cfg.get("seed", 0) + 31337)
    except Exception as e:
        print(f"[robust] vizdoom unavailable ({e}) — synthetic eval")
        import numpy as _np
        obs = _np.random.default_rng(31337).random((args.n, input_dim), dtype=_np.float32)
    with torch.no_grad():
        a_clean = ann(torch.as_tensor(obs)).argmax(1)
    report = {"kinds": {}, "obs_mode": cfg.get("obs_mode", "mean"),
              "input_dim": input_dim}
    for kind in ["gauss01", "gauss02", "occlude", "bright", "dark"]:
        po = perturb(obs, kind, size=size)
        with torch.no_grad():
            a_ann = ann(torch.as_tensor(po)).argmax(1)
            sp = poisson_encode(po, T=args.t, seed=2)
            a_snn = snn(torch.as_tensor(sp)).argmax(1)
        report["kinds"][kind] = {
            "ann_stability": float((a_clean == a_ann).float().mean()),
            "snn_ann_agreement": float((a_ann == a_snn).float().mean()),
        }
        print(f"[robust] {kind}: ann_stable={report['kinds'][kind]['ann_stability']:.3f} "
              f"snn_agree={report['kinds'][kind]['snn_ann_agreement']:.3f}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness_report.json").write_text(json.dumps(report, indent=2))
    print(f"[robust] saved to {out / 'robustness_report.json'}")


if __name__ == "__main__":
    main()
