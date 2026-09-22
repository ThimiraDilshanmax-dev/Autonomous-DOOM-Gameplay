"""Stage 3 (Week 6): surrogate-gradient fine-tuning of a converted SNN.

Distills a trained ANN (Q-targets) into a Poisson-driven SnnMlp with BPTT
through snnTorch LIF neurons. Evaluates ANN/SNN action agreement each
epoch, keeps the best student, and refreshes the Cerebra export bundle.

Run (WSL2):
    python src/train/finetune_snn.py \
        --checkpoint training_basic4000/checkpoints/final.pth \
        --config configs/basic.yaml --out-dir finetune_basic --epochs 8
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))


def build_student(ann, size: int, hidden: int, n_actions: int,
                  T: int, beta: float, thresholds: list[float],
                  input_dim: int | None = None):
    import torch
    from src.models.snn_mlp import SnnMlp
    if input_dim is None:
        # Infer from teacher: fc1.in_features is the source of truth and
        # keeps mean (784) and stack (3136) checkpoints compatible.
        try:
            input_dim = int(ann.net[0].in_features)
        except Exception:
            input_dim = size * size
    snn = SnnMlp(input_dim, hidden, n_actions, T=T, beta=beta,
                 threshold=thresholds[0])
    snn.fc1.load_state_dict(ann.net[0].state_dict())
    snn.fc2.load_state_dict(ann.net[2].state_dict())
    snn.fc3.load_state_dict(ann.net[4].state_dict())
    for lif, thr in zip([snn.lif1, snn.lif2], thresholds):
        try:
            if torch.is_tensor(getattr(lif, "threshold", None)):
                with torch.no_grad():
                    lif.threshold.fill_(thr)
            else:
                lif.threshold = thr
        except Exception:
            lif.threshold = thr
    return snn


def encode_batch(obs: np.ndarray, T: int, gain: float = 1.0):
    import torch
    from src.env.poisson_encoder import poisson_encode
    # Obs-shape agnostic (784 or 3136+). Optional gain compensates for
    # darkened/low-contrast navigation frames (see robustness report).
    if gain != 1.0:
        import numpy as np
        obs = np.clip(obs.astype(np.float32) * gain, 0.0, 1.0)
    spikes = poisson_encode(obs, T=T, seed=None)  # fresh sample each call
    return torch.as_tensor(spikes, dtype=torch.float32)


def finetune(checkpoint: str, config_path: str = "configs/basic.yaml",
             out_dir: str = "finetune_basic", epochs: int = 8,
             T_train: int = 25, lr: float = 1e-3, batch: int = 32,
             n_train: int = 1024, n_val: int = 256,
             percentile: float = 99.0, threshold_scale: float = 0.7,
             beta: float = 0.5):
    import json
    import yaml
    import torch
    from src.models.ann_mlp import AnnMlp
    from src.train.convert_ann_to_snn import (
        collect_observations, calibrate_thresholds, agreement)
    from src.export.cerebra_export import export_snn

    cfg = yaml.safe_load(open(config_path))
    size, hidden, n_actions = cfg["encode_size"], cfg["hidden_size"], cfg["n_actions"]
    from src.env.preprocessing import obs_dim
    input_dim = obs_dim(size, int(cfg.get("stack_frames", 4)),
                        cfg.get("obs_mode", "mean"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ann = AnnMlp(input_dim, hidden, n_actions)
    try:
        ann.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    except RuntimeError as e:
        raise RuntimeError(
            f"checkpoint {checkpoint} incompatible with {config_path} "
            f"(input_dim={input_dim}, hidden={hidden}). Old health checkpoints "
            f"trained with mean/784 must be retrained with stack/3136. Original: {e}")
    ann.eval()

    try:
        obs_all = collect_observations(cfg, n_train + n_val, seed=cfg.get("seed", 0))
    except Exception as e:
        print(f"[finetune] vizdoom unavailable ({e}) — synthetic data for offline test")
        rng = np.random.default_rng(0)
        obs_all = rng.random((n_train + n_val, input_dim), dtype=np.float32)
    tr, va = obs_all[:n_train], obs_all[n_train:]
    q_teacher_tr = ann(torch.as_tensor(tr)).detach()
    print(f"[finetune] data: {len(tr)} train / {len(va)} val", flush=True)

    thresholds = [t * threshold_scale
                  for t in calibrate_thresholds(ann, tr, percentile)]
    snn = build_student(ann, size, hidden, n_actions, T_train, beta, thresholds)
    opt = torch.optim.Adam(snn.parameters(), lr=lr)
    base = agreement(ann, snn, va, T_train)
    print(f"[finetune] start agreement={base['agreement']:.3f}", flush=True)

    best, best_state = base["agreement"], None
    history = []
    for ep in range(epochs):
        snn.train()
        perm = np.random.default_rng(ep).permutation(len(tr))
        tot = 0.0
        for i in range(0, len(tr), batch):
            idx = perm[i:i + batch]
            x = encode_batch(tr[idx], T_train)
            y = q_teacher_tr[idx]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(snn(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
        snn.eval()
        m = agreement(ann, snn, va, T_train)
        history.append({"epoch": ep, "loss": tot / (len(tr) // batch),
                        "agreement": m["agreement"]})
        print(f"[finetune] ep={ep} loss={tot / (len(tr) // batch):.4f} "
              f"agree={m['agreement']:.3f}", flush=True)
        if m["agreement"] > best:
            best = m["agreement"]
            best_state = {k: v.cpu().clone() for k, v in snn.state_dict().items()}
    if best_state is not None:
        snn.load_state_dict(best_state, strict=False)
    torch.save(snn.state_dict(), out / "snn_finetuned.pth")

    # final T-sweep on the fine-tuned weights
    report = {"checkpoint": checkpoint, "best_val_agreement": best,
              "history": history, "sweeps": []}
    for T in (50, 25):
        probe = build_student(ann, size, hidden, n_actions, T, beta, thresholds)
        probe.load_state_dict(snn.state_dict(), strict=False)
        # copy linear weights (T-independent); keep probe's own LIF buffers
        probe.fc1.load_state_dict(snn.fc1.state_dict())
        probe.fc2.load_state_dict(snn.fc2.state_dict())
        probe.fc3.load_state_dict(snn.fc3.state_dict())
        m = agreement(ann, probe, va, T)
        m["T"] = T
        report["sweeps"].append(m)
        print(f"[finetune] T={T} agreement={m['agreement']:.3f}", flush=True)
        if T == 50:
            export_snn(probe, out_dir=str(out / "cerebra_bundle"), beta=beta,
                       threshold=thresholds[0], T=T)
    (out / "finetune_report.json").write_text(json.dumps(report, indent=2))
    print(f"[finetune] best={best:.3f} saved to {out}")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SNN surrogate fine-tuning (distillation)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/basic.yaml")
    p.add_argument("--out-dir", default="finetune_basic")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--t-train", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    finetune(args.checkpoint, args.config, args.out_dir, args.epochs,
             args.t_train, args.lr, args.batch)
