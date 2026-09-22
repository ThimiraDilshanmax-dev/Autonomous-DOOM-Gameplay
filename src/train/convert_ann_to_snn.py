"""Stage 2 (Weeks 4-5): ANN -> SNN conversion with threshold calibration.

Copies trained AnnMlp weights into SnnMlp (same topology), sets LIF
thresholds from activation percentiles on real ViZDoom observations
(falls back to synthetic inputs without vizdoom), then sweeps inference
timesteps T = {50, 25} reporting ANN/SNN action agreement.

Run (WSL2):
    python src/train/convert_ann_to_snn.py \
        --checkpoint training_basic1000/checkpoints/best.pth \
        --config configs/basic.yaml --out-dir conversion_basic
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))


def collect_observations(cfg, n: int = 512, seed: int = 0) -> np.ndarray:
    """Real frame-stack observations from random ViZDoom rollouts.

    Scenario-aware: uses the correct action table (health needs
    TURN/MOVE_FORWARD, not basic MOVE/ATTACK) and the configured
    obs_mode (mean vs stack) so calibration matches training.
    """
    from src.env.vizdoom_wrapper import create_game, get_action_table_for_scenario
    from src.env.preprocessing import FrameStack, stack_to_obs
    import random
    rng = random.Random(seed)
    scenario = cfg.get("scenario", "basic")
    obs_mode = cfg.get("obs_mode", "mean")
    actions = get_action_table_for_scenario(scenario)
    game = create_game(scenario)
    stacker = FrameStack(cfg.get("stack_frames", 4), cfg["encode_size"])
    obs = []
    while len(obs) < n:
        game.new_episode()
        o = stack_to_obs(stacker.reset(game.get_state().screen_buffer), obs_mode)
        obs.append(o)
        done = game.is_episode_finished()
        steps = 0
        while not done and len(obs) < n:
            game.make_action(actions[rng.randrange(len(actions))],
                             cfg.get("frame_skip", 4))
            done = game.is_episode_finished()
            if not done:
                o = stack_to_obs(stacker.step(game.get_state().screen_buffer), obs_mode)
                obs.append(o)
            steps += 1
            if steps > 500:
                break
    game.close()
    return np.stack(obs[:n])


def calibrate_thresholds(ann, obs: np.ndarray, percentile: float = 99.0) -> list[float]:
    """Per-layer LIF thresholds from ANN pre-activation percentiles."""
    import torch
    ann.eval()
    thresholds = []
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32)
        for fc in [ann.net[0], ann.net[2]]:
            pre = fc(x)
            thr = float(np.percentile(pre.cpu().numpy(), percentile))
            thresholds.append(max(thr, 0.1))
            x = torch.relu(pre)
    return thresholds


def agreement(ann, snn, obs: np.ndarray, T: int) -> dict:
    """Action agreement + output MSE between ANN and Poisson-driven SNN.

    Obs-shape agnostic: works for mean (B,784) and stack (B,3136) inputs.
    Poisson encoding is applied directly to the flat vector so no temporal
    info is collapsed here either.
    """
    import torch
    from src.env.poisson_encoder import poisson_encode
    ann.eval()
    with torch.no_grad():
        q_ann = ann(torch.as_tensor(obs, dtype=torch.float32))
        a_ann = q_ann.argmax(dim=1)
    spikes = poisson_encode(obs, T=T, seed=0)
    # spikes: (T, B, D) — already matches SNN input (T, B, input_dim).
    with torch.no_grad():
        q_snn = snn(torch.as_tensor(spikes, dtype=torch.float32))
        a_snn = q_snn.argmax(dim=1)
    agree = float((a_ann == a_snn).float().mean())
    mse = float(((q_ann - q_snn) ** 2).mean())
    # degeneracy diagnostics: constant policies agree vacuously, silent SNNs
    # output the same vector for every action
    hist = torch.bincount(a_ann, minlength=q_ann.shape[1]).tolist()
    snn_std = float(q_snn.std())
    return {"agreement": agree, "mse": mse,
            "ann_action_hist": hist,
            "snn_output_std": snn_std,
            "silent_warning": snn_std < 1e-6,
            "energy_proxy_spike_steps": T * snn.neuron_count()}


def convert(checkpoint: str, config_path: str = "configs/basic.yaml",
            out_dir: str = "conversion_basic", timesteps: tuple = (50, 25),
            percentile: float = 99.0, n_calib: int = 512, beta: float = 0.5,
            threshold_scale: float = 1.0):
    import json
    import yaml
    import torch
    from src.models.ann_mlp import AnnMlp
    from src.models.snn_mlp import SnnMlp
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
            f"(input_dim={input_dim}). Health moved mean/784 -> stack/3136; "
            f"retrain health from scratch. Original: {e}")
    print(f"[convert] loaded ANN {checkpoint}")

    try:
        obs = collect_observations(cfg, n_calib, seed=cfg.get("seed", 0))
        print(f"[convert] collected {len(obs)} real observations")
    except Exception as e:
        print(f"[convert] vizdoom unavailable ({e}) — synthetic calibration")
        rng = np.random.default_rng(0)
        obs = rng.random((n_calib, size * size), dtype=np.float32)

    thresholds = [t * threshold_scale
                  for t in calibrate_thresholds(ann, obs, percentile)]
    print(f"[convert] thresholds p{percentile} x{threshold_scale}: {thresholds}")

    report = {"checkpoint": checkpoint, "percentile": percentile,
              "threshold_scale": threshold_scale,
              "thresholds": thresholds, "beta": beta, "sweeps": []}
    for T in timesteps:
        snn = SnnMlp(input_dim, hidden, n_actions, T=T, beta=beta,
                     threshold=thresholds[0])
        # copy ANN weights into SNN linear layers
        snn.fc1.load_state_dict(ann.net[0].state_dict())
        snn.fc2.load_state_dict(ann.net[2].state_dict())
        snn.fc3.load_state_dict(ann.net[4].state_dict())
        # per-layer thresholds (snnTorch keeps threshold as a buffer tensor)
        for lif, thr in zip([snn.lif1, snn.lif2], thresholds):
            try:
                if torch.is_tensor(getattr(lif, "threshold", None)):
                    with torch.no_grad():
                        lif.threshold.fill_(thr)
                else:
                    lif.threshold = thr
            except Exception:
                lif.threshold = thr
        m = agreement(ann, snn, obs, T)
        m["T"] = T
        report["sweeps"].append(m)
        print(f"[convert] T={T} agreement={m['agreement']:.3f} mse={m['mse']:.4f}", flush=True)
        if T == timesteps[0]:
            torch.save(snn.state_dict(), out / "snn_weights.pth")
            export_snn(snn, out_dir=str(out / "cerebra_bundle"), beta=beta,
                       threshold=thresholds[0], T=T)
    (out / "conversion_report.json").write_text(json.dumps(report, indent=2))
    print(f"[convert] report saved to {out / 'conversion_report.json'}")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ANN->SNN conversion with calibration")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/basic.yaml")
    p.add_argument("--out-dir", default="conversion_basic")
    p.add_argument("--timesteps", type=int, nargs="+", default=[50, 25])
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--n-calib", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--threshold-scale", type=float, default=1.0)
    args = p.parse_args()
    convert(args.checkpoint, args.config, args.out_dir, tuple(args.timesteps),
            args.percentile, args.n_calib, args.beta, args.threshold_scale)
