"""Live gameplay: greedy agent plays ViZDoom with visible window + mp4 recording.

Run under WSL2 (WSLg provides the visible window on your Windows desktop):
    ~/snn-venv/bin/python /mnt/c/Users/Admin/Desktop/SNN/tools/wsl_play_live.py \
        --ckpt /mnt/c/Users/Admin/Desktop/SNN/training_basic4000/checkpoints/best.pth \
        --scenario basic --episodes 3 --visible --out /mnt/c/Users/Admin/Desktop/SNN/videos/live_play.mp4

Use --snn-weights to play the spiking network instead of the ANN baseline.
"""
import sys
from pathlib import Path as _P
for _cand in ("/mnt/c/Users/Admin/Desktop/SNN", "C:/Users/Admin/Desktop/SNN",
              str(_P(__file__).resolve().parents[1])):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

import time
import numpy as np


def play(weight_path, scenario="basic", episodes=3, episode_timeout=300,
         encode_size=28, stack_frames=4, obs_mode="mean", hidden=128,
         n_actions=3, frame_skip=4, visible=True, out="videos/live_play.mp4",
         sleep=0.02, snn_weights=None, snn_T=25):
    import torch
    from src.env.vizdoom_wrapper import create_game, get_action_table_for_scenario
    from src.env.preprocessing import FrameStack, obs_dim, stack_to_obs
    from src.models.ann_mlp import AnnMlp

    input_dim = obs_dim(encode_size, stack_frames, obs_mode)
    actions = get_action_table_for_scenario(scenario)
    assert len(actions) == n_actions, f"{scenario}: table {len(actions)} vs {n_actions}"

    use_snn = snn_weights is not None
    if use_snn:
        from src.train.finetune_snn import build_student
        from src.train.convert_ann_to_snn import calibrate_thresholds, collect_observations
        from src.env.poisson_encoder import poisson_encode
        ann = AnnMlp(input_dim, hidden, n_actions)
        ann.load_state_dict(torch.load(weight_path, map_location="cpu"))
        ann.eval()
        try:
            calib = collect_observations(
                {"scenario": scenario, "encode_size": encode_size,
                 "stack_frames": stack_frames, "obs_mode": obs_mode,
                 "frame_skip": frame_skip, "seed": 0}, 256, seed=7)
        except Exception as e:
            print(f"[play] calib fallback synthetic ({e})", flush=True)
            calib = np.random.default_rng(7).random((256, input_dim)).astype(np.float32)
        thresholds = [t * 0.7 for t in calibrate_thresholds(ann, calib)]
        net = build_student(ann, encode_size, hidden, n_actions, snn_T, 0.5, thresholds)
        net.load_state_dict(torch.load(snn_weights, map_location="cpu"), strict=False)
        net.eval()
        print(f"[play] SNN mode T={snn_T} thresholds={[round(t,3) for t in thresholds]}", flush=True)
    else:
        net = AnnMlp(input_dim, hidden, n_actions)
        if weight_path != "random":
            net.load_state_dict(torch.load(weight_path, map_location="cpu"))
        net.eval()
        print(f"[play] ANN mode ckpt={weight_path}", flush=True)

    game = create_game(scenario, visible=visible, episode_timeout=episode_timeout)
    st = FrameStack(stack_frames, encode_size)

    out_path = str(out)
    _P(out_path).parent.mkdir(parents=True, exist_ok=True)
    vw = None
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        print("[play] cv2 missing — no video recording", flush=True)
        _has_cv2 = False

    for ep in range(episodes):
        game.new_episode()
        obs = stack_to_obs(st.reset(game.get_state().screen_buffer), obs_mode)
        done = game.is_episode_finished()
        steps, frames = 0, []
        t0 = time.time()
        while not done:
            if weight_path == "random" and not use_snn:
                import random
                a = random.randrange(n_actions)
            elif use_snn:
                import torch as _t
                with _t.no_grad():
                    sp = poisson_encode(obs, T=snn_T, seed=None)
                    q = net(_t.as_tensor(sp, dtype=_t.float32))
                    a = int(q.argmax())
            else:
                with torch.no_grad():
                    a = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax())
            game.make_action(actions[a], frame_skip)
            done = game.is_episode_finished()
            if not done:
                buf = game.get_state().screen_buffer
                frames.append(np.array(buf))
                obs = stack_to_obs(st.step(buf), obs_mode)
            steps += 1
            if sleep > 0:
                time.sleep(sleep)
            if steps > 3000:
                break
        dt = time.time() - t0
        r = float(game.get_total_reward())
        print(f"[play] ep={ep} reward={r:.1f} steps={steps} "
              f"time={dt:.1f}s {'SNN' if use_snn else 'ANN'}", flush=True)
        if _has_cv2 and frames:
            h, w, _ = frames[0].shape
            if vw is None:
                vw = __import__("cv2").VideoWriter(
                    out_path, __import__("cv2").VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            for f in frames:
                vw.write(__import__("cv2").cvtColor(f, __import__("cv2").COLOR_RGB2BGR))
            print(f"[play] ep={ep} appended {len(frames)} frames -> {out_path}", flush=True)
    if vw is not None:
        vw.release()
        print(f"[play] video saved: {out_path}", flush=True)
    game.close()
    print("PLAY_DONE", flush=True)


if __name__ == "__main__":
    import argparse
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/c/Users/Admin/Desktop/SNN/training_basic4000/checkpoints/best.pth")
    p.add_argument("--config", default="/mnt/c/Users/Admin/Desktop/SNN/configs/basic.yaml")
    p.add_argument("--scenario", default="basic")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument("--visible", action="store_true", default=True)
    p.add_argument("--no-visible", dest="visible", action="store_false")
    p.add_argument("--out", default="/mnt/c/Users/Admin/Desktop/SNN/videos/live_play.mp4")
    p.add_argument("--sleep", type=float, default=0.02)
    p.add_argument("--snn-weights", default=None)
    p.add_argument("--snn-T", type=int, default=25)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    play(args.ckpt, args.scenario, args.episodes, args.episode_timeout,
         int(cfg.get("encode_size", 28)), int(cfg.get("stack_frames", 4)),
         cfg.get("obs_mode", "mean"), int(cfg.get("hidden_size", 128)),
         int(cfg.get("n_actions", 3)), int(cfg.get("frame_skip", 4)),
         args.visible, args.out, args.sleep, args.snn_weights, args.snn_T)
