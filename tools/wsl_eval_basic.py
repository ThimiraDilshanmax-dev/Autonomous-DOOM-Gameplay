"""Greedy evaluation of trained checkpoints (scenario-aware)."""
import sys
from pathlib import Path as _P
for _cand in ("/mnt/c/Users/Admin/Desktop/SNN", "C:/Users/Admin/Desktop/SNN",
              str(_P(__file__).resolve().parents[1])):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)
import random
import torch
from src.env.vizdoom_wrapper import create_game, get_action_table_for_scenario
from src.env.preprocessing import FrameStack, obs_dim, stack_to_obs
from src.models.ann_mlp import AnnMlp

def evaluate(weight_path, scenario="basic", episodes=50, n_actions=3,
             episode_timeout=300, encode_size=28, stack_frames=4,
             obs_mode="mean", hidden=128, frame_skip=4):
    input_dim = obs_dim(encode_size, stack_frames, obs_mode)
    net = AnnMlp(input_dim, hidden, n_actions)
    actions = get_action_table_for_scenario(scenario)
    assert len(actions) == n_actions, f"{scenario}: table {len(actions)} vs {n_actions}"
    if weight_path != "random":
        net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()
    game = create_game(scenario, episode_timeout=episode_timeout)
    st = FrameStack(stack_frames, encode_size)
    rewards, steps_l, kills = [], [], 0
    for ep in range(episodes):
        game.new_episode()
        obs = stack_to_obs(st.reset(game.get_state().screen_buffer), obs_mode)
        done = game.is_episode_finished()
        steps = 0
        while not done:
            if weight_path == "random":
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    a = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax())
            game.make_action(actions[a], frame_skip)
            done = game.is_episode_finished()
            if not done:
                obs = stack_to_obs(st.step(game.get_state().screen_buffer), obs_mode)
            steps += 1
            if steps > 3000:
                break
        r = float(game.get_total_reward())
        rewards.append(r)
        steps_l.append(steps)
        if scenario == "health_gathering":
            # Survival is success in health: count full-timeout survivals.
            if steps >= episode_timeout - 1:
                kills += 1
        elif r > 0:
            kills += 1
    game.close()
    import numpy as np
    tag = weight_path.split("/")[-1]
    print(f"[{scenario}] {tag}: pos_rate={kills}/{episodes} "
          f"mean_reward={np.mean(rewards):.1f} mean_steps={np.mean(steps_l):.1f} "
          f"obs={obs_mode} dim={input_dim}", flush=True)

if __name__ == "__main__":
    import argparse
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="basic")
    p.add_argument("--config", default=None)
    p.add_argument("--n-actions", type=int, default=3)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument("--ckpt", nargs="+", default=["random"])
    p.add_argument("--episodes", type=int, default=50)
    args = p.parse_args()
    # If a config is given, mirror its obs geometry so health (stack/3136)
    # checkpoints evaluate correctly instead of assuming basic (mean/784).
    enc, kf, om, hid, fskip = 28, 4, "mean", 128, 4
    na = args.n_actions
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        enc = int(cfg.get("encode_size", enc))
        kf = int(cfg.get("stack_frames", kf))
        om = cfg.get("obs_mode", om)
        hid = int(cfg.get("hidden_size", hid))
        fskip = int(cfg.get("frame_skip", fskip))
        na = int(cfg.get("n_actions", na))
    for c in args.ckpt:
        evaluate(c, args.scenario, args.episodes, na,
                 args.episode_timeout, enc, kf, om, hid, fskip)
    print("EVAL_DONE")
