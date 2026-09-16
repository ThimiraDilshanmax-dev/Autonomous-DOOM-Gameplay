"""Greedy evaluation of trained checkpoints (any 3-button scenario)."""
import sys
sys.path.insert(0, "/mnt/c/Users/Admin/Desktop/SNN")
import random
import torch
from src.env.vizdoom_wrapper import create_game, get_action_table
from src.env.preprocessing import FrameStack
from src.models.ann_mlp import AnnMlp

def evaluate(weight_path, scenario="basic", episodes=50, n_actions=3,
             episode_timeout=300):
    net = AnnMlp(28 * 28, 128, n_actions)
    actions = get_action_table(n_actions)
    if weight_path != "random":
        net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()
    game = create_game(scenario, episode_timeout=episode_timeout)
    st = FrameStack(4, 28)
    rewards, steps_l, kills = [], [], 0
    for ep in range(episodes):
        game.new_episode()
        obs = st.reset(game.get_state().screen_buffer).mean(axis=0).reshape(-1)
        done = game.is_episode_finished()
        steps = 0
        while not done:
            if weight_path == "random":
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    a = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax())
            game.make_action(actions[a], 4)
            done = game.is_episode_finished()
            if not done:
                obs = st.step(game.get_state().screen_buffer).mean(axis=0).reshape(-1)
            steps += 1
            if steps > 2000:
                break
        r = float(game.get_total_reward())
        rewards.append(r)
        steps_l.append(steps)
        if r > 0:
            kills += 1
    game.close()
    import numpy as np
    tag = weight_path.split("/")[-1]
    print(f"[{scenario}] {tag}: pos_rate={kills}/{episodes} "
          f"mean_reward={np.mean(rewards):.1f} mean_steps={np.mean(steps_l):.1f}", flush=True)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="basic")
    p.add_argument("--n-actions", type=int, default=3)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument("--ckpt", nargs="+", default=["random"])
    p.add_argument("--episodes", type=int, default=50)
    args = p.parse_args()
    for c in args.ckpt:
        evaluate(c, args.scenario, args.episodes, args.n_actions,
                 args.episode_timeout)
    print("EVAL_DONE")
