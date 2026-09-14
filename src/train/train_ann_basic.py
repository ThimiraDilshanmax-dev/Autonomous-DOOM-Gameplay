"""Week 2-3: minimal DQN on ANN MLP (works on CPU, no vizdoom needed for smoke).

Real ViZDoom loop lives in run_vizdoom_dqn() — call it from WSL2/Colab.
`python tools/smoke_test.py` exercises only the offline parts.
"""
from __future__ import annotations
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def dqn_loss_fn(q_net, target_net, batch, gamma: float = 0.99):
    """Pure-torch DQN loss; importing torch lazily keeps numpy smoke green."""
    import torch
    import torch.nn.functional as F
    s, a, r, ns, d = batch
    s = torch.as_tensor(s, dtype=torch.float32)
    ns = torch.as_tensor(ns, dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.int64)
    r = torch.as_tensor(r, dtype=torch.float32)
    d = torch.as_tensor(d, dtype=torch.float32)
    q = q_net(s).gather(1, a.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        q_next = target_net(ns).max(dim=1).values
        target = r + gamma * q_next * (1.0 - d)
    return F.mse_loss(q, target)


def run_vizdoom_dqn(config_path: str = "configs/basic.yaml", episodes: int = 50):
    """Full training loop — run under WSL2/Linux or Colab only."""
    import yaml, torch
    from src.env.vizdoom_wrapper import create_game, BASIC_ACTIONS
    from src.env.preprocessing import FrameStack
    from src.models.ann_mlp import AnnMlp

    cfg = yaml.safe_load(open(config_path))
    size, n_actions = cfg["encode_size"], cfg["n_actions"]
    input_dim = size * size  # single encoded frame (mean of stack) for MLP
    q_net = AnnMlp(input_dim, cfg["hidden_size"], n_actions)
    target = AnnMlp(input_dim, cfg["hidden_size"], n_actions)
    target.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg["learning_rate"])
    buf = ReplayBuffer()
    game = create_game(cfg["scenario"])
    stacker = FrameStack(cfg["stack_frames"], size)
    eps = 1.0

    for ep in range(episodes):
        game.new_episode()
        state = game.get_state()
        obs = stacker.reset(state.screen_buffer).mean(axis=0).reshape(-1)
        done = game.is_episode_finished()
        while not done:
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    a = int(q_net(torch.as_tensor(obs).unsqueeze(0)).argmax())
            game.make_action(BASIC_ACTIONS[a], cfg["frame_skip"])
            done = game.is_episode_finished()
            r = game.get_last_reward() if not done else -10.0
            nobs = obs if done else stacker.step(
                game.get_state().screen_buffer).mean(axis=0).reshape(-1)
            buf.push(obs, a, r, nobs, float(done))
            obs = nobs
            if len(buf) >= 512:
                loss = dqn_loss_fn(q_net, target, buf.sample(32), cfg["gamma"])
                opt.zero_grad()
                loss.backward()
                opt.step()
        target.load_state_dict(q_net.state_dict())
        eps = max(0.1, eps * 0.995)
        print(f"ep={ep} reward={game.get_total_reward():.1f} eps={eps:.2f}", flush=True)
    game.close()
    return q_net
