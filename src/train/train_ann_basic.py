"""Week 2-3: DQN on ANN MLP (works on CPU, no vizdoom needed for smoke).

Real ViZDoom loop lives in run_vizdoom_dqn() — call it from WSL2/Colab.
`python tools/smoke_test.py` exercises only the offline parts.

Full-quality solo pipeline: checkpoints, gameplay videos, TensorBoard,
score table. Run:

    python src/train/train_ann_basic.py --config configs/basic.yaml --episodes 200
"""
from __future__ import annotations
import random
from collections import deque
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))


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
    dev = next(q_net.parameters()).device
    s = torch.as_tensor(s, dtype=torch.float32, device=dev)
    ns = torch.as_tensor(ns, dtype=torch.float32, device=dev)
    a = torch.as_tensor(a, dtype=torch.int64, device=dev)
    r = torch.as_tensor(r, dtype=torch.float32, device=dev)
    d = torch.as_tensor(d, dtype=torch.float32, device=dev)
    q = q_net(s).gather(1, a.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        q_next = target_net(ns).max(dim=1).values
        target = r + gamma * q_next * (1.0 - d)
    return F.mse_loss(q, target)


def set_seed(seed: int = 0):
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _save_video(frames: list, path: str, fps: int = 20):
    """Write RGB frames to mp4. Skips gracefully if cv2 is missing."""
    try:
        import cv2
    except ImportError:
        print(f"[train] cv2 missing — skipping video {path}")
        return
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"[train] video saved: {path} ({len(frames)} frames)")


def run_vizdoom_dqn(config_path: str = "configs/basic.yaml", episodes: int = 50,
                     out_dir: str = ".", save_every: int = 25,
                     record_video: bool = True, resume_from: str | None = None):
    """Full training loop — run under WSL2/Linux or Colab only.

    Saves: checkpoints/ep{N}.pth + best.pth + final.pth, videos/ep{N}.mp4,
    scores.json, TensorBoard logs under runs/.
    Returns the trained q_net.
    """
    import time
    import yaml
    import torch
    from src.env.vizdoom_wrapper import create_game, get_action_table
    from src.env.preprocessing import FrameStack
    from src.models.ann_mlp import AnnMlp
    from src.eval.evaluate import score_summary

    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg.get("seed", 0))
    out = Path(out_dir)
    ckpt_dir, vid_dir = out / "checkpoints", out / "videos"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    size, n_actions = cfg["encode_size"], cfg["n_actions"]
    hidden = cfg.get("hidden_size", 128)
    input_dim = size * size  # single encoded frame (mean of stack) for MLP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_net = AnnMlp(input_dim, hidden, n_actions).to(device)
    target = AnnMlp(input_dim, hidden, n_actions).to(device)
    target.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.get("learning_rate", 2.5e-4))
    if resume_from:
        sd = torch.load(resume_from, map_location=device)
        q_net.load_state_dict(sd)
        target.load_state_dict(sd)
        print(f"[train] resumed from {resume_from}")
    gamma = cfg.get("gamma", 0.99)
    frame_skip = cfg.get("frame_skip", 4)

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(out / "runs" / f"{cfg['scenario']}_{time.strftime('%Y%m%d_%H%M%S')}"))
    except Exception as e:
        print(f"[train] tensorboard disabled ({e})")
        tb = None

    scenario_actions = get_action_table(n_actions)
    game = create_game(cfg["scenario"],
                       episode_timeout=int(cfg.get("episode_timeout", 300)))
    got = len(game.get_available_buttons())
    assert got == n_actions, (
        f"{cfg['scenario']}: cfg n_actions={n_actions} but env exposes {got}")
    buf = ReplayBuffer(cfg.get("replay_capacity", 20000))
    try:
        import vizdoom as vzd
        _HIT = vzd.GameVariable.HITCOUNT
    except Exception:
        _HIT = None
    hit_bonus = float(cfg.get("hit_bonus", 0.0))
    stacker = FrameStack(cfg.get("stack_frames", 4), size)
    eps = cfg.get("eps_start", 1.0)
    eps_end, eps_decay = cfg.get("eps_end", 0.1), cfg.get("eps_decay", 0.995)

    rewards, losses, kills = [], [], []
    best_reward = -1e9

    for ep in range(episodes):
        game.new_episode()
        state = game.get_state()
        obs = stacker.reset(state.screen_buffer).mean(axis=0).reshape(-1)
        done = game.is_episode_finished()
        try:
            prev_hits = float(game.get_game_variable(_HIT)) if _HIT is not None else 0.0
        except Exception:
            prev_hits = 0.0
        ep_loss, steps = 0.0, 0
        while not done:
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    a = int(q_net(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax())
            game.make_action(scenario_actions[a], frame_skip)
            done = game.is_episode_finished()
            # NOTE: always take the env's last reward — even on the final
            # step. Overwriting it with a fixed terminal penalty erases the
            # kill bonus from replay and the agent learns never to shoot.
            # Rewards are scaled (Mnih-style) so MSE targets stay small.
            r = float(game.get_last_reward()) * cfg.get("reward_scale", 1.0)
            if hit_bonus and _HIT is not None:
                try:
                    hits = float(game.get_game_variable(_HIT))
                    r += hit_bonus * (hits - prev_hits)
                    prev_hits = hits
                except Exception:
                    pass
            nobs = obs if done else stacker.step(
                game.get_state().screen_buffer).mean(axis=0).reshape(-1)
            buf.push(obs, a, r, nobs, float(done))
            obs = nobs
            steps += 1
            if len(buf) >= cfg.get("learn_start", 512):
                loss = dqn_loss_fn(q_net, target, buf.sample(32), gamma)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()
                ep_loss += float(loss.detach())
            if steps > cfg.get("max_steps", 2000):
                break
        target.load_state_dict(q_net.state_dict())
        eps = max(eps_end, eps * eps_decay)
        total = float(game.get_total_reward())
        rewards.append(total)
        kills.append(1.0 if total > 0 else 0.0)  # basic/defend: positive total ⇒ frag(s)
        avg_loss = ep_loss / max(steps, 1)
        losses.append(avg_loss)
        print(f"ep={ep} reward={total:.1f} avg_loss={avg_loss:.4f} eps={eps:.2f} steps={steps}", flush=True)
        if tb is not None:
            tb.add_scalar("reward/episode", total, ep)
            tb.add_scalar("loss/avg", avg_loss, ep)
            tb.add_scalar("eps", eps, ep)

        is_best = total > best_reward
        if is_best:
            best_reward = total
            torch.save(q_net.state_dict(), ckpt_dir / "best.pth")
        if (ep + 1) % save_every == 0 or (ep + 1) == episodes:
            torch.save(q_net.state_dict(), ckpt_dir / f"ep{ep + 1}.pth")
            if record_video:
                # greedy rollout for the video (no exploration, no training)
                game.new_episode()
                frames = []
                st = game.get_state()
                vo = stacker.reset(st.screen_buffer).mean(axis=0).reshape(-1)
                vd = game.is_episode_finished()
                while not vd:
                    frames.append(np.array(game.get_state().screen_buffer))
                    with torch.no_grad():
                        va = int(q_net(torch.as_tensor(vo, device=device).unsqueeze(0)).argmax())
                    game.make_action(scenario_actions[va], frame_skip)
                    vd = game.is_episode_finished()
                    if not vd:
                        vo = stacker.step(game.get_state().screen_buffer).mean(axis=0).reshape(-1)
                    if len(frames) > cfg.get("max_steps", 2000):
                        break
                _save_video(frames, str(vid_dir / f"ep{ep + 1}_{cfg['scenario']}.mp4"))

    torch.save(q_net.state_dict(), ckpt_dir / "final.pth")
    summary = score_summary(rewards, kills)
    summary.update({"best_reward": best_reward, "scenario": cfg["scenario"],
                    "episodes": episodes, "device": device,
                    "hit_bonus": hit_bonus})
    import json
    (out / "scores.json").write_text(json.dumps(summary, indent=2))
    print("SCORES:", summary, flush=True)
    if tb is not None:
        tb.close()
    game.close()
    return q_net


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="DQN training for ViZDoom ANN baseline")
    p.add_argument("--config", default="configs/basic.yaml")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--resume-from", default=None)
    args = p.parse_args()
    run_vizdoom_dqn(args.config, episodes=args.episodes, out_dir=args.out_dir,
                     save_every=args.save_every, record_video=not args.no_video,
                     resume_from=args.resume_from)
