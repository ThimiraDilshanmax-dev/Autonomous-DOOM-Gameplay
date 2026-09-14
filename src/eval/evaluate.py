"""Evaluation helpers: score table + video hook (Week 3 / Week 7)."""
from __future__ import annotations


def score_summary(episode_rewards: list[float], kills: list[float] | None = None) -> dict:
    import numpy as np
    out = {
        "episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
    }
    if kills:
        out["mean_kills"] = float(np.mean(kills))
        out["max_kills"] = float(np.max(kills))
    return out


def record_note(path: str = "videos/"):
    print(f"[eval] Save gameplay mp4s under {path} (Week 3 baseline, Week 7 SNN).")
