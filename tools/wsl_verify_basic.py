"""WSL Week-1 verification: ViZDoom loop + preprocessing (no torch needed)."""
import sys
from pathlib import Path as _P
for _cand in ("/mnt/c/Users/Admin/Desktop/SNN", "C:/Users/Admin/Desktop/SNN",
              str(_P(__file__).resolve().parents[1])):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)
import random
import numpy as np
from src.env.vizdoom_wrapper import create_game, get_action_table_for_scenario
from src.env.preprocessing import FrameStack, preprocess, stack_to_obs

def run_random(scenario="basic", episodes=3):
    actions = get_action_table_for_scenario(scenario)
    game = create_game(scenario)
    stacker = FrameStack(k=4, size=28)
    rewards = []
    for ep in range(episodes):
        game.new_episode()
        state = game.get_state()
        obs = stacker.reset(state.screen_buffer)
        assert obs.shape == (4, 28, 28), obs.shape
        # Validate both obs paths used by training.
        _ = stack_to_obs(obs, "mean")
        _ = stack_to_obs(obs, "stack")
        steps = 0
        while not game.is_episode_finished():
            a = random.randrange(len(actions))
            game.make_action(actions[a], 4)
            if not game.is_episode_finished():
                obs = stacker.step(game.get_state().screen_buffer)
            steps += 1
            if steps > 500:
                break
        total = game.get_total_reward()
        rewards.append(float(total))
        print(f"ep={ep} steps={steps} reward={total:.1f} actions={len(actions)}", flush=True)
    game.close()
    print(f"SCENARIO={scenario} mean_reward={np.mean(rewards):.2f} rewards={rewards}")
    return rewards

if __name__ == "__main__":
    print("=== basic ===")
    run_random("basic", episodes=3)
    print("=== defend_the_center ===")
    run_random("defend_the_center", episodes=2)
    print("=== health_gathering ===")
    run_random("health_gathering", episodes=2)
    print("WSL_VERIFY_OK")
