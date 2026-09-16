"""WSL Week-1 verification: ViZDoom loop + preprocessing (no torch needed)."""
import sys
sys.path.insert(0, "/mnt/c/Users/Admin/Desktop/SNN")
import random
import numpy as np
from src.env.vizdoom_wrapper import create_game, BASIC_ACTIONS
from src.env.preprocessing import FrameStack, preprocess

def run_random(scenario="basic", episodes=3):
    game = create_game(scenario)
    stacker = FrameStack(k=4, size=28)
    rewards = []
    for ep in range(episodes):
        game.new_episode()
        state = game.get_state()
        obs = stacker.reset(state.screen_buffer)
        assert obs.shape == (4, 28, 28), obs.shape
        steps = 0
        while not game.is_episode_finished():
            a = random.randrange(len(BASIC_ACTIONS))
            game.make_action(BASIC_ACTIONS[a], 4)
            if not game.is_episode_finished():
                obs = stacker.step(game.get_state().screen_buffer)
            steps += 1
            if steps > 500:
                break
        total = game.get_total_reward()
        rewards.append(float(total))
        print(f"ep={ep} steps={steps} reward={total:.1f}", flush=True)
    game.close()
    print(f"SCENARIO={scenario} mean_reward={np.mean(rewards):.2f} rewards={rewards}")
    return rewards

if __name__ == "__main__":
    print("=== basic ===")
    run_random("basic", episodes=3)
    print("=== defend_the_center ===")
    run_random("defend_the_center", episodes=2)
    print("WSL_VERIFY_OK")
