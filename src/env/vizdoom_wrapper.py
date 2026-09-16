"""ViZDoom wrapper. vizdoom import is lazy so smoke tests run without it.

IMPORTANT: run training under WSL2/Linux or Colab. Native Windows ViZDoom
wheels are unreliable — tools/check_env.py will warn you.
"""
from __future__ import annotations


BASIC_ACTIONS = [
    [True, False, False],   # MOVE_LEFT
    [False, True, False],   # MOVE_RIGHT
    [False, False, True],   # ATTACK
]

# deadly_corridor.cfg exposes 7 buttons (move + turn, no strafe-fire combo).
# Single-button action table keeps the mapping positional and explicit.
CORRIDOR_ACTIONS = [
    [True, False, False, False, False, False, False],   # MOVE_LEFT
    [False, True, False, False, False, False, False],   # MOVE_RIGHT
    [False, False, True, False, False, False, False],   # ATTACK
    [False, False, False, True, False, False, False],   # MOVE_FORWARD
    [False, False, False, False, True, False, False],   # MOVE_BACKWARD
    [False, False, False, False, False, True, False],   # TURN_LEFT
    [False, False, False, False, False, False, True],   # TURN_RIGHT
]

ACTION_TABLES = {3: BASIC_ACTIONS, 7: CORRIDOR_ACTIONS}


def get_action_table(n_actions: int):
    try:
        return ACTION_TABLES[n_actions]
    except KeyError:
        raise ValueError(f"no action table for {n_actions} buttons (have {sorted(ACTION_TABLES)})")

SCENARIO_FILES = {
    "basic": ("basic.wad", "basic.cfg"),
    "defend_the_center": ("defend_the_center.wad", "defend_the_center.cfg"),
    "health_gathering": ("health_gathering.wad", "health_gathering.cfg"),
    "deadly_corridor": ("deadly_corridor.wad", "deadly_corridor.cfg"),
}


def create_game(scenario: str = "basic", visible: bool = False, sound: bool = False,
                episode_timeout: int = 300):
    try:
        import vizdoom as vzd
    except ImportError as e:
        raise ImportError(
            "vizdoom is not installed here. Install with `pip install vizdoom` "
            "under WSL2/Linux or Colab (see requirements.txt). "
            f"Original error: {e}"
        ) from e

    game = vzd.DoomGame()
    wad, cfg_name = SCENARIO_FILES.get(scenario, ("basic.wad", "basic.cfg"))
    import os
    scenarios_dir = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
    # Load the scenario .cfg first: it defines available buttons, rewards,
    # game variables and episode timeout. Without it, make_action() is a
    # no-op and get_last_reward() is always 0.
    cfg_path = os.path.join(scenarios_dir, cfg_name)
    if os.path.isfile(cfg_path):
        game.load_config(cfg_path)
    # Absolute WAD path (the cfg only has a relative one).
    wad_path = os.path.join(scenarios_dir, wad)
    if os.path.isfile(wad_path):
        game.set_doom_scenario_path(wad_path)
    n_buttons = len(game.get_available_buttons())
    if n_buttons != 3:
        print(f"[wrapper] note: scenario {scenario!r} exposes {n_buttons} buttons; "
              f"BASIC_ACTIONS covers the first 3 (MOVE_LEFT, MOVE_RIGHT, ATTACK).")
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_episode_timeout(episode_timeout)
    game.set_window_visible(visible)
    game.set_sound_enabled(sound)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    return game
