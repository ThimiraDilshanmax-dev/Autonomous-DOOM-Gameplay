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

SCENARIO_FILES = {
    "basic": "basic.wad",
    "defend_the_center": "defend_the_center.wad",
    "health_gathering": "health_gathering.wad",
    "deadly_corridor": "deadly_corridor.wad",
}


def create_game(scenario: str = "basic", visible: bool = False, sound: bool = False):
    try:
        import vizdoom as vzd
    except ImportError as e:
        raise ImportError(
            "vizdoom is not installed here. Install with `pip install vizdoom` "
            "under WSL2/Linux or Colab (see requirements.txt). "
            f"Original error: {e}"
        ) from e

    game = vzd.DoomGame()
    wad = SCENARIO_FILES.get(scenario, "basic.wad")
    try:
        import os
        scenarios_dir = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
        wad_path = os.path.join(scenarios_dir, wad)
        game.set_doom_scenario_path(wad_path)
    except Exception:
        pass  # let ViZDoom raise a clear error if missing
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_episode_timeout(300)
    game.set_window_visible(visible)
    game.set_sound_enabled(sound)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    return game
