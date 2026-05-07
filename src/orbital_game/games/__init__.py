"""Game catalog package — typed Game subtypes + builders."""

from __future__ import annotations

from orbital_game.games.base import Game, NoGame
from orbital_game.games.get_off_my_lawn import (
    GetOffMyLawn,
    GetOffMyLawnReward,
    GetOffMyLawnTermination,
    make_get_off_my_lawn,
)
from orbital_game.games.lady_bandit_guard import LadyBanditGuard, make_lady_bandit_guard
from orbital_game.games.observation_blocking import (
    ObservationBlocking,
    ObservationBlockingReward,
    make_observation_blocking,
)
from orbital_game.games.pursuit_evasion import (
    PursuitEvasion,
    PursuitEvasionReward,
    PursuitEvasionTermination,
    make_pursuit_evasion,
)
from orbital_game.games.sun_blocking import (
    SunBlocking,
    SunBlockingReward,
    make_sun_blocking,
)

__all__ = [
    "Game",
    "GetOffMyLawn",
    "GetOffMyLawnReward",
    "GetOffMyLawnTermination",
    "LadyBanditGuard",
    "NoGame",
    "ObservationBlocking",
    "ObservationBlockingReward",
    "PursuitEvasion",
    "PursuitEvasionReward",
    "PursuitEvasionTermination",
    "SunBlocking",
    "SunBlockingReward",
    "make_game",
    "make_get_off_my_lawn",
    "make_lady_bandit_guard",
    "make_observation_blocking",
    "make_pursuit_evasion",
    "make_sun_blocking",
]

from orbital_game.registry import GameKey


def make_game(key: GameKey, **kwargs):
    """Dispatch to the per-game builder by `GameKey`.

    Example:
        cfg = make_game(GameKey.LADY_BANDIT_GUARD, breach_radius_m=5.0, catch_radius_m=50.0)
    """
    builders = {
        GameKey.LADY_BANDIT_GUARD: make_lady_bandit_guard,
        GameKey.PURSUIT_EVASION: make_pursuit_evasion,
        GameKey.SUN_BLOCKING: make_sun_blocking,
        GameKey.OBSERVATION_BLOCKING: make_observation_blocking,
        GameKey.GET_OFF_MY_LAWN: make_get_off_my_lawn,
    }
    if key not in builders:
        raise KeyError(f"No builder for {key!r}")
    return builders[key](**kwargs)
