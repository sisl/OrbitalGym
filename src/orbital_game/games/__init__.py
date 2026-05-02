"""Game catalog package — typed Game subtypes + builders."""

from __future__ import annotations

from orbital_game.games.base import Game, NoGame
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
    "make_lady_bandit_guard",
    "make_observation_blocking",
    "make_pursuit_evasion",
    "make_sun_blocking",
]

from orbital_game.registry import GameKey


def make_game(key: GameKey, **kwargs):
    """Dispatch to the per-game builder by `GameKey`.

    Example:
        cfg = make_game(GameKey.LADY_BANDIT_GUARD, breach_distance_m=15.0)
    """
    builders = {
        GameKey.LADY_BANDIT_GUARD: make_lady_bandit_guard,
        GameKey.PURSUIT_EVASION: make_pursuit_evasion,
        GameKey.SUN_BLOCKING: make_sun_blocking,
        GameKey.OBSERVATION_BLOCKING: make_observation_blocking,
    }
    if key not in builders:
        raise KeyError(f"No builder for {key!r}")
    return builders[key](**kwargs)
