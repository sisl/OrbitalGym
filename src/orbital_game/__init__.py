"""orbital-game — JAX-native framework for guard/bandit POMDP research.

NOTE: At import time, we configure astrojax (and JAX) for float64 precision.
This is required because astrojax defaults to float32 internally even when
JAX's jax_enable_x64 flag is set; without this call, KOE/ECI conversions
produce ~7m residuals on Earth-orbit-scale problems. Power users who want
float32 for GPU throughput can call `astrojax.config.set_dtype(jnp.float32)`
after import.

Public API surface:

    Core types:
        ScenarioConfig, VehicleParamsSpec, ReferenceOrbitState
        OrbitalGameEnv, EnvState, SingleAgentView
        Side, BySide, Actions, SideOutput, StepOutput, SideTrajectory, Trajectory

    Rollout + logging:
        rollout, rollout_single_agent, episode_mask
        save_run, load_run

    Game catalog (typed knob bundles + builders):
        Game, NoGame, GameKey
        LadyBanditGuard, PursuitEvasion, SunBlocking, ObservationBlocking
        make_lady_bandit_guard, make_pursuit_evasion, make_sun_blocking,
        make_observation_blocking, make_game

    Adapters (see orbital_game.adapters):
        POMDPAdapter, GymnasiumAdapter, PettingZooAdapter
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from astrojax.config import set_dtype as _set_astrojax_dtype


def set_precision(dtype) -> None:
    """Set both astrojax's internal dtype and JAX's `jax_enable_x64` flag.

    `astrojax.config.set_dtype(jnp.float64)` enables `jax_enable_x64` as a side
    effect, but `set_dtype(jnp.float32)` does NOT toggle it back. This helper
    keeps the two settings consistent so callers can flip precision with a
    single call. Use `set_precision(jnp.float32)` when targeting MPS (which is
    float32-only) or for GPU throughput; use `set_precision(jnp.float64)` for
    sub-mm orbit precision (the package default at import time).

    No package-level cache invalidation is needed: typed-instance dynamics
    (`KeplerianEciDynamics`, `AstrojaxOrbitDynamics`) build their Epoch / RHS
    state in ``__post_init__``, so a fresh dtype is picked up the next time
    a new env is constructed. Existing instances retain their original dtype
    — flip the dtype *before* building the env you want to run on the new
    precision.
    """
    _set_astrojax_dtype(dtype)
    jax.config.update("jax_enable_x64", dtype is jnp.float64 or dtype == jnp.float64)


_set_astrojax_dtype(jnp.float64)

# Imports below intentionally follow the dtype configuration: orbital_game
# submodules import astrojax helpers, and we want them to see float64.
from orbital_game import adapters  # noqa: E402
from orbital_game.config import ScenarioConfig, VehicleParamsSpec  # noqa: E402
from orbital_game.env.core import EnvState, OrbitalGameEnv  # noqa: E402
from orbital_game.env.single_agent import SingleAgentView  # noqa: E402
from orbital_game.env.types import (  # noqa: E402
    Actions,
    BySide,
    Side,
    SideOutput,
    SideTrajectory,
    StepOutput,
    Trajectory,
)
from orbital_game.games import (  # noqa: E402
    Game,
    LadyBanditGuard,
    NoGame,
    ObservationBlocking,
    PursuitEvasion,
    SunBlocking,
    make_game,
    make_lady_bandit_guard,
    make_observation_blocking,
    make_pursuit_evasion,
    make_sun_blocking,
)
from orbital_game.logging.reader import load_run  # noqa: E402
from orbital_game.logging.writer import save_run  # noqa: E402
from orbital_game.reference_orbit import ReferenceOrbitState  # noqa: E402
from orbital_game.registry import GameKey  # noqa: E402
from orbital_game.rollout import (  # noqa: E402
    episode_mask,
    rollout,
    rollout_single_agent,
)

POMDPAdapter = adapters.POMDPAdapter
GymnasiumAdapter = adapters.GymnasiumAdapter
PettingZooAdapter = adapters.PettingZooAdapter

__all__ = [
    "Actions",
    "BySide",
    "EnvState",
    "Game",
    "set_precision",
    "GameKey",
    "GymnasiumAdapter",
    "LadyBanditGuard",
    "NoGame",
    "ObservationBlocking",
    "OrbitalGameEnv",
    "POMDPAdapter",
    "PettingZooAdapter",
    "PursuitEvasion",
    "ReferenceOrbitState",
    "ScenarioConfig",
    "Side",
    "SideOutput",
    "SideTrajectory",
    "SingleAgentView",
    "StepOutput",
    "SunBlocking",
    "Trajectory",
    "VehicleParamsSpec",
    "adapters",
    "episode_mask",
    "load_run",
    "make_game",
    "make_lady_bandit_guard",
    "make_observation_blocking",
    "make_pursuit_evasion",
    "make_sun_blocking",
    "rollout",
    "rollout_single_agent",
    "save_run",
]
