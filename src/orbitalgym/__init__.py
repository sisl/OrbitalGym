"""OrbitalGym — JAX-native framework for guard/bandit POMDP research.

NOTE: At import time, we pin two defaults that make the package work
out-of-the-box on every JAX backend (CPU, CUDA, MPS):

1. **Backend: CPU** — we set ``JAX_PLATFORMS=cpu`` via ``os.environ.setdefault``
   *before* importing JAX. This is a no-op when the user (or another module)
   already set the env var or has already imported JAX, so power users
   who want CUDA or MPS keep their explicit choice. Targeting CPU by default
   avoids surprise selection of the experimental ``jax-mps`` backend on
   Apple Silicon when it happens to be installed.

2. **Precision: float32** — we call ``set_precision(jnp.float32)`` so
   ``jax_enable_x64`` is OFF and astrojax's internal dtype is float32. This
   keeps Python-scalar promotion (``jnp.deg2rad(15.0)``) on float32 so MPS
   device-puts succeed.

   Trade-off: astrojax's KOE/ECI conversions hit a ~7 m precision floor in
   float32 on Earth-orbit-scale problems. Code that uses ``KEPLERIAN_ECI``
   or full-force ``ASTROJAX_*`` dynamics — or any path that requires
   sub-meter orbit accuracy — must opt into float64::

       import jax.numpy as jnp
       import orbitalgym

       orbitalgym.set_precision(jnp.float64)   # before building any env
       env = orbitalgym.OrbitalGymEnv(cfg)

   Pure HCW relative-motion dynamics (``HCW_RT``, ``HCW_RTN``) are fine in
   float32; the in-package HCW closed form has no precision floor.

Public API surface:

    Core types:
        ScenarioConfig, VehicleParamsSpec, ReferenceOrbitState
        OrbitalGymEnv, EnvState, SingleAgentView
        Side, BySide, Actions, SideOutput, StepOutput, SideTrajectory, Trajectory

    Rollout + logging:
        rollout, rollout_single_agent, episode_mask
        save_run, load_run

    Game catalog (typed knob bundles + builders):
        Game, NoGame, GameKey
        LadyBanditGuard, PursuitEvasion, SunBlocking, ObservationBlocking,
        GetOffMyLawn
        make_lady_bandit_guard, make_pursuit_evasion, make_sun_blocking,
        make_observation_blocking, make_get_off_my_lawn, make_game

    Adapters (see orbitalgym.adapters):
        POMDPAdapter, GymnasiumAdapter, PettingZooAdapter
"""

from __future__ import annotations

import os

# Pin CPU as the default JAX backend before JAX imports. ``setdefault`` keeps
# any pre-existing env var (e.g. ``JAX_PLATFORMS=cuda`` or ``cpu,mps``) the
# user or shell has set. This only takes effect if orbitalgym is imported
# *before* JAX; if JAX is already imported, this is a no-op.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from astrojax.config import set_dtype as _set_astrojax_dtype  # noqa: E402


def set_precision(dtype) -> None:
    """Set both astrojax's internal dtype and JAX's `jax_enable_x64` flag.

    `astrojax.config.set_dtype(jnp.float64)` enables `jax_enable_x64` as a side
    effect, but `set_dtype(jnp.float32)` does NOT toggle it back. This helper
    keeps the two settings consistent so callers can flip precision with a
    single call. Call ``set_precision(jnp.float64)`` for sub-mm orbit precision
    (required by KOE/ECI conversions inside ``KEPLERIAN_ECI`` and full-force
    astrojax dynamics); the package default is ``jnp.float32`` for backend
    portability.

    No package-level cache invalidation is needed: typed-instance dynamics
    (`KeplerianEciDynamics`, `AstrojaxOrbitDynamics`) build their Epoch / RHS
    state in ``__post_init__``, so a fresh dtype is picked up the next time
    a new env is constructed. Existing instances retain their original dtype
    — flip the dtype *before* building the env you want to run on the new
    precision.
    """
    _set_astrojax_dtype(dtype)
    jax.config.update("jax_enable_x64", dtype is jnp.float64 or dtype == jnp.float64)


# Default to float32 + x64 OFF. Users who need orbit-grade precision call
# ``set_precision(jnp.float64)`` after import (see the module docstring).
set_precision(jnp.float32)

# Imports below intentionally follow the dtype configuration: orbitalgym
# submodules import astrojax helpers, and we want them to see float64.
from orbitalgym import adapters  # noqa: E402
from orbitalgym.config import ScenarioConfig, VehicleParamsSpec  # noqa: E402
from orbitalgym.env.core import EnvState, OrbitalGymEnv  # noqa: E402
from orbitalgym.env.single_agent import SingleAgentView  # noqa: E402
from orbitalgym.env.types import (  # noqa: E402
    Actions,
    BySide,
    Side,
    SideOutput,
    SideTrajectory,
    StepOutput,
    Trajectory,
)
from orbitalgym.games import (  # noqa: E402
    Game,
    GetOffMyLawn,
    LadyBanditGuard,
    NoGame,
    ObservationBlocking,
    PursuitEvasion,
    SunBlocking,
    make_game,
    make_get_off_my_lawn,
    make_lady_bandit_guard,
    make_observation_blocking,
    make_pursuit_evasion,
    make_sun_blocking,
)
from orbitalgym.logging.reader import load_run  # noqa: E402
from orbitalgym.logging.writer import save_run  # noqa: E402
from orbitalgym.reference_orbit import ReferenceOrbitState  # noqa: E402
from orbitalgym.registry import GameKey  # noqa: E402
from orbitalgym.rollout import (  # noqa: E402
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
    "GetOffMyLawn",
    "GymnasiumAdapter",
    "LadyBanditGuard",
    "NoGame",
    "ObservationBlocking",
    "OrbitalGymEnv",
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
    "make_get_off_my_lawn",
    "make_lady_bandit_guard",
    "make_observation_blocking",
    "make_pursuit_evasion",
    "make_sun_blocking",
    "rollout",
    "rollout_single_agent",
    "save_run",
]
