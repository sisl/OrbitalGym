"""orbital-game — JAX-native framework for guard/bandit POMDP research.

NOTE: At import time, we configure astrojax (and JAX) for float64 precision.
This is required because astrojax defaults to float32 internally even when
JAX's jax_enable_x64 flag is set; without this call, KOE/ECI conversions
produce ~7m residuals on Earth-orbit-scale problems. Power users who want
float32 for GPU throughput can call `astrojax.config.set_dtype(jnp.float32)`
after import.
"""

from __future__ import annotations

import jax.numpy as jnp
from astrojax.config import set_dtype as _set_astrojax_dtype

_set_astrojax_dtype(jnp.float64)

# Imports below intentionally follow the dtype configuration: orbital_game
# submodules import astrojax helpers, and we want them to see float64.
from orbital_game.config import ScenarioConfig, VehicleParamsSpec  # noqa: E402
from orbital_game.env.core import EnvState, OrbitalGameEnv  # noqa: E402
from orbital_game.logging.reader import load_run  # noqa: E402
from orbital_game.logging.writer import save_run  # noqa: E402
from orbital_game.reference_orbit import ReferenceOrbitState  # noqa: E402
from orbital_game.rollout import Trajectory, episode_mask, rollout  # noqa: E402

__all__ = [
    "EnvState",
    "OrbitalGameEnv",
    "ReferenceOrbitState",
    "ScenarioConfig",
    "Trajectory",
    "VehicleParamsSpec",
    "episode_mask",
    "load_run",
    "rollout",
    "save_run",
]
