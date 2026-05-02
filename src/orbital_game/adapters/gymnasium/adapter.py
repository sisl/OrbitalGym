"""GymnasiumAdapter — wraps SingleAgentView in a gymnasium.Env.

The adapter holds env state + opponent policy state as mutable Python
attributes, materializes JAX arrays as numpy at the boundary, and infers
Box action/observation spaces from the wrapped env.

Action / observation spaces:
    - For PER_SIDE-scope obs (default), the obs is a 1-D Box with bounds
      [-inf, +inf] and shape (obs_dim,).
    - For PER_VEHICLE-scope obs, the leading vehicle axis is flattened
      into a single Box (centralized-controller framing).
    - The action is shaped (n_controlled_vehicles, action_dim) flattened
      to a 1-D Box for the same reason.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.single_agent import SingleAgentView
from orbital_game.env.types import Side


def _action_dim_from_dynamics(dynamics_key) -> int:
    """Mirror env/core.py's _dyn_action_dim — HCW_RT → 2, HCW_RTN → 3."""
    from orbital_game.registry import DynamicsKey

    return 2 if dynamics_key is DynamicsKey.HCW_RT else 3


class GymnasiumAdapter(gym.Env):
    """A gymnasium.Env wrapping a SingleAgentView."""

    metadata = {"render_modes": []}

    def __init__(self, env: OrbitalGameEnv, *, seed: int = 0):
        self.env = env
        self.view = SingleAgentView(env)
        self._seed = seed
        self._rng_key = jax.random.PRNGKey(seed)
        self._state: Any = None
        self._opp_ps: Any = None

        controlled = self.view.controlled_side
        n_controlled = env.config.n_guards if controlled is Side.GUARD else env.config.n_bandits
        action_dim = _action_dim_from_dynamics(env.config.truth_dynamics)
        self._n_controlled = n_controlled
        self._action_dim = action_dim

        # Probe a reset to discover observation shape.
        probe_state, probe_obs, _ps = self.view.reset(jax.random.PRNGKey(0))
        obs_shape = tuple(probe_obs.shape)

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_controlled * action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        # Call super().reset to satisfy gymnasium's np_random initialization contract.
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._rng_key = jax.random.PRNGKey(seed)
        self._rng_key, k_reset = jax.random.split(self._rng_key, 2)
        state, obs, opp_ps = self.view.reset(k_reset)
        self._state = state
        self._opp_ps = opp_ps
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        action_arr = jnp.asarray(action, dtype=jnp.float32).reshape(
            self._n_controlled, self._action_dim
        )
        self._rng_key, k_step = jax.random.split(self._rng_key, 2)
        next_state, next_obs, reward, done, next_opp_ps, info = self.view.step(
            k_step, self._state, action_arr, self._opp_ps
        )
        self._state = next_state
        self._opp_ps = next_opp_ps
        terminated = bool(done)
        truncated = False
        return (
            np.asarray(next_obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            dict(info) if info else {},
        )
