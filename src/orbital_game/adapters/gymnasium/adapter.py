"""GymnasiumAdapter — wraps SingleAgentView in a gymnasium.Env.

The adapter holds env state + opponent policy state as mutable Python
attributes, materializes JAX arrays as numpy at the boundary, and infers
Box action/observation spaces from the wrapped env.

Action / observation spaces:
    - For PER_SIDE-scope obs (default), the obs is a 1-D Box with bounds
      [-inf, +inf] and shape (obs_dim,).
    - For PER_VEHICLE-scope obs, the leading vehicle axis is flattened
      into a single Box (centralized-controller framing).
    - The action is a 1-D Box sized by `command_flat_dim(command_cls)` —
      the concatenation of every Command-component field, flattened.
      Adapters round-trip via `unflatten_command` before handing the
      Command pytree to the env.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from orbital_game.adapters._command_flatten import command_flat_dim, unflatten_command
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.single_agent import SingleAgentView
from orbital_game.env.types import Side


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

        controlled_command_cls = (
            env.guard_command_cls
            if env.config.controlled_side is Side.GUARD
            else env.bandit_command_cls
        )
        self._controlled_command_cls = controlled_command_cls
        action_flat_dim = command_flat_dim(controlled_command_cls)
        self._action_flat_dim = action_flat_dim

        # Probe a reset to discover observation shape. SingleAgentView.reset
        # already returns a flattened jax.Array.
        probe_state, probe_obs, _ps = self.view.reset(jax.random.PRNGKey(0))
        obs_shape = tuple(probe_obs.shape)

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(action_flat_dim,),
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
        action_flat = jnp.asarray(action, dtype=jnp.float32).reshape(self._action_flat_dim)
        controlled_command = unflatten_command(self._controlled_command_cls, action_flat)
        self._rng_key, k_step = jax.random.split(self._rng_key, 2)
        next_state, next_obs, reward, done, next_opp_ps, info = self.view.step(
            k_step, self._state, controlled_command, self._opp_ps
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
