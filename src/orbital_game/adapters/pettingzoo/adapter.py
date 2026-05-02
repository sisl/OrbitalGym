"""PettingZooAdapter — wraps OrbitalGameEnv as a pettingzoo.ParallelEnv.

Agent IDs: 'guard_0', 'guard_1', ..., 'bandit_0', 'bandit_1', ...
Per-vehicle slicing for PER_VEHICLE scope; broadcast for PER_SIDE scope.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide


def _action_dim_from_dynamics(dynamics_key) -> int:
    from orbital_game.registry import DynamicsKey

    return 2 if dynamics_key is DynamicsKey.HCW_RT else 3


class PettingZooAdapter(ParallelEnv):
    """Multi-agent adapter using pettingzoo's Parallel API.

    Each vehicle is a separate agent. Agent IDs are `<side>_<index>`
    (e.g. 'guard_0', 'bandit_0'). All agents step simultaneously.
    """

    metadata = {"render_modes": [], "name": "orbital_game_v0"}

    def __init__(self, env: OrbitalGameEnv, *, seed: int = 0):
        self.env = env
        self._seed = seed
        self._rng_key = jax.random.PRNGKey(seed)
        self._state: Any = None

        n_g = env.config.n_guards
        n_b = env.config.n_bandits
        action_dim = _action_dim_from_dynamics(env.config.truth_dynamics)
        self._action_dim = action_dim
        self._n_guards = n_g
        self._n_bandits = n_b

        self._guard_ids = [f"guard_{i}" for i in range(n_g)]
        self._bandit_ids = [f"bandit_{i}" for i in range(n_b)]
        self.possible_agents = list(self._guard_ids + self._bandit_ids)
        self.agents = list(self.possible_agents)

        # Discover obs shapes via a probe reset.
        probe_state, probe_outputs = env.reset(jax.random.PRNGKey(0))
        guard_obs = probe_outputs.guard.obs
        bandit_obs = probe_outputs.bandit.obs

        self._guard_obs_shape = self._per_agent_obs_shape(guard_obs, n_g)
        self._bandit_obs_shape = self._per_agent_obs_shape(bandit_obs, n_b)

        guard_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._guard_obs_shape, dtype=np.float32
        )
        bandit_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._bandit_obs_shape, dtype=np.float32
        )
        action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)

        self.observation_spaces = {
            **{aid: guard_obs_space for aid in self._guard_ids},
            **{aid: bandit_obs_space for aid in self._bandit_ids},
        }
        self.action_spaces = {aid: action_space for aid in self.possible_agents}

    @staticmethod
    def _per_agent_obs_shape(side_obs: jax.Array, n_side: int) -> tuple[int, ...]:
        """Determine the per-agent obs shape from a side's obs."""
        if side_obs.ndim >= 1 and side_obs.shape[0] == n_side and n_side > 1:
            return tuple(side_obs.shape[1:])
        return tuple(side_obs.shape)

    @staticmethod
    def _slice_or_broadcast(side_obs: jax.Array, n_side: int, idx: int) -> jax.Array:
        """Slice the per-side obs to a per-agent obs (or broadcast for PER_SIDE)."""
        if side_obs.ndim >= 1 and side_obs.shape[0] == n_side and n_side > 1:
            return side_obs[idx]
        return side_obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = seed
            self._rng_key = jax.random.PRNGKey(seed)
        self._rng_key, k_reset = jax.random.split(self._rng_key, 2)
        state, outputs = self.env.reset(k_reset)
        self._state = state
        self.agents = list(self.possible_agents)
        obs_dict = self._build_obs_dict(outputs)
        info_dict = {aid: {} for aid in self.possible_agents}
        return obs_dict, info_dict

    def step(self, actions: dict):
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        guard_actions = jnp.stack(
            [jnp.asarray(actions[aid], dtype=jnp.float32) for aid in self._guard_ids]
        )
        bandit_actions = jnp.stack(
            [jnp.asarray(actions[aid], dtype=jnp.float32) for aid in self._bandit_ids]
        )
        env_actions = Actions(sides=BySide(guard=guard_actions, bandit=bandit_actions))
        self._rng_key, k_step = jax.random.split(self._rng_key, 2)
        step_out = self.env.step(k_step, self._state, env_actions)
        self._state = step_out.state

        obs_dict = self._build_obs_dict(step_out.outputs)

        reward_dict = {}
        for i, aid in enumerate(self._guard_ids):
            reward_dict[aid] = self._scalar_reward(step_out.outputs.guard.reward, self._n_guards, i)
        for i, aid in enumerate(self._bandit_ids):
            reward_dict[aid] = self._scalar_reward(
                step_out.outputs.bandit.reward, self._n_bandits, i
            )

        episode_done = bool(step_out.episode_done)
        terminated_dict = {aid: episode_done for aid in self.possible_agents}
        truncated_dict = {aid: False for aid in self.possible_agents}
        info_dict = {aid: {} for aid in self.possible_agents}

        if episode_done:
            self.agents = []

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def _build_obs_dict(self, outputs: BySide) -> dict:
        d = {}
        guard_obs = outputs.guard.obs
        for i, aid in enumerate(self._guard_ids):
            d[aid] = np.asarray(
                self._slice_or_broadcast(guard_obs, self._n_guards, i),
                dtype=np.float32,
            )
        bandit_obs = outputs.bandit.obs
        for i, aid in enumerate(self._bandit_ids):
            d[aid] = np.asarray(
                self._slice_or_broadcast(bandit_obs, self._n_bandits, i),
                dtype=np.float32,
            )
        return d

    @staticmethod
    def _scalar_reward(side_reward: jax.Array, n_side: int, idx: int) -> float:
        """PER_SIDE reward: shape (); PER_VEHICLE: shape (n_side,). Either way -> float."""
        if side_reward.ndim == 0:
            return float(side_reward)
        return float(side_reward[idx])

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
