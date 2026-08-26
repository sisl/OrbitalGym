"""PettingZooAdapter — wraps OrbitalGymEnv as a pettingzoo.ParallelEnv.

Agent IDs: 'guard_0', 'guard_1', ..., 'bandit_0', 'bandit_1', ...
Each agent sees its own row of the per-pair observation tensor.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from orbitalgym.adapters._command_flatten import command_flat_dim, unflatten_command
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide
from orbitalgym.observations.types import flatten_observations_per_agent


class PettingZooAdapter(ParallelEnv):
    """Multi-agent adapter using pettingzoo's Parallel API.

    Each vehicle is a separate agent. Agent IDs are `<side>_<index>`
    (e.g. 'guard_0', 'bandit_0'). All agents step simultaneously.
    """

    metadata = {"render_modes": [], "name": "orbitalgym_v0"}

    def __init__(self, env: OrbitalGymEnv, *, seed: int = 0):
        self.env = env
        self._seed = seed
        self._rng_key = jax.random.PRNGKey(seed)
        self._state: Any = None

        n_g = env.config.n_guards
        n_b = env.config.n_bandits
        self._n_guards = n_g
        self._n_bandits = n_b

        # Per-side Command pytree classes — built by env from the configured
        # action components. Each agent's flat action is a slice of the
        # side's flat layout (per_agent_dim = side_flat_dim // n_side).
        self._guard_command_cls = env.guard_command_cls
        self._bandit_command_cls = env.bandit_command_cls
        self._guard_side_flat_dim = command_flat_dim(self._guard_command_cls)
        self._bandit_side_flat_dim = command_flat_dim(self._bandit_command_cls)
        # Per-agent flat dim = side flat dim / n_side. Component fields are
        # all leading-axis n_side, so this division is exact.
        #
        # Note on multi-component layout: per-agent flat actions are
        # concatenated across agents in `step`, then `unflatten_command`
        # reconstitutes the side's Command. `unflatten_command` uses
        # field-major layout (all-agents-of-field-1, then all-agents-of-field-2),
        # which matches the natural concat order only when each side has at most
        # one Command field. Currently every configured scenario uses
        # ImpulsiveManeuver only (single `dv` field), so the ordering coincides.
        # Multi-component PettingZoo support will need an agent-major <->
        # field-major reshuffle at the per-agent boundary.
        guard_per_agent_dim = self._guard_side_flat_dim // n_g
        bandit_per_agent_dim = self._bandit_side_flat_dim // n_b

        self._guard_ids = [f"guard_{i}" for i in range(n_g)]
        self._bandit_ids = [f"bandit_{i}" for i in range(n_b)]
        self.possible_agents = list(self._guard_ids + self._bandit_ids)
        self.agents = list(self.possible_agents)

        # Discover per-agent obs shapes via a probe reset.
        probe_state, probe_outputs = env.reset(jax.random.PRNGKey(0))
        guard_per_agent = flatten_observations_per_agent(probe_outputs.guard.obs)
        bandit_per_agent = flatten_observations_per_agent(probe_outputs.bandit.obs)

        self._guard_obs_shape = (guard_per_agent.shape[-1],)
        self._bandit_obs_shape = (bandit_per_agent.shape[-1],)

        guard_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._guard_obs_shape, dtype=np.float32
        )
        bandit_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._bandit_obs_shape, dtype=np.float32
        )
        guard_action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(guard_per_agent_dim,), dtype=np.float32
        )
        bandit_action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(bandit_per_agent_dim,), dtype=np.float32
        )

        self.observation_spaces = {
            **{aid: guard_obs_space for aid in self._guard_ids},
            **{aid: bandit_obs_space for aid in self._bandit_ids},
        }
        self.action_spaces = {
            **{aid: guard_action_space for aid in self._guard_ids},
            **{aid: bandit_action_space for aid in self._bandit_ids},
        }

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
        # Per-agent flat actions stack to a side-flat vector, then unflatten
        # into the side's Command pytree.
        guard_flat = jnp.concatenate(
            [jnp.asarray(actions[aid], dtype=jnp.float32).reshape(-1) for aid in self._guard_ids]
        )
        bandit_flat = jnp.concatenate(
            [jnp.asarray(actions[aid], dtype=jnp.float32).reshape(-1) for aid in self._bandit_ids]
        )
        guard_command = unflatten_command(self._guard_command_cls, guard_flat)
        bandit_command = unflatten_command(self._bandit_command_cls, bandit_flat)
        env_actions = Actions(sides=BySide(guard=guard_command, bandit=bandit_command))
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
        guard_per_agent = flatten_observations_per_agent(outputs.guard.obs)
        for i, aid in enumerate(self._guard_ids):
            d[aid] = np.asarray(guard_per_agent[i], dtype=np.float32)
        bandit_per_agent = flatten_observations_per_agent(outputs.bandit.obs)
        for i, aid in enumerate(self._bandit_ids):
            d[aid] = np.asarray(bandit_per_agent[i], dtype=np.float32)
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
