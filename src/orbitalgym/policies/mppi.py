"""MPPIPolicy: model-predictive path integral control over continuous delta-v.

Samples ``n_samples`` control sequences around a warm-started mean, rolls each
through the adapter with the opponent modeled by ``opponent_model``, scores
them by negative discounted side reward (plus an optional discounted terminal
value), standardizes the costs by their spread, and updates the mean with
softmax weights at ``temperature`` applied to the standardized costs. The mean
sequence is the policy state, shifted by one step after each call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.adapters._command_flatten import flatten_command
from orbitalgym.belief.flatten import belief_mean_to_flat_state
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.observations.types import flatten_observations
from orbitalgym.registry import PolicyKey, register


def _clip_norm(u: jax.Array, dv_max: float) -> jax.Array:
    """Scale each (..., dv_dim) vector down to magnitude ``dv_max`` when larger."""
    norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, dv_max / jnp.maximum(norm, 1e-12))
    return u * scale


@register(PolicyKey.MPPI)
@dataclass(frozen=True)
class MPPIPolicy:
    """Sampling-based receding-horizon planner conforming to the Policy protocol."""

    env_model: Any
    side: Side
    opponent_model: Any
    n_samples: int = 256
    horizon: int = 10
    temperature: float = 1.0
    noise_sigma: float = 0.1
    dv_max: float = 0.5
    terminal_value_fn: Callable[[jax.Array], jax.Array] | None = None
    # Per-planning-step discount. ``None`` reads ``env_model.discount()``,
    # the adapter's per-macro-step discount.
    discount: float | None = None
    n_vehicles: int = 0
    command_cls: Any = None
    template_env_state: Any = None

    def __post_init__(self) -> None:
        if self.command_cls is None or self.n_vehicles == 0:
            raise ValueError("MPPIPolicy requires command_cls and n_vehicles > 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

    @property
    def dv_dim(self) -> int:
        return int(self.command_cls.zeros(self.n_vehicles).dv.shape[-1])

    def init_state(self) -> jax.Array:
        return jnp.zeros((self.horizon, self.n_vehicles, self.dv_dim), dtype=jnp.float32)

    def _flat_state(self, agent_view: Any) -> jax.Array:
        if isinstance(agent_view, jax.Array):
            return agent_view
        belief = agent_view.inner if hasattr(agent_view, "inner") else agent_view
        mean = belief.mean
        if mean.ndim == 1:
            return mean
        if self.template_env_state is None:
            raise ValueError("MPPIPolicy needs template_env_state to plan from a belief view")
        return belief_mean_to_flat_state(mean, self.side, self.env_model, self.template_env_state)

    def _rollout_cost(self, s0: jax.Array, u_seq: jax.Array, key: jax.Array) -> jax.Array:
        adapter = self.env_model
        env = adapter.env
        opp_side = self.side.opposite()
        opp_obs_fn = (
            env.guard_observation_fn if opp_side is Side.GUARD else env.bandit_observation_fn
        )
        opp_cls = env.guard_command_cls if opp_side is Side.GUARD else env.bandit_command_cls
        n_opp = env.config.n_guards if opp_side is Side.GUARD else env.config.n_bandits
        own_template = self.command_cls.zeros(self.n_vehicles)
        identity = Actions(
            sides=BySide(
                guard=env.guard_command_cls.zeros(env.config.n_guards),
                bandit=env.bandit_command_cls.zeros(env.config.n_bandits),
            )
        )

        def step(carry, inputs):
            (s,) = carry
            u, k = inputs
            k_opp, k_step = jax.random.split(k)
            state = adapter.unpack(s)
            opp_obs = opp_obs_fn(state, identity, opp_side, env.config, k_opp, state.t)
            opp_cmd, _ = self.opponent_model(None, flatten_observations(opp_obs), k_opp, state.t)
            own_cmd = own_template.replace(dv=u.astype(own_template.dv.dtype))
            opp_flat = flatten_command(opp_cls.zeros(n_opp).replace(dv=opp_cmd.dv))
            own_flat = flatten_command(own_cmd)
            a_flat = (
                jnp.concatenate([own_flat, opp_flat])
                if self.side is Side.GUARD
                else jnp.concatenate([opp_flat, own_flat])
            )
            s_next, r = adapter.step(s, a_flat, k_step, self.side)
            return (s_next,), r

        keys = jax.random.split(key, self.horizon)
        (s_h,), rewards = jax.lax.scan(step, (s0,), (u_seq, keys))
        gamma = adapter.discount() if self.discount is None else self.discount
        weights = jnp.asarray(gamma, rewards.dtype) ** jnp.arange(self.horizon, dtype=rewards.dtype)
        cost = -jnp.sum(weights * rewards)
        if self.terminal_value_fn is not None:
            cost = cost - jnp.asarray(
                gamma, rewards.dtype
            ) ** self.horizon * self.terminal_value_fn(s_h)
        return cost

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        del t
        s0 = self._flat_state(agent_view)
        u_mean = policy_state if policy_state is not None else self.init_state()
        k_noise, k_roll = jax.random.split(key)
        eps = self.noise_sigma * jax.random.normal(
            k_noise, (self.n_samples,) + u_mean.shape, u_mean.dtype
        )
        u = _clip_norm(u_mean[None] + eps, self.dv_max)  # (K, H, n, dv_dim)
        roll_keys = jax.random.split(k_roll, self.n_samples)
        costs = jax.vmap(lambda uk, kk: self._rollout_cost(s0, uk, kk))(u, roll_keys)
        centered = costs - jnp.min(costs)
        scale = jnp.maximum(jnp.std(costs), 1e-6)
        weights = jax.nn.softmax(-centered / (scale * self.temperature))
        u_new = jnp.einsum("k,khnd->hnd", weights, u).astype(u_mean.dtype)
        action = u_new[0]
        next_state = jnp.concatenate([u_new[1:], jnp.zeros_like(u_new[:1])], axis=0)
        cmd = self.command_cls.zeros(self.n_vehicles).replace(
            dv=action.astype(self.command_cls.zeros(1).dv.dtype)
        )
        return cmd, next_state
