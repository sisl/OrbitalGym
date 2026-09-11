"""MPPIPolicy: model-predictive path integral control over continuous delta-v.

Samples ``n_samples`` control sequences around a warm-started mean, rolls each
through the adapter with the opponent modeled by ``opponent_model``, scores
them by negative discounted side reward (plus an optional discounted terminal
value), standardizes the costs by their spread, and updates the mean with
softmax weights at ``temperature`` applied to the standardized costs. The mean
sequence is the policy state, shifted by one step after each call.

Multi-vehicle teams (``n_vehicles > 1``):

- ``coordination="joint"`` (default) — one sampler over the whole fleet's
  Delta-v sequence, shape ``(horizon, n_vehicles, dv_dim)``. Every sample
  perturbs all vehicles at once, so the weighted mean is a coordinated plan.
- ``coordination="independent"`` — one sampler per own vehicle, vmapped over
  the fleet. Vehicle ``i`` perturbs only its own ``(horizon, dv_dim)``
  sequence; its teammates' Delta-v at every rollout step comes from
  ``teammate_model``, the same-side analogue of ``opponent_model``. The
  per-vehicle first actions are stacked into one fleet command. Sampling cost
  is linear in ``n_vehicles`` and each vehicle keeps its own warm start (its
  slice of the shared mean array), at the price of no coordination between
  teammates.

For ``n_vehicles == 1`` the two modes are the same computation, and the
independent mode dispatches to the joint path so the results match exactly.

``teammate_model`` is evaluated at every rollout step rather than once at the
root: it costs one policy evaluation per step, the same order as the opponent
model already on that path, and it keeps the modelled teammates reactive to
the trajectory each sample explores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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
    """Sampling-based receding-horizon planner conforming to the Policy protocol.

    See the module docstring for the joint and independent coordination modes.
    """

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
    # Modeling prior for fields absent from the belief and optional planning
    # context. Independent roots overlay only current self telemetry, clock
    # and local estimated dwell; joint roots overlay team telemetry.
    template_env_state: Any = None

    # Multi-vehicle coordination. "joint" samples the whole fleet's Delta-v
    # sequence at once; "independent" runs one sampler per own vehicle with
    # teammates modelled by `teammate_model`. Equivalent for n_vehicles == 1.
    coordination: Literal["joint", "independent"] = "joint"
    # Required when coordination=="independent" and n_vehicles>1. A Policy
    # whose `__call__` outputs the whole self-side fleet's per-vehicle Delta-v;
    # the planning vehicle's slot is overwritten by its sampled Delta-v.
    teammate_model: Any = None

    def __post_init__(self) -> None:
        if self.command_cls is None or self.n_vehicles == 0:
            raise ValueError("MPPIPolicy requires command_cls and n_vehicles > 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.coordination not in ("joint", "independent"):
            raise ValueError(
                f"MPPIPolicy.coordination must be 'joint' or 'independent', "
                f"got {self.coordination!r}."
            )
        if (
            self.coordination == "independent"
            and self.n_vehicles > 1
            and self.teammate_model is None
        ):
            raise ValueError(
                "coordination='independent' with n_vehicles>1 requires a "
                "`teammate_model` (a Policy returning the whole self side's "
                "Delta-v during the rollouts). ZeroControl(n_vehicles=n_self, "
                "command_cls=env.<side>_command_cls) is a cheap default."
            )

    @property
    def dv_dim(self) -> int:
        return int(self.command_cls.zeros(self.n_vehicles).dv.shape[-1])

    def init_state(self) -> jax.Array:
        return jnp.zeros((self.horizon, self.n_vehicles, self.dv_dim), dtype=jnp.float32)

    def _flat_state(
        self, agent_view: Any, observer_index: Any = 0, *, joint: bool = False
    ) -> jax.Array:
        if isinstance(agent_view, jax.Array):
            return agent_view
        belief = agent_view.inner if hasattr(agent_view, "inner") else agent_view
        mean = belief.mean
        if mean.ndim == 1:
            return mean
        if self.template_env_state is None:
            raise ValueError("MPPIPolicy needs template_env_state to plan from a belief view")
        return belief_mean_to_flat_state(
            mean,
            self.side,
            self.env_model,
            self.template_env_state,
            observer_index=observer_index,
            planning_context=getattr(agent_view, "planning_context", None),
            joint=joint,
        )

    def _rollout_cost(
        self,
        s0: jax.Array,
        u_seq: jax.Array,
        key: jax.Array,
        self_index: jax.Array | None = None,
    ) -> jax.Array:
        """Discounted cost of one sampled control sequence.

        ``self_index`` picks the mode. ``None`` means ``u_seq`` is the whole
        fleet's ``(horizon, n_vehicles, dv_dim)`` sequence. An index means
        ``u_seq`` is that one vehicle's ``(horizon, dv_dim)`` sequence and the
        rest of the fleet is driven by ``teammate_model``.
        """
        adapter = self.env_model
        env = adapter.env
        side = self.side
        opp_side = side.opposite()
        self_obs_fn = env.guard_observation_fn if side is Side.GUARD else env.bandit_observation_fn
        opp_obs_fn = (
            env.guard_observation_fn if opp_side is Side.GUARD else env.bandit_observation_fn
        )
        opp_cls = env.guard_command_cls if opp_side is Side.GUARD else env.bandit_command_cls
        n_opp = env.config.n_guards if opp_side is Side.GUARD else env.config.n_bandits
        own_template = self.command_cls.zeros(self.n_vehicles)
        teammate_model = self.teammate_model
        identity = Actions(
            sides=BySide(
                guard=env.guard_command_cls.zeros(env.config.n_guards),
                bandit=env.bandit_command_cls.zeros(env.config.n_bandits),
            )
        )

        def own_dv(s: jax.Array, u: jax.Array, k: jax.Array) -> jax.Array:
            """Whole self-side Delta-v for one step. Shape ``(n_vehicles, dv_dim)``."""
            if self_index is None:
                return u.astype(own_template.dv.dtype)
            state = adapter.unpack(s)
            self_obs = self_obs_fn(state, identity, side, env.config, k, state.t)
            cmd, _ = teammate_model(None, flatten_observations(self_obs), k, state.t)
            fleet = cmd.dv.astype(own_template.dv.dtype)
            return fleet.at[self_index].set(u.astype(fleet.dtype))

        def step(carry, inputs):
            (s,) = carry
            u, k = inputs
            k_opp, k_team, k_step = jax.random.split(k, 3)
            state = adapter.unpack(s)
            opp_obs = opp_obs_fn(state, identity, opp_side, env.config, k_opp, state.t)
            opp_cmd, _ = self.opponent_model(None, flatten_observations(opp_obs), k_opp, state.t)
            own_cmd = own_template.replace(dv=own_dv(s, u, k_team))
            opp_flat = flatten_command(opp_cls.zeros(n_opp).replace(dv=opp_cmd.dv))
            own_flat = flatten_command(own_cmd)
            a_flat = (
                jnp.concatenate([own_flat, opp_flat])
                if side is Side.GUARD
                else jnp.concatenate([opp_flat, own_flat])
            )
            s_next, r = adapter.step(s, a_flat, k_step, side)
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

    def _weighted_mean(self, u: jax.Array, costs: jax.Array) -> jax.Array:
        """Softmax-weighted average of the sampled sequences ``u`` over its leading axis."""
        centered = costs - jnp.min(costs)
        scale = jnp.maximum(jnp.std(costs), 1e-6)
        weights = jax.nn.softmax(-centered / (scale * self.temperature))
        return jnp.tensordot(weights, u, axes=(0, 0))

    def _plan_joint(self, s0: jax.Array, u_mean: jax.Array, key: jax.Array) -> jax.Array:
        """One sampler over the fleet. ``u_mean`` and the result are ``(H, n, dv_dim)``."""
        k_noise, k_roll = jax.random.split(key)
        eps = self.noise_sigma * jax.random.normal(
            k_noise, (self.n_samples,) + u_mean.shape, u_mean.dtype
        )
        u = _clip_norm(u_mean[None] + eps, self.dv_max)  # (K, H, n, dv_dim)
        roll_keys = jax.random.split(k_roll, self.n_samples)
        costs = jax.vmap(lambda uk, kk: self._rollout_cost(s0, uk, kk))(u, roll_keys)
        return self._weighted_mean(u, costs).astype(u_mean.dtype)

    def _plan_one_vehicle(
        self,
        s0: jax.Array,
        u_mean_i: jax.Array,
        key: jax.Array,
        self_index: jax.Array,
    ) -> jax.Array:
        """Sampler for one vehicle. ``u_mean_i`` and the result are ``(H, dv_dim)``."""
        k_noise, k_roll = jax.random.split(key)
        eps = self.noise_sigma * jax.random.normal(
            k_noise, (self.n_samples,) + u_mean_i.shape, u_mean_i.dtype
        )
        u = _clip_norm(u_mean_i[None] + eps, self.dv_max)  # (K, H, dv_dim)
        roll_keys = jax.random.split(k_roll, self.n_samples)
        costs = jax.vmap(lambda uk, kk: self._rollout_cost(s0, uk, kk, self_index))(u, roll_keys)
        return self._weighted_mean(u, costs).astype(u_mean_i.dtype)

    def _plan_independent(self, s0: jax.Array, u_mean: jax.Array, key: jax.Array) -> jax.Array:
        """One sampler per own vehicle. ``u_mean`` and the result are ``(H, n, dv_dim)``."""
        keys = jax.random.split(key, self.n_vehicles)
        per_vehicle = jax.vmap(self._plan_one_vehicle, in_axes=(None, 1, 0, 0))(
            s0, u_mean, keys, jnp.arange(self.n_vehicles)
        )  # (n, H, dv_dim)
        return jnp.swapaxes(per_vehicle, 0, 1)

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        del t
        u_mean = policy_state if policy_state is not None else self.init_state()
        if self.coordination == "joint" or self.n_vehicles == 1:
            u_new = self._plan_joint(self._flat_state(agent_view, joint=True), u_mean, key)
        else:
            indices = jnp.arange(self.n_vehicles)
            roots = jax.vmap(lambda i: self._flat_state(agent_view, i))(indices)
            keys = jax.random.split(key, self.n_vehicles)
            plans = jax.vmap(self._plan_one_vehicle, in_axes=(0, 1, 0, 0))(
                roots, u_mean, keys, indices
            )
            u_new = jnp.swapaxes(plans, 0, 1)
        action = u_new[0]
        next_state = jnp.concatenate([u_new[1:], jnp.zeros_like(u_new[:1])], axis=0)
        cmd = self.command_cls.zeros(self.n_vehicles).replace(
            dv=action.astype(self.command_cls.zeros(1).dv.dtype)
        )
        return cmd, next_state
