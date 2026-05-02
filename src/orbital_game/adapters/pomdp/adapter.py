"""POMDPAdapter — POMDPPlanners-shape interface over OrbitalGameEnv.

Exposes a duck-typed protocol matching POMDPPlanners (Python POMDPs.jl-shape):
    transition(s_flat, a_flat, key)   → s_flat'
    observation(s, a, s', side)       → obs (per-side)
    reward(s, a, s', side)            → scalar
    discount()                        → float
    initialstate(key)                 → s_flat (sampled)
    states_dim, action_dim_per_side   → properties

State is exposed as a flat JAX vector. The vector packs `StateLayout.flatten`
(per-side guard/bandit truth) followed by two scalar tail entries: ``t`` and
``step``. The reference orbit and ``ic_valid`` flag are captured from the
env config at ``__init__`` time and treated as per-scenario constants.

The adapter is **pure**: ``transition``, ``observation``, and ``reward`` are
deterministic functions of their flat-vector arguments and never mutate
adapter attributes. They compose with ``jax.vmap`` and ``jax.lax.scan``
without leaking tracers, which makes vectorised search loops (e.g. random
shooting over K candidates × H horizon) compile to a single JIT call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.env.core import EnvState, OrbitalGameEnv
from orbital_game.env.types import Actions, BySide, Side
from orbital_game.observations.types import flatten_observations


def _action_dim_from_dynamics(dynamics_key) -> int:
    from orbital_game.registry import DynamicsKey

    return 2 if dynamics_key is DynamicsKey.HCW_RT else 3


class POMDPAdapter:
    """Flat-state-vector adapter conforming to POMDPPlanners-shape.

    Pure: every public method is a function of its arguments alone. Safe
    to drive under ``jax.vmap`` / ``jax.lax.scan`` / ``jax.jit``.
    """

    def __init__(self, env: OrbitalGameEnv):
        self.env = env
        self.layout = env.layout
        self._action_dim = _action_dim_from_dynamics(env.config.truth_dynamics)
        # Per-scenario constants — captured once, never mutated. The flat
        # state vector carries only (guards, bandits, t, step); reference
        # orbit and ic_valid are scenario-level metadata that don't change
        # across a planning rollout.
        self._reference_orbit = env.config.reference_orbit
        # ic_valid defaults to True; the planner-facing path doesn't model
        # IC-rejection failures, which only matter at reset time.
        self._ic_valid_default = jnp.asarray(True)

    @property
    def states_dim(self) -> int:
        """Flat-state dimensionality: ``layout.flat_dim + 2`` (t, step tail)."""
        return self.layout.flat_dim + 2

    @property
    def action_dim_per_side(self) -> int:
        return self._action_dim

    def discount(self) -> float:
        """Episode-bounded → 1.0. Override per-game if needed."""
        return 1.0

    # ---- pack / unpack ----

    def _pack(self, state: EnvState) -> jax.Array:
        """EnvState → flat vector. t and step are appended as float scalars."""
        flat_xy = self.layout.flatten(state.guards, state.bandits)
        # Cast t and step to a uniform float dtype matching flat_xy so
        # concatenation works whether x64 is enabled or not.
        tail = jnp.stack(
            [
                jnp.asarray(state.t, flat_xy.dtype),
                jnp.asarray(state.step, flat_xy.dtype),
            ]
        )
        return jnp.concatenate([flat_xy, tail])

    def _unpack(self, s_flat: jax.Array) -> EnvState:
        """Flat vector → EnvState (reference_orbit / ic_valid from __init__)."""
        flat_xy = s_flat[:-2]
        t = s_flat[-2]
        step = s_flat[-1].astype(jnp.int32)
        guards, bandits = self.layout.unflatten(flat_xy)
        return EnvState(
            t=t,
            step=step,
            guards=guards,
            bandits=bandits,
            reference_orbit=self._reference_orbit,
            ic_valid=self._ic_valid_default,
        )

    # ---- POMDPPlanners-shape interface ----

    def initialstate(self, key: jax.Array) -> jax.Array:
        """Sample an initial state and return its flat vector."""
        state, _outs = self.env.reset(key)
        return self._pack(state)

    def transition(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Apply one env step. ``a_flat`` is the concatenated guard+bandit actions."""
        state = self._unpack(s_flat)
        actions = self._make_actions(a_flat)
        step_out = self.env.step(key, state, actions)
        return self._pack(step_out.state)

    def observation(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        s_next_flat: jax.Array,
        side: Side,
    ) -> jax.Array:
        """Return the side's observation at the next state."""
        del a_flat
        next_state = self._unpack(s_next_flat)
        obs_fn = (
            self.env.guard_observation_fn if side is Side.GUARD else self.env.bandit_observation_fn
        )
        return flatten_observations(
            obs_fn(next_state, side, self.env.config, jax.random.PRNGKey(0), next_state.t)
        )

    def reward(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        s_next_flat: jax.Array,
        side: Side,
    ) -> jax.Array:
        """Return the side's per-step reward."""
        prev_state = self._unpack(s_flat)
        next_state = self._unpack(s_next_flat)
        actions = self._make_actions(a_flat)
        return self.env.reward_fn(
            prev_state, actions, next_state, side, self.env.config, prev_state.t
        )

    # ---- helpers ----

    def _make_actions(self, a_flat: jax.Array) -> Actions:
        n_g = self.env.config.n_guards
        n_b = self.env.config.n_bandits
        d = self._action_dim
        guard_action = a_flat[: n_g * d].reshape(n_g, d)
        bandit_action = a_flat[n_g * d : n_g * d + n_b * d].reshape(n_b, d)
        return Actions(sides=BySide(guard=guard_action, bandit=bandit_action))
