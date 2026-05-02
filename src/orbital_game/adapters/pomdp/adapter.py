"""POMDPAdapter — POMDPPlanners-shape interface over OrbitalGameEnv.

Exposes a duck-typed protocol matching POMDPPlanners (Python POMDPs.jl-shape):
    transition(s_flat, a_flat, key)   → s_flat'
    observation(s, a, s', side)       → obs (per-side)
    reward(s, a, s', side)            → scalar
    discount()                        → float
    initialstate(key)                 → s_flat (sampled)
    states_dim, action_dim_per_side   → properties

State is exposed as a flat numpy/jax vector via StateLayout.flatten/unflatten.
The adapter exposes per-side observation and reward methods so consumers
can pick which side is the "agent" at consumption time.

Note: per-step env state has scalars (t, step) and the per-side pytrees
(guards, bandits) plus a reference_orbit. The flat-vector interface here
flattens ONLY the per-side guard/bandit state — the scalar fields t/step
and reference_orbit are read from a stashed "context state" carried
separately. POMDPPlanners typically expects pure functions of (state, action)
→ next_state, but our env's t/step advance deterministically, so we
maintain them via the adapter's bookkeeping.
"""

from __future__ import annotations

from typing import Any

import jax

from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide, Side
from orbital_game.observations.types import coerce_obs_to_flat_array


def _action_dim_from_dynamics(dynamics_key) -> int:
    from orbital_game.registry import DynamicsKey

    return 2 if dynamics_key is DynamicsKey.HCW_RT else 3


class POMDPAdapter:
    """Flat-state-vector adapter conforming to POMDPPlanners-shape."""

    def __init__(self, env: OrbitalGameEnv):
        self.env = env
        self.layout = env.layout
        self._action_dim = _action_dim_from_dynamics(env.config.truth_dynamics)
        self._last_state: Any = None

    @property
    def states_dim(self) -> int:
        """Flat-state dimensionality from StateLayout."""
        return self.layout.flat_dim

    @property
    def action_dim_per_side(self) -> int:
        return self._action_dim

    def discount(self) -> float:
        """Episode-bounded → 1.0. Override per-game if needed."""
        return 1.0

    def initialstate(self, key: jax.Array) -> jax.Array:
        """Sample initial state and return flat vector. Stash full state for
        downstream calls that need t/step/reference_orbit."""
        state, _outs = self.env.reset(key)
        self._last_state = state
        return self.layout.flatten(state.guards, state.bandits)

    def transition(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Apply one env step. a_flat is the concatenated guard+bandit actions."""
        if self._last_state is None:
            raise RuntimeError("transition() called before initialstate()")
        guards, bandits = self.layout.unflatten(s_flat)
        state = self._last_state.replace(guards=guards, bandits=bandits)
        n_g = self.env.config.n_guards
        n_b = self.env.config.n_bandits
        d = self._action_dim
        guard_action = a_flat[: n_g * d].reshape(n_g, d)
        bandit_action = a_flat[n_g * d : n_g * d + n_b * d].reshape(n_b, d)
        actions = Actions(sides=BySide(guard=guard_action, bandit=bandit_action))
        step_out = self.env.step(key, state, actions)
        self._last_state = step_out.state
        return self.layout.flatten(step_out.state.guards, step_out.state.bandits)

    def observation(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        s_next_flat: jax.Array,
        side: Side,
    ) -> jax.Array:
        """Return the side's observation at the next state."""
        del a_flat
        if self._last_state is None:
            raise RuntimeError("observation() called before initialstate()")
        guards_next, bandits_next = self.layout.unflatten(s_next_flat)
        next_state = self._last_state.replace(guards=guards_next, bandits=bandits_next)
        obs_fn = (
            self.env.guard_observation_fn if side is Side.GUARD else self.env.bandit_observation_fn
        )
        return coerce_obs_to_flat_array(
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
        if self._last_state is None:
            raise RuntimeError("reward() called before initialstate()")
        guards_prev, bandits_prev = self.layout.unflatten(s_flat)
        guards_next, bandits_next = self.layout.unflatten(s_next_flat)
        prev_state = self._last_state.replace(guards=guards_prev, bandits=bandits_prev)
        next_state = self._last_state.replace(guards=guards_next, bandits=bandits_next)
        n_g = self.env.config.n_guards
        n_b = self.env.config.n_bandits
        d = self._action_dim
        guard_action = a_flat[: n_g * d].reshape(n_g, d)
        bandit_action = a_flat[n_g * d : n_g * d + n_b * d].reshape(n_b, d)
        actions = Actions(sides=BySide(guard=guard_action, bandit=bandit_action))
        return self.env.reward_fn(
            prev_state, actions, next_state, side, self.env.config, prev_state.t
        )
