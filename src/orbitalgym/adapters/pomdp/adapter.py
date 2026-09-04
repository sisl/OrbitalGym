"""POMDPAdapter — POMDPPlanners-shape interface over OrbitalGymEnv.

Exposes a duck-typed protocol matching POMDPPlanners (Python POMDPs.jl-shape):
    transition(s_flat, a_flat, key)   → s_flat'
    observation(s, a, s', side)       → obs (per-side)
    reward(s, a, s', side)            → scalar
    step(s_flat, a_flat, key, side)   → (s_flat', reward)
    discount()                        → float
    initialstate(key)                 → s_flat (sampled)
    states_dim, action_dim_per_side   → properties

``action_repeat = k`` makes one planner action a *macro step*: the same
``a_flat`` is applied for k consecutive env steps, so the planner searches
``k * dt`` seconds per tree edge while the env still applies its own
per-step delta-v cap on every substep. ``reward`` returns the discounted
sum of the k per-substep rewards and ``discount()`` returns the
per-macro-step discount, which keeps the planner's returns on the same
scale as the env's own accumulated reward.

State is exposed as a flat JAX vector. The vector packs `StateLayout.flatten`
(per-side guard/bandit truth) followed by two scalar tail entries: ``t`` and
``step``. The reference orbit and ``ic_valid`` flag are captured from the
env config at ``__init__`` time and treated as per-scenario constants.

Actions are exposed as a flat vector that concatenates the guard side's
flat-Command layout followed by the bandit side's flat-Command layout. The
adapter unflattens back into the per-side Command pytrees before invoking
``env.step``. Sizes come from
``command_flat_dim(env.{guard,bandit}_command_cls)``.

The adapter is **pure**: ``transition``, ``observation``, and ``reward`` are
deterministic functions of their flat-vector arguments and never mutate
adapter attributes. They compose with ``jax.vmap`` and ``jax.lax.scan``
without leaking tracers, which makes vectorised search loops (e.g. random
shooting over K candidates × H horizon) compile to a single JIT call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbitalgym.adapters._command_flatten import command_flat_dim, unflatten_command
from orbitalgym.env.core import EnvState, OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.observations.types import flatten_observations


class POMDPAdapter:
    """Flat-state-vector adapter conforming to POMDPPlanners-shape.

    Pure: every public method is a function of its arguments alone. Safe
    to drive under ``jax.vmap`` / ``jax.lax.scan`` / ``jax.jit``.
    """

    def __init__(self, env: OrbitalGymEnv, action_repeat: int = 1, discount: float = 1.0):
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be >= 1, got {action_repeat}")
        if not 0.0 < discount <= 1.0:
            raise ValueError(f"discount must lie in (0, 1], got {discount}")
        self.env = env
        self.layout = env.layout
        # A planner action spans `action_repeat` env steps. `discount` is the
        # per-env-step discount; `discount()` reports the per-macro-step one.
        self.action_repeat = int(action_repeat)
        self.step_discount = float(discount)
        self.macro_dt = env.config.dt * action_repeat
        # Per-side Command pytree classes — built by env from the configured
        # action components. Each side's flat-action width is the sum of
        # every component field's per-agent size, multiplied by n_agents.
        self._guard_command_cls = env.guard_command_cls
        self._bandit_command_cls = env.bandit_command_cls
        self._guard_flat_dim = command_flat_dim(self._guard_command_cls)
        self._bandit_flat_dim = command_flat_dim(self._bandit_command_cls)
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
        """Per-agent flat action width.

        Both sides must agree on the per-agent dim for this property to be
        meaningful (planners that call this typically assume a single
        per-agent action width). When sides have different Command shapes,
        callers should use ``guard_action_flat_dim`` / ``bandit_action_flat_dim``
        directly.
        """
        guard_per_agent = self._guard_flat_dim // self.env.config.n_guards
        bandit_per_agent = self._bandit_flat_dim // self.env.config.n_bandits
        if guard_per_agent != bandit_per_agent:
            raise ValueError(
                "action_dim_per_side is ambiguous when guard and bandit Commands "
                f"have different per-agent widths (guard={guard_per_agent}, "
                f"bandit={bandit_per_agent}). Use guard_action_flat_dim / "
                "bandit_action_flat_dim directly."
            )
        return guard_per_agent

    @property
    def guard_action_flat_dim(self) -> int:
        """Total flat width of the guard side's Command pytree."""
        return self._guard_flat_dim

    @property
    def bandit_action_flat_dim(self) -> int:
        """Total flat width of the bandit side's Command pytree."""
        return self._bandit_flat_dim

    def discount(self) -> float:
        """Per-macro-step discount: ``step_discount ** action_repeat``."""
        return self.step_discount**self.action_repeat

    # ---- pack / unpack ----

    def pack(self, state: EnvState) -> jax.Array:
        """Public alias for `_pack`. Use this from planners and tests."""
        return self._pack(state)

    def unpack(self, s_flat: jax.Array) -> EnvState:
        """Public alias for `_unpack`."""
        return self._unpack(s_flat)

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
        """Apply ``action_repeat`` env steps of ``a_flat``.

        ``a_flat`` is the concatenated guard+bandit action. Every substep
        goes through the env, so each one applies the env's own per-step
        delta-v cap: a macro action is k capped burns in one direction,
        exactly what the env executes when a policy repeats the command.
        Substep keys are splits of ``key``. Once a substep reports
        ``episode_done`` the state is frozen for the remaining substeps.
        """
        s_next, _reward = self._macro_step(s_flat, a_flat, key, None)
        return s_next

    def observation(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        s_next_flat: jax.Array,
        side: Side,
    ) -> jax.Array:
        """Return the side's observation at the next state."""
        next_state = self._unpack(s_next_flat)
        actions = self._make_actions(a_flat)
        obs_fn = (
            self.env.guard_observation_fn if side is Side.GUARD else self.env.bandit_observation_fn
        )
        return flatten_observations(
            obs_fn(next_state, actions, side, self.env.config, jax.random.PRNGKey(0), next_state.t)
        )

    def reward(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        s_next_flat: jax.Array,
        side: Side,
    ) -> jax.Array:
        """Return the side's reward for the macro step.

        With ``action_repeat = 1`` this is the single-step reward over the
        supplied ``(s, a, s')`` triple. With ``action_repeat = k`` it is the
        sum of the k per-substep rewards discounted by ``step_discount`` per
        substep, zero after a terminal substep; the intermediate states are
        not available from this signature, so the substeps are re-simulated
        from ``s_flat`` and ``s_next_flat`` is ignored. Re-simulation is
        exact: translational and attitude dynamics are deterministic given
        the state and action, and observation noise keys never enter the
        reward. Prefer :meth:`step`, which returns the next state and the
        reward from one rollout.
        """
        if self.action_repeat == 1:
            prev_state = self._unpack(s_flat)
            next_state = self._unpack(s_next_flat)
            actions = self._make_actions(a_flat)
            return self.env.reward_fn(
                prev_state, actions, next_state, side, self.env.config, prev_state.t
            )
        _s_next, reward = self._macro_step(s_flat, a_flat, key=jax.random.PRNGKey(0), side=side)
        return reward

    def step(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        key: jax.Array,
        side: Side,
    ) -> tuple[jax.Array, jax.Array]:
        """Advance one macro step and return ``(s_next_flat, reward)``.

        The planner-facing entry point: one rollout serves both the next
        state and the side's discounted macro-step reward, so a search does
        not simulate the substeps twice.
        """
        return self._macro_step(s_flat, a_flat, key, side)

    def _macro_step(
        self,
        s_flat: jax.Array,
        a_flat: jax.Array,
        key: jax.Array,
        side: Side | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Roll ``action_repeat`` substeps of ``a_flat``, freezing on termination.

        Returns the flat state after the last substep and, when ``side`` is
        given, the discounted sum of the per-substep rewards. The substep
        count is a Python int, so the loop unrolls under trace.
        """
        actions = self._make_actions(a_flat)
        keys = [key] if self.action_repeat == 1 else list(jax.random.split(key, self.action_repeat))
        s_cur = s_flat
        reward = jnp.zeros((), dtype=s_flat.dtype)
        done = jnp.asarray(False)
        substep_discount = jnp.asarray(1.0, dtype=s_flat.dtype)
        for i, k in enumerate(keys):
            prev_state = self._unpack(s_cur)
            step_out = self.env.step(k, prev_state, actions)
            s_new = self._pack(step_out.state)
            if side is not None:
                r = self.env.reward_fn(
                    prev_state, actions, step_out.state, side, self.env.config, prev_state.t
                )
                r = r.astype(s_flat.dtype) * substep_discount
                reward = reward + (r if i == 0 else jnp.where(done, 0.0, r))
            s_cur = s_new if i == 0 else jnp.where(done, s_cur, s_new)
            if i + 1 < self.action_repeat:
                done = jnp.logical_or(done, step_out.episode_done)
                substep_discount = substep_discount * self.step_discount
        return s_cur, reward

    # ---- helpers ----

    def _make_actions(self, a_flat: jax.Array) -> Actions:
        """Split a_flat into per-side slices and unflatten each into a Command."""
        guard_flat = a_flat[: self._guard_flat_dim]
        bandit_flat = a_flat[self._guard_flat_dim : self._guard_flat_dim + self._bandit_flat_dim]
        guard_command = unflatten_command(self._guard_command_cls, guard_flat)
        bandit_command = unflatten_command(self._bandit_command_cls, bandit_flat)
        return Actions(sides=BySide(guard=guard_command, bandit=bandit_command))
