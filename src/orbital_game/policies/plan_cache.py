"""PlanCachePolicy — contact-gated finite-horizon plan-caching wrapper.

Holds a fixed-length plan in `policy_state` (``plan: (H, n_vehicles, dv_dim)``)
that is refreshed on each upload event (entry into a contact window with
the lag and upload-delay conditions satisfied). Every tick the policy
emits ``plan[step_in_plan]`` if ``step_in_plan < plan_horizon``, else a
zero command — modelling "operator loses comms / plan expires, vehicle
goes safe-mode" rather than indefinitely replaying a stale Δv.

`plan_horizon` should be tuned to span more than the longest expected
inter-contact gap (typically a couple of orbits' worth of ticks). Plans
shorter than the longest gap will leave the agent in safe-mode for a
portion of every off-contact period; plans longer than the longest gap
keep the cached Δv applied through every gap.

See `superpowers/specs/2026-05-07-ground-station-comms-design.md` for
the operational model and lag semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from orbital_game.groundstations.network import ContactSchedule


@flax.struct.dataclass
class PlanCacheState:
    """Cached H-tick plan and the cursor into it.

    Each tick the policy emits ``plan[step_in_plan]`` if
    ``step_in_plan < plan_horizon``, else a zero command. On an upload
    event the plan is refreshed (a fresh H-tick rollout from the lagged
    belief) and ``step_in_plan`` resets to 0.

    ``belief_history`` is a ring buffer of recent post-sync beliefs of
    length ``replan_contacts_lag + 1``. The lagged-belief lookup at
    upload time reads ``belief_history[0]`` (oldest = most lagged).

    ``inner_state`` carries the wrapped inner planner's state across
    uploads. Stateless inner planners pass ``None`` (default) and the
    field is treated as a static null pytree leaf. Stateful inner
    planners (e.g. an MCTSPolicy that returns its tree, or a learned
    policy with RNN state) seed it via ``init_state(inner_init_state=...)``.
    """

    plan: jax.Array  # (plan_horizon, n_vehicles, dv_dim)
    step_in_plan: jax.Array  # () int32 — clamped at plan_horizon to indicate "expired"
    belief_history: jax.Array  # (lag + 1, *belief_shape)
    in_contact_prev: jax.Array  # () bool
    t_entered_contact: jax.Array  # () float
    uploaded_this_contact: jax.Array  # () bool
    contact_count: jax.Array  # () int32
    inner_state: Any = flax.struct.field(default=None, pytree_node=True)


@flax.struct.dataclass
class _LaggedView:
    """Minimal Belief-shaped view exposing `.mean` for the inner planner.

    Module-level so `jax.lax.cond` and `jax.lax.scan` branches that build
    it share an identical pytree structure (closures over a locally
    defined dataclass would each create a distinct treedef, which the
    transformations reject).
    """

    mean: jax.Array


@dataclass(frozen=True)
class PlanCachePolicy:
    """Contact-gated finite-horizon plan-caching wrapper.

    Wraps any inner Policy. At each upload event (entry into a contact
    window with the lag and upload-delay conditions satisfied), invokes
    the inner planner ``plan_horizon`` times against the lagged belief
    and stores the resulting H-tick command sequence. Each tick the
    policy emits ``plan[step_in_plan]`` if the cursor is still inside
    the plan, else a zero command (plan expired).

    Default ``replan_contacts_lag=1`` matches the operational picture:
    the belief synced this contact drives the plan uplinked next contact.

    ``plan_horizon`` is the number of ticks the cached plan covers.
    Tune it to span slightly more than the longest expected inter-contact
    gap so every off-contact period stays plan-active. Plans shorter
    than the longest gap will leave the vehicle in safe-mode (zero Δv)
    for a portion of every off-contact period.

    For a stateless inner planner (typical for LQR), all H slots are
    nearly identical (the lagged belief doesn't evolve across the H
    inner calls), so the plan effectively encodes "apply this Δv for
    H ticks, then go safe". For a stateful inner planner (e.g. a
    rollout-based optimizer that produces a sequence of differing
    commands), each slot can carry a different Δv.
    """

    inner: Any  # a Policy
    plan_horizon: int  # H ticks per plan
    schedule: ContactSchedule
    dt: float
    replan_contacts_lag: int = 1
    upload_delay_s: float = 0.0
    n_vehicles: int = 0
    command_cls: Any = None

    def __post_init__(self) -> None:
        if self.command_cls is None:
            raise ValueError("PlanCachePolicy requires command_cls")
        if self.n_vehicles == 0:
            raise ValueError("PlanCachePolicy requires n_vehicles > 0")
        if self.plan_horizon < 1:
            raise ValueError(f"plan_horizon must be >= 1, got {self.plan_horizon}")
        if self.replan_contacts_lag < 0:
            raise ValueError(f"replan_contacts_lag must be >= 0, got {self.replan_contacts_lag}")
        if self.upload_delay_s < 0.0:
            raise ValueError(f"upload_delay_s must be >= 0, got {self.upload_delay_s}")

    def init_state(
        self,
        belief: Any,
        inner_init_state: Any = None,
        key: jax.Array | None = None,
    ) -> PlanCacheState:
        """Build an initial ``PlanCacheState`` whose plan is computed
        from the initial belief — so the agent acts immediately at
        ``t=0``, not after the first upload event.

        The initial plan covers the first ``plan_horizon`` ticks; if the
        first upload event hasn't fired by tick ``plan_horizon``, the
        cursor falls off the end of the plan and the policy emits zero
        Δv until the first upload arrives.

        ``inner_init_state`` is forwarded into the inner planner so
        stateful inners (e.g. MCTS) start from a well-defined seed.
        ``key`` is required when the inner planner is non-deterministic
        (e.g. MCTS); for deterministic LQR-style inners pass any
        deterministic key. Defaults to ``jax.random.key(0)`` so the
        method is callable in pure-deterministic setups without
        threading a key through.
        """
        if key is None:
            key = jax.random.key(0)
        mean = belief.mean
        # Build an initial belief_history ring filled with the current belief
        # so the lagged-belief lookup at the first upload doesn't pull from
        # an all-zeros initial entry.
        belief_history_init = jnp.broadcast_to(
            mean[None, ...], (self.replan_contacts_lag + 1,) + mean.shape
        ).astype(mean.dtype)
        # Compute the initial plan by invoking the H-step rollout against
        # the initial belief.
        init_plan, init_inner_state = self._compute_plan(
            belief_history_init, inner_init_state, key, jnp.asarray(0.0)
        )
        return PlanCacheState(
            plan=init_plan.astype(jnp.float32),
            step_in_plan=jnp.asarray(0, dtype=jnp.int32),
            belief_history=belief_history_init,
            in_contact_prev=jnp.asarray(False),
            t_entered_contact=jnp.asarray(0.0),
            uploaded_this_contact=jnp.asarray(False),
            contact_count=jnp.asarray(0, dtype=jnp.int32),
            inner_state=init_inner_state,
        )

    def __call__(
        self,
        policy_state: PlanCacheState,
        agent_view: Any,  # ContactAwareBelief
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, PlanCacheState]:
        in_contact = jnp.any(agent_view.contact)
        # Edge detect: rising edge means a new contact just started.
        rising = jnp.logical_and(in_contact, jnp.logical_not(policy_state.in_contact_prev))

        # On rising edge: bump contact_count, set t_entered, reset upload flag,
        # roll the belief ring buffer and write the freshest belief at the head.
        new_contact_count = jnp.where(
            rising, policy_state.contact_count + 1, policy_state.contact_count
        )
        new_t_entered = jnp.where(rising, t, policy_state.t_entered_contact)
        new_uploaded_flag = jnp.where(
            rising, jnp.asarray(False), policy_state.uploaded_this_contact
        )

        def _push_ring(ring):
            # Roll left (oldest drops, newest at the end).
            rolled = jnp.roll(ring, shift=-1, axis=0)
            return rolled.at[-1].set(agent_view.inner.mean.astype(ring.dtype))

        new_history = jax.lax.cond(rising, _push_ring, lambda r: r, policy_state.belief_history)

        # Upload trigger.
        delay_satisfied = (t - new_t_entered) >= self.upload_delay_s
        lag_satisfied = new_contact_count > self.replan_contacts_lag
        should_upload = jnp.logical_and(
            in_contact,
            jnp.logical_and(
                jnp.logical_and(delay_satisfied, lag_satisfied),
                jnp.logical_not(new_uploaded_flag),
            ),
        )

        plan_dtype = policy_state.plan.dtype

        # `jax.lax.cond` so this is JIT/scan-safe.
        def _do_upload():
            new_plan, next_inner = self._compute_plan(new_history, policy_state.inner_state, key, t)
            return new_plan.astype(plan_dtype), next_inner

        def _no_upload():
            return policy_state.plan, policy_state.inner_state

        new_plan, new_inner_state = jax.lax.cond(should_upload, _do_upload, _no_upload)
        new_uploaded_flag = jnp.logical_or(new_uploaded_flag, should_upload)

        # On upload: cursor resets to 0. Otherwise: advance by 1.
        new_step = jnp.where(
            should_upload,
            jnp.asarray(0, dtype=jnp.int32),
            policy_state.step_in_plan + 1,
        )

        # Emit plan[step_in_plan] if step < H, else zero (plan expired).
        plan_active = new_step < self.plan_horizon
        # Safe-index: clamp to H-1 to read the array, then mask out via
        # jnp.where. This avoids out-of-bounds reads when the cursor has
        # run past the end.
        safe_idx = jnp.minimum(new_step, jnp.asarray(self.plan_horizon - 1, dtype=jnp.int32))
        cmd_dv = jnp.where(plan_active, new_plan[safe_idx], jnp.zeros_like(new_plan[0]))

        cmd = self.command_cls.zeros(self.n_vehicles).replace(
            dv=cmd_dv.astype(self.command_cls.zeros(self.n_vehicles).dv.dtype)
        )

        new_state = PlanCacheState(
            plan=new_plan,
            step_in_plan=new_step,
            belief_history=new_history,
            in_contact_prev=in_contact,
            t_entered_contact=new_t_entered,
            uploaded_this_contact=new_uploaded_flag,
            contact_count=new_contact_count,
            inner_state=new_inner_state,
        )
        return cmd, new_state

    def _compute_plan(
        self,
        history: jax.Array,
        inner_state: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, Any]:
        """Roll out the inner planner ``plan_horizon`` ticks against the
        lagged belief.

        Returns ``(plan, new_inner_state)`` with shapes
        ``(plan_horizon, n_vehicles, dv_dim)`` and the same pytree
        structure as ``inner_state``.

        For a stateless inner the H slots are all the same Δv (no
        carry-state evolves across the rollout). For a stateful inner
        each slot can differ.
        """
        lagged_mean = history[0]  # oldest belief at index 0

        def step(carry, h_idx):
            inner_state_h, key_h = carry
            view = _LaggedView(mean=lagged_mean)
            cmd, next_inner = self.inner(inner_state_h, view, key_h, t + h_idx * self.dt)
            next_key = jax.random.fold_in(key_h, h_idx)
            return (next_inner, next_key), cmd.dv

        init_carry = (inner_state, key)
        final_carry, dvs = jax.lax.scan(step, init_carry, jnp.arange(self.plan_horizon))
        final_inner, _ = final_carry
        return dvs, final_inner
