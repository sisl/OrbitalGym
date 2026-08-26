from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
import pytest

from orbitalgym.belief.contact_aware import ContactAwareBelief
from orbitalgym.groundstations import ContactSchedule
from orbitalgym.policies.plan_cache import PlanCachePolicy


# Minimal stand-in command class — mirrors what build_command_class produces.
@flax.struct.dataclass
class _StubCommand:
    dv: jax.Array

    @classmethod
    def zeros(cls, n: int) -> "_StubCommand":
        return cls(dv=jnp.zeros((n, 2)))

    def replace(self, **kwargs):
        return self.__class__(**{**{"dv": self.dv}, **kwargs})


@dataclass(frozen=True)
class _RecordingInner:
    """Inner planner that emits a constant command. Has NO Python side effects
    so it survives `jax.lax.cond` and `jax.lax.scan` (both trace once).

    Tests verify upload behavior by inspecting `state.plan` and the emitted
    command — uploaded plans contain the inner's constant; non-uploaded
    plans keep the prior cache.
    """

    n_vehicles: int = 0
    command_cls: Any = None

    def __call__(self, ps, view, key, t):
        del view, key, t  # unused; this stub is constant.
        cmd = self.command_cls.zeros(self.n_vehicles).replace(
            dv=jnp.ones((self.n_vehicles, 2)) * 0.1
        )
        return cmd, ps


@flax.struct.dataclass
class _StubBelief:
    mean: jax.Array


def _belief(n_obs=1, n_total=2):
    return _StubBelief(mean=jnp.zeros((n_obs, n_total, 6), dtype=jnp.float32))


def _schedule(*windows, pad_to=8):
    n = len(windows)
    pad = [(-1.0, -1.0)] * (pad_to - n)
    return ContactSchedule(
        windows=jnp.asarray(list(windows) + pad, dtype=jnp.float32),
        n_valid=jnp.asarray(n),
        station_ix=jnp.asarray(list(range(n)) + [-1] * (pad_to - n)),
    )


def test_plan_cache_init_populates_plan_from_initial_belief():
    """init_state runs the inner planner H times against the initial belief
    so the agent acts immediately from t=0 (instead of free-drifting until
    the first upload event)."""
    inner = _RecordingInner(n_vehicles=1, command_cls=_StubCommand)
    sch = _schedule((0.0, 100.0))
    policy = PlanCachePolicy(
        inner=inner,
        plan_horizon=5,
        schedule=sch,
        dt=10.0,
        replan_contacts_lag=1,
        n_vehicles=1,
        command_cls=_StubCommand,
    )
    state = policy.init_state(belief=_belief())
    assert state.plan.shape == (5, 1, 2)
    assert int(state.step_in_plan) == 0
    # Stateless inner emits constant 0.1 — every plan slot should match.
    assert jnp.allclose(state.plan, jnp.ones((5, 1, 2)) * 0.1)


def test_plan_cache_emits_zero_after_horizon_expires():
    """Once step_in_plan >= H without an upload, the policy emits zero —
    modeling 'plan expired, vehicle goes safe-mode'."""
    inner = _RecordingInner(n_vehicles=1, command_cls=_StubCommand)
    sch = _schedule((0.0, 50.0))  # one short contact window at the start
    H = 3  # noqa: N806
    policy = PlanCachePolicy(
        inner=inner,
        plan_horizon=H,
        schedule=sch,
        dt=10.0,
        replan_contacts_lag=0,
        n_vehicles=1,
        command_cls=_StubCommand,
    )
    state = policy.init_state(belief=_belief())
    view_on = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([True]))
    view_off = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([False]))

    # Tick at t=0 (in contact, lag=0): upload fires; step resets to 0; emit slot 0.
    cmd, state = policy(state, view_on, jax.random.key(0), jnp.asarray(0.0))
    assert float(jnp.linalg.norm(cmd.dv)) > 0.0
    assert int(state.step_in_plan) == 0
    # Tick at t=10 (still in contact, already uploaded): step→1, plan still active.
    cmd, state = policy(state, view_on, jax.random.key(0), jnp.asarray(10.0))
    assert float(jnp.linalg.norm(cmd.dv)) > 0.0
    assert int(state.step_in_plan) == 1
    # Tick at t=60 (off contact): step→2, still active (plan_horizon=3).
    cmd, state = policy(state, view_off, jax.random.key(0), jnp.asarray(60.0))
    assert float(jnp.linalg.norm(cmd.dv)) > 0.0
    assert int(state.step_in_plan) == 2
    # Tick at t=70: step→3 == plan_horizon → plan expired, emit zero.
    cmd, state = policy(state, view_off, jax.random.key(0), jnp.asarray(70.0))
    assert float(jnp.linalg.norm(cmd.dv)) == 0.0
    assert int(state.step_in_plan) == 3
    # Tick at t=80: still expired.
    cmd, state = policy(state, view_off, jax.random.key(0), jnp.asarray(80.0))
    assert float(jnp.linalg.norm(cmd.dv)) == 0.0


def test_plan_cache_lag1_first_contact_does_not_upload():
    """At lag=1, the *first* contact must not trigger a new upload —
    state.plan stays at the initial-belief plan computed at init time."""
    inner = _RecordingInner(n_vehicles=1, command_cls=_StubCommand)
    sch = _schedule((0.0, 100.0), (300.0, 400.0))
    policy = PlanCachePolicy(
        inner=inner,
        plan_horizon=5,
        schedule=sch,
        dt=10.0,
        replan_contacts_lag=1,
        n_vehicles=1,
        command_cls=_StubCommand,
    )
    state = policy.init_state(belief=_belief())
    initial_plan = state.plan.copy()
    view = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([True]))
    _, state = policy(state, view, jax.random.key(0), jnp.asarray(0.0))
    # No NEW upload during the first contact: plan must equal the snapshot.
    assert jnp.allclose(state.plan, initial_plan), (
        "First-contact tick at lag=1 must not trigger a new upload"
    )
    # contact_count advanced; uploaded_this_contact stays False at lag=1.
    assert int(state.contact_count) == 1
    assert bool(state.uploaded_this_contact) is False


def test_plan_cache_lag1_second_contact_uploads_and_resets_step():
    """At lag=1, the second contact triggers an upload — state.plan
    refreshes and step_in_plan resets to 0."""
    inner = _RecordingInner(n_vehicles=1, command_cls=_StubCommand)
    sch = _schedule((0.0, 100.0), (300.0, 400.0))
    policy = PlanCachePolicy(
        inner=inner,
        plan_horizon=5,
        schedule=sch,
        dt=10.0,
        replan_contacts_lag=1,
        n_vehicles=1,
        command_cls=_StubCommand,
    )
    state = policy.init_state(belief=_belief())

    # Tick 1: in first contact (t=10) — no upload at lag=1
    view = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([True]))
    _, state = policy(state, view, jax.random.key(0), jnp.asarray(10.0))
    # Tick 2: gap (t=200, no contact)
    view_off = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([False]))
    _, state = policy(state, view_off, jax.random.key(0), jnp.asarray(200.0))
    # Tick 3: enter second contact (t=300) — upload should fire, step → 0
    cmd, state = policy(state, view, jax.random.key(0), jnp.asarray(300.0))
    assert float(jnp.linalg.norm(cmd.dv)) > 0.0
    assert int(state.step_in_plan) == 0  # reset on upload
    # Plan now matches inner's constant 0.1.
    assert jnp.allclose(state.plan, jnp.ones((5, 1, 2)) * 0.1)


def test_plan_cache_executes_cached_during_off_contact():
    inner = _RecordingInner(n_vehicles=1, command_cls=_StubCommand)
    sch = _schedule((0.0, 100.0), (300.0, 400.0))
    policy = PlanCachePolicy(
        inner=inner,
        plan_horizon=20,  # plenty of slots so plan doesn't expire in this test
        schedule=sch,
        dt=10.0,
        replan_contacts_lag=0,
        n_vehicles=1,
        command_cls=_StubCommand,
    )
    state = policy.init_state(belief=_belief())
    view_on = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([True]))
    view_off = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([False]))
    # First contact: should upload (lag=0, contact_count=1>0)
    _, state = policy(state, view_on, jax.random.key(0), jnp.asarray(0.0))
    plan_after_upload = state.plan.copy()
    # Off contact: must NOT trigger a re-upload — plan unchanged
    cmd, state = policy(state, view_off, jax.random.key(0), jnp.asarray(150.0))
    assert jnp.allclose(state.plan, plan_after_upload), (
        "Off-contact tick must not modify the cached plan"
    )
    # Step advanced; emitted command non-zero
    assert int(state.step_in_plan) > 0
    assert float(jnp.linalg.norm(cmd.dv)) > 0.0


def test_plan_cache_rejects_negative_lag():
    with pytest.raises(ValueError, match="replan_contacts_lag"):
        PlanCachePolicy(
            inner=_RecordingInner(),
            plan_horizon=5,
            schedule=_schedule((0.0, 100.0)),
            dt=10.0,
            replan_contacts_lag=-1,
            n_vehicles=1,
            command_cls=_StubCommand,
        )


def test_plan_cache_rejects_negative_upload_delay():
    with pytest.raises(ValueError, match="upload_delay_s"):
        PlanCachePolicy(
            inner=_RecordingInner(),
            plan_horizon=5,
            schedule=_schedule((0.0, 100.0)),
            dt=10.0,
            replan_contacts_lag=0,
            upload_delay_s=-1.0,
            n_vehicles=1,
            command_cls=_StubCommand,
        )


def test_plan_cache_rejects_zero_horizon():
    with pytest.raises(ValueError, match="plan_horizon"):
        PlanCachePolicy(
            inner=_RecordingInner(),
            plan_horizon=0,
            schedule=_schedule((0.0, 100.0)),
            dt=10.0,
            replan_contacts_lag=0,
            n_vehicles=1,
            command_cls=_StubCommand,
        )


@flax.struct.dataclass
class _CounterState:
    """Stateful pytree for `_StatefulInner`. Module-level so the dtype/shape
    is stable across `lax.cond` branches and `lax.scan` carries."""

    count: jax.Array


@dataclass(frozen=True)
class _StatefulInner:
    """Inner planner that increments a counter on every call and emits a
    Δv proportional to the (post-increment) count.

    Lets us verify that `inner_state` is threaded through across uploads
    AND across the H-step plan-rollout — we should see the counter grow
    monotonically.
    """

    n_vehicles: int = 0
    command_cls: Any = None

    def __call__(self, ps, view, key, t):
        del view, key, t
        new_ps = _CounterState(count=ps.count + jnp.asarray(1, dtype=ps.count.dtype))
        cmd = self.command_cls.zeros(self.n_vehicles).replace(
            dv=jnp.ones((self.n_vehicles, 2)) * 0.1 * new_ps.count.astype(jnp.float32)
        )
        return cmd, new_ps


def test_plan_cache_threads_inner_state_across_plan_and_uploads():
    """Stateful inner planners must have their state threaded across both
    the H-step plan-rollout AND the cross-upload boundary.

    With H=4, init_state runs the inner 4 times → counter 0→4. Each
    subsequent upload runs another H=4 steps. After 3 uploads at lag=0
    the total inner-call count is 4 + 4 + 4 + 4 = 16."""
    inner = _StatefulInner(n_vehicles=1, command_cls=_StubCommand)
    sch = _schedule((0.0, 100.0), (300.0, 400.0), (600.0, 700.0))
    H = 4  # noqa: N806
    policy = PlanCachePolicy(
        inner=inner,
        plan_horizon=H,
        schedule=sch,
        dt=10.0,
        replan_contacts_lag=0,
        n_vehicles=1,
        command_cls=_StubCommand,
    )
    state = policy.init_state(
        belief=_belief(),
        inner_init_state=_CounterState(count=jnp.asarray(0, dtype=jnp.int32)),
    )
    # init_state ran 4 inner calls.
    assert int(state.inner_state.count) == 4
    view_on = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([True]))
    view_off = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([False]))

    _, state = policy(state, view_on, jax.random.key(0), jnp.asarray(0.0))  # upload 2: 4→8
    _, state = policy(state, view_off, jax.random.key(0), jnp.asarray(200.0))  # no upload
    _, state = policy(state, view_on, jax.random.key(0), jnp.asarray(300.0))  # upload 3: 8→12
    _, state = policy(state, view_off, jax.random.key(0), jnp.asarray(500.0))
    _, state = policy(state, view_on, jax.random.key(0), jnp.asarray(600.0))  # upload 4: 12→16

    assert int(state.inner_state.count) == 16


def test_plan_cache_multi_step_plan_evolves_with_stateful_inner():
    """A stateful inner planner produces H different commands across the
    plan slots — verify each slot's Δv matches the expected counter step."""
    inner = _StatefulInner(n_vehicles=1, command_cls=_StubCommand)
    sch = _schedule((0.0, 100.0))
    H = 4  # noqa: N806
    policy = PlanCachePolicy(
        inner=inner,
        plan_horizon=H,
        schedule=sch,
        dt=10.0,
        replan_contacts_lag=0,
        n_vehicles=1,
        command_cls=_StubCommand,
    )
    state = policy.init_state(
        belief=_belief(),
        inner_init_state=_CounterState(count=jnp.asarray(0, dtype=jnp.int32)),
    )
    # init_state's H-step rollout: counter advances 0→4.
    # Each plan slot's Δv = 0.1 * (slot_index + 1) per axis.
    plan_norms = jnp.linalg.norm(state.plan, axis=-1).flatten()
    # slot 0 (counter went 0→1): |dv| = 0.1 * sqrt(2) ≈ 0.1414
    # slot 1 (counter went 1→2): 0.2 * sqrt(2)
    # slot 2 (counter went 2→3): 0.3 * sqrt(2)
    # slot 3 (counter went 3→4): 0.4 * sqrt(2)
    expected = jnp.asarray([0.1, 0.2, 0.3, 0.4]) * jnp.sqrt(2.0)
    assert jnp.allclose(plan_norms, expected, atol=1e-5)


def test_plan_cache_works_under_jit():
    """Lock in JIT compatibility — the rollout driver uses jax.lax.scan
    which traces once."""
    inner = _RecordingInner(n_vehicles=1, command_cls=_StubCommand)
    sch = _schedule((0.0, 100.0), (300.0, 400.0))
    policy = PlanCachePolicy(
        inner=inner,
        plan_horizon=5,
        schedule=sch,
        dt=10.0,
        replan_contacts_lag=0,
        n_vehicles=1,
        command_cls=_StubCommand,
    )
    state = policy.init_state(belief=_belief())
    view = ContactAwareBelief(inner=_belief(), contact=jnp.asarray([True]))

    @jax.jit
    def jit_call(ps, v, key, t):
        return policy(ps, v, key, t)

    cmd, state = jit_call(state, view, jax.random.key(0), jnp.asarray(0.0))
    # Plan should now contain the inner's constant 0.1 — proves upload fired
    # under JIT.
    assert float(jnp.linalg.norm(cmd.dv)) > 0.0
