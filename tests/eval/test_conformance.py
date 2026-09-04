"""Conformance check for externally supplied solvers."""

import dataclasses

import flax
import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.eval.conformance import check_policy_conforms
from orbitalgym.groundstations import ContactSchedule
from orbitalgym.policies.plan_cache import PlanCachePolicy
from orbitalgym.policies.zero import ZeroControl


def _env():
    cfg = make_lady_bandit_guard(n_guards=2)
    env = OrbitalGymEnv(cfg)
    init = ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=8)
    return env, init


def _schedule(*windows, pad_to=8):
    n = len(windows)
    pad = [(-1.0, -1.0)] * (pad_to - n)
    return ContactSchedule(
        windows=jnp.asarray(list(windows) + pad, dtype=jnp.float32),
        n_valid=jnp.asarray(n),
        station_ix=jnp.asarray(list(range(n)) + [-1] * (pad_to - n)),
    )


def _plan_cache_policy(env):
    inner = dataclasses.replace(ZeroControl(), command_cls=env.guard_command_cls, n_vehicles=2)
    return PlanCachePolicy(
        inner=inner,
        plan_horizon=5,
        schedule=_schedule((0.0, 100.0)),
        dt=env.config.dt,
        replan_contacts_lag=1,
        n_vehicles=2,
        command_cls=env.guard_command_cls,
    )


def test_zero_control_conforms():
    env, init = _env()
    policy = dataclasses.replace(ZeroControl(), command_cls=env.guard_command_cls, n_vehicles=2)
    report = check_policy_conforms(policy, env, Side.GUARD, init)
    assert report.command_ok and report.traceable and report.rollout_ok, report.message


def test_wrong_shape_is_reported():
    env, init = _env()

    @flax.struct.dataclass
    class Bad:
        def __call__(self, ps, view, key, t):
            return env.guard_command_cls.zeros(1), ps  # one vehicle instead of two

    report = check_policy_conforms(Bad(), env, Side.GUARD, init)
    assert not report.command_ok
    assert "leading axis" in report.message


def test_policy_state_requiring_arguments_is_reported_without_crashing():
    """PlanCachePolicy.init_state needs a belief; auto-resolving it must not raise."""
    env, init = _env()
    policy = _plan_cache_policy(env)
    report = check_policy_conforms(policy, env, Side.GUARD, init)
    assert report.message == "policy.init_state requires arguments; pass init_policy_state="
    assert not report.rollout_ok


def test_policy_conforms_with_explicit_init_policy_state():
    """Passing init_policy_state= lets a PlanCachePolicy conform fully."""
    env, init = _env()
    policy = _plan_cache_policy(env)
    state, _ = env.reset(jax.random.PRNGKey(0))
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))
    ps0 = policy.init_state(belief=belief)
    report = check_policy_conforms(policy, env, Side.GUARD, init, init_policy_state=ps0)
    assert report.command_ok and report.traceable and report.rollout_ok, report.message
