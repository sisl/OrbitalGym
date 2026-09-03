"""Conformance check for externally supplied solvers."""

import dataclasses

import flax

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.eval.conformance import check_policy_conforms
from orbitalgym.policies.zero import ZeroControl


def _env():
    cfg = make_lady_bandit_guard(n_guards=2)
    env = OrbitalGymEnv(cfg)
    init = ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=8)
    return env, init


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
