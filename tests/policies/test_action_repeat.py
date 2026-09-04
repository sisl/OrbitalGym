"""ActionRepeatPolicy: hold an inner command for several env steps."""

from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
import pytest

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.eval.conformance import check_policy_conforms
from orbitalgym.policies.action_repeat import ActionRepeatPolicy
from orbitalgym.policies.zero import ZeroControl


@flax.struct.dataclass
class _StubCommand:
    dv: jax.Array

    @classmethod
    def zeros(cls, n: int) -> "_StubCommand":
        return cls(dv=jnp.zeros((n, 2)))

    def replace(self, **kwargs):
        return self.__class__(**{**{"dv": self.dv}, **kwargs})


@dataclass(frozen=True)
class _CountingInner:
    """Emits ``dv = call_index`` and carries the call index as policy state."""

    n_vehicles: int = 1
    command_cls: Any = None

    def init_state(self) -> jax.Array:
        return jnp.asarray(0, dtype=jnp.int32)

    def __call__(self, ps, view, key, t):
        del view, key, t
        dv = jnp.full((self.n_vehicles, 2), ps.astype(jnp.float32) + 1.0)
        return self.command_cls.zeros(self.n_vehicles).replace(dv=dv), ps + 1


def _wrapper(repeat=3):
    inner = _CountingInner(n_vehicles=1, command_cls=_StubCommand)
    return ActionRepeatPolicy(inner=inner, repeat=repeat, n_vehicles=1, command_cls=_StubCommand)


def test_inner_runs_only_every_repeat_ticks():
    policy = _wrapper(repeat=3)
    ps = policy.init_state()
    emitted = []
    for i in range(7):
        cmd, ps = policy(ps, None, jax.random.PRNGKey(i), jnp.asarray(0.0))
        emitted.append(float(cmd.dv[0, 0]))
    # Inner called on ticks 0, 3, 6; the cached command is replayed between.
    assert emitted == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0]
    assert int(ps.inner_state) == 3


def test_repeat_one_calls_the_inner_every_tick():
    policy = _wrapper(repeat=1)
    ps = policy.init_state()
    emitted = []
    for i in range(4):
        cmd, ps = policy(ps, None, jax.random.PRNGKey(i), jnp.asarray(0.0))
        emitted.append(float(cmd.dv[0, 0]))
    assert emitted == [1.0, 2.0, 3.0, 4.0]


def test_init_state_follows_the_rollout_init_convention():
    policy = _wrapper()
    ps = policy.init_state(None, None, jax.random.PRNGKey(0))
    assert int(ps.step) == 0
    assert jnp.all(ps.cached_dv == 0.0)
    assert int(ps.inner_state) == 0


def test_rejects_bad_construction():
    with pytest.raises(ValueError, match="repeat"):
        ActionRepeatPolicy(inner=_CountingInner(), repeat=0, n_vehicles=1, command_cls=_StubCommand)
    with pytest.raises(ValueError, match="command_cls"):
        ActionRepeatPolicy(inner=_CountingInner(), repeat=2, n_vehicles=1)


def test_wrapper_passes_conformance():
    env = OrbitalGymEnv(make_lady_bandit_guard(max_horizon_s=200.0))
    inner = ZeroControl(n_vehicles=env.config.n_guards, command_cls=env.guard_command_cls)
    policy = ActionRepeatPolicy(
        inner=inner,
        repeat=3,
        n_vehicles=env.config.n_guards,
        command_cls=env.guard_command_cls,
    )
    report = check_policy_conforms(
        policy,
        env,
        Side.GUARD,
        ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=8),
    )
    assert report.command_ok, report.message
    assert report.traceable, report.message
    assert report.rollout_ok, report.message
