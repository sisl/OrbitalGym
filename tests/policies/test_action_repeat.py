"""ActionRepeatPolicy: hold an inner command for several env steps."""

from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
import pytest

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters._command_flatten import flatten_command
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.env.types import Actions, BySide
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


def test_rollout_matches_the_macro_step_adapter():
    """The flown trajectory equals the searched one at every macro boundary."""
    repeat = 3
    n_macro = 4
    env = OrbitalGymEnv(make_lady_bandit_guard(max_horizon_s=600.0))
    adapter = POMDPAdapter(env, action_repeat=repeat)
    n_g, n_b = env.config.n_guards, env.config.n_bandits

    dv_table = jnp.array(
        [[0.02, 0.0, 0.0], [0.0, 0.03, 0.0], [-0.01, 0.0, 0.02], [0.0, -0.02, 0.0]]
    )
    dv_dim = int(env.guard_command_cls.zeros(n_g).dv.shape[-1])

    @dataclass(frozen=True)
    class _ScriptedInner:
        n_vehicles: int = 0
        command_cls: Any = None

        def init_state(self):
            return jnp.asarray(0, dtype=jnp.int32)

        def __call__(self, ps, view, key, t):
            del view, key, t
            dv = jnp.broadcast_to(
                dv_table[ps % dv_table.shape[0], :dv_dim], (self.n_vehicles, dv_dim)
            )
            return self.command_cls.zeros(self.n_vehicles).replace(dv=dv), ps + 1

    inner = _ScriptedInner(n_vehicles=n_g, command_cls=env.guard_command_cls)
    policy = ActionRepeatPolicy(
        inner=inner, repeat=repeat, n_vehicles=n_g, command_cls=env.guard_command_cls
    )

    state, _ = env.reset(jax.random.PRNGKey(0))
    ps = policy.init_state()
    macro_keys = jax.random.split(jax.random.PRNGKey(7), n_macro)
    searched = adapter.pack(state)

    for j in range(n_macro):
        substep_keys = jax.random.split(macro_keys[j], repeat)
        macro_cmd = None
        for k_sub in substep_keys:
            cmd, ps = policy(ps, adapter.pack(state), jax.random.PRNGKey(0), state.t)
            if macro_cmd is None:
                macro_cmd = cmd
            # The wrapper holds one command for the whole macro step.
            assert jnp.array_equal(cmd.dv, macro_cmd.dv)
            actions = Actions(sides=BySide(guard=cmd, bandit=env.bandit_command_cls.zeros(n_b)))
            state = env.step(k_sub, state, actions).state

        a_flat = jnp.concatenate(
            [flatten_command(macro_cmd), flatten_command(env.bandit_command_cls.zeros(n_b))]
        )
        searched = adapter.transition(searched, a_flat, macro_keys[j])
        assert jnp.allclose(searched, adapter.pack(state))
