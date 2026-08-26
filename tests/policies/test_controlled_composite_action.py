"""CompositeActionPolicy — sum of learned offset and scripted base actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.actions.assemble import build_command_class
from orbitalgym.actions.components import Communicate, ImpulsiveManeuver
from orbitalgym.policies import ZeroControl
from orbitalgym.policies.controlled import CompositeActionPolicy
from orbitalgym.registry import Frame
from tests.policies._helpers import make_impulsive_maneuver_command_cls


def _impulsive_rtn() -> ImpulsiveManeuver:
    return ImpulsiveManeuver(
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=False,
    )


@dataclass(frozen=True)
class _ConstantOffset:
    """Test fixture: emits a fixed +x offset action."""

    n_vehicles: int = 1
    command_cls: Any = None

    def __call__(self, policy_state, obs, key, t):
        dv = jnp.broadcast_to(jnp.array([0.05, 0.0, 0.0]), (self.n_vehicles, 3))
        cmd = self.command_cls.zeros(self.n_vehicles).replace(dv=dv)
        return cmd, policy_state


def test_composite_sums_base_and_offset_actions():
    cmd_cls = make_impulsive_maneuver_command_cls(1)
    offset = _ConstantOffset(n_vehicles=1, command_cls=cmd_cls)
    base = ZeroControl(n_vehicles=1, command_cls=cmd_cls)
    p = CompositeActionPolicy(base=base, offset=offset, n_vehicles=1, command_cls=cmd_cls)
    obs = jnp.zeros(12)
    cmd, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # ZeroControl gives 0; _ConstantOffset gives [0.05, 0, 0]; sum = [0.05, 0, 0].
    assert jnp.array_equal(cmd.dv[0], jnp.array([0.05, 0.0, 0.0]))


@dataclass(frozen=True)
class _CommsActiveBase:
    """Test fixture: emits zero dv but `active=True` on the Communicate slot."""

    n_vehicles: int = 1
    command_cls: Any = None

    def __call__(self, policy_state, obs, key, t):
        active = jnp.ones((self.n_vehicles,), dtype=jnp.bool_)
        cmd = self.command_cls.zeros(self.n_vehicles).replace(active=active)
        return cmd, policy_state


def test_composite_preserves_base_non_dv_fields():
    """Non-`dv` Command fields propagate from the base policy through the
    composite. A base that emits `(dv=0, active=True)` should yield
    `active=True` on the combined command — even when the offset's `active`
    is False (offset is a `dv`-only contribution by contract).
    """
    cmd_cls = build_command_class(
        (_impulsive_rtn(), Communicate()), n_agents=1, class_name="TestComposite"
    )
    base = _CommsActiveBase(n_vehicles=1, command_cls=cmd_cls)
    offset = _ConstantOffset(n_vehicles=1, command_cls=cmd_cls)
    p = CompositeActionPolicy(base=base, offset=offset, n_vehicles=1, command_cls=cmd_cls)
    obs = jnp.zeros(12)
    cmd, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # dv summed: base 0 + offset 0.05.
    assert float(cmd.dv[0, 0]) == 0.05
    # active preserved from base.
    assert bool(cmd.active[0]) is True
