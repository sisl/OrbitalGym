"""HeuristicWithFallbackPolicy — routes between a primary and a fallback
based on a confidence scalar threaded through `policy_state`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.actions.assemble import build_command_class
from orbital_game.actions.components import Communicate, ImpulsiveManeuver
from orbital_game.policies import ZeroControl
from orbital_game.policies.controlled import HeuristicWithFallbackPolicy
from orbital_game.registry import Frame
from tests.policies._helpers import make_impulsive_maneuver_command_cls


def _impulsive_rtn() -> ImpulsiveManeuver:
    return ImpulsiveManeuver(
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=False,
    )


@dataclass(frozen=True)
class _ConstantConfidencePrimary:
    """Test fixture: emits +1 thrust on +x. policy_state is a confidence float."""

    n_vehicles: int = 1
    command_cls: Any = None

    def __call__(self, policy_state, obs, key, t):
        dv = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (self.n_vehicles, 3))
        cmd = self.command_cls.zeros(self.n_vehicles).replace(dv=dv)
        return cmd, policy_state


def test_routes_to_primary_when_confidence_above_threshold():
    cmd_cls = make_impulsive_maneuver_command_cls(1)
    primary = _ConstantConfidencePrimary(n_vehicles=1, command_cls=cmd_cls)
    fallback = ZeroControl(n_vehicles=1, command_cls=cmd_cls)
    p = HeuristicWithFallbackPolicy(
        primary=primary,
        fallback=fallback,
        threshold=0.5,
        n_vehicles=1,
        command_cls=cmd_cls,
    )
    obs = jnp.zeros(12)
    confidence = jnp.asarray(0.9)
    cmd, _ = p(confidence, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert float(cmd.dv[0, 0]) == 1.0


def test_routes_to_fallback_when_confidence_below_threshold():
    cmd_cls = make_impulsive_maneuver_command_cls(1)
    primary = _ConstantConfidencePrimary(n_vehicles=1, command_cls=cmd_cls)
    fallback = ZeroControl(n_vehicles=1, command_cls=cmd_cls)
    p = HeuristicWithFallbackPolicy(
        primary=primary,
        fallback=fallback,
        threshold=0.5,
        n_vehicles=1,
        command_cls=cmd_cls,
    )
    obs = jnp.zeros(12)
    confidence = jnp.asarray(0.1)
    cmd, _ = p(confidence, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert float(cmd.dv[0, 0]) == 0.0


@dataclass(frozen=True)
class _CommsActivePrimary:
    """Primary fixture that emits `(dv=[1,0,0], active=True)`."""

    n_vehicles: int = 1
    command_cls: Any = None

    def __call__(self, policy_state, obs, key, t):
        dv = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (self.n_vehicles, 3))
        active = jnp.ones((self.n_vehicles,), dtype=jnp.bool_)
        cmd = self.command_cls.zeros(self.n_vehicles).replace(dv=dv, active=active)
        return cmd, policy_state


def test_preserves_primary_non_dv_fields_regardless_of_route():
    """Non-`dv` Command fields always come from the **primary**, even when
    confidence routes the dv to the fallback. Fallback's non-`dv` fields
    are dropped by design.
    """
    cmd_cls = build_command_class(
        (_impulsive_rtn(), Communicate()), n_agents=1, class_name="TestRouting"
    )
    primary = _CommsActivePrimary(n_vehicles=1, command_cls=cmd_cls)
    fallback = ZeroControl(n_vehicles=1, command_cls=cmd_cls)
    p = HeuristicWithFallbackPolicy(
        primary=primary,
        fallback=fallback,
        threshold=0.5,
        n_vehicles=1,
        command_cls=cmd_cls,
    )
    obs = jnp.zeros(12)
    # Low confidence -> dv from fallback (zeros), but `active` from primary (True).
    confidence = jnp.asarray(0.1)
    cmd, _ = p(confidence, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert float(cmd.dv[0, 0]) == 0.0
    assert bool(cmd.active[0]) is True
    # High confidence -> dv from primary too.
    confidence = jnp.asarray(0.9)
    cmd, _ = p(confidence, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert float(cmd.dv[0, 0]) == 1.0
    assert bool(cmd.active[0]) is True
