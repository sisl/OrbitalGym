"""Tests for cfg.action_frame and step()'s action-frame conversion (Task 8.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def test_action_frame_default_is_rtn():
    from orbital_game.registry import Frame
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg()
    assert cfg.action_frame is Frame.RTN


def test_action_in_rtn_with_eci_truth_rotates_correctly():
    """Policy emits RTN-frame Δv; truth is ECI; Δv reaches ECI as a rotated vector."""
    from orbital_game.env.core import OrbitalGameEnv
    from orbital_game.env.types import Actions, BySide
    from orbital_game.frames.conversions import convert_action
    from orbital_game.registry import DynamicsKey, Frame, StateComponentKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg(
        truth_dynamics=DynamicsKey.KEPLERIAN_ECI,
        policy_dynamics=DynamicsKey.HCW_RTN,
        guard_components=(StateComponentKey.ECI,),
        bandit_components=(StateComponentKey.ECI,),
        action_frame=Frame.RTN,
        dt=1.0,
    )
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # 1 m/s radial impulse in RTN.
    dv_rtn = jnp.array([[1.0, 0.0, 0.0]])
    guard_cmd_with = env.guard_command_cls.zeros(1).replace(dv=dv_rtn)
    guard_cmd_zero = env.guard_command_cls.zeros(1)
    bandit_cmd_zero = env.bandit_command_cls.zeros(1)
    actions_with = Actions(sides=BySide(guard=guard_cmd_with, bandit=bandit_cmd_zero))
    actions_zero = Actions(sides=BySide(guard=guard_cmd_zero, bandit=bandit_cmd_zero))

    out_with = env.step(jax.random.PRNGKey(0), state, actions_with)
    out_zero = env.step(jax.random.PRNGKey(0), state, actions_zero)

    # Use the START-of-step reference (the same reference the env uses for the
    # rotation): the impulsive Δv is applied at the start of the interval, so
    # the rotation is anchored at the pre-step reference.
    ref_pre = state.reference_orbit  # pre-step
    ref6 = jnp.concatenate([ref_pre.position_eci, ref_pre.velocity_eci])
    expected_dv_eci = convert_action(dv_rtn, Frame.RTN, Frame.ECI, ref6)

    impulse_eci = (out_with.state.guards.eci - out_zero.state.guards.eci)[:, 3:]
    np.testing.assert_allclose(impulse_eci, expected_dv_eci, atol=1e-5)


def test_action_frame_defaults_to_truth_frame_eci():
    from orbital_game.registry import DynamicsKey, Frame, StateComponentKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg(
        truth_dynamics=DynamicsKey.KEPLERIAN_ECI,
        guard_components=(StateComponentKey.ECI,),
        bandit_components=(StateComponentKey.ECI,),
    )
    assert cfg.action_frame is Frame.ECI


def test_explicit_action_frame_rtn_with_rt_truth_rejected():
    import pytest

    from orbital_game.registry import DynamicsKey, Frame, StateComponentKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    with pytest.raises(ValueError, match="lossy"):
        _minimal_cfg(
            truth_dynamics=DynamicsKey.HCW_RT,
            guard_components=(StateComponentKey.RT,),
            bandit_components=(StateComponentKey.RT,),
            action_frame=Frame.RTN,
        )
