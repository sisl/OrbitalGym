import jax.numpy as jnp
import pytest

from orbitalgym.config import ScenarioConfig
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    StateComponentKey,
)
from tests.helpers.minimal_scenario import minimal_scenario_kwargs


def _params():
    return AttitudeParams(
        inertia_diag=jnp.ones(3),
        omega_max=jnp.ones(3),
    )


def test_attitude_components_must_appear_together_xor_neither():
    """Attitude xor BodyRates is invalid."""
    with pytest.raises(ValueError, match="ATTITUDE.*BODY_RATES"):
        ScenarioConfig(
            **minimal_scenario_kwargs(
                guard_components=(StateComponentKey.RTN, StateComponentKey.ATTITUDE),
                bandit_components=(StateComponentKey.RTN,),
                attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
                guard_attitude_params=_params(),
                bandit_attitude_params=_params(),
            )
        )


def test_attitude_components_require_dynamics_key():
    with pytest.raises(ValueError, match="attitude_dynamics_key"):
        ScenarioConfig(
            **minimal_scenario_kwargs(
                guard_components=(
                    StateComponentKey.RTN,
                    StateComponentKey.ATTITUDE,
                    StateComponentKey.BODY_RATES,
                ),
                bandit_components=(
                    StateComponentKey.RTN,
                    StateComponentKey.ATTITUDE,
                    StateComponentKey.BODY_RATES,
                ),
                attitude_dynamics_key=None,
            )
        )


def test_dynamics_key_requires_attitude_components():
    with pytest.raises(ValueError, match="ATTITUDE.*BODY_RATES"):
        ScenarioConfig(
            **minimal_scenario_kwargs(
                guard_components=(StateComponentKey.RTN,),
                bandit_components=(StateComponentKey.RTN,),
                attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
                guard_attitude_params=_params(),
                bandit_attitude_params=_params(),
            )
        )


def test_dynamics_key_requires_attitude_params():
    with pytest.raises(ValueError, match="attitude_params"):
        ScenarioConfig(
            **minimal_scenario_kwargs(
                guard_components=(
                    StateComponentKey.RTN,
                    StateComponentKey.ATTITUDE,
                    StateComponentKey.BODY_RATES,
                ),
                bandit_components=(
                    StateComponentKey.RTN,
                    StateComponentKey.ATTITUDE,
                    StateComponentKey.BODY_RATES,
                ),
                attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
                guard_attitude_params=None,
                bandit_attitude_params=None,
            )
        )


def test_attitude_control_requires_attitude_state():
    with pytest.raises(ValueError, match="ATTITUDE_CONTROL"):
        ScenarioConfig(
            **minimal_scenario_kwargs(
                guard_components=(StateComponentKey.RTN,),
                bandit_components=(StateComponentKey.RTN,),
                guard_action_components=(ActionComponentKey.ATTITUDE_CONTROL,),
            )
        )


def test_full_attitude_config_validates_and_auto_extends_applied_torque():
    """Happy path."""
    cfg = ScenarioConfig(
        **minimal_scenario_kwargs(
            guard_components=(
                StateComponentKey.RTN,
                StateComponentKey.ATTITUDE,
                StateComponentKey.BODY_RATES,
            ),
            bandit_components=(
                StateComponentKey.RTN,
                StateComponentKey.ATTITUDE,
                StateComponentKey.BODY_RATES,
            ),
            attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
            guard_attitude_params=_params(),
            bandit_attitude_params=_params(),
        )
    )
    assert StateComponentKey.APPLIED_TORQUE in cfg.guard_components_extended
    assert StateComponentKey.APPLIED_TORQUE in cfg.bandit_components_extended


def test_attitude_control_torque_max_threads_to_action_component():
    """attitude_control_torque_max on config flows to AttitudeControl(torque_max=...)."""
    from orbitalgym.actions.components import AttitudeControl
    from orbitalgym.env.core import OrbitalGymEnv

    cfg = ScenarioConfig(
        **minimal_scenario_kwargs(
            guard_components=(
                StateComponentKey.RTN,
                StateComponentKey.ATTITUDE,
                StateComponentKey.BODY_RATES,
            ),
            bandit_components=(
                StateComponentKey.RTN,
                StateComponentKey.ATTITUDE,
                StateComponentKey.BODY_RATES,
            ),
            guard_action_components=(ActionComponentKey.ATTITUDE_CONTROL,),
            bandit_action_components=(ActionComponentKey.ATTITUDE_CONTROL,),
            attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
            guard_attitude_params=_params(),
            bandit_attitude_params=_params(),
            attitude_control_torque_max=(0.5, 0.5, 0.5),
        )
    )
    env = OrbitalGymEnv(cfg)
    guard_comps = env.guard_action_component_instances
    [att_ctrl] = [c for c in guard_comps if isinstance(c, AttitudeControl)]
    assert att_ctrl.torque_max == (0.5, 0.5, 0.5)


def test_attitude_control_torque_max_default_is_none():
    cfg = ScenarioConfig(
        **minimal_scenario_kwargs(
            guard_components=(
                StateComponentKey.RTN,
                StateComponentKey.ATTITUDE,
                StateComponentKey.BODY_RATES,
            ),
            bandit_components=(
                StateComponentKey.RTN,
                StateComponentKey.ATTITUDE,
                StateComponentKey.BODY_RATES,
            ),
            guard_action_components=(ActionComponentKey.ATTITUDE_CONTROL,),
            bandit_action_components=(ActionComponentKey.ATTITUDE_CONTROL,),
            attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
            guard_attitude_params=_params(),
            bandit_attitude_params=_params(),
        )
    )
    assert cfg.attitude_control_torque_max is None
