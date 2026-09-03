"""Config rules for PointAt and kinematic attitude."""

import pytest

from orbitalgym import make_lady_bandit_guard
from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
from orbitalgym.registry import ActionComponentKey, AttitudeDynamicsKey, StateComponentKey

ATT = (
    StateComponentKey.RTN,
    StateComponentKey.MASS,
    StateComponentKey.ATTITUDE,
    StateComponentKey.BODY_RATES,
)


def test_vehicle_params_default_slew_rate_is_zero():
    assert VehicleParamsSpec(100.0, 220.0, 5.0).slew_rate_rad_s == 0.0


def test_kinematic_attitude_needs_no_attitude_params():
    cfg = make_lady_bandit_guard(
        guard_components=ATT,
        attitude_dynamics_key=AttitudeDynamicsKey.KINEMATIC,
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.POINT_AT,
        ),
    )
    assert cfg.pointing_boresight_body == (1.0, 0.0, 0.0)


def test_point_at_requires_attitude_state():
    with pytest.raises(ValueError, match="POINT_AT"):
        make_lady_bandit_guard(
            guard_action_components=(
                ActionComponentKey.IMPULSIVE_MANEUVER,
                ActionComponentKey.POINT_AT,
            ),
        )


def test_point_at_and_attitude_control_are_exclusive():
    with pytest.raises(ValueError, match="POINT_AT"):
        make_lady_bandit_guard(
            guard_components=ATT,
            attitude_dynamics_key=AttitudeDynamicsKey.KINEMATIC,
            guard_action_components=(
                ActionComponentKey.IMPULSIVE_MANEUVER,
                ActionComponentKey.POINT_AT,
                ActionComponentKey.ATTITUDE_CONTROL,
            ),
        )


def test_pointing_fields_round_trip_json():
    cfg = make_lady_bandit_guard(
        guard_components=ATT,
        attitude_dynamics_key=AttitudeDynamicsKey.KINEMATIC,
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.POINT_AT,
        ),
        guard_params=VehicleParamsSpec(100.0, 220.0, 5.0, slew_rate_rad_s=0.02),
        pointing_boresight_body=(0.0, 1.0, 0.0),
    )
    cfg2 = ScenarioConfig.from_json(cfg.to_json())
    assert cfg2.guard_params.slew_rate_rad_s == 0.02
    assert cfg2.pointing_boresight_body == (0.0, 1.0, 0.0)
    assert cfg2.attitude_dynamics_key is AttitudeDynamicsKey.KINEMATIC
