from orbital_game.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    ObservationFnKey,
    StateComponentKey,
)


def test_state_component_keys():
    assert StateComponentKey.APPLIED_DV.value == "applied_dv"
    assert StateComponentKey.APPLIED_TORQUE.value == "applied_torque"


def test_action_component_keys():
    assert ActionComponentKey.ATTITUDE_CONTROL.value == "attitude_control"


def test_observation_fn_keys():
    assert ObservationFnKey.CONICAL.value == "conical_observation"


def test_attitude_dynamics_keys():
    assert AttitudeDynamicsKey.RIGID_BODY.value == "rigid_body_attitude"
