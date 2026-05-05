from orbital_game.registry import ActionComponentKey


def test_action_component_key_values():
    assert ActionComponentKey.IMPULSIVE_MANEUVER.value == "impulsive_maneuver"
    assert ActionComponentKey.COMMUNICATE.value == "communicate"
