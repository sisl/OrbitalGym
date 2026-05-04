"""Tests for the three-role dynamics split (Task 6.1).

`planning_dynamics` is removed; replaced by `policy_dynamics`. A new
`belief_dynamics` field is added with a None-default that resolves to
`policy_dynamics` via `belief_dynamics_resolved` in __post_init__.
"""

from __future__ import annotations

import pytest


def test_policy_dynamics_replaces_planning():
    from orbital_game.registry import DynamicsKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg()
    assert cfg.policy_dynamics is DynamicsKey.HCW_RTN
    # planning_dynamics is removed.
    with pytest.raises(TypeError):
        _minimal_cfg(planning_dynamics=DynamicsKey.HCW_RTN)


def test_belief_dynamics_defaults_to_policy():
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg()
    assert cfg.belief_dynamics_resolved is cfg.policy_dynamics


def test_belief_dynamics_explicit_override():
    from orbital_game.registry import DynamicsKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg(belief_dynamics=DynamicsKey.HCW_RTN, policy_dynamics=DynamicsKey.HCW_RTN)
    assert cfg.belief_dynamics_resolved is DynamicsKey.HCW_RTN


def test_extended_components_not_in_json():
    import json

    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg()
    raw = json.loads(cfg.to_json())
    assert "guard_components_extended" not in raw
    assert "bandit_components_extended" not in raw
