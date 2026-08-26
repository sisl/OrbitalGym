"""Tests for APPLIED_DV auto-extension in ScenarioConfig (Task 2.1).

Every spatial-frame scenario must have APPLIED_DV on the extended components
so env.step's translational-dynamics block always has an applied_dv field to
consume.
"""

from __future__ import annotations

from orbitalgym.config import ScenarioConfig
from orbitalgym.registry import (
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from tests.helpers.minimal_scenario import minimal_scenario_kwargs


def test_applied_dv_auto_extended_for_rt():
    cfg = ScenarioConfig(
        **minimal_scenario_kwargs(
            truth_dynamics=DynamicsKey.HCW_RT,
            action_frame=Frame.RT,
            guard_components=(StateComponentKey.RT,),
            bandit_components=(StateComponentKey.RT,),
        )
    )
    assert StateComponentKey.APPLIED_DV in cfg.guard_components_extended
    assert StateComponentKey.APPLIED_DV in cfg.bandit_components_extended


def test_applied_dv_auto_extended_for_rtn():
    cfg = ScenarioConfig(
        **minimal_scenario_kwargs(
            truth_dynamics=DynamicsKey.HCW_RTN,
            action_frame=Frame.RTN,
            guard_components=(StateComponentKey.RTN,),
            bandit_components=(StateComponentKey.RTN,),
        )
    )
    assert StateComponentKey.APPLIED_DV in cfg.guard_components_extended
    assert StateComponentKey.APPLIED_DV in cfg.bandit_components_extended


def test_applied_dv_appears_only_once():
    """User explicitly listed it — auto-extension must not duplicate."""
    cfg = ScenarioConfig(
        **minimal_scenario_kwargs(
            truth_dynamics=DynamicsKey.HCW_RTN,
            action_frame=Frame.RTN,
            guard_components=(StateComponentKey.RTN, StateComponentKey.APPLIED_DV),
            bandit_components=(StateComponentKey.RTN,),
        )
    )
    assert list(cfg.guard_components_extended).count(StateComponentKey.APPLIED_DV) == 1
