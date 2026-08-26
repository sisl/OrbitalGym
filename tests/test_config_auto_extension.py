"""Tests for ScenarioConfig auto-extension of per-side components (Task 6.2).

The config computes `guard_components_extended` and `bandit_components_extended`
in __post_init__ from the union of frames consumed by the three per-side resolved
dynamics roles (truth, policy, belief_dynamics_resolved) plus Frame.RTN for viz.
The extended tuples are the canonical input to `_COMP_LOOKUP` for state-class
assembly.
"""

from __future__ import annotations


def test_no_extension_when_all_roles_share_frame():
    from orbitalgym.registry import StateComponentKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg()
    # APPLIED_DV is auto-extended onto every spatial-frame side as the
    # transient carrier from action components to translational dynamics.
    assert cfg.guard_components_extended == (
        StateComponentKey.RTN,
        StateComponentKey.APPLIED_DV,
    )
    assert cfg.bandit_components_extended == (
        StateComponentKey.RTN,
        StateComponentKey.APPLIED_DV,
    )


def test_extension_when_truth_eci_policy_rtn():
    from orbitalgym.registry import DynamicsKey, StateComponentKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg(
        truth_dynamics=DynamicsKey.KEPLERIAN_ECI,
        policy_dynamics=DynamicsKey.HCW_RTN,
        guard_components=(StateComponentKey.ECI,),
        bandit_components=(StateComponentKey.ECI,),
    )
    # ECI is canonical (truth); RTN is added as a derived view.
    assert StateComponentKey.ECI in cfg.guard_components_extended
    assert StateComponentKey.RTN in cfg.guard_components_extended


def test_viz_always_forces_rtn():
    """Even when no role needs RTN, viz does — RTN must be in the extended set."""
    from orbitalgym.registry import DynamicsKey, StateComponentKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg(
        truth_dynamics=DynamicsKey.KEPLERIAN_ECI,
        policy_dynamics=DynamicsKey.KEPLERIAN_ECI,
        belief_dynamics=DynamicsKey.KEPLERIAN_ECI,
        guard_components=(StateComponentKey.ECI,),
        bandit_components=(StateComponentKey.ECI,),
    )
    assert StateComponentKey.RTN in cfg.guard_components_extended


def test_frame_less_policy_dynamics_raises():
    """Per the fail-fast rule, a callable without Frame metadata in a policy
    role must error at config construction, not silently fall through."""
    import pytest

    from tests.test_config_resolved_dynamics import _minimal_cfg

    def my_bad_dyn(state, dv, params, dt):
        return state  # has no .frame, no .kind

    with pytest.raises(ValueError, match="Frame metadata"):
        _minimal_cfg(policy_dynamics=my_bad_dyn)
