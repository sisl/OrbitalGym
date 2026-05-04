"""Gallery — verify every documented import path resolves."""

from __future__ import annotations


def test_every_gallery_class_imports_from_documented_path():
    from orbital_game.policies import ZeroControl
    from orbital_game.policies.controlled import (
        BeliefConditionedPolicy,
        CompositeActionPolicy,
        HeuristicWithFallbackPolicy,
    )
    from orbital_game.policies.heuristic import (
        JitteredPolicy,
        LeadInterceptPursuer,
        OrthogonalEvader,
        SunTrackerBlocker,
    )

    assert ZeroControl()
    assert OrthogonalEvader()
    assert LeadInterceptPursuer()
    assert SunTrackerBlocker()
    assert JitteredPolicy(base=ZeroControl(), sigma=0.0)
    assert HeuristicWithFallbackPolicy(primary=ZeroControl(), fallback=ZeroControl())
    assert CompositeActionPolicy(base=ZeroControl(), offset=ZeroControl())
    assert BeliefConditionedPolicy(base=ZeroControl())
