"""Gallery — verify every documented import path resolves."""

from __future__ import annotations


def test_every_gallery_class_imports_from_documented_path():
    import jax.numpy as jnp

    from orbitalgym.policies import UniformRandomDiscretePolicy, ZeroControl
    from orbitalgym.policies.controlled import (
        CompositeActionPolicy,
        HeuristicWithFallbackPolicy,
    )
    from orbitalgym.policies.heuristic import (
        GlideslopeIntercept,
        GlideslopeToLady,
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
    assert UniformRandomDiscretePolicy(action_grid=jnp.zeros((1, 2)))
    assert GlideslopeToLady is not None
    assert GlideslopeIntercept is not None
