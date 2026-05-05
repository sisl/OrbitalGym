import importlib

import pytest

from orbital_game import registry as _registry
from orbital_game.registry import (
    DynamicsKey,
    DynamicsKind,
    Frame,
    _clear_registry_for_tests,
    register,
    resolve,
)


def test_frame_values():
    assert Frame.RT.value == "rt"
    assert Frame.RTN.value == "rtn"
    assert Frame.ECI.value == "eci"


def test_dynamics_kind_values():
    assert DynamicsKind.RELATIVE.value == "relative"
    assert DynamicsKind.ABSOLUTE.value == "absolute"


@pytest.fixture(autouse=True)
def _clear_registry():
    """Snapshot the registry, clear it for the test, then restore.

    The dynamics module's @register decorators only run once per process (Python
    import cache). If we left the registry empty after teardown, downstream tests
    that resolve dynamics keys would fail. Snapshot/restore keeps test isolation
    *within* this module while preserving the package-level registrations that
    other tests depend on.
    """
    fwd = dict(_registry._REGISTRY)
    rev = dict(_registry._REVERSE)
    games_fwd = dict(_registry._GAME_REGISTRY)
    games_rev = dict(_registry._GAME_REVERSE)
    _clear_registry_for_tests()
    try:
        yield
    finally:
        _clear_registry_for_tests()
        _registry._REGISTRY.update(fwd)
        _registry._REVERSE.update(rev)
        _registry._GAME_REGISTRY.update(games_fwd)
        _registry._GAME_REVERSE.update(games_rev)


def test_register_attaches_frame_and_kind():
    @register(DynamicsKey.HCW_RTN, frame=Frame.RTN, kind=DynamicsKind.RELATIVE)
    def my_dyn(state, dv, params, dt):
        return state

    fn = resolve(DynamicsKey.HCW_RTN)
    assert fn.frame is Frame.RTN
    assert fn.kind is DynamicsKind.RELATIVE


def test_register_without_frame_kind_still_works():
    # Non-DynamicsKey keys (e.g. PolicyKey) don't represent dynamics, so
    # frame/kind metadata is optional. DynamicsKey itself requires both —
    # see test_register_dynamics_key_requires_frame / _kind below.
    from orbital_game.registry import PolicyKey

    @register(PolicyKey.ZERO_CONTROL)
    def f(state, dv, params, dt):
        return state

    assert resolve(PolicyKey.ZERO_CONTROL) is f
    assert getattr(f, "frame", None) is None
    assert getattr(f, "kind", None) is None


def test_register_dynamics_key_requires_frame():
    """A DynamicsKey registration without frame= should raise ValueError."""
    import pytest

    from orbital_game.registry import DynamicsKey, DynamicsKind, register

    with pytest.raises(ValueError, match="frame=Frame"):

        @register(DynamicsKey.HCW_RT, kind=DynamicsKind.RELATIVE)
        def my_bad_dyn(state, dv, params, dt):
            return state


def test_register_dynamics_key_requires_kind():
    """A DynamicsKey registration without kind= should raise ValueError."""
    import pytest

    from orbital_game.registry import DynamicsKey, Frame, register

    with pytest.raises(ValueError, match="kind=DynamicsKind"):

        @register(DynamicsKey.HCW_RT, frame=Frame.RT)
        def my_bad_dyn(state, dv, params, dt):
            return state


def test_register_non_dynamics_key_allows_no_metadata():
    """Non-DynamicsKey (e.g. PolicyKey) registrations work without frame/kind."""
    from dataclasses import dataclass

    from orbital_game.registry import PolicyKey, register, resolve

    @register(PolicyKey.ZERO_CONTROL)
    @dataclass(frozen=True)
    class FakePolicy:
        knob: bool = False

    fn = resolve(PolicyKey.ZERO_CONTROL)
    assert fn is FakePolicy


def test_hcw_rt_has_frame_and_kind():
    # The autouse _clear_registry fixture wipes the registry dicts before
    # this test runs, but Python's import cache means orbital_game.dynamics.hcw
    # was already imported (its @register decorators ran once at process start).
    # Reload the module so the decorators re-run and re-populate the registry.
    from orbital_game.dynamics import hcw

    importlib.reload(hcw)
    fn = resolve(DynamicsKey.HCW_RT)
    assert fn.frame is Frame.RT
    assert fn.kind is DynamicsKind.RELATIVE


def test_hcw_rtn_has_frame_and_kind():
    from orbital_game.dynamics import hcw

    importlib.reload(hcw)
    fn = resolve(DynamicsKey.HCW_RTN)
    assert fn.frame is Frame.RTN
    assert fn.kind is DynamicsKind.RELATIVE
