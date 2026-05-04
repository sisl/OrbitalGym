"""Tests for ScenarioConfig validation — frame-driven dynamics/component check."""

from __future__ import annotations


def test_validator_reads_frame_from_dynamics():
    """Validator should consult dynamics.frame, not a hard-coded enum table."""
    # Importing orbital_game.dynamics.hcw triggers its @register decorators,
    # which attach the Frame/DynamicsKind metadata to the registered callables.
    import orbital_game.dynamics.hcw  # noqa: F401
    from orbital_game.registry import DynamicsKey, DynamicsKind, Frame, resolve

    fn = resolve(DynamicsKey.HCW_RTN)
    assert fn.frame is Frame.RTN
    assert fn.kind is DynamicsKind.RELATIVE
