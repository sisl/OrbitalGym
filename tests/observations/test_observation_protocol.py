"""ObservationFn.__call__ accepts an actions: Actions parameter."""

import inspect

from orbital_game.observations.base import ObservationFn


def test_observation_fn_signature_has_actions():
    sig = inspect.signature(ObservationFn.__call__)
    assert "actions" in sig.parameters
