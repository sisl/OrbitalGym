"""Distance-cap termination — ends the episode when any vehicle drifts too far.

Useful as a *runaway-drift guard* in long-horizon demos (e.g. the
``examples/lbg_groundstations_delayed_planning.py`` notebook). HCW
relative-frame dynamics with poorly-tuned controllers can secularly drift
a satellite to absurd distances given enough time; capping at, say,
10 km from the reference origin keeps the demo on-screen and surfaces a
controller-tuning bug rather than letting it accumulate silently.

Composes naturally with other terminations via :class:`AnyOfTermination`
(also in this module): the episode ends on the FIRST event that fires.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.registry import TerminationFnKey, register


def _positions(side_state):
    """Return per-vehicle (N, d) positions in whichever planar frame the side carries.

    Mirrors the helper in :mod:`orbital_game.termination.lbg_events` so the
    two terminations agree on which slice of state counts as "position".
    """
    if hasattr(side_state, "rtn"):
        return side_state.rtn[:, :3]
    return side_state.rt[:, :2]


@register(TerminationFnKey.MAX_DISTANCE)
@dataclass(frozen=True)
class MaxDistanceTermination:
    """Terminate when ANY vehicle's distance from the RTN/RT origin exceeds a cap.

    The lady (in the LBG framing) sits at the rotating-frame origin, so the
    distance from the origin is also the distance from the lady. Both guard
    and bandit positions are checked; the first to exceed ``max_distance_m``
    ends the episode.
    """

    max_distance_m: float = 10000.0

    def __call__(self, state, params, t) -> jax.Array:
        del params, t
        guard_pos = _positions(state.guards)
        bandit_pos = _positions(state.bandits)
        all_pos = jnp.concatenate([guard_pos, bandit_pos], axis=0)
        distances = jnp.linalg.norm(all_pos, axis=-1)
        return jnp.any(distances > self.max_distance_m)


@register(TerminationFnKey.ANY_OF)
@dataclass(frozen=True)
class AnyOfTermination:
    """Compose multiple terminations with a logical OR.

    Returns ``True`` as soon as any wrapped termination fires. Useful for
    layering a runaway-drift guard on top of an event-driven termination
    (catch / breach / max-steps) without rewriting either.

    Stored as a tuple so the dataclass remains frozen and hashable; build
    from any sequence at construction time.
    """

    terminations: tuple

    def __init__(self, terminations: Sequence):
        # Frozen dataclasses block attribute assignment; route via object.__setattr__
        # so we can normalise list/tuple inputs into a tuple.
        object.__setattr__(self, "terminations", tuple(terminations))

    def __call__(self, state, params, t) -> jax.Array:
        result = jnp.asarray(False)
        for term in self.terminations:
            result = jnp.logical_or(result, term(state, params, t))
        return result
