"""Reference termination: pure step-cap (no spatial event)."""

from __future__ import annotations

from dataclasses import dataclass

import jax

from orbital_game.registry import TerminationFnKey, register


@register(TerminationFnKey.MAX_STEPS_ONLY)
@dataclass(frozen=True)
class MaxStepsOnly:
    """Episode ends when ``state.step >= params.max_steps``. No spatial event.

    The default termination for ``NoGame``. Reads ``max_steps`` from the cfg
    at call time, so it can be constructed without a cfg in scope — which is
    what lets each ``Game`` subclass build its default termination cleanly
    inside ``default_termination_fn``.
    """

    def __call__(self, state, params, t) -> jax.Array:
        del t
        return state.step >= params.max_steps
