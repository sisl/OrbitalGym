"""CommsLeakObservation — bandit-side leak channel triggered by guard comms.

Emits a per-(bandit, guard) channel whose visibility mask is gated on
`actions.sides.guard.active.any()`. When any guard communicates, every
bandit gets a high-precision measurement of every guard's truth state.
When no guard communicates, the channel is all-invisible (zero-information).

This is the LBG-specific instantiation of the action-driven observation
pattern enabled by Phase 3. The same pattern supports any "agent A's
action makes agent B observe X" semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from orbital_game.belief._common import _truth_arrays_for_side
from orbital_game.observations.types import Observation
from orbital_game.registry import ObservationFnKey, register


@register(ObservationFnKey.COMMS_LEAK)
@dataclass(frozen=True)
class CommsLeakObservation:
    """Bandit-side observation channel gated on guard `communicate.active`."""

    layout: Any
    epsilon: float = 1e-6

    def __call__(self, env_state, actions, side, params, key, t):
        del params, key, t
        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        n_self = own_truth.shape[0]
        n_tgt = opp_truth.shape[0]
        n_total = n_self + n_tgt
        d = self.layout.dynamics_state_dim
        # Match the canonical (N_self, N_total, d) layout used by FullObservation,
        # OnboardGPSObservation, and RangeLimitedObservation. Own-side columns
        # (k < n_self) are filled with the own-side truth so the obs array is
        # well-defined, but the visibility mask is False for them — a comms-leak
        # observer never observes itself via this channel.
        stacked = jnp.concatenate([own_truth, opp_truth], axis=0)
        obs = jnp.broadcast_to(stacked[None, :, :], (n_self, n_total, d))
        H = jnp.eye(d)  # noqa: N806
        R = jnp.eye(d) * self.epsilon  # noqa: N806

        from orbital_game.env.types import Side as _Side

        opposing_mask = jnp.arange(n_total) >= n_self
        opposing_mask = jnp.broadcast_to(opposing_mask[None, :], (n_self, n_total))
        if side is _Side.BANDIT:
            # Fail fast: requires Communicate component on the guard side.
            # Wiring check at env-construction time lives in
            # OrbitalGameEnv.__init__ — this access just raises clearly if
            # the user bypassed the env (e.g. bare unit-test invocation).
            any_active = actions.sides.guard.active.any()
            gated = jnp.broadcast_to(any_active, (n_self, n_total))
            visible = jnp.logical_and(opposing_mask, gated)
        else:
            visible = jnp.zeros((n_self, n_total), dtype=bool)

        return (Observation(obs=obs, visible=visible, obs_matrix=H, obs_noise=R),)
