"""Controlled-side policy cookbook — source-of-truth for snippets in
docs/extending/controlled-policy-cookbook.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


def test_controlled_cookbook_walkthrough():
    # --8<-- [start:imports]

    from orbital_game.actions.assemble import build_command_class
    from orbital_game.actions.components import ImpulsiveManeuver
    from orbital_game.policies import ZeroControl
    from orbital_game.policies.controlled import (
        BeliefConditionedPolicy,
        CompositeActionPolicy,
        HeuristicWithFallbackPolicy,
    )
    from orbital_game.registry import Frame
    # --8<-- [end:imports]

    # An ImpulsiveManeuver-only Command class is enough for the cookbook patterns
    # (each emits a single dv field, no other components).
    cmd_cls = build_command_class((ImpulsiveManeuver,), 1, "_CookbookCommand")

    # --8<-- [start:scripted-with-fallback-pattern]
    @dataclass(frozen=True)
    class _Primary:
        n_vehicles: int = 1
        command_cls: Any = None

        def __call__(self, ps, obs, key, t):
            dv = jnp.broadcast_to(jnp.array([0.05, 0.0, 0.0]), (self.n_vehicles, 3))
            return self.command_cls.zeros(self.n_vehicles).replace(dv=dv), ps

    primary = _Primary(command_cls=cmd_cls)
    fallback = ZeroControl(command_cls=cmd_cls, n_vehicles=1)
    policy = HeuristicWithFallbackPolicy(
        primary=primary,
        fallback=fallback,
        threshold=0.5,
        n_vehicles=1,
        command_cls=cmd_cls,
    )
    # Pass confidence through policy_state. High confidence → primary.
    confident = jnp.asarray(0.9)
    cmd_high, _ = policy(confident, jnp.zeros(12), jax.random.PRNGKey(0), jnp.asarray(0))
    # Low confidence → fallback (ZeroControl).
    unconfident = jnp.asarray(0.1)
    cmd_low, _ = policy(unconfident, jnp.zeros(12), jax.random.PRNGKey(0), jnp.asarray(0))
    # --8<-- [end:scripted-with-fallback-pattern]

    assert float(cmd_high.dv[0, 0]) == 0.05
    assert float(cmd_low.dv[0, 0]) == 0.0

    # --8<-- [start:composite-action-pattern]
    @dataclass(frozen=True)
    class _LearnedOffset:
        n_vehicles: int = 1
        command_cls: Any = None

        def __call__(self, ps, obs, key, t):
            # In real use: a NN forward pass.
            dv = jnp.broadcast_to(jnp.array([0.01, 0.0, 0.0]), (self.n_vehicles, 3))
            return self.command_cls.zeros(self.n_vehicles).replace(dv=dv), ps

    base = ZeroControl(command_cls=cmd_cls, n_vehicles=1)
    composite = CompositeActionPolicy(
        base=base, offset=_LearnedOffset(command_cls=cmd_cls), n_vehicles=1, command_cls=cmd_cls
    )
    cmd, _ = composite(None, jnp.zeros(12), jax.random.PRNGKey(0), jnp.asarray(0))
    # --8<-- [end:composite-action-pattern]

    assert float(cmd.dv[0, 0]) == 0.01

    # --8<-- [start:belief-conditioned-pattern]
    @dataclass(frozen=True)
    class _BeliefAware:
        """Sees augmented obs (belief prepended) and uses the belief mean."""

        n_vehicles: int = 1
        command_cls: Any = None

        def __call__(self, ps, obs, key, t):
            # First 3 obs floats are the belief mean (prepended by wrapper).
            dv = jnp.broadcast_to(obs[:3] * 0.001, (self.n_vehicles, 3))
            return self.command_cls.zeros(self.n_vehicles).replace(dv=dv), ps

    base_aware = _BeliefAware(command_cls=cmd_cls)
    wrapped = BeliefConditionedPolicy(base=base_aware, n_vehicles=1, command_cls=cmd_cls)
    obs_raw = jnp.zeros(6)
    belief_mean = jnp.array([10.0, 20.0, 30.0])
    state = (belief_mean, None)  # (belief_mean, base_state)
    bc_cmd, _ = wrapped(state, obs_raw, jax.random.PRNGKey(0), jnp.asarray(0))
    # --8<-- [end:belief-conditioned-pattern]

    # Suppress unused warning — Frame is imported for completeness in the snippet.
    _ = Frame
    assert jnp.allclose(bc_cmd.dv[0], jnp.array([0.01, 0.02, 0.03]))
