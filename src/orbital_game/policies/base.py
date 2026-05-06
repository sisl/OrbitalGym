"""Policy protocol — role-agnostic.

A Policy is a pure callable mapping ``(policy_state, agent_view, key, t)`` to
a tuple ``(command, next_policy_state)``. Stateful by default; stateless
policies pass ``None`` for ``policy_state``. The same concrete Policy class
can be wired as the guard policy, bandit policy, or learned controlled side
— only the per-side Command class and ``n_vehicles`` injection differ at
env-build.

``agent_view`` is whatever the side's perception pipeline produced this
tick. For the obs-only :func:`orbital_game.rollout.rollout` driver it is
the flattened observation vector. For the
:func:`orbital_game.rollout.belief_rollout` driver it is the per-side
:class:`orbital_game.belief.base.Belief` (an object exposing ``mean``).
Policies that only consume the state estimate read ``agent_view.mean``;
memoryless reactive policies treat ``agent_view`` as the obs vector. The
distinction is duck-typed; no separate ``Planner`` protocol is needed for
state-aware planners.

The returned ``command`` is a per-side Command pytree (built by
``actions.assemble.build_command_class``), not a raw ``(N, action_dim)``
array. Each component contributes its own field(s) to the Command;
``ZeroControl`` emits the identity (zeros) for every component.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax


class Policy(Protocol):
    """Structural protocol for a role-agnostic policy.

    Implementations are typically frozen dataclasses with a ``command_cls``
    field (the per-side Command class) and an ``n_vehicles`` field
    (defaulting to ``None``/0) that the env populates via
    ``dataclasses.replace`` at construction. The call returns a Command
    pytree — never a raw array.
    """

    def __call__(
        self,
        policy_state: Any,
        agent_view: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]: ...
