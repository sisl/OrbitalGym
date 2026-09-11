"""Construct modeled planning states from an explicitly selected observer.

Joint planning uses observer 0 as the representative estimate; callers must
fuse beliefs beforehand when shared information is intended. Without optional
PlanningContext, untracked fields retain the supplied template values.
"""

from __future__ import annotations

from typing import Any

import jax

from orbitalgym.env.types import Side


def belief_mean_to_env_state(
    belief_mean: jax.Array,
    side: Side,
    env: Any,
    template_env_state: Any,
    *,
    observer_index: int | jax.Array = 0,
    planning_context: Any = None,
    joint: bool = False,
) -> Any:
    """Overlay local motion, authorized own telemetry, clock and estimated dwell.

    Independent roots read only telemetry slot ``observer_index``; modeled
    teammates and opponents retain template extras. Motion always comes from
    the selected belief row, including self motion. Context contains no true
    opponent state or true global event counters.
    """
    if belief_mean.ndim != 3:
        raise ValueError(f"belief_mean must be (N_obs, N_total, d); got shape {belief_mean.shape}")
    n_self = belief_mean.shape[0]
    row = belief_mean[observer_index]
    own_dyn, opp_dyn = row[:n_self], row[n_self:]
    truth_field = {"RT": "rt", "RTN": "rtn", "ECI": "eci"}[env.truth_frame.name]
    own_name, opp_name = ("guards", "bandits") if side is Side.GUARD else ("bandits", "guards")
    own = getattr(template_env_state, own_name)
    opp = getattr(template_env_state, opp_name)
    own = own.replace(**{truth_field: own_dyn.astype(getattr(own, truth_field).dtype)})
    opp = opp.replace(**{truth_field: opp_dyn.astype(getattr(opp, truth_field).dtype)})
    extras = {}
    if planning_context is not None:
        for name, values in planning_context.own_telemetry.items():
            prior = getattr(own, name)
            current = values.astype(prior.dtype)
            extras[name] = (
                current if joint else prior.at[observer_index].set(current[observer_index])
            )
        own = own.replace(**extras)
        extras = dict(
            t=planning_context.t,
            step=planning_context.step,
            dwell_catch=planning_context.dwell_catch[observer_index],
            dwell_breach=planning_context.dwell_breach[observer_index],
        )
    return template_env_state.replace(**{own_name: own, opp_name: opp}, **extras)


def belief_mean_to_flat_state(
    belief_mean: jax.Array,
    side: Side,
    adapter: Any,
    template_env_state: Any,
    *,
    observer_index: int | jax.Array = 0,
    planning_context: Any = None,
    joint: bool = False,
) -> jax.Array:
    """Pack a modeled root; default observer/template semantics remain compatible."""
    return adapter.pack(
        belief_mean_to_env_state(
            belief_mean,
            side,
            adapter.env,
            template_env_state,
            observer_index=observer_index,
            planning_context=planning_context,
            joint=joint,
        )
    )
