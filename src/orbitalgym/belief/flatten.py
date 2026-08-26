"""Adapter from a Belief's per-pair mean to a flat state vector.

`MCTSPolicy` (and any planner whose ``__call__`` consumes ``adapter.pack(state)``)
expects a 1-D flat state. KF/EKF belief means are shaped ``(N_obs, N_total, d)``
— one row per (observer, tracked-entity) pair. This module bridges the two
worlds without baking belief-awareness into the planner.

Approach: take observer 0's row of the belief mean (which carries that
observer's view of every tracked entity, own + opposing), write those
dynamics-state arrays into the per-side truth fields of a *template*
``EnvState``, then call ``adapter.pack`` to flatten consistently with how
the env produces its flat-state representation. Components that the belief
does not track (mass, attitude, body rates, …) are inherited from the
template — typically the env's reset state — so any planner that reads
those tail components sees scenario-default values.

Usage::

    from orbitalgym.belief.flatten import belief_mean_to_flat_state

    env, _ = env.reset(jax.random.key(0))[0], None  # capture once
    s_flat = belief_mean_to_flat_state(
        belief.mean, side, adapter, template_env_state=env_state,
    )
    cmd, _ = mcts_policy(None, s_flat, key, t)

Limitations:

- The belief is assumed to track the env's truth frame. Mixed-frame
  scenarios (e.g. RTN truth but RT-only belief) require a frame-aware
  conversion before this call.
- Per-vehicle "extras" (mass, attitude, …) come from the template and do
  not vary across the planning rollout. Planners that depend on those
  components should be designed accordingly.
"""

from __future__ import annotations

from typing import Any

import jax

from orbitalgym.env.types import Side


def belief_mean_to_flat_state(
    belief_mean: jax.Array,  # (N_obs, N_total, d)
    side: Side,
    adapter: Any,  # POMDPAdapter — has .pack(state) and .env
    template_env_state: Any,
) -> jax.Array:
    """Convert observer-0's per-target view to a flat state vector matching
    ``adapter.pack(template_env_state)``'s layout.

    ``belief_mean`` carries observer-0's estimate of all (n_self + n_opp)
    target dynamics states (each of length ``d``). We write these into the
    per-side truth fields of ``template_env_state`` and use
    ``adapter.pack`` to flatten the result. The flat-state layout is
    therefore identical to what ``adapter.pack(true_state)`` produces, so
    the planner never sees a "belief-shaped" state directly.
    """
    if belief_mean.ndim != 3:
        raise ValueError(f"belief_mean must be (N_obs, N_total, d); got shape {belief_mean.shape}")

    n_obs, n_total, _ = belief_mean.shape
    n_self = n_obs

    # Observer 0's view of all targets. Targets [0, n_self) are this side's
    # vehicles; targets [n_self, n_total) are the opposing side's.
    obs0 = belief_mean[0]  # (n_total, d)
    self_dyn = obs0[:n_self]
    opp_dyn = obs0[n_self:]

    if side is Side.GUARD:
        guard_dyn, bandit_dyn = self_dyn, opp_dyn
    else:
        guard_dyn, bandit_dyn = opp_dyn, self_dyn

    # Determine the truth field name from the env. `env.truth_frame` is set
    # by `OrbitalGymEnv.__init__`. We compute the field name without
    # importing the private `_truth_field` helper to keep the dependency
    # surface narrow.
    truth_frame = adapter.env.truth_frame
    truth_field = {
        "RT": "rt",
        "RTN": "rtn",
        "ECI": "eci",
    }[truth_frame.name]

    new_guards = template_env_state.guards.replace(
        **{truth_field: guard_dyn.astype(getattr(template_env_state.guards, truth_field).dtype)}
    )
    new_bandits = template_env_state.bandits.replace(
        **{truth_field: bandit_dyn.astype(getattr(template_env_state.bandits, truth_field).dtype)}
    )
    new_env_state = template_env_state.replace(guards=new_guards, bandits=new_bandits)
    return adapter.pack(new_env_state)
