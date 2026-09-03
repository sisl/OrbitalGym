"""Batched belief rollouts from a bank, reduced to per-episode metrics on device."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from orbitalgym.env.types import BySide
from orbitalgym.eval.metrics import EpisodeMetrics, lbg_episode_metrics
from orbitalgym.rollout import belief_rollout


def evaluate_bank(
    env: Any,
    cfg: Any,
    bank_states: Any,
    key: jax.Array,
    n_steps: int,
    *,
    policies: BySide,
    init_policy_state_fns: BySide,
    belief_initializers: BySide,
    belief_updaters: BySide,
    **rollout_kwargs: Any,
) -> EpisodeMetrics:
    """Run one belief rollout per bank episode under ``jax.vmap`` and return metrics.

    Episode ``i`` starts from ``bank_states`` index ``i`` with PRNG key
    ``split(key, n)[i]``. Only metrics come back to the host; trajectories
    stay on device and are discarded.
    """
    n_episodes = int(bank_states.ic_valid.shape[0])
    keys = jax.random.split(key, n_episodes)

    def _one(initial_state, k):
        traj, _ = belief_rollout(
            env,
            policies,
            init_policy_state_fns,
            belief_initializers,
            belief_updaters,
            k,
            n_steps,
            initial_state=initial_state,
            **rollout_kwargs,
        )
        return lbg_episode_metrics(traj, cfg)

    return jax.vmap(_one)(bank_states, keys)


def metrics_to_records(metrics: EpisodeMetrics, **constants: Any) -> list[dict[str, Any]]:
    """One plain-Python dict per episode, with ``constants`` copied into every row."""
    fields = {
        "outcome": np.asarray(metrics.outcome).astype(int),
        "steps": np.asarray(metrics.steps).astype(int),
        "min_d_gb": np.asarray(metrics.min_d_gb).astype(float),
        "min_d_bl": np.asarray(metrics.min_d_bl).astype(float),
        "dv_guard": np.asarray(metrics.dv_guard).astype(float),
        "dv_bandit": np.asarray(metrics.dv_bandit).astype(float),
        "link_events_guard": np.asarray(metrics.link_events_guard).astype(int),
        "ic_valid": np.asarray(metrics.ic_valid).astype(bool),
    }
    n = int(fields["outcome"].shape[0])
    rows = []
    for i in range(n):
        row = {"episode": i}
        row.update({k: v[i].item() for k, v in fields.items()})
        row.update(constants)
        rows.append(row)
    return rows
