"""Batched belief rollouts from a bank, reduced to per-episode metrics on device."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from orbitalgym.env.types import BySide
from orbitalgym.eval.metrics import EpisodeMetrics, lbg_episode_metrics, pe_episode_metrics
from orbitalgym.games.lady_bandit_guard import LadyBanditGuard
from orbitalgym.games.pursuit_evasion import PursuitEvasion
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
    commit_radius_m: float = 2500.0,
    detect_error_m: float = 100.0,
    **rollout_kwargs: Any,
) -> EpisodeMetrics:
    """Run one belief rollout per bank episode under ``jax.vmap`` and return metrics.

    Episode ``i`` starts from ``bank_states`` index ``i`` with PRNG key
    ``split(key, n)[i]``. Only metrics come back to the host; trajectories
    and belief logs stay on device and are discarded.

    ``commit_radius_m`` is the bandit-to-lady distance that marks the
    commit step, and ``detect_error_m`` the threshold for
    ``time_to_belief_error_below_threshold_guard``. Sensor detection time
    uses logged visibility independently of this threshold. For PE, commitment
    has no meaning and ``commit_radius_m`` is ignored. Native PE capture is
    scored as ``Outcome.PE_CAPTURE`` (bandit success), and only a reached game
    deadline counts as ``TIMEOUT`` (guard survival); shorter prefixes remain
    ``UNRESOLVED``. Other games require a game-specific evaluator.
    """
    if isinstance(cfg.game, LadyBanditGuard):
        episode_metrics = lbg_episode_metrics
    elif isinstance(cfg.game, PursuitEvasion):
        episode_metrics = pe_episode_metrics
    else:
        raise ValueError(f"evaluate_bank does not support game {type(cfg.game).__name__}")
    n_episodes = int(bank_states.ic_valid.shape[0])
    keys = jax.random.split(key, n_episodes)

    def _one(initial_state, k):
        traj, belief_history = belief_rollout(
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
        return episode_metrics(
            traj,
            cfg,
            belief_guard=belief_history.guard,
            commit_radius_m=commit_radius_m,
            detect_error_m=detect_error_m,
        )

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
        "in_cone_fraction_guard": np.asarray(metrics.in_cone_fraction_guard).astype(float),
        "belief_err_guard": np.asarray(metrics.belief_err_guard).astype(float),
        "belief_err_guard_at_commit": np.asarray(metrics.belief_err_guard_at_commit).astype(float),
        "time_to_detect_guard": np.asarray(metrics.time_to_detect_guard).astype(float),
        "time_to_belief_error_below_threshold_guard": np.asarray(
            metrics.time_to_belief_error_below_threshold_guard
        ).astype(float),
        "belief_age_guard": np.asarray(metrics.belief_age_guard).astype(float),
        "dwell_catch_steps": np.asarray(metrics.dwell_catch_steps).astype(int),
        "dwell_breach_steps": np.asarray(metrics.dwell_breach_steps).astype(int),
    }
    n = int(fields["outcome"].shape[0])
    rows = []
    for i in range(n):
        row = {"episode": i}
        row.update({k: v[i].item() for k, v in fields.items()})
        row.update(constants)
        rows.append(row)
    return rows
