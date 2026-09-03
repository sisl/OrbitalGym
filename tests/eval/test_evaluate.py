"""Batched evaluation from a stored bank."""

import jax

from orbitalgym.eval.bank import sample_bank
from orbitalgym.eval.evaluate import evaluate_bank, metrics_to_records
from orbitalgym.eval.metrics import Outcome


def test_evaluate_bank_returns_per_episode_metrics(make_minimal_lbg_env_with_kf):
    env, belief_init, belief_upd, init_ps_fns, policies = make_minimal_lbg_env_with_kf()
    states = sample_bank(env, n_episodes=4, seed=11)
    metrics = evaluate_bank(
        env,
        env.config,
        states,
        jax.random.PRNGKey(0),
        n_steps=6,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
    )
    assert metrics.outcome.shape == (4,)
    assert metrics.dv_guard.shape == (4,)
    assert all(int(o) == Outcome.TIMEOUT for o in metrics.outcome)
    assert all(int(s) == 6 for s in metrics.steps)


def test_metrics_to_records_adds_constants(make_minimal_lbg_env_with_kf):
    env, belief_init, belief_upd, init_ps_fns, policies = make_minimal_lbg_env_with_kf()
    states = sample_bank(env, n_episodes=2, seed=3)
    metrics = evaluate_bank(
        env,
        env.config,
        states,
        jax.random.PRNGKey(0),
        n_steps=3,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
    )
    rows = metrics_to_records(metrics, bank="lbg-1v1", regime="DEC")
    assert len(rows) == 2
    assert rows[0]["episode"] == 0 and rows[1]["episode"] == 1
    assert rows[0]["regime"] == "DEC"
    assert isinstance(rows[0]["outcome"], int)
    assert isinstance(rows[0]["dv_guard"], float)
