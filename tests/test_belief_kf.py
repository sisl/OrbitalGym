"""Tests for belief/kf.py — KFBelief + KFBeliefUpdater + initializers.

New shape (Task 4):
  belief.mean: (N_obs, N_total, d)
  belief.cov:  (N_obs, N_total, d, d)
Updater consumes tuple[Observation, ...] and applies sequential corrections
gated by per-pair visibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbital_game.belief.kf import (
    KFBelief,
    KFBeliefUpdater,
    KFFromTruthInitializer,
    KFUniformDefaultInitializer,
)
from orbital_game.env.types import Side
from orbital_game.observations.types import Observation


def _make_belief(n_obs=1, n_total=2, d=4, mean_fill=0.0, cov_scale=1.0):
    mean = jnp.full((n_obs, n_total, d), mean_fill)
    cov = jnp.broadcast_to(jnp.eye(d) * cov_scale, (n_obs, n_total, d, d))
    return KFBelief(mean=mean, cov=cov)


def _identity_updater(d=4, u_dim=2):
    return KFBeliefUpdater(
        stm=jnp.eye(d),
        control_matrix=jnp.zeros((d, u_dim)),
        process_noise=jnp.zeros((d, d)),
    )


def _identity_obs_channel(n_obs, n_total, m=4, d=4, visible_value=True, obs_value=0.0):
    return Observation(
        obs=jnp.full((n_obs, n_total, m), obs_value),
        visible=jnp.full((n_obs, n_total), visible_value, dtype=bool),
        obs_matrix=jnp.eye(m, d) if m == d else jnp.eye(m, d),
        obs_noise=jnp.eye(m) * 0.01,
    )


def test_predict_only_no_observation_channels_inflates_cov_by_q_per_pair():
    belief = _make_belief(n_obs=1, n_total=3, d=4, cov_scale=1.0)
    Q = jnp.eye(4) * 0.1  # noqa: N806
    updater = KFBeliefUpdater(
        stm=jnp.eye(4),
        control_matrix=jnp.zeros((4, 2)),
        process_noise=Q,
    )
    out = updater(
        belief,
        observations=(),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    expected_cov = jnp.broadcast_to(jnp.eye(4) + Q, (1, 3, 4, 4))
    assert jnp.allclose(out.cov, expected_cov)


def test_b_times_u_applies_only_to_self_pair():
    """For observer i, predict adds B @ action[i] only to mean[i, i]."""
    n_obs = 2
    n_total = 3
    d = 4
    belief = _make_belief(n_obs=n_obs, n_total=n_total, d=d, mean_fill=0.0, cov_scale=1.0)
    B = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])  # noqa: N806
    updater = KFBeliefUpdater(
        stm=jnp.eye(d),
        control_matrix=B,
        process_noise=jnp.zeros((d, d)),
    )
    action = jnp.array([[10.0, 20.0], [30.0, 40.0]])  # (n_obs, u_dim)
    out = updater(
        belief, observations=(), action=action, side=Side.GUARD, key=jax.random.PRNGKey(0)
    )
    # Observer 0: only mean[0, 0] gets B @ action[0]; mean[0, k!=0] stays 0.
    assert jnp.allclose(out.mean[0, 0], jnp.array([10.0, 20.0, 0.0, 0.0]))
    assert jnp.allclose(out.mean[0, 1], jnp.zeros(d))
    assert jnp.allclose(out.mean[0, 2], jnp.zeros(d))
    # Observer 1: only mean[1, 1] gets B @ action[1].
    assert jnp.allclose(out.mean[1, 0], jnp.zeros(d))
    assert jnp.allclose(out.mean[1, 1], jnp.array([30.0, 40.0, 0.0, 0.0]))
    assert jnp.allclose(out.mean[1, 2], jnp.zeros(d))


def test_visible_pair_correction_reduces_cov():
    belief = _make_belief(n_obs=1, n_total=2, d=4, cov_scale=1.0)
    updater = _identity_updater(d=4, u_dim=2)
    chan = _identity_obs_channel(n_obs=1, n_total=2, m=4, d=4, visible_value=True, obs_value=0.0)
    out = updater(
        belief,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    # Both pairs visible: trace of cov shrinks below initial (=4.0).
    assert float(jnp.trace(out.cov[0, 0])) < 4.0
    assert float(jnp.trace(out.cov[0, 1])) < 4.0


def test_invisible_pair_keeps_predict_only_cov():
    belief = _make_belief(n_obs=1, n_total=2, d=4, cov_scale=1.0)
    updater = _identity_updater(d=4, u_dim=2)
    visible = jnp.array([[True, False]])
    chan = Observation(
        obs=jnp.zeros((1, 2, 4)),
        visible=visible,
        obs_matrix=jnp.eye(4),
        obs_noise=jnp.eye(4) * 0.01,
    )
    out = updater(
        belief,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    assert float(jnp.trace(out.cov[0, 0])) < 4.0  # corrected
    # Predict-only with Q=0 → cov unchanged from initial identity.
    assert jnp.allclose(out.cov[0, 1], jnp.eye(4))


def test_sequential_channels_compose():
    """Two channels both visible should drive cov below either alone."""
    belief = _make_belief(n_obs=1, n_total=1, d=4, cov_scale=1.0)
    updater = _identity_updater(d=4, u_dim=2)
    chan1 = _identity_obs_channel(n_obs=1, n_total=1, m=4, d=4, visible_value=True, obs_value=0.0)
    chan2 = _identity_obs_channel(n_obs=1, n_total=1, m=4, d=4, visible_value=True, obs_value=0.0)
    out_one = updater(
        belief,
        observations=(chan1,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    out_two = updater(
        belief,
        observations=(chan1, chan2),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    assert float(jnp.trace(out_two.cov[0, 0])) < float(jnp.trace(out_one.cov[0, 0]))


def test_kf_rejects_channel_with_obs_fn_set():
    belief = _make_belief(n_obs=1, n_total=1, d=4, cov_scale=1.0)
    updater = _identity_updater(d=4, u_dim=2)
    chan = Observation(
        obs=jnp.zeros((1, 1, 1)),
        visible=jnp.ones((1, 1), dtype=bool),
        obs_matrix=jnp.zeros((1, 4)),
        obs_noise=jnp.eye(1) * 0.01,
        obs_fn=lambda x: jnp.array([jnp.linalg.norm(x[:3])]),
    )
    with pytest.raises(TypeError, match="obs_fn"):
        updater(
            belief,
            observations=(chan,),
            action=jnp.zeros((1, 2)),
            side=Side.GUARD,
            key=jax.random.PRNGKey(0),
        )


# --- Initializer tests ---


class _Layout:
    """Stub layout that exposes per-side dynamics state arrays.

    For the new shape, initializers need: own-side states (N_self, d) and
    opposing-side states (N_tgt, d). They concatenate them along the
    tracked-entity axis.
    """

    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


class _Env:
    """Stub env state with raw RTN arrays per side."""

    def __init__(self, guards_rtn, bandits_rtn):
        from types import SimpleNamespace

        self.guards = SimpleNamespace(rtn=guards_rtn)
        self.bandits = SimpleNamespace(rtn=bandits_rtn)


def test_kf_from_truth_initializes_mean_to_truth_per_observer():
    """For observer i in the guard side, mean[i, k<N_self] = guard truth k;
    mean[i, k>=N_self] = bandit truth (k - N_self)."""
    n_guards = 2
    n_bandits = 3
    d = 6
    guards_rtn = jnp.arange(n_guards * d, dtype=jnp.float64).reshape(n_guards, d)
    bandits_rtn = jnp.arange(n_bandits * d, dtype=jnp.float64).reshape(n_bandits, d) + 100.0
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)

    init = KFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(d))
    belief = init(env_state=env, side=Side.GUARD, key=jax.random.PRNGKey(0))

    assert belief.mean.shape == (n_guards, n_guards + n_bandits, d)
    assert belief.cov.shape == (n_guards, n_guards + n_bandits, d, d)
    # Every observer i sees the same truth populated at every tracked index.
    for i in range(n_guards):
        for k in range(n_guards):
            assert jnp.allclose(belief.mean[i, k], guards_rtn[k])
        for j in range(n_bandits):
            assert jnp.allclose(belief.mean[i, n_guards + j], bandits_rtn[j])
        for k in range(n_guards + n_bandits):
            assert jnp.allclose(belief.cov[i, k], jnp.eye(d))


def test_kf_uniform_default_ignores_env_state():
    n_guards = 2
    n_bandits = 3
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    default_mean = jnp.arange(d, dtype=jnp.float64) + 7.0
    init = KFUniformDefaultInitializer(
        layout=layout,
        default_mean=default_mean,
        variance_diag=jnp.ones(d) * 2.0,
    )
    belief = init(env_state=None, side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert belief.mean.shape == (n_guards, n_guards + n_bandits, d)
    for i in range(n_guards):
        for k in range(n_guards + n_bandits):
            assert jnp.allclose(belief.mean[i, k], default_mean)
            assert jnp.allclose(belief.cov[i, k], jnp.eye(d) * 2.0)
