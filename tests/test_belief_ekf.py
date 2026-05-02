"""Tests for belief/ekf.py — EKFBelief + EKFBeliefUpdater + initializers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.belief.ekf import (
    EKFBelief,
    EKFBeliefUpdater,
    EKFFromTruthInitializer,
    EKFUniformDefaultInitializer,
)
from orbital_game.env.types import Side
from orbital_game.observations.types import Observation


def _make_belief(n_obs=1, n_total=2, d=4, mean_fill=0.0, cov_scale=1.0):
    mean = jnp.full((n_obs, n_total, d), mean_fill)
    cov = jnp.broadcast_to(jnp.eye(d) * cov_scale, (n_obs, n_total, d, d))
    return EKFBelief(mean=mean, cov=cov)


def test_ekf_with_identity_dynamics_and_linear_obs_matches_kf_predict_only():
    def identity_dynamics(x, u, dt):
        del u, dt
        return x

    upd = EKFBeliefUpdater(
        dynamics_fn=identity_dynamics,
        process_noise=jnp.zeros((4, 4)),
        dt=1.0,
    )
    belief = _make_belief(n_obs=1, n_total=2, d=4, mean_fill=0.0, cov_scale=1.0)
    out = upd(
        belief,
        observations=(),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(out.cov, belief.cov)
    assert jnp.allclose(out.mean, belief.mean)


def test_ekf_with_nonlinear_obs_fn_produces_finite_posdef_cov():
    def identity_dynamics(x, u, dt):
        del u, dt
        return x

    def range_to_origin(x):
        return jnp.array([jnp.linalg.norm(x[:3])])

    upd = EKFBeliefUpdater(
        dynamics_fn=identity_dynamics,
        process_noise=jnp.zeros((6, 6)),
        dt=1.0,
    )
    mean = jnp.array([[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]]])
    cov = jnp.broadcast_to(jnp.eye(6) * 1.0, (1, 2, 6, 6))
    belief = EKFBelief(mean=mean, cov=cov)
    chan = Observation(
        obs=jnp.full((1, 2, 1), jnp.sqrt(3.0)),
        visible=jnp.ones((1, 2), dtype=bool),
        obs_matrix=jnp.zeros((1, 6)),
        obs_noise=jnp.eye(1) * 0.01,
        obs_fn=range_to_origin,
    )
    out = upd(
        belief,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    assert bool(jnp.all(jnp.isfinite(out.cov)))
    asymmetry = jnp.max(jnp.abs(out.cov - jnp.swapaxes(out.cov, -1, -2)))
    assert float(asymmetry) < 1e-6
    diag = jnp.diagonal(out.cov, axis1=-2, axis2=-1)
    assert bool(jnp.all(diag > 0))


def test_ekf_with_linear_dynamics_and_obs_matches_kf_numerically():
    """EKF reduced to KF when both f and h are linear."""
    from orbital_game.belief.kf import KFBelief, KFBeliefUpdater

    F = jnp.eye(4) + jnp.diag(jnp.array([0.1, 0.0, 0.0, 0.0]))  # noqa: N806
    Q = jnp.eye(4) * 0.01  # noqa: N806
    H = jnp.eye(4)  # noqa: N806

    def linear_dynamics(x, u, dt):
        del u, dt
        return F @ x

    ekf = EKFBeliefUpdater(dynamics_fn=linear_dynamics, process_noise=Q, dt=1.0)
    kf = KFBeliefUpdater(stm=F, control_matrix=jnp.zeros((4, 2)), process_noise=Q)

    mean = jnp.array([[[1.0, 2.0, 3.0, 4.0]]])
    cov = jnp.broadcast_to(jnp.eye(4) * 1.5, (1, 1, 4, 4))
    ekf_belief = EKFBelief(mean=mean, cov=cov)
    kf_belief = KFBelief(mean=mean, cov=cov)

    chan = Observation(
        obs=jnp.array([[[1.0, 2.0, 3.0, 4.0]]]),
        visible=jnp.ones((1, 1), dtype=bool),
        obs_matrix=H,
        obs_noise=jnp.eye(4) * 0.05,
    )
    ekf_out = ekf(
        ekf_belief,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    kf_out = kf(
        kf_belief,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(ekf_out.mean, kf_out.mean, atol=1e-9)
    assert jnp.allclose(ekf_out.cov, kf_out.cov, atol=1e-9)


# --- Initializer tests ---


class _Layout:
    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


class _Env:
    def __init__(self, guards_rtn, bandits_rtn):
        from types import SimpleNamespace

        self.guards = SimpleNamespace(rtn=guards_rtn)
        self.bandits = SimpleNamespace(rtn=bandits_rtn)


def test_ekf_from_truth_initializer():
    n_guards = 2
    n_bandits = 1
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    guards_rtn = jnp.arange(n_guards * d, dtype=jnp.float64).reshape(n_guards, d)
    bandits_rtn = jnp.arange(n_bandits * d, dtype=jnp.float64).reshape(n_bandits, d) + 100.0
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)

    init = EKFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(d))
    belief = init(env_state=env, side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert isinstance(belief, EKFBelief)
    assert belief.mean.shape == (n_guards, n_guards + n_bandits, d)
    assert belief.cov.shape == (n_guards, n_guards + n_bandits, d, d)


def test_ekf_uniform_default_initializer():
    n_guards = 2
    n_bandits = 1
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    init = EKFUniformDefaultInitializer(
        layout=layout,
        default_mean=jnp.zeros(d),
        variance_diag=jnp.ones(d),
    )
    belief = init(env_state=None, side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert isinstance(belief, EKFBelief)
    assert belief.mean.shape == (n_guards, n_guards + n_bandits, d)
