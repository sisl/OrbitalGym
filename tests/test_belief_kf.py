"""Tests for belief/kf.py — KFBelief + KFBeliefUpdater + initializers.

Linear-Gaussian sanity: predict preserves mean for zero control; update reduces
covariance along observed directions. Initializer tests confirm mean/cov layout.

Note: these tests still use the OLD scalar (D,) belief shape. Task 4 rewrites
them for the per-(observer, target) shape.
"""

import jax
import jax.numpy as jnp

from orbital_game.belief.kf import (
    KFBelief,
    KFBeliefUpdater,
    KFFromTruthInitializer,
    KFUniformDefaultInitializer,
)
from orbital_game.env.types import Side


def test_predict_preserves_mean_under_identity_stm():
    mean = jnp.array([1.0, 2.0, 3.0, 4.0])
    cov = jnp.eye(4)
    belief = KFBelief(mean=mean, cov=cov)
    upd = KFBeliefUpdater(
        stm=jnp.eye(4),
        control_matrix=jnp.zeros((4, 2)),
        process_noise=jnp.zeros((4, 4)),
        obs_matrix=jnp.eye(4),
        obs_noise=jnp.eye(4),
    )
    predicted = upd.predict(belief, action=jnp.zeros(2))
    assert jnp.allclose(predicted.mean, mean)


def test_predict_inflates_covariance_by_process_noise():
    belief = KFBelief(mean=jnp.zeros(2), cov=jnp.eye(2))
    Q = jnp.eye(2) * 0.1  # noqa: N806
    upd = KFBeliefUpdater(
        stm=jnp.eye(2),
        control_matrix=jnp.zeros((2, 1)),
        process_noise=Q,
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
    )
    predicted = upd.predict(belief, action=jnp.zeros(1))
    assert jnp.allclose(predicted.cov, jnp.eye(2) + Q)


def test_update_reduces_covariance_along_observed_directions():
    belief = KFBelief(mean=jnp.zeros(2), cov=jnp.eye(2))
    H = jnp.array([[1.0, 0.0]])  # noqa: N806
    R = jnp.array([[0.01]])  # noqa: N806
    upd = KFBeliefUpdater(
        stm=jnp.eye(2),
        control_matrix=jnp.zeros((2, 1)),
        process_noise=jnp.zeros((2, 2)),
        obs_matrix=H,
        obs_noise=R,
    )
    observation = jnp.array([0.5])
    updated = upd.correct(belief, observation)
    assert updated.cov[0, 0] < 1.0
    assert jnp.isclose(updated.cov[1, 1], 1.0, atol=1e-6)


def test_joseph_and_simple_forms_agree_at_optimal_k():
    belief = KFBelief(mean=jnp.zeros(3), cov=jnp.eye(3) * 2.0)
    H = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # noqa: N806
    R = jnp.diag(jnp.array([0.05, 0.1]))  # noqa: N806
    kwargs = dict(
        stm=jnp.eye(3),
        control_matrix=jnp.zeros((3, 1)),
        process_noise=jnp.zeros((3, 3)),
        obs_matrix=H,
        obs_noise=R,
    )
    observation = jnp.array([0.3, -0.2])

    joseph = KFBeliefUpdater(**kwargs, use_joseph_form=True).correct(belief, observation)
    simple = KFBeliefUpdater(**kwargs, use_joseph_form=False).correct(belief, observation)

    assert jnp.allclose(joseph.mean, simple.mean, atol=1e-7)
    assert jnp.allclose(joseph.cov, simple.cov, atol=1e-7)


def test_joseph_form_preserves_symmetry():
    belief = KFBelief(mean=jnp.zeros(4), cov=jnp.eye(4) * 1.5)
    H = jnp.array([[1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.3]])  # noqa: N806
    R = jnp.diag(jnp.array([0.001, 0.001]))  # noqa: N806
    upd = KFBeliefUpdater(
        stm=jnp.eye(4),
        control_matrix=jnp.zeros((4, 1)),
        process_noise=jnp.zeros((4, 4)),
        obs_matrix=H,
        obs_noise=R,
        use_joseph_form=True,
    )
    updated = upd.correct(belief, observation=jnp.zeros(2))
    asymmetry = jnp.max(jnp.abs(updated.cov - updated.cov.T))
    assert asymmetry < 1e-6


# --- Initializer tests (folded in from old test_belief_initializers.py) ---


class _Layout:
    flat_dim = 4

    def flatten(self, d, i):
        return jnp.concatenate([d, i])


class _Env:
    guards = jnp.array([1.0, 2.0])
    bandits = jnp.array([3.0, 4.0])


def test_kf_from_truth_uses_env_state_as_mean():
    init = KFFromTruthInitializer(layout=_Layout(), variance_diag=jnp.ones(4))
    belief = init(env_state=_Env(), side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert jnp.allclose(belief.mean, jnp.array([1.0, 2.0, 3.0, 4.0]))
    assert jnp.allclose(belief.cov, jnp.eye(4))


def test_kf_uniform_default_ignores_env_state():
    init = KFUniformDefaultInitializer(
        default_mean=jnp.array([10.0, 20.0, 30.0, 40.0]),
        variance_diag=jnp.array([1.0, 2.0, 3.0, 4.0]),
    )
    belief = init(env_state=_Env(), side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert jnp.allclose(belief.mean, jnp.array([10.0, 20.0, 30.0, 40.0]))
    assert jnp.allclose(belief.cov, jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0])))
