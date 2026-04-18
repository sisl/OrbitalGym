"""Tests for belief/gaussian.py — GaussianBelief + GaussianKalmanUpdater.

Linear-Gaussian sanity: predict preserves mean for zero control; update reduces
covariance along observed directions.
"""

import jax.numpy as jnp

from orbital_game.belief.gaussian import GaussianBelief, GaussianKalmanUpdater


def test_predict_preserves_mean_under_identity_stm():
    mean = jnp.array([1.0, 2.0, 3.0, 4.0])
    cov = jnp.eye(4)
    belief = GaussianBelief(mean=mean, cov=cov)
    upd = GaussianKalmanUpdater(
        stm=jnp.eye(4),
        control_matrix=jnp.zeros((4, 2)),
        process_noise=jnp.zeros((4, 4)),
        obs_matrix=jnp.eye(4),
        obs_noise=jnp.eye(4),
    )
    predicted = upd.predict(belief, action=jnp.zeros(2))
    assert jnp.allclose(predicted.mean, mean)


def test_predict_inflates_covariance_by_process_noise():
    belief = GaussianBelief(mean=jnp.zeros(2), cov=jnp.eye(2))
    Q = jnp.eye(2) * 0.1  # noqa: N806
    upd = GaussianKalmanUpdater(
        stm=jnp.eye(2),
        control_matrix=jnp.zeros((2, 1)),
        process_noise=Q,
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
    )
    predicted = upd.predict(belief, action=jnp.zeros(1))
    assert jnp.allclose(predicted.cov, jnp.eye(2) + Q)


def test_update_reduces_covariance_along_observed_directions():
    belief = GaussianBelief(mean=jnp.zeros(2), cov=jnp.eye(2))
    # Observe only the first component.
    H = jnp.array([[1.0, 0.0]])  # noqa: N806
    R = jnp.array([[0.01]])  # noqa: N806
    upd = GaussianKalmanUpdater(
        stm=jnp.eye(2),
        control_matrix=jnp.zeros((2, 1)),
        process_noise=jnp.zeros((2, 2)),
        obs_matrix=H,
        obs_noise=R,
    )
    observation = jnp.array([0.5])
    updated = upd.correct(belief, observation)
    # Covariance of first component should drop below prior variance (1.0).
    assert updated.cov[0, 0] < 1.0
    # Second component uncertainty effectively unchanged.
    assert jnp.isclose(updated.cov[1, 1], 1.0, atol=1e-6)


def test_joseph_and_simple_forms_agree_at_optimal_k():
    """At optimal K, Joseph form and simple form produce identical posterior covariance
    (they are algebraically equivalent; Joseph only diverges from simple when K is non-optimal
    or when floating-point drift has accumulated). This test verifies that the user-facing
    opt-out (use_joseph_form=False) gives the textbook-matching result for a single step."""
    belief = GaussianBelief(mean=jnp.zeros(3), cov=jnp.eye(3) * 2.0)
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

    joseph = GaussianKalmanUpdater(**kwargs, use_joseph_form=True).correct(belief, observation)
    simple = GaussianKalmanUpdater(**kwargs, use_joseph_form=False).correct(belief, observation)

    assert jnp.allclose(joseph.mean, simple.mean, atol=1e-7)
    assert jnp.allclose(joseph.cov, simple.cov, atol=1e-7)


def test_joseph_form_preserves_symmetry():
    """Joseph form should produce a numerically-symmetric posterior covariance even
    when the simple form might leak asymmetry through matmul rounding. Check that
    |cov - cov.T| is essentially zero for Joseph form."""
    belief = GaussianBelief(mean=jnp.zeros(4), cov=jnp.eye(4) * 1.5)
    # Slightly ill-conditioned observation to stress numerics.
    H = jnp.array([[1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.3]])  # noqa: N806
    R = jnp.diag(jnp.array([0.001, 0.001]))  # noqa: N806
    upd = GaussianKalmanUpdater(
        stm=jnp.eye(4),
        control_matrix=jnp.zeros((4, 1)),
        process_noise=jnp.zeros((4, 4)),
        obs_matrix=H,
        obs_noise=R,
        use_joseph_form=True,
    )
    updated = upd.correct(belief, observation=jnp.zeros(2))
    asymmetry = jnp.max(jnp.abs(updated.cov - updated.cov.T))
    # JAX defaults to float32; machine epsilon ~1e-7. Joseph form holds asymmetry
    # at the epsilon level; simple form can drift measurably more over many steps.
    assert asymmetry < 1e-6
