"""In-depth: Belief. Source-of-truth for snippets in
docs/in-depth/belief.md.
"""

from __future__ import annotations


def test_indepth_belief_kf_shape():
    # --8<-- [start:kf-belief-shape]
    import jax
    import jax.numpy as jnp

    from orbital_game import OrbitalGameEnv, Side, make_pursuit_evasion
    from orbital_game.belief.kf import KFFromTruthInitializer

    cfg = make_pursuit_evasion(n_guards=1, n_bandits=2, seed=0)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    d = env.layout.dynamics_state_dim
    init = KFFromTruthInitializer(
        layout=env.layout,
        variance_diag=jnp.ones(d) * 1e-2,
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))

    # Belief mean shape: (N_obs, N_total, d)  where
    # N_obs = vehicles on the requesting side
    # N_total = N_obs + N_targets
    n_obs = cfg.n_guards
    n_total = cfg.n_guards + cfg.n_bandits
    assert belief.mean.shape == (n_obs, n_total, d)
    assert belief.cov.shape == (n_obs, n_total, d, d)
    # --8<-- [end:kf-belief-shape]

    # --8<-- [start:belief-indexing]
    # Guard 0's estimate of bandit 1 lives at:
    bandit1_estimate_mean = belief.mean[0, cfg.n_guards + 1]  # (d,)
    bandit1_estimate_cov = belief.cov[0, cfg.n_guards + 1]  # (d, d)
    # --8<-- [end:belief-indexing]

    assert bandit1_estimate_mean.shape == (d,)
    assert bandit1_estimate_cov.shape == (d, d)


def test_indepth_belief_pf_shape():
    # --8<-- [start:pf-belief-shape]
    import jax
    import jax.numpy as jnp

    from orbital_game import OrbitalGameEnv, Side, make_lady_bandit_guard
    from orbital_game.belief.pf import ParticleFilterRingInitializer
    from orbital_game.reference_orbit import mean_motion as ref_mean_motion
    from orbital_game.registry import DynamicsKey, StateComponentKey

    # RT-plane LBG ring scenario — the canonical PF use case.
    cfg = make_lady_bandit_guard(
        n_guards=1,
        n_bandits=1,
        guard_components=(StateComponentKey.RT,),
        bandit_components=(StateComponentKey.RT,),
        truth_dynamics=DynamicsKey.HCW_RT,
        policy_dynamics=DynamicsKey.HCW_RT,
        dt=10.0,
        max_horizon_s=200.0,
    )
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    K = 128  # noqa: N806  # standard PF notation for particle count
    init = ParticleFilterRingInitializer(
        layout=env.layout,
        ring_radius_m=2000.0,
        mean_motion_rad_s=float(ref_mean_motion(cfg.reference_orbit)),
        n_particles=K,
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))

    n_obs = cfg.n_guards
    n_total = cfg.n_guards + cfg.n_bandits
    d = env.layout.dynamics_state_dim
    assert belief.particles.shape == (n_obs, n_total, K, d)
    assert belief.log_weights.shape == (n_obs, n_total, K)
    assert belief.n_eff.shape == (n_obs, n_total)
    assert belief.weight_entropy.shape == (n_obs, n_total)
    assert belief.resampled.shape == (n_obs, n_total)
    # mean is a property — same shape as KF/EKF mean.
    assert belief.mean.shape == (n_obs, n_total, d)

    # Initial state: uniform weights → max N_eff and max entropy, no resamples.
    assert bool(jnp.allclose(belief.n_eff, float(K)))
    assert bool(jnp.allclose(belief.weight_entropy, jnp.log(float(K))))
    assert not bool(jnp.any(belief.resampled))
    # --8<-- [end:pf-belief-shape]
