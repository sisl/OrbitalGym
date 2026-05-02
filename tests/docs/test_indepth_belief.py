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
