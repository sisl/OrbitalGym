"""In-depth: State layout. Source-of-truth for snippets in
docs/in-depth/state-layout.md.
"""

from __future__ import annotations


def test_indepth_state_layout():
    # --8<-- [start:basic-rtn-fields]
    import jax
    import jax.numpy as jnp

    from orbital_game import OrbitalGameEnv, make_lady_bandit_guard

    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1, seed=0)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # state.guards.rtn has shape (N_g, 6) — one row per guard, six columns:
    # [r, theta, n, r_dot, theta_dot, n_dot]  (RTN position then velocity)
    rtn = state.guards.rtn
    assert rtn.shape == (cfg.n_guards, 6)

    # Position columns:
    pos = rtn[:, 0:3]  # (N_g, 3)
    vel = rtn[:, 3:6]  # (N_g, 3)
    # --8<-- [end:basic-rtn-fields]

    # --8<-- [start:propellant-mass]
    # If the side carries the Mass component, propellant_mass has shape (N_side,).
    if hasattr(state.guards, "propellant_mass"):
        m_g = state.guards.propellant_mass
        assert m_g.shape == (cfg.n_guards,)
    # --8<-- [end:propellant-mass]

    # --8<-- [start:flat-flatten]
    # POMDPAdapter flattens state into a 1-D vector for solver consumption.
    flat = env.layout.flatten(state.guards, state.bandits)
    rebuilt_g, rebuilt_b = env.layout.unflatten(flat)
    assert bool(jnp.allclose(rebuilt_g.rtn, state.guards.rtn))
    # --8<-- [end:flat-flatten]

    # Sanity:
    assert pos.shape == (cfg.n_guards, 3)
    assert vel.shape == (cfg.n_guards, 3)
    assert flat.ndim == 1
