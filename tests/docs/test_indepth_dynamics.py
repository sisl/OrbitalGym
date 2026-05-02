"""In-depth: Dynamics. Source-of-truth for snippets in
docs/in-depth/dynamics.md.
"""

from __future__ import annotations


def test_indepth_dynamics_hcw_rtn_step():
    # --8<-- [start:hcw-rtn-step]
    import jax
    import jax.numpy as jnp

    from orbital_game import OrbitalGameEnv, make_lady_bandit_guard

    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1, seed=0)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # `env.truth_dynamics` is the bound dynamics callable. It takes the raw
    # per-side dynamics array (shape (N_side, 6) for HCW-RTN), a Δv impulse
    # of shape (N_side, action_dim), the per-side `VehicleParams`, and a
    # scalar `dt`. It returns the next-step raw array of the same shape.
    guard_rtn = state.guards.rtn  # (N_g, 6)
    zero_dv = jnp.zeros((cfg.n_guards, 3))
    next_guard_rtn = env.truth_dynamics(guard_rtn, zero_dv, env.guard_params, cfg.dt)
    assert next_guard_rtn.shape == guard_rtn.shape

    # The same callable handles the bandit side — symmetric API.
    bandit_rtn = state.bandits.rtn  # (N_b, 6)
    next_bandit_rtn = env.truth_dynamics(
        bandit_rtn,
        jnp.zeros((cfg.n_bandits, 3)),
        env.bandit_params,
        cfg.dt,
    )
    assert next_bandit_rtn.shape == bandit_rtn.shape
    # --8<-- [end:hcw-rtn-step]


def test_indepth_dynamics_actuator_composition():
    # --8<-- [start:actuator-composition]
    import jax
    import jax.numpy as jnp

    from orbital_game import Actions, BySide, OrbitalGameEnv, make_lady_bandit_guard

    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1, seed=0)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # `env.step` runs the actuator (action → applied Δv) and then the
    # dynamics step (state + Δv → next state) for each side.
    actions = Actions(
        sides=BySide(
            guard=jnp.zeros((cfg.n_guards, 3)),
            bandit=jnp.zeros((cfg.n_bandits, 3)),
        )
    )
    step_output = env.step(jax.random.PRNGKey(1), state, actions)

    # The dynamics step preserves the per-side state shape.
    assert step_output.state.guards.rtn.shape == state.guards.rtn.shape
    assert step_output.state.bandits.rtn.shape == state.bandits.rtn.shape
    # --8<-- [end:actuator-composition]
