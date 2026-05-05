"""env.step under (IMPULSIVE_MANEUVER,) action components produces a next
state with pinned numerical values from a one-step rollout (parity with
the implementation that existed at Phase 2.5 / Task 2.6)."""

import jax
import jax.numpy as jnp

from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide
from orbital_game.games.lady_bandit_guard import make_lady_bandit_guard


def test_step_with_impulsive_maneuver_component_matches_legacy():
    cfg = make_lady_bandit_guard()
    env = OrbitalGameEnv(cfg)
    key = jax.random.PRNGKey(0)
    state, _ = env.reset(key)

    # Construct guard/bandit Commands with a known Δv.
    guard_cmd = env.guard_command_cls.zeros(cfg.n_guards).replace(dv=jnp.array([[0.1, 0.0, 0.0]]))
    bandit_cmd = env.bandit_command_cls.zeros(cfg.n_bandits).replace(
        dv=jnp.array([[-0.05, 0.0, 0.0]])
    )
    actions = Actions(sides=BySide(guard=guard_cmd, bandit=bandit_cmd))
    out = env.step(jax.random.PRNGKey(1), state, actions)

    # Sanity assertions.
    assert out.state.t == jnp.asarray(cfg.dt)
    # Sanity: guard radial position changed from initial.
    assert not jnp.allclose(out.state.guards.rtn[0, :3], state.guards.rtn[0, :3])

    # Pinned numerical values captured from a one-step rollout. ImpulsiveManeuver
    # wraps `convert_action` + `truth_dynamics` + rocket-equation propellant
    # deduction. Any change to ImpulsiveManeuver semantics will change these
    # and trip this test.
    expected_guard_rtn = jnp.array(
        [
            [
                -9.97174180e02,
                2.19047019e01,
                0.0,
                1.12022880e-01,
                2.18928444e00,
                0.0,
            ]
        ]
    )
    expected_bandit_rtn = jnp.array(
        [
            [
                1.00860843e03,
                -2.21502606e01,
                0.0,
                -6.21576606e-02,
                -2.21438821e00,
                0.0,
            ]
        ]
    )
    expected_guard_propellant = jnp.array([9.99490154])

    assert jnp.allclose(out.state.guards.rtn, expected_guard_rtn, atol=1e-5, rtol=1e-6)
    assert jnp.allclose(out.state.bandits.rtn, expected_bandit_rtn, atol=1e-5, rtol=1e-6)
    assert jnp.allclose(out.state.guards.propellant_mass, expected_guard_propellant, atol=1e-7)
