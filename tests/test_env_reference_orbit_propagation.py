import jax
import jax.numpy as jnp


def _zero_actions(env):
    from orbitalgym.env.types import Actions, BySide

    return Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(env.config.n_guards),
            bandit=env.bandit_command_cls.zeros(env.config.n_bandits),
        )
    )


def test_reference_orbit_advances_under_keplerian():
    """Default reference dynamics is KEPLERIAN_ECI for relative truth.

    Position must change between steps.
    """
    from orbitalgym.env.core import OrbitalGymEnv
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg(dt=60.0, max_horizon_s=600.0)
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    initial_pos = state.reference_orbit.position_eci
    actions = _zero_actions(env)
    next_step = env.step(jax.random.PRNGKey(1), state, actions)
    next_pos = next_step.state.reference_orbit.position_eci
    delta = float(jnp.linalg.norm(next_pos - initial_pos))
    assert delta > 1.0, (
        f"Reference orbit position should advance ~7 km in 60s of LEO motion; got {delta}m"
    )


def test_hcw_state_finite_after_propagation():
    """HCW relative state must remain finite (no NaN) when the reference orbit propagates."""
    from orbitalgym.env.core import OrbitalGymEnv
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg()
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(42))
    actions = _zero_actions(env)
    out = env.step(jax.random.PRNGKey(1), state, actions)
    assert not jnp.any(jnp.isnan(out.state.guards.rtn))
