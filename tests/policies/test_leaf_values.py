"""Reward-scale leaf values for search on LBG."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.policies.leaf_values import bandit_leaf_value, guard_leaf_value

ALPHA = 1e-3
R_CATCH = 1000.0
R_BREACH = 1000.0
CATCH_RADIUS_M = 50.0
BREACH_RADIUS_M = 5.0
V_CLOSE = 0.5
N_REMAINING = 32

PARAMS = dict(
    alpha=ALPHA,
    r_catch=R_CATCH,
    r_breach=R_BREACH,
    catch_radius_m=CATCH_RADIUS_M,
    breach_radius_m=BREACH_RADIUS_M,
    v_close_guard_mps=V_CLOSE,
    v_close_bandit_mps=V_CLOSE,
    n_remaining=N_REMAINING,
)


def _adapter(action_repeat=1, discount=0.99):
    env = OrbitalGymEnv(make_lady_bandit_guard())
    return POMDPAdapter(env, action_repeat=action_repeat, discount=discount)


def _state_with(adapter, guard_r, bandit_r):
    state, _ = adapter.env.reset(jax.random.PRNGKey(0))
    guards = state.guards.replace(rtn=jnp.array([[guard_r, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    bandits = state.bandits.replace(rtn=jnp.array([[bandit_r, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    return adapter.pack(state.replace(guards=guards, bandits=bandits))


def test_guard_value_rises_as_the_guard_closes_on_the_bandit():
    adapter = _adapter()
    value = guard_leaf_value(adapter, **PARAMS)
    near = _state_with(adapter, guard_r=950.0, bandit_r=1000.0)
    far = _state_with(adapter, guard_r=100.0, bandit_r=1000.0)
    assert float(value(near)) > float(value(far))


def test_guard_value_falls_as_the_bandit_closes_on_the_lady():
    adapter = _adapter()
    value = guard_leaf_value(adapter, **PARAMS)
    # Guard parked far from both, so only the breach term moves.
    safe = _state_with(adapter, guard_r=-5000.0, bandit_r=1000.0)
    threatened = _state_with(adapter, guard_r=-5000.0, bandit_r=100.0)
    assert float(value(threatened)) < float(value(safe))


def test_bandit_value_rises_as_the_bandit_closes_on_the_lady():
    adapter = _adapter()
    value = bandit_leaf_value(adapter, **PARAMS)
    # Guard parked far from both, so only the breach term moves.
    close = _state_with(adapter, guard_r=-5000.0, bandit_r=100.0)
    distant = _state_with(adapter, guard_r=-5000.0, bandit_r=1000.0)
    assert float(value(close)) > float(value(distant))


def test_bandit_value_falls_as_the_guard_closes_on_the_bandit():
    adapter = _adapter()
    value = bandit_leaf_value(adapter, **PARAMS)
    chased = _state_with(adapter, guard_r=950.0, bandit_r=1000.0)
    free = _state_with(adapter, guard_r=0.0, bandit_r=1000.0)
    assert float(value(chased)) < float(value(free))


def test_values_reduce_to_the_shaping_sum_at_huge_distances():
    """Terminal terms discount away when both events are far in the future."""
    adapter = _adapter(discount=0.9)
    g = adapter.discount()
    horizon = (1.0 - g**N_REMAINING) / (1.0 - g)
    d = 1.0e7
    s = _state_with(adapter, guard_r=0.0, bandit_r=d)

    assert jnp.allclose(guard_leaf_value(adapter, **PARAMS)(s), -ALPHA * d * horizon)
    assert jnp.allclose(bandit_leaf_value(adapter, **PARAMS)(s), -ALPHA * d * horizon)


def test_terminal_bonus_is_undiscounted_inside_the_radius():
    adapter = _adapter(discount=0.9)
    g = adapter.discount()
    # Guard on top of the bandit (inside the catch radius) and the bandit far
    # from the lady: t_catch is zero, so r_catch is paid in full.
    s = _state_with(adapter, guard_r=1.0e7, bandit_r=1.0e7)
    t_breach = (1.0e7 - BREACH_RADIUS_M) / (V_CLOSE * adapter.macro_dt)
    expected = R_CATCH - R_BREACH * g**t_breach
    assert jnp.allclose(guard_leaf_value(adapter, **PARAMS)(s), expected)


def test_undiscounted_horizon_uses_the_arithmetic_sum():
    adapter = _adapter(discount=1.0)
    assert adapter.discount() == 1.0
    s = _state_with(adapter, guard_r=0.0, bandit_r=200.0)
    value = float(guard_leaf_value(adapter, **PARAMS)(s))
    assert jnp.isfinite(value)
    assert jnp.allclose(value, -ALPHA * 200.0 * N_REMAINING + R_CATCH - R_BREACH)


def test_macro_step_shortens_the_estimated_time_to_an_event():
    """A longer macro step covers the same distance in fewer macro steps."""
    fine = _adapter(action_repeat=1)
    coarse = _adapter(action_repeat=4)
    params = dict(PARAMS, discount=0.99)
    v_fine = float(bandit_leaf_value(fine, **params)(_state_with(fine, 0.0, 500.0)))
    v_coarse = float(bandit_leaf_value(coarse, **params)(_state_with(coarse, 0.0, 500.0)))
    assert v_coarse > v_fine
