"""Reward-scale leaf values for search on LBG."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.policies.leaf_values import (
    bandit_leaf_value,
    bandit_leaf_value_from_game,
    guard_leaf_value,
    guard_leaf_value_from_game,
)
from orbitalgym.rewards.lbg_zero_sum import lbg_potential
from orbitalgym.rewards.reference import DistanceToReferenceOrbit

SHAPING_GAIN = 1.0
SHAPING_SCALE_M = 300.0
HOME_WEIGHT = 0.0
R_CATCH = 1000.0
R_BREACH = 1000.0
CATCH_RADIUS_M = 50.0
BREACH_RADIUS_M = 5.0
V_CLOSE = 0.5
N_REMAINING = 32

PARAMS = dict(
    shaping_gain=SHAPING_GAIN,
    shaping_scale_m=SHAPING_SCALE_M,
    home_weight=HOME_WEIGHT,
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


def _potential(adapter, s_flat):
    return float(lbg_potential(adapter.unpack(s_flat), SHAPING_SCALE_M, HOME_WEIGHT))


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


def test_the_two_sides_are_exact_mirrors():
    adapter = _adapter()
    s = _state_with(adapter, guard_r=-800.0, bandit_r=400.0)
    guard = float(guard_leaf_value(adapter, **PARAMS)(s))
    bandit = float(bandit_leaf_value(adapter, **PARAMS)(s))
    assert guard + bandit == pytest.approx(0.0, abs=1e-4)


def test_the_leaf_is_the_terminal_estimate_plus_the_potential():
    """The value-initialization form: the leaf carries Phi rather than cancelling it."""
    adapter = _adapter()
    s = _state_with(adapter, guard_r=-1000.0, bandit_r=800.0)
    bare = float(guard_leaf_value(adapter, **dict(PARAMS, shaping_gain=0.0))(s))
    with_phi = float(guard_leaf_value(adapter, **PARAMS)(s))
    assert with_phi == pytest.approx(bare + SHAPING_GAIN * _potential(adapter, s), abs=1e-3)


def test_the_gain_scales_the_potential_the_leaf_carries():
    adapter = _adapter()
    s = _state_with(adapter, guard_r=-1000.0, bandit_r=800.0)
    bare = float(guard_leaf_value(adapter, **dict(PARAMS, shaping_gain=0.0))(s))
    doubled = float(guard_leaf_value(adapter, **dict(PARAMS, shaping_gain=2.0))(s))
    assert doubled == pytest.approx(bare + 2.0 * _potential(adapter, s), abs=1e-3)


def test_the_potential_ranks_leaves_whose_terminal_estimates_are_equal():
    """Beyond the leaf horizon both bonuses weigh nothing and only Phi separates states."""
    adapter = _adapter(discount=1.0)
    value = guard_leaf_value(adapter, **PARAMS)
    # Both events sit past N * v_close * macro_dt, so both weights are zero.
    close = _state_with(adapter, guard_r=-1000.0, bandit_r=3000.0)
    distant = _state_with(adapter, guard_r=-3000.0, bandit_r=3000.0)
    bare = guard_leaf_value(adapter, **dict(PARAMS, shaping_gain=0.0))
    assert float(bare(close)) == pytest.approx(float(bare(distant)), abs=1e-9)
    assert float(value(close)) > float(value(distant))
    assert float(value(close)) - float(value(distant)) == pytest.approx(
        SHAPING_GAIN * (_potential(adapter, close) - _potential(adapter, distant)), abs=1e-4
    )


def test_values_reduce_to_the_potential_at_huge_distances():
    """Terminal terms discount away when both events are far in the future."""
    adapter = _adapter(discount=0.9)
    d = 1.0e7
    s = _state_with(adapter, guard_r=-d, bandit_r=d)
    phi = _potential(adapter, s)

    assert jnp.allclose(guard_leaf_value(adapter, **PARAMS)(s), SHAPING_GAIN * phi)
    assert jnp.allclose(bandit_leaf_value(adapter, **PARAMS)(s), -SHAPING_GAIN * phi)


def test_terminal_bonus_is_undiscounted_inside_the_radius():
    adapter = _adapter(discount=0.9)
    g = adapter.discount()
    # Guard on top of the bandit (inside the catch radius) and the bandit far
    # from the lady: t_catch is zero, so r_catch is paid in full.
    s = _state_with(adapter, guard_r=1.0e7, bandit_r=1.0e7)
    t_breach = (1.0e7 - BREACH_RADIUS_M) / (V_CLOSE * adapter.macro_dt)
    expected = R_CATCH - R_BREACH * g**t_breach + SHAPING_GAIN * _potential(adapter, s)
    assert jnp.allclose(guard_leaf_value(adapter, **PARAMS)(s), expected)


def _linear_weight(t_hit):
    return max(0.0, 1.0 - t_hit / N_REMAINING)


def test_undiscounted_terminal_weights_stay_distance_aware():
    """At g == 1 the bonuses fall off linearly in the estimated time to go."""
    adapter = _adapter(discount=1.0)
    assert adapter.discount() == 1.0
    per_macro_step_m = V_CLOSE * adapter.macro_dt
    s = _state_with(adapter, guard_r=0.0, bandit_r=200.0)
    value = float(guard_leaf_value(adapter, **PARAMS)(s))
    w_catch = _linear_weight((200.0 - CATCH_RADIUS_M) / per_macro_step_m)
    w_breach = _linear_weight((200.0 - BREACH_RADIUS_M) / per_macro_step_m)
    expected = R_CATCH * w_catch - R_BREACH * w_breach - SHAPING_GAIN * _potential(adapter, s)
    assert jnp.isfinite(value)
    assert jnp.allclose(value, expected)


def test_undiscounted_terminal_terms_vary_with_distance():
    """A nearer catch is worth more at g == 1, not the same constant bonus."""
    adapter = _adapter(discount=1.0)
    value = guard_leaf_value(adapter, **PARAMS)
    per_macro_step_m = V_CLOSE * adapter.macro_dt
    s_near = _state_with(adapter, guard_r=980.0, bandit_r=1000.0)
    s_far = _state_with(adapter, guard_r=800.0, bandit_r=1000.0)
    near = float(value(s_near))
    far = float(value(s_far))
    # Both catches sit inside the horizon, so only the catch weight and the
    # potential differ, and they pull the same way.
    gap = R_CATCH * (
        _linear_weight(max(20.0 - CATCH_RADIUS_M, 0.0) / per_macro_step_m)
        - _linear_weight((200.0 - CATCH_RADIUS_M) / per_macro_step_m)
    ) + SHAPING_GAIN * (_potential(adapter, s_near) - _potential(adapter, s_far))
    assert near > far
    assert jnp.allclose(near - far, gap)


def test_macro_step_shortens_the_estimated_time_to_an_event():
    """A longer macro step covers the same distance in fewer macro steps."""
    fine = _adapter(action_repeat=1, discount=1.0)
    coarse = _adapter(action_repeat=4, discount=1.0)
    v_fine = float(bandit_leaf_value(fine, **PARAMS)(_state_with(fine, -500.0, 500.0)))
    v_coarse = float(bandit_leaf_value(coarse, **PARAMS)(_state_with(coarse, -500.0, 500.0)))
    assert v_coarse > v_fine


def test_from_game_matches_explicit_reward_weights():
    cfg = make_lady_bandit_guard()
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env)
    s = _state_with(adapter, guard_r=-800.0, bandit_r=400.0)
    speeds = dict(v_close_guard_mps=V_CLOSE, v_close_bandit_mps=V_CLOSE, n_remaining=N_REMAINING)
    explicit = dict(
        shaping_gain=cfg.reward_fn.shaping_gain,
        shaping_scale_m=cfg.reward_fn.shaping_scale_m,
        home_weight=cfg.reward_fn.home_weight,
        r_catch=cfg.reward_fn.r_catch,
        r_breach=cfg.reward_fn.r_breach,
        catch_radius_m=cfg.reward_fn.catch_radius_m,
        breach_radius_m=cfg.reward_fn.breach_radius_m,
    )
    assert float(guard_leaf_value_from_game(adapter, cfg, **speeds)(s)) == float(
        guard_leaf_value(adapter, **explicit, **speeds)(s)
    )
    assert float(bandit_leaf_value_from_game(adapter, cfg, **speeds)(s)) == float(
        bandit_leaf_value(adapter, **explicit, **speeds)(s)
    )


def test_from_game_tracks_a_retuned_reward():
    cfg = make_lady_bandit_guard()
    retuned = dataclasses.replace(
        cfg, reward_fn=dataclasses.replace(cfg.reward_fn, shaping_gain=25.0)
    )
    adapter = POMDPAdapter(OrbitalGymEnv(cfg))
    s = _state_with(adapter, guard_r=-800.0, bandit_r=400.0)
    speeds = dict(v_close_guard_mps=V_CLOSE, v_close_bandit_mps=V_CLOSE, n_remaining=N_REMAINING)
    assert float(guard_leaf_value_from_game(adapter, retuned, **speeds)(s)) != float(
        guard_leaf_value_from_game(adapter, cfg, **speeds)(s)
    )


def test_from_game_rejects_a_reward_it_cannot_read():
    cfg = make_lady_bandit_guard()
    adapter = POMDPAdapter(OrbitalGymEnv(cfg))
    other = dataclasses.replace(cfg, reward_fn=DistanceToReferenceOrbit())
    with pytest.raises(TypeError, match="LbgZeroSumReward"):
        guard_leaf_value_from_game(
            adapter, other, v_close_guard_mps=V_CLOSE, v_close_bandit_mps=V_CLOSE
        )
