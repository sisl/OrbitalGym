"""Tests for LbgZeroSumReward — mirrored dense shaping plus within-step events."""

from __future__ import annotations

import jax.numpy as jnp

from orbitalgym.env.types import Side
from orbitalgym.rewards.lbg_zero_sum import LbgZeroSumReward
from orbitalgym.termination.lbg_events import LbgEventTermination


class _Cfg:
    def __init__(self, max_steps: int = 100, dt: float = 10.0):
        self.max_steps = max_steps
        self.dt = dt


class _Side:
    def __init__(self, rtn):
        self.rtn = rtn


class _State:
    def __init__(self, guards, bandits, step: int = 0):
        self.guards = guards
        self.bandits = bandits
        self.step = jnp.asarray(step)


def _state(g_xyz, b_xyz, step=0):
    g = jnp.asarray(g_xyz, dtype=jnp.float32)
    b = jnp.asarray(b_xyz, dtype=jnp.float32)
    g_full = jnp.concatenate([g, jnp.zeros_like(g)], axis=-1)
    b_full = jnp.concatenate([b, jnp.zeros_like(b)], axis=-1)
    return _State(_Side(g_full), _Side(b_full), step=step)


_FAR_GUARD = [[5000.0, 0.0, 0.0]]


def _rewards(reward, prev, nxt, cfg):
    guard = reward(prev, None, nxt, Side.GUARD, cfg, nxt.step)
    bandit = reward(prev, None, nxt, Side.BANDIT, cfg, nxt.step)
    return float(guard), float(bandit)


def test_quiet_step_is_dense_shaping_only():
    reward = LbgZeroSumReward()
    prev = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]])
    nxt = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]], step=1)
    guard, bandit = _rewards(reward, prev, nxt, _Cfg())
    assert abs(guard - (-4.0)) < 1e-3
    assert abs(bandit - (-1.0)) < 1e-3


def test_breach_bonus_paid_in_the_step_termination_fires():
    """The reward and the termination read the same within-step event."""
    reward = LbgZeroSumReward(breach_radius_m=5.0, breach_speed_mps=0.5)
    term = LbgEventTermination(breach_radius_m=5.0, breach_speed_mps=0.5)
    cfg = _Cfg(dt=40.0)
    prev = _state(_FAR_GUARD, [[-6.0, 4.9, 0.0]])
    nxt = _state(_FAR_GUARD, [[6.0, 4.9, 0.0]], step=1)

    assert bool(term(prev, nxt, cfg, nxt.step))
    guard, bandit = _rewards(reward, prev, nxt, cfg)
    assert bandit > reward.r_breach - 1.0
    assert guard < -reward.r_breach + 1.0


def test_breach_bonus_withheld_when_speed_gate_blocks_termination():
    reward = LbgZeroSumReward(breach_radius_m=5.0, breach_speed_mps=0.5)
    term = LbgEventTermination(breach_radius_m=5.0, breach_speed_mps=0.5)
    cfg = _Cfg(dt=4.0)
    prev = _state(_FAR_GUARD, [[-6.0, 4.9, 0.0]])
    nxt = _state(_FAR_GUARD, [[6.0, 4.9, 0.0]], step=1)

    assert not bool(term(prev, nxt, cfg, nxt.step))
    guard, bandit = _rewards(reward, prev, nxt, cfg)
    assert bandit < 0.0
    assert guard < 0.0


def test_catch_bonus_paid_in_the_step_termination_fires():
    reward = LbgZeroSumReward(catch_radius_m=50.0, breach_radius_m=5.0)
    term = LbgEventTermination(breach_radius_m=5.0, catch_radius_m=50.0)
    cfg = _Cfg()
    prev = _state([[0.0, 0.0, 0.0]], [[-400.0, 30.0, 0.0]])
    nxt = _state([[0.0, 0.0, 0.0]], [[400.0, 30.0, 0.0]], step=1)

    assert bool(term(prev, nxt, cfg, nxt.step))
    guard, bandit = _rewards(reward, prev, nxt, cfg)
    assert guard > reward.r_catch - 1.0
    assert bandit < -reward.r_catch + 1.0


def test_rewards_are_zero_sum():
    reward = LbgZeroSumReward(alpha=0.0, catch_radius_m=50.0, breach_radius_m=5.0)
    cfg = _Cfg()
    prev = _state([[0.0, 0.0, 0.0]], [[-400.0, 30.0, 0.0]])
    nxt = _state([[0.0, 0.0, 0.0]], [[400.0, 30.0, 0.0]], step=1)
    guard, bandit = _rewards(reward, prev, nxt, cfg)
    assert abs(guard + bandit) < 1e-5


def test_repelled_step_pays_the_guard_the_catch_bonus():
    """A bandit driven beyond the escape radius pays the same as a catch."""
    reward = LbgZeroSumReward(escape_radius_m=5000.0)
    prev = _state(_FAR_GUARD, [[6000.0, 0.0, 0.0]])
    nxt = _state(_FAR_GUARD, [[6000.0, 10.0, 0.0]], step=1)
    guard, bandit = _rewards(reward, prev, nxt, _Cfg())
    assert guard > reward.r_catch - 10.0
    assert bandit < -(reward.r_catch - 10.0)


def test_repelled_and_termination_agree():
    """The reward pays the bonus on exactly the step the termination fires."""
    reward = LbgZeroSumReward(escape_radius_m=5000.0)
    term = LbgEventTermination(breach_radius_m=5.0, escape_radius_m=5000.0)
    prev = _state(_FAR_GUARD, [[4000.0, 0.0, 0.0]])
    nxt = _state(_FAR_GUARD, [[4000.0, 10.0, 0.0]], step=1)
    guard, _ = _rewards(reward, prev, nxt, _Cfg())
    assert guard < reward.r_catch
    assert not bool(term(prev, nxt, _Cfg(), nxt.step))
