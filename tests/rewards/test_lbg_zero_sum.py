"""Tests for LbgZeroSumReward — mirrored dense shaping plus within-step events."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from orbitalgym.actions.components import G0
from orbitalgym.config import VehicleParamsSpec
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

    inside_prev = _state(_FAR_GUARD, [[4000.0, 0.0, 0.0]])
    inside_next = _state(_FAR_GUARD, [[4000.0, 10.0, 0.0]], step=1)
    guard, bandit = _rewards(reward, inside_prev, inside_next, _Cfg())
    assert guard < reward.r_catch
    assert bandit > -reward.r_catch
    assert not bool(term(inside_prev, inside_next, _Cfg(), inside_next.step))

    outside_prev = _state(_FAR_GUARD, [[6000.0, 0.0, 0.0]])
    outside_next = _state(_FAR_GUARD, [[6000.0, 10.0, 0.0]], step=1)
    guard, bandit = _rewards(reward, outside_prev, outside_next, _Cfg())
    assert guard == pytest.approx(reward.r_catch, abs=10.0)
    assert bandit == pytest.approx(-reward.r_catch, abs=10.0)
    assert bool(term(outside_prev, outside_next, _Cfg(), outside_next.step))


class _MassSide:
    def __init__(self, rtn, propellant_mass):
        self.rtn = rtn
        self.propellant_mass = jnp.asarray(propellant_mass, dtype=jnp.float32)


class _FueledCfg(_Cfg):
    def __init__(self, guard_params, bandit_params, max_steps: int = 100, dt: float = 10.0):
        super().__init__(max_steps=max_steps, dt=dt)
        self.guard_params = guard_params
        self.bandit_params = bandit_params


def _mass_state(g_xyz, b_xyz, g_prop, b_prop, step=0):
    plain = _state(g_xyz, b_xyz, step=step)
    return _State(
        _MassSide(plain.guards.rtn, g_prop),
        _MassSide(plain.bandits.rtn, b_prop),
        step=step,
    )


def _burn_propellant(propellant_kg, dv_mps, vehicle):
    """Propellant left after an impulse, under the env's rocket-equation update."""
    wet = vehicle.dry_mass_kg + propellant_kg
    assert dv_mps < vehicle.max_thrust_n * 10.0 / wet
    return propellant_kg - wet * (1.0 - jnp.exp(-dv_mps / (vehicle.isp_s * G0)))


def test_dv_cost_charges_the_burning_side_its_delta_v():
    guard_params = VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=50.0)
    bandit_params = VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=20.0)
    cfg = _FueledCfg(guard_params, bandit_params)
    dv_mps = 0.25

    prev = _mass_state(_FAR_GUARD, [[1000.0, 0.0, 0.0]], [10.0], [8.0])
    nxt = _mass_state(
        _FAR_GUARD,
        [[1000.0, 0.0, 0.0]],
        [_burn_propellant(10.0, dv_mps, guard_params)],
        [8.0],
        step=1,
    )

    free_guard, free_bandit = _rewards(LbgZeroSumReward(), prev, nxt, cfg)
    charged_guard, charged_bandit = _rewards(LbgZeroSumReward(dv_cost=3.0), prev, nxt, cfg)

    assert charged_guard == pytest.approx(free_guard - 3.0 * dv_mps, abs=1e-4)
    assert charged_bandit == pytest.approx(free_bandit, abs=1e-6)


def test_dv_cost_of_zero_reproduces_the_massless_rewards():
    cfg = _FueledCfg(
        VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=50.0),
        VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=20.0),
    )
    fueled_prev = _mass_state(_FAR_GUARD, [[1000.0, 0.0, 0.0]], [10.0], [8.0])
    fueled_next = _mass_state(_FAR_GUARD, [[1000.0, 0.0, 0.0]], [9.0], [7.5], step=1)

    plain_prev = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]])
    plain_next = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]], step=1)

    assert _rewards(LbgZeroSumReward(), fueled_prev, fueled_next, cfg) == pytest.approx(
        _rewards(LbgZeroSumReward(), plain_prev, plain_next, _Cfg())
    )


def test_dv_cost_is_skipped_for_a_side_without_a_mass_component():
    """The bandit here has no propellant trace, so it pays nothing."""
    cfg = _FueledCfg(
        VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=50.0),
        VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=20.0),
    )
    plain = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]])
    guard_only_prev = _State(_MassSide(plain.guards.rtn, [10.0]), plain.bandits)
    guard_only_next = _State(_MassSide(plain.guards.rtn, [9.5]), plain.bandits, step=1)

    _, free_bandit = _rewards(LbgZeroSumReward(), guard_only_prev, guard_only_next, cfg)
    charged_guard, charged_bandit = _rewards(
        LbgZeroSumReward(dv_cost=1.0), guard_only_prev, guard_only_next, cfg
    )
    assert charged_bandit == pytest.approx(free_bandit)
    assert charged_guard < 0.0
