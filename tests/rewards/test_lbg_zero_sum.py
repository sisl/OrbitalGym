"""Tests for LbgZeroSumReward — potential shaping, per-side costs, and events."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from orbitalgym.actions.components import G0
from orbitalgym.config import VehicleParamsSpec
from orbitalgym.env.types import Side
from orbitalgym.rewards.lbg_zero_sum import (
    LbgZeroSumReward,
    guard_separation_cost,
    lbg_potential,
)
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
# Off the lady by more than lady_keepout_m, and inside the catch radius of a
# bandit crossing the along-track axis at 30 m.
_CATCHING_GUARD = [[0.0, 0.0, -30.0]]


def _rewards(reward, prev, nxt, cfg):
    guard = reward(prev, None, nxt, Side.GUARD, cfg, nxt.step)
    bandit = reward(prev, None, nxt, Side.BANDIT, cfg, nxt.step)
    return float(guard), float(bandit)


def test_a_frozen_step_pays_nothing():
    """No motion means no potential difference, and no event to pay for."""
    reward = LbgZeroSumReward()
    prev = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]])
    nxt = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]], step=1)
    guard, bandit = _rewards(reward, prev, nxt, _Cfg())
    assert guard == pytest.approx(0.0, abs=1e-4)
    assert bandit == pytest.approx(0.0, abs=1e-4)


def test_potential_rises_as_the_guard_closes_and_falls_as_the_bandit_does():
    reward = LbgZeroSumReward()
    far = reward.potential(_state([[2000.0, 0.0, 0.0]], [[1000.0, 0.0, 0.0]]))
    near = reward.potential(_state([[1100.0, 0.0, 0.0]], [[1000.0, 0.0, 0.0]]))
    threatened = reward.potential(_state([[2000.0, 0.0, 0.0]], [[100.0, 0.0, 0.0]]))
    assert float(near) > float(far)
    assert float(threatened) < float(far)


def test_home_weight_penalizes_a_guard_that_leaves_the_lady():
    state = _state([[2000.0, 0.0, 0.0]], [[1000.0, 0.0, 0.0]])
    home = float(lbg_potential(state, 300.0, 0.5))
    away = float(lbg_potential(state, 300.0, 0.0))
    assert home == pytest.approx(away - 0.5 * 2000.0 / 300.0, abs=1e-4)


def _shaping_sequence():
    """A guard closing on a bandit that is itself closing on the lady."""
    return [
        _state([[g, 0.0, 0.0]], [[b, 0.0, 0.0]], step=i)
        for i, (g, b) in enumerate(
            [(3000.0, 2000.0), (2600.0, 1700.0), (2100.0, 1500.0), (1600.0, 1400.0)]
        )
    ]


def test_shaping_telescopes_to_the_endpoint_potentials():
    """At g == 1 the shaping rewards over any sequence sum to Phi(s_T) - Phi(s_0)."""
    reward = LbgZeroSumReward(shaping_discount=1.0, r_catch=0.0, r_breach=0.0)
    cfg = _Cfg()
    states = _shaping_sequence()
    total = sum(
        float(reward(prev, None, nxt, Side.GUARD, cfg, nxt.step))
        for prev, nxt in zip(states[:-1], states[1:], strict=True)
    )
    expected = float(reward.potential(states[-1]) - reward.potential(states[0]))
    assert total == pytest.approx(expected, abs=1e-4)


def test_shaping_is_zero_sum_between_the_sides():
    reward = LbgZeroSumReward(r_catch=0.0, r_breach=0.0, separation_cost=0.0)
    cfg = _Cfg()
    states = _shaping_sequence()
    for prev, nxt in zip(states[:-1], states[1:], strict=True):
        guard, bandit = _rewards(reward, prev, nxt, cfg)
        assert guard + bandit == pytest.approx(0.0, abs=1e-5)
        assert guard != pytest.approx(0.0, abs=1e-3)


def test_shaping_gain_of_zero_leaves_a_terminal_only_reward():
    reward = LbgZeroSumReward(shaping_gain=0.0, separation_cost=0.0)
    cfg = _Cfg()
    states = _shaping_sequence()
    for prev, nxt in zip(states[:-1], states[1:], strict=True):
        assert _rewards(reward, prev, nxt, cfg) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_shaping_discount_scales_the_next_state_potential():
    reward = LbgZeroSumReward(shaping_discount=0.9, r_catch=0.0, r_breach=0.0)
    cfg = _Cfg()
    prev, nxt = _shaping_sequence()[:2]
    guard, _ = _rewards(reward, prev, nxt, cfg)
    expected = float(0.9 * reward.potential(nxt) - reward.potential(prev))
    assert guard == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    ("distance_m", "expected"),
    [(0.0, 10.0), (10.0, 2.5), (20.0, 0.0), (25.0, 0.0)],
)
def test_lady_keepout_hinge_is_squared_and_stops_at_the_radius(distance_m, expected):
    state = _state([[distance_m, 0.0, 0.0]], [[5000.0, 0.0, 0.0]])
    cost = guard_separation_cost(state, 20.0, 20.0, 10.0)
    assert float(cost) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    ("distance_m", "expected"),
    [(0.0, 10.0), (10.0, 2.5), (20.0, 0.0), (25.0, 0.0)],
)
def test_guard_pair_hinge_is_squared_and_stops_at_the_radius(distance_m, expected):
    state = _state([[5000.0, 0.0, 0.0], [5000.0 + distance_m, 0.0, 0.0]], [[9000.0, 0.0, 0.0]])
    cost = guard_separation_cost(state, 20.0, 20.0, 10.0)
    assert float(cost) == pytest.approx(expected, abs=1e-5)


def test_a_single_guard_pays_only_the_lady_term():
    """One guard has no pairs, so a nil keepout leaves nothing to charge."""
    state = _state([[5.0, 0.0, 0.0]], [[5000.0, 0.0, 0.0]])
    assert float(guard_separation_cost(state, 20.0, 0.0, 10.0)) == pytest.approx(0.0)
    assert float(guard_separation_cost(state, 20.0, 20.0, 10.0)) > 0.0


def test_the_separation_cost_falls_on_the_guard_alone():
    reward = LbgZeroSumReward(shaping_gain=0.0)
    cfg = _Cfg()
    prev = _state([[5.0, 0.0, 0.0]], [[3000.0, 0.0, 0.0]])
    nxt = _state([[5.0, 0.0, 0.0]], [[3000.0, 0.0, 0.0]], step=1)
    guard, bandit = _rewards(reward, prev, nxt, cfg)
    expected = float(guard_separation_cost(nxt, 20.0, 20.0, 10.0))
    assert expected > 0.0
    assert guard == pytest.approx(-expected, abs=1e-4)
    assert bandit == pytest.approx(0.0, abs=1e-6)


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
    assert abs(guard) < reward.r_breach
    assert abs(bandit) < reward.r_breach


def test_catch_bonus_paid_in_the_step_termination_fires():
    reward = LbgZeroSumReward(catch_radius_m=50.0, breach_radius_m=5.0)
    term = LbgEventTermination(breach_radius_m=5.0, catch_radius_m=50.0)
    cfg = _Cfg()
    prev = _state(_CATCHING_GUARD, [[-400.0, 30.0, 0.0]])
    nxt = _state(_CATCHING_GUARD, [[400.0, 30.0, 0.0]], step=1)

    assert bool(term(prev, nxt, cfg, nxt.step))
    guard, bandit = _rewards(reward, prev, nxt, cfg)
    assert guard > reward.r_catch - 1.0
    assert bandit < -reward.r_catch + 1.0


def test_rewards_are_zero_sum_without_the_per_side_costs():
    """Shaping and terminal events mirror; only fuel and separation break it."""
    reward = LbgZeroSumReward(separation_cost=0.0, catch_radius_m=50.0, breach_radius_m=5.0)
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
