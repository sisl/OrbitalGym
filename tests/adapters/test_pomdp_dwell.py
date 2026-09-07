"""Dwell counters ride in the POMDP adapter's flat state across substeps."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.belief.flatten import belief_mean_to_flat_state
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec

_DWELL_STEPS = 5
_ACTION_REPEAT = 3
_R_CATCH = 1000.0


def _setup():
    """A co-located guard and bandit, so every step is inside the catch radius."""
    cfg = make_lady_bandit_guard(
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=0.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=0.0,
            ),
        ),
        catch_dwell_steps=_DWELL_STEPS,
        shaping_gain=0.0,
        max_horizon_s=200.0,
    )
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env, action_repeat=_ACTION_REPEAT)
    state, _ = env.reset(jax.random.PRNGKey(0))
    action = jnp.zeros(adapter.guard_action_flat_dim + adapter.bandit_action_flat_dim)
    return env, adapter, state, action


def _displaced(state, metres: float):
    """The same state with the bandit pushed radially away from the guard."""
    offset = jnp.asarray([metres, 0.0, 0.0])
    rtn = state.bandits.rtn.at[:, :3].add(offset)
    return state.replace(bandits=state.bandits.replace(rtn=rtn))


def test_flat_state_round_trips_the_counters():
    _env, adapter, state, _action = _setup()
    packed = adapter.pack(
        state.replace(dwell_catch=jnp.asarray([4]), dwell_breach=jnp.asarray([2]))
    )
    restored = adapter.unpack(packed)
    assert packed.shape == (adapter.states_dim,)
    assert restored.dwell_catch.tolist() == [4]
    assert restored.dwell_breach.tolist() == [2]
    assert restored.dwell_catch.dtype == jnp.int32


def test_dwell_accumulates_across_substeps_and_macro_steps():
    """A hold of five steps completes on the second macro step of three."""
    _env, adapter, state, action = _setup()
    s0 = adapter.pack(state)

    s1, r1 = adapter.step(s0, action, jax.random.PRNGKey(1), Side.GUARD)
    assert adapter.unpack(s1).dwell_catch.tolist() == [_ACTION_REPEAT]
    assert float(r1) == pytest.approx(0.0)

    s2, r2 = adapter.step(s1, action, jax.random.PRNGKey(2), Side.GUARD)
    assert adapter.unpack(s2).dwell_catch.tolist() == [_DWELL_STEPS]
    assert int(adapter.unpack(s2).step) == _DWELL_STEPS
    assert float(r2) == pytest.approx(_R_CATCH)


def test_reward_re_simulation_sees_the_dwell_event():
    """``reward`` re-runs the substeps, so it pays the same bonus as ``step``."""
    _env, adapter, state, action = _setup()
    s0 = adapter.pack(state)
    s1, _ = adapter.step(s0, action, jax.random.PRNGKey(1), Side.GUARD)
    s2, r_step = adapter.step(s1, action, jax.random.PRNGKey(2), Side.GUARD)
    r_reward = adapter.reward(s1, action, s2, Side.GUARD)
    assert float(r_reward) == pytest.approx(float(r_step))
    assert float(r_reward) == pytest.approx(_R_CATCH)


def test_a_root_inside_a_hold_fires_on_the_first_substep():
    """A planner rooted at a counter of four completes the dwell one step later."""
    _env, adapter, state, action = _setup()
    s0 = adapter.pack(state.replace(dwell_catch=jnp.asarray([_DWELL_STEPS - 1])))
    s1, r1 = adapter.step(s0, action, jax.random.PRNGKey(1), Side.GUARD)
    assert adapter.unpack(s1).dwell_catch.tolist() == [_DWELL_STEPS]
    assert int(adapter.unpack(s1).step) == 1
    assert float(r1) == pytest.approx(_R_CATCH)


def test_a_step_outside_the_radius_restarts_the_hold():
    """One macro step spent outside resets the counter, and the hold starts over."""
    _env, adapter, state, action = _setup()
    outside = _displaced(state.replace(dwell_catch=jnp.asarray([_DWELL_STEPS - 1])), 5000.0)

    s1, r1 = adapter.step(adapter.pack(outside), action, jax.random.PRNGKey(1), Side.GUARD)
    assert adapter.unpack(s1).dwell_catch.tolist() == [0]
    assert float(r1) == pytest.approx(0.0)

    # Back on top of the guard, the hold restarts from zero: had the counter
    # survived at four, the catch would have fired on the next substep.
    rejoined = adapter.unpack(s1)
    rejoined = rejoined.replace(bandits=rejoined.bandits.replace(rtn=rejoined.guards.rtn))
    s2, r2 = adapter.step(adapter.pack(rejoined), action, jax.random.PRNGKey(2), Side.GUARD)
    assert adapter.unpack(s2).dwell_catch.tolist() == [_ACTION_REPEAT]
    assert float(r2) == pytest.approx(0.0)


def test_counters_slice_in_order_for_several_bandits():
    """Catch counters occupy the first per-bandit block, breach the second."""
    cfg = make_lady_bandit_guard(n_bandits=2, catch_dwell_steps=3)
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env)
    state, _ = env.reset(jax.random.PRNGKey(0))
    packed = adapter.pack(
        state.replace(dwell_catch=jnp.asarray([1, 2]), dwell_breach=jnp.asarray([3, 4]))
    )
    assert adapter.states_dim == env.layout.flat_dim + 2 + 4
    assert packed[-4:].tolist() == [1.0, 2.0, 3.0, 4.0]
    restored = adapter.unpack(packed)
    assert restored.dwell_catch.tolist() == [1, 2]
    assert restored.dwell_breach.tolist() == [3, 4]


def test_transition_with_counters_is_jittable_and_vmappable():
    """A batch of roots at different points in a hold advances under jit + vmap."""
    _env, adapter, state, action = _setup()
    roots = jnp.stack(
        [
            adapter.pack(state.replace(dwell_catch=jnp.asarray([0]))),
            adapter.pack(state.replace(dwell_catch=jnp.asarray([1]))),
        ]
    )
    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    step = jax.jit(jax.vmap(adapter.transition, in_axes=(0, None, 0)))
    out = step(roots, action, keys)
    counters = [adapter.unpack(s).dwell_catch.tolist() for s in out]
    assert counters == [[_ACTION_REPEAT], [_ACTION_REPEAT + 1]]


def _belief_mean(state):
    """Observer 0's view of both vehicles, taken from the truth state."""
    return jnp.concatenate([state.guards.rtn, state.bandits.rtn], axis=0)[None]


def test_a_belief_root_takes_its_counters_from_the_template():
    """The belief does not track the dwell, so the template supplies it."""
    _env, adapter, state, _action = _setup()
    template = state.replace(dwell_catch=jnp.asarray([4]), dwell_breach=jnp.asarray([1]))
    s_flat = belief_mean_to_flat_state(_belief_mean(state), Side.GUARD, adapter, template)
    restored = adapter.unpack(s_flat)
    assert restored.dwell_catch.tolist() == [4]
    assert restored.dwell_breach.tolist() == [1]


def test_a_belief_root_on_a_reset_template_starts_the_dwell_at_zero():
    """A reset template leaves a search blind to a hold already under way."""
    _env, adapter, state, _action = _setup()
    s_flat = belief_mean_to_flat_state(_belief_mean(state), Side.GUARD, adapter, state)
    restored = adapter.unpack(s_flat)
    assert restored.dwell_catch.tolist() == [0]
    assert restored.dwell_breach.tolist() == [0]
