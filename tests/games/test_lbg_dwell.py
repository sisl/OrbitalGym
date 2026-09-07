"""Dwell-based catch and breach: counters, events, rewards, and batching."""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp
import pytest

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.env.types import BySide, Side
from orbitalgym.eval.metrics import Outcome, lbg_episode_metrics
from orbitalgym.games.proximity import (
    advance_dwell,
    lbg_dwell_step,
    lbg_events,
    lbg_events_with_dwell,
)
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.reference_orbit import ReferenceOrbitState
from orbitalgym.rewards.lbg_zero_sum import LbgZeroSumReward
from orbitalgym.rollout import rollout
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec
from orbitalgym.termination.lbg_events import LbgEventTermination

_REFERENCE_ORBIT = ReferenceOrbitState(
    position_eci=jnp.array([7000e3, 0.0, 0.0]),
    velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
)

_CATCH_RADIUS_M = 50.0
_BREACH_RADIUS_M = 5.0
_FAR = 5000.0


class _Cfg:
    """Scenario stand-in carrying the fields the event functions read."""

    def __init__(self, max_steps: int = 100, dt: float = 10.0):
        self.max_steps = max_steps
        self.dt = dt
        self.reference_orbit = _REFERENCE_ORBIT


@flax.struct.dataclass
class _Side:
    rtn: jax.Array


@flax.struct.dataclass
class _State:
    guards: _Side
    bandits: _Side
    step: jax.Array
    dwell_catch: jax.Array
    dwell_breach: jax.Array


def _rtn(xyz) -> jax.Array:
    pos = jnp.asarray(xyz, dtype=jnp.float32)
    return jnp.concatenate([pos, jnp.zeros_like(pos)], axis=-1)


def _state(
    guard_x: float, bandit_x: float, step: int = 0, dwell=(0, 0), bandit_y: float = 0.0
) -> _State:
    return _State(
        guards=_Side(_rtn([[guard_x, 0.0, 0.0]])),
        bandits=_Side(_rtn([[bandit_x, bandit_y, 0.0]])),
        step=jnp.asarray(step),
        dwell_catch=jnp.asarray([dwell[0]], dtype=jnp.int32),
        dwell_breach=jnp.asarray([dwell[1]], dtype=jnp.int32),
    )


def _run(
    bandit_x_by_step,
    guard_x: float = 0.0,
    dt: float = 10.0,
    bandit_y: float = 0.0,
    catch_speed_mps: float = float("inf"),
):
    """Walk the bandit through the scripted positions, advancing the counters.

    Returns one ``(entering, leaving)`` pair per step, the second carrying
    the counters that step arrived at.
    """
    state = _state(guard_x, bandit_x_by_step[0], bandit_y=bandit_y)
    leaving = []
    for i, bandit_x in enumerate(bandit_x_by_step[1:], start=1):
        nxt = _state(guard_x, bandit_x, step=i, bandit_y=bandit_y)
        dwell_catch, dwell_breach = lbg_dwell_step(
            state,
            nxt,
            dt,
            _CATCH_RADIUS_M,
            catch_speed_mps,
            _BREACH_RADIUS_M,
            float("inf"),
        )
        nxt = nxt.replace(dwell_catch=dwell_catch, dwell_breach=dwell_breach)
        leaving.append((state, nxt))
        state = nxt
    return leaving


def test_advance_dwell_increments_inside_and_resets_outside():
    count = jnp.asarray([3, 3], dtype=jnp.int32)
    inside = jnp.asarray([True, False])
    assert advance_dwell(count, inside).tolist() == [4, 0]


@pytest.mark.parametrize(
    ("guard_x", "bandit_x"),
    [(0.0, 10.0), (0.0, _FAR), (_FAR, 1.0), (_FAR, _FAR)],
)
def test_zero_dwell_matches_the_single_step_events(guard_x, bandit_x):
    """With both dwells zero the events are the single-step ones, unchanged."""
    prev, nxt = _run([bandit_x, bandit_x], guard_x=guard_x)[0]
    args = (prev, nxt, 10.0, _CATCH_RADIUS_M, float("inf"), _BREACH_RADIUS_M, float("inf"))
    plain = lbg_events(*args)
    dwelled = lbg_events_with_dwell(*args, 0, 0)
    for a, b in zip(plain, dwelled, strict=True):
        assert jnp.array_equal(a, b)


def test_catch_fires_on_the_step_the_dwell_completes():
    """Three steps inside the catch radius: the catch fires on the third."""
    term = LbgEventTermination(
        breach_radius_m=_BREACH_RADIUS_M,
        catch_radius_m=_CATCH_RADIUS_M,
        catch_dwell_steps=3,
    )
    fired = []
    counts = []
    for prev, nxt in _run([10.0, 10.0, 10.0, 10.0]):
        fired.append(bool(term(prev, nxt, _Cfg(), nxt.step)))
        counts.append(int(nxt.dwell_catch[0]))
    assert counts == [1, 2, 3]
    assert fired == [False, False, True]


def test_leaving_the_catch_radius_resets_the_dwell():
    """A step spent outside sends the counter back to zero and nothing fires.

    The step the bandit departs on still counts: its closest approach is the
    position it started from, which is inside the radius. The step after it,
    spent wholly outside, is the one that breaks the hold.
    """
    term = LbgEventTermination(
        breach_radius_m=_BREACH_RADIUS_M,
        catch_radius_m=_CATCH_RADIUS_M,
        catch_dwell_steps=3,
    )
    fired = []
    counts = []
    for prev, nxt in _run([10.0, 10.0, _FAR, _FAR, 10.0, 10.0]):
        fired.append(bool(term(prev, nxt, _Cfg(), nxt.step)))
        counts.append(int(nxt.dwell_catch[0]))
    assert counts == [1, 2, 0, 1, 2]
    assert not any(fired)


def test_breach_fires_on_the_step_the_dwell_completes():
    """The same rule at the lady, with the guard parked far away."""
    term = LbgEventTermination(breach_radius_m=_BREACH_RADIUS_M, breach_dwell_steps=2)
    fired = []
    counts = []
    for prev, nxt in _run([1.0, 1.0, 1.0], guard_x=_FAR):
        fired.append(bool(term(prev, nxt, _Cfg(), nxt.step)))
        counts.append(int(nxt.dwell_breach[0]))
    assert counts == [1, 2]
    assert fired == [False, True]


def test_leaving_the_breach_radius_resets_the_dwell():
    term = LbgEventTermination(breach_radius_m=_BREACH_RADIUS_M, breach_dwell_steps=3)
    fired = []
    counts = []
    for prev, nxt in _run([1.0, 1.0, _FAR, _FAR, 1.0], guard_x=_FAR):
        fired.append(bool(term(prev, nxt, _Cfg(), nxt.step)))
        counts.append(int(nxt.dwell_breach[0]))
    assert counts == [1, 2, 0, 1]
    assert not any(fired)


def test_speed_gate_resets_the_dwell_mid_hold():
    """A step that fails the speed gate breaks the hold even inside the radius."""
    term = LbgEventTermination(
        breach_radius_m=_BREACH_RADIUS_M,
        catch_radius_m=_CATCH_RADIUS_M,
        catch_speed_mps=1.0,
        catch_dwell_steps=3,
    )
    # The bandit holds a 20 m cross-track offset, well outside the breach
    # radius, and crosses the guard's radius at 4 m/s on the second step.
    fired = []
    counts = []
    for prev, nxt in _run([-20.0, -20.0, 20.0, 20.0, 20.0], bandit_y=20.0, catch_speed_mps=1.0):
        fired.append(bool(term(prev, nxt, _Cfg(), nxt.step)))
        counts.append(int(nxt.dwell_catch[0]))
    assert counts == [1, 0, 1, 2]
    assert not any(fired)


def test_terminal_bonus_is_paid_on_the_dwell_step_only():
    """The guard's catch bonus lands once, on the step the dwell completes."""
    reward = LbgZeroSumReward(
        shaping_gain=0.0,
        separation_cost=0.0,
        lady_keepout_m=0.0,
        catch_radius_m=_CATCH_RADIUS_M,
        breach_radius_m=_BREACH_RADIUS_M,
        catch_dwell_steps=3,
    )
    cfg = _Cfg()
    guard = []
    bandit = []
    for prev, nxt in _run([10.0, 10.0, 10.0, 10.0]):
        guard.append(float(reward(prev, None, nxt, Side.GUARD, cfg, prev.step)))
        bandit.append(float(reward(prev, None, nxt, Side.BANDIT, cfg, prev.step)))
    assert guard == pytest.approx([0.0, 0.0, reward.r_catch])
    assert bandit == pytest.approx([0.0, 0.0, -reward.r_catch])


def test_dwell_events_are_jittable_and_vmappable():
    """Counters are traced integer arrays, so jit and vmap both go through."""
    term = LbgEventTermination(
        breach_radius_m=_BREACH_RADIUS_M,
        catch_radius_m=_CATCH_RADIUS_M,
        catch_dwell_steps=3,
    )
    cfg = _Cfg()

    def one(prev, nxt):
        dwell_catch, dwell_breach = lbg_dwell_step(
            prev, nxt, cfg.dt, _CATCH_RADIUS_M, float("inf"), _BREACH_RADIUS_M, float("inf")
        )
        nxt = nxt.replace(dwell_catch=dwell_catch, dwell_breach=dwell_breach)
        return term(prev, nxt, cfg, nxt.step), nxt.dwell_catch

    # Two episodes at once, both two steps into a hold: one completes the
    # third step inside the radius, the other spends it wholly outside.
    prev = jax.tree.map(
        lambda a, b: jnp.stack([a, b]),
        _state(0.0, 10.0, step=2, dwell=(2, 0)),
        _state(0.0, _FAR, step=2, dwell=(2, 0)),
    )
    nxt = jax.tree.map(
        lambda a, b: jnp.stack([a, b]),
        _state(0.0, 10.0, step=3),
        _state(0.0, _FAR, step=3),
    )
    done, counts = jax.jit(jax.vmap(one))(prev, nxt)
    assert done.tolist() == [True, False]
    assert counts.tolist() == [[3], [0]]


def _run_env(n_steps: int, **game_kwargs):
    """One rollout of a scenario whose guard starts on top of the bandit."""
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
        max_horizon_s=n_steps * 10.0,
        **game_kwargs,
    )
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(n_vehicles=1, command_cls=env.guard_command_cls),
        bandit=ZeroControl(n_vehicles=1, command_cls=env.bandit_command_cls),
    )
    init = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
    traj = rollout(env, policies, init, jax.random.PRNGKey(0), n_steps=n_steps)
    return traj, cfg


def test_env_catch_dwell_delays_the_episode_end():
    """The co-located pair ends the episode on step 1 without a dwell, step 3 with one."""
    traj, cfg = _run_env(5)
    instant = lbg_episode_metrics(traj, cfg)
    assert int(instant.outcome) == Outcome.CATCH
    assert int(instant.steps) == 1
    assert int(instant.dwell_catch_steps) == 1

    traj, cfg = _run_env(5, catch_dwell_steps=3)
    dwelled = lbg_episode_metrics(traj, cfg)
    assert int(dwelled.outcome) == Outcome.CATCH
    assert int(dwelled.steps) == 3
    assert int(dwelled.dwell_catch_steps) == 3


def test_env_dwell_longer_than_the_horizon_times_out():
    """A dwell the episode has no room to complete leaves the outcome a timeout."""
    traj, cfg = _run_env(3, catch_dwell_steps=9)
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.outcome) == Outcome.TIMEOUT
    assert int(m.steps) == 3
    assert int(m.dwell_catch_steps) == 3


def test_env_reset_zeroes_the_dwell_counters():
    cfg = make_lady_bandit_guard(catch_dwell_steps=2)
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert state.dwell_catch.shape == (cfg.n_bandits,)
    assert state.dwell_catch.tolist() == [0]
    assert state.dwell_breach.tolist() == [0]
