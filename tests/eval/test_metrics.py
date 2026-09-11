"""Outcome classification and resource accounting for LBG trajectories."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.env.types import BySide
from orbitalgym.eval.metrics import Outcome, lbg_episode_metrics
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.rollout import rollout
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec
from orbitalgym.termination.lbg_events import LbgEventTermination
from orbitalgym.termination.max_distance import AnyOfTermination, MaxDistanceTermination


@dataclass(frozen=True)
class ConstantGuardBurn:
    """Commands a fixed-magnitude delta-v every step, for delta-v accounting tests."""

    command_cls: Any = None
    n_vehicles: int = 0
    dv_mps: float = 0.1

    def __call__(self, policy_state, agent_view, key, t):
        del agent_view, key, t
        zero = self.command_cls.zeros(self.n_vehicles)
        dim = zero.dv.shape[-1]
        dv = jnp.zeros((self.n_vehicles, dim)).at[:, 0].set(self.dv_mps)
        return zero.replace(dv=dv), policy_state


def _ic(guard_radius_m: float, bandit_radius_m: float, bandit_phase: float) -> ICSpec:
    return ICSpec(
        guard_sampler=RelativeEllipse(
            radial_ellipse_m=guard_radius_m,
            phase_rad=0.0,
            sigma_radial_ellipse_m=0.0,
            mass_sampler=ConstantMass(propellant_mass_kg=10.0),
        ),
        bandit_sampler=RelativeEllipse(
            radial_ellipse_m=bandit_radius_m,
            phase_rad=bandit_phase,
            sigma_radial_ellipse_m=0.0,
        ),
    )


def _run(ic: ICSpec, n_steps: int = 5, **game_kwargs):
    cfg = make_lady_bandit_guard(ic_sampler=ic, max_horizon_s=n_steps * 10.0, **game_kwargs)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(n_vehicles=1, command_cls=env.guard_command_cls),
        bandit=ZeroControl(n_vehicles=1, command_cls=env.bandit_command_cls),
    )
    init = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
    traj = rollout(env, policies, init, jax.random.PRNGKey(0), n_steps=n_steps)
    return traj, cfg


def test_timeout_when_nobody_reaches_anything():
    traj, cfg = _run(_ic(1000.0, 1000.0, jnp.pi))
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.outcome) == Outcome.TIMEOUT
    assert int(m.steps) == 5
    assert float(m.min_d_gb) > cfg.game.catch_radius_m
    assert float(m.min_d_bl) > cfg.game.breach_radius_m


def test_breach_when_bandit_starts_inside_breach_radius():
    traj, cfg = _run(_ic(1000.0, 1.0, 0.0))
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.outcome) == Outcome.BREACH
    assert int(m.steps) == 1
    assert float(m.min_d_bl) < cfg.game.breach_radius_m


def test_catch_when_guard_and_bandit_start_together():
    traj, cfg = _run(_ic(1000.0, 1000.0, 0.0))
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.outcome) == Outcome.CATCH
    assert int(m.steps) == 1


def test_zero_control_uses_no_delta_v():
    traj, cfg = _run(_ic(1000.0, 1000.0, jnp.pi))
    m = lbg_episode_metrics(traj, cfg)
    assert float(m.dv_guard) == 0.0
    assert float(m.dv_bandit) == 0.0
    assert int(m.link_events_guard) == 0
    assert bool(m.ic_valid)


def test_link_events_guard_counts_rising_edges_not_steps_in_contact():
    traj, cfg = _run(_ic(1000.0, 1000.0, jnp.pi), n_steps=5)
    # Two contiguous contact spans -> two rising edges, though contact holds
    # for three of the five masked steps.
    contact_guard = jnp.array([[False], [True], [True], [False], [True]])
    contact = BySide(guard=contact_guard, bandit=jnp.zeros_like(contact_guard))
    traj = traj.replace(contact=contact)
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.link_events_guard) == 2


def test_metrics_vmap_over_batch():
    traj, cfg = _run(_ic(1000.0, 1000.0, jnp.pi))
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), traj)
    m = jax.vmap(lambda t: lbg_episode_metrics(t, cfg))(batched)
    assert m.outcome.shape == (2,)
    assert m.dv_guard.shape == (2,)


def test_breach_when_termination_is_the_last_step():
    traj, cfg = _run(_ic(1000.0, 1.0, 0.0), n_steps=1)
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.outcome) == Outcome.BREACH
    assert int(m.steps) == 1
    assert float(m.min_d_bl) < cfg.game.breach_radius_m
    assert float(m.dv_guard) == 0.0


def _burn_traj(n_steps: int, dv_mps: float = 0.1):
    ic = _ic(1000.0, 1000.0, jnp.pi)
    cfg = make_lady_bandit_guard(ic_sampler=ic, max_horizon_s=n_steps * 10.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ConstantGuardBurn(command_cls=env.guard_command_cls, n_vehicles=1, dv_mps=dv_mps),
        bandit=ZeroControl(n_vehicles=1, command_cls=env.bandit_command_cls),
    )
    init = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
    return rollout(env, policies, init, jax.random.PRNGKey(0), n_steps=n_steps), cfg


def test_delta_v_is_reported_for_a_side_with_mass():
    traj, cfg = _burn_traj(2)
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.steps) == 2
    assert float(m.dv_guard) == pytest.approx(0.2, abs=1e-3)


def test_delta_v_of_a_full_length_episode_counts_the_last_burn():
    traj, cfg = _burn_traj(10)
    m = lbg_episode_metrics(traj, cfg)
    applied = float(jnp.sum(jnp.linalg.norm(traj.applied_dv.guard, axis=-1)))
    assert int(m.steps) == 10
    assert float(m.dv_guard) == pytest.approx(applied, rel=1e-3)


def _two_guard_traj(guard_rtn, bandit_rtn, guard_final, bandit_final, cfg):
    """One-step trajectory whose geometry is set by hand.

    The rollout supplies the machinery metrics needs (mask, delta-v trace,
    ic_valid); only the positions are overwritten, so the classification is
    read off exactly the segment the test describes.
    """
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls),
        bandit=ZeroControl(n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls),
    )
    init = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
    traj = rollout(env, policies, init, jax.random.PRNGKey(0), n_steps=1)

    def _set(state, guards_xyz, bandits_xyz):
        guards = state.guards.replace(rtn=state.guards.rtn.at[..., :3].set(guards_xyz))
        bandits = state.bandits.replace(rtn=state.bandits.rtn.at[..., :3].set(bandits_xyz))
        return state.replace(guards=guards, bandits=bandits)

    return traj.replace(
        env_state=_set(traj.env_state, jnp.asarray(guard_rtn)[None], jnp.asarray(bandit_rtn)[None]),
        final_state=_set(traj.final_state, jnp.asarray(guard_final), jnp.asarray(bandit_final)),
    )


def test_catch_uses_per_pair_speed_gate_not_the_closest_pair():
    """A slow pair at 4 m catches even though a faster pair passes closer."""
    cfg = make_lady_bandit_guard(
        n_guards=2,
        ic_sampler=_ic(1000.0, 1000.0, jnp.pi),
        max_horizon_s=10.0,
        catch_radius_m=5.0,
        catch_speed_mps=0.5,
        breach_radius_m=1.0,
    )
    bandit_prev = [[1000.0, 0.0, 0.0]]
    bandit_next = [[1000.0, 0.0, 0.0]]
    # Guard 0 holds a 4 m standoff (zero relative speed); guard 1 sweeps past
    # at 2 m and 4 m/s.
    guard_prev = [[1004.0, 0.0, 0.0], [980.0, 2.0, 0.0]]
    guard_next = [[1004.0, 0.0, 0.0], [1020.0, 2.0, 0.0]]

    traj = _two_guard_traj(guard_prev, bandit_prev, guard_next, bandit_next, cfg)
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.outcome) == Outcome.CATCH
    assert float(m.min_d_gb) == pytest.approx(2.0, abs=1e-6)


def test_fast_arrival_inside_the_radius_at_the_last_step_is_not_a_breach():
    """The final segment is a real transition, so its speed gate still applies."""
    cfg = make_lady_bandit_guard(
        n_guards=1,
        ic_sampler=_ic(1000.0, 1000.0, jnp.pi),
        max_horizon_s=10.0,
        catch_radius_m=1.0,
        breach_radius_m=50.0,
        breach_speed_mps=0.5,
    )
    guard_prev = [[5000.0, 0.0, 0.0]]
    guard_next = [[5000.0, 0.0, 0.0]]
    # The bandit ends 14.1 m from the lady — well inside 50 m — but crosses at
    # 21 m/s, far above the gate.
    bandit_prev = [[-200.0, 10.0, 0.0]]
    bandit_next = [[10.0, 10.0, 0.0]]

    traj = _two_guard_traj(guard_prev, bandit_prev, guard_next, bandit_next, cfg)
    m = lbg_episode_metrics(traj, cfg)
    assert float(jnp.linalg.norm(jnp.asarray(bandit_next[0]))) < cfg.game.breach_radius_m
    assert int(m.outcome) == Outcome.TIMEOUT


def test_metrics_require_the_final_state():
    traj, cfg = _run(_ic(1000.0, 1000.0, jnp.pi))
    with pytest.raises(ValueError, match="final_state"):
        lbg_episode_metrics(traj.replace(final_state=None), cfg)


def test_repelled_when_the_bandit_is_pushed_past_the_escape_radius():
    """An early termination with neither catch nor breach classifies as REPELLED."""
    traj, cfg = _run(_ic(1000.0, 6000.0, jnp.pi), n_steps=5, escape_radius_m=5000.0)
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.outcome) == Outcome.REPELLED
    assert int(m.steps) == 1


def test_timeout_is_not_repelled():
    """Running out the clock without an event stays TIMEOUT."""
    traj, cfg = _run(_ic(1000.0, 1000.0, jnp.pi), n_steps=5, escape_radius_m=5000.0)
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.outcome) == Outcome.TIMEOUT
    assert int(m.steps) == 5


def test_early_stop_without_a_repel_gate_is_a_timeout():
    """A drift cap ending the episode early is not a repulsion."""
    traj, cfg = _run(
        _ic(1000.0, 6000.0, jnp.pi),
        n_steps=5,
        termination_fn=AnyOfTermination(
            [
                LbgEventTermination(breach_radius_m=5.0, catch_radius_m=50.0),
                MaxDistanceTermination(max_distance_m=5000.0),
            ]
        ),
    )
    m = lbg_episode_metrics(traj, cfg)
    assert int(m.steps) == 1
    assert int(m.outcome) == Outcome.TIMEOUT


@pytest.mark.parametrize("counter", ["dwell_catch", "dwell_breach"])
def test_longest_dwell_survives_a_reset_and_ignores_padding(counter):
    traj, cfg = _run(_ic(1000.0, 1000.0, jnp.pi), n_steps=6)
    # Live leaving counters are [1, 2, 0, 1]; padding later reaches 50.
    traj = traj.replace(
        env_state=traj.env_state.replace(**{counter: jnp.array([[0], [1], [2], [0], [1], [50]])}),
        final_state=traj.final_state.replace(**{counter: jnp.array([51])}),
        episode_done=jnp.array([False, False, False, True, True, True]),
    )
    m = jax.jit(lambda t: lbg_episode_metrics(t, cfg))(traj)
    assert int(getattr(m, counter + "_steps")) == 2


@pytest.mark.parametrize("counter", ["dwell_catch", "dwell_breach"])
def test_longest_dwell_includes_the_final_transition(counter):
    traj, cfg = _run(_ic(1000.0, 1000.0, jnp.pi), n_steps=2)
    traj = traj.replace(
        env_state=traj.env_state.replace(**{counter: jnp.array([[0], [1]])}),
        final_state=traj.final_state.replace(**{counter: jnp.array([2])}),
    )
    m = lbg_episode_metrics(traj, cfg)
    assert int(getattr(m, counter + "_steps")) == 2
