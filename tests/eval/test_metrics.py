"""Outcome classification and resource accounting for LBG trajectories."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.env.types import BySide
from orbitalgym.eval.metrics import Outcome, lbg_episode_metrics
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.rollout import rollout
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec


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


def _run(ic: ICSpec, n_steps: int = 5):
    cfg = make_lady_bandit_guard(ic_sampler=ic, max_horizon_s=n_steps * 10.0)
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
