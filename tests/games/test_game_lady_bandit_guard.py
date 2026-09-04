"""Tests for the LadyBanditGuard game preset."""

from __future__ import annotations

import jax
import pytest

from orbitalgym.config import ScenarioConfig
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide
from orbitalgym.games import LadyBanditGuard, make_lady_bandit_guard
from orbitalgym.games.base import NoGame
from orbitalgym.registry import DynamicsKey, GameKey, StateComponentKey, resolve_game
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec


def test_lady_bandit_guard_registered():
    assert resolve_game(GameKey.LADY_BANDIT_GUARD) is LadyBanditGuard


def test_make_lady_bandit_guard_default_returns_lbg_game():
    cfg = make_lady_bandit_guard()
    assert isinstance(cfg.game, LadyBanditGuard)
    assert cfg.game.breach_radius_m == 5.0
    assert cfg.game.catch_radius_m == 50.0


def test_lady_bandit_guard_breach_radius_threads_through_termination():
    """The breach_radius_m knob is wired into the termination function via game default."""
    cfg = make_lady_bandit_guard(breach_radius_m=42.0, catch_radius_m=80.0)
    assert isinstance(cfg.game, LadyBanditGuard)
    assert cfg.game.breach_radius_m == 42.0
    assert cfg.game.catch_radius_m == 80.0
    assert cfg.termination_fn.breach_radius_m == 42.0
    assert cfg.termination_fn.catch_radius_m == 80.0


def test_lady_bandit_guard_serialize_roundtrip():
    cfg = make_lady_bandit_guard(breach_radius_m=15.0, catch_radius_m=30.0)
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert isinstance(cfg2.game, LadyBanditGuard)
    assert cfg2.game.breach_radius_m == 15.0
    assert cfg2.game.catch_radius_m == 30.0


def test_lady_bandit_guard_not_no_game():
    cfg = make_lady_bandit_guard()
    assert not isinstance(cfg.game, NoGame)


def test_repel_knobs_thread_through_termination_and_reward():
    cfg = make_lady_bandit_guard(
        escape_radius_m=5000.0,
        repel_on_empty_tank=True,
        bandit_components=(StateComponentKey.RTN, StateComponentKey.MASS),
    )
    assert cfg.termination_fn.escape_radius_m == 5000.0
    assert cfg.termination_fn.repel_on_empty_tank is True
    assert cfg.reward_fn.escape_radius_m == 5000.0
    assert cfg.reward_fn.repel_on_empty_tank is True


def test_repel_on_empty_tank_requires_a_mass_component():
    with pytest.raises(ValueError, match="repel_on_empty_tank"):
        make_lady_bandit_guard(
            repel_on_empty_tank=True,
            bandit_components=(StateComponentKey.RTN,),
        )


def test_repel_on_empty_tank_accepts_a_mass_tracked_bandit():
    cfg = make_lady_bandit_guard(
        repel_on_empty_tank=True,
        bandit_components=(StateComponentKey.RTN, StateComponentKey.MASS),
    )
    assert cfg.game.repel_on_empty_tank is True


def test_repel_on_empty_tank_rejects_non_hcw_rtn_truth_dynamics():
    with pytest.raises(ValueError, match="HCW-RTN"):
        make_lady_bandit_guard(
            repel_on_empty_tank=True,
            bandit_components=(StateComponentKey.RT, StateComponentKey.MASS),
            guard_components=(StateComponentKey.RT, StateComponentKey.MASS),
            truth_dynamics=DynamicsKey.HCW_RT,
            policy_dynamics=DynamicsKey.HCW_RT,
        )


def test_repel_on_empty_tank_steps_under_jit():
    cfg = make_lady_bandit_guard(
        repel_on_empty_tank=True,
        bandit_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=300.0, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=3000.0, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
            ),
        ),
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )
    out = jax.jit(env.step)(jax.random.PRNGKey(1), state, actions)
    assert bool(out.episode_done) is False
