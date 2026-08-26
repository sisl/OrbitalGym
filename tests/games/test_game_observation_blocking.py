"""Tests for the ObservationBlocking game preset."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbitalgym.config import ScenarioConfig
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.games import (
    ObservationBlocking,
    ObservationBlockingReward,
    make_observation_blocking,
)
from orbitalgym.games.base import NoGame
from orbitalgym.registry import GameKey, resolve_game


def test_observation_blocking_registered():
    assert resolve_game(GameKey.OBSERVATION_BLOCKING) is ObservationBlocking


def test_ob_target_ecef_precomputed():
    """The geodetic→ECEF conversion runs once in __post_init__."""
    g = ObservationBlocking(target_lat_deg=37.4, target_lon_deg=-122.2)
    assert g.target_ecef_m.shape == (3,)
    # Should be at roughly Earth radius.
    assert 6.0e6 < float(jnp.linalg.norm(g.target_ecef_m)) < 7.0e6


def test_ob_serialize_roundtrip():
    """Round-trip preserves user knobs; target_ecef_m re-derived in __post_init__."""
    cfg = make_observation_blocking(
        target_lat_deg=40.0,
        target_lon_deg=-105.0,
        min_elevation_deg=10.0,
    )
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert isinstance(cfg2.game, ObservationBlocking)
    assert cfg2.game.target_lat_deg == 40.0
    assert cfg2.game.target_lon_deg == -105.0
    assert cfg2.game.min_elevation_deg == 10.0
    # Recomputed ECEF should match.
    assert float(jnp.linalg.norm(cfg2.game.target_ecef_m - cfg.game.target_ecef_m)) < 1e-3


def test_ob_reward_zero_sum():
    cfg = make_observation_blocking()
    env = OrbitalGymEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )
    out = env.step(jax.random.PRNGKey(1), state, actions)
    r_g = float(out.outputs.guard.reward)
    r_b = float(out.outputs.bandit.reward)
    assert abs(r_g + r_b) < 1e-6


def test_ob_reward_in_signed_unit_range():
    """KSP-DG SB1 form ⇒ bandit reward in [-1, 1]."""
    cfg = make_observation_blocking()
    env = OrbitalGymEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )
    out = env.step(jax.random.PRNGKey(1), state, actions)
    r_b = float(out.outputs.bandit.reward)
    assert -1.0 - 1e-6 <= r_b <= 1.0 + 1e-6


def test_ob_reward_rejects_wrong_game():
    class _Cfg:
        game = NoGame()

    fn = ObservationBlockingReward()
    try:
        fn(None, None, None, Side.BANDIT, _Cfg(), None)
    except TypeError as e:
        assert "ObservationBlocking" in str(e)
    else:
        raise AssertionError("expected TypeError")
