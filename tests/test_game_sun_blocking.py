"""Tests for the SunBlocking game preset."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide, Side
from orbital_game.games import SunBlocking, SunBlockingReward, make_sun_blocking
from orbital_game.games.base import NoGame
from orbital_game.registry import GameKey, resolve_game


def test_sun_blocking_registered():
    assert resolve_game(GameKey.SUN_BLOCKING) is SunBlocking


def test_make_sb_default_knobs():
    cfg = make_sun_blocking()
    assert isinstance(cfg.game, SunBlocking)
    assert cfg.game.angle_sigma_deg == 5.0


def test_sb_serialize_roundtrip():
    cfg = make_sun_blocking(angle_sigma_deg=2.5)
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert isinstance(cfg2.game, SunBlocking)
    assert cfg2.game.angle_sigma_deg == 2.5


def test_sb_reward_zero_sum():
    """Guard and bandit rewards sum to 0."""
    cfg = make_sun_blocking()
    env = OrbitalGameEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    actions = Actions(sides=BySide(
        guard=jnp.zeros((cfg.n_guards, 3)),
        bandit=jnp.zeros((cfg.n_bandits, 3)),
    ))
    out = env.step(jax.random.PRNGKey(1), state, actions)
    r_g = float(out.outputs.guard.reward)
    r_b = float(out.outputs.bandit.reward)
    assert abs(r_g + r_b) < 1e-6


def test_sb_reward_in_unit_range_for_bandit():
    """exp(-angle²/2σ²) gated on ordering ⇒ bandit reward in [0, 1]."""
    cfg = make_sun_blocking()
    env = OrbitalGameEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    actions = Actions(sides=BySide(
        guard=jnp.zeros((cfg.n_guards, 3)),
        bandit=jnp.zeros((cfg.n_bandits, 3)),
    ))
    out = env.step(jax.random.PRNGKey(1), state, actions)
    r_b = float(out.outputs.bandit.reward)
    assert 0.0 <= r_b <= 1.0


def test_sb_reward_rejects_wrong_game():
    class _Cfg:
        game = NoGame()

    fn = SunBlockingReward()
    try:
        fn(None, None, None, Side.BANDIT, _Cfg(), None)
    except TypeError as e:
        assert "SunBlocking" in str(e)
    else:
        raise AssertionError("expected TypeError")
