"""Tests for the PursuitEvasion game preset."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide, Side
from orbital_game.games import (
    PursuitEvasion,
    PursuitEvasionReward,
    PursuitEvasionTermination,
    make_pursuit_evasion,
)
from orbital_game.games.base import NoGame
from orbital_game.registry import GameKey, resolve_game


def test_pursuit_evasion_registered():
    assert resolve_game(GameKey.PURSUIT_EVASION) is PursuitEvasion


def test_make_pe_default_knobs():
    cfg = make_pursuit_evasion()
    assert isinstance(cfg.game, PursuitEvasion)
    assert cfg.game.capture_distance_m == 10.0


def test_pe_serialize_roundtrip():
    cfg = make_pursuit_evasion(capture_distance_m=25.0)
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert isinstance(cfg2.game, PursuitEvasion)
    assert cfg2.game.capture_distance_m == 25.0


def test_pe_reward_is_zero_sum():
    """Bandit reward = -|r|; guard reward = +|r|. Sum is 0."""
    cfg = make_pursuit_evasion()
    env = OrbitalGameEnv(cfg)
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


def test_pe_termination_triggers_on_capture():
    """If guards and bandits are placed at the same point, capture immediately."""
    cfg = make_pursuit_evasion(capture_distance_m=1000.0)
    env = OrbitalGameEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    # Force the bandit to be co-located with the guard.
    state = state.replace(bandits=state.bandits.replace(rtn=state.guards.rtn))
    done = cfg.termination_fn(state, cfg, state.t)
    assert bool(done) is True


def test_pe_reward_rejects_wrong_game():
    """PE reward raises TypeError when cfg.game is not PursuitEvasion."""

    class _Cfg:
        game = NoGame()

    fn = PursuitEvasionReward()
    try:
        fn(None, None, None, Side.GUARD, _Cfg(), None)
    except TypeError as e:
        assert "PursuitEvasion" in str(e)
    else:
        raise AssertionError("expected TypeError")


def test_pe_termination_rejects_wrong_game():
    """PE termination raises TypeError when cfg.game is not PursuitEvasion."""

    class _Cfg:
        game = NoGame()
        max_steps = 100

    fn = PursuitEvasionTermination()
    try:
        # Need a state with .step, .guards.rtn, .bandits.rtn
        class _Guards:
            rtn = jnp.zeros((1, 6))

        class _Bandits:
            rtn = jnp.zeros((1, 6))

        class _State:
            step = jnp.asarray(0)
            guards = _Guards()
            bandits = _Bandits()

        fn(_State(), _Cfg(), None)
    except TypeError as e:
        assert "PursuitEvasion" in str(e)
    else:
        raise AssertionError("expected TypeError")
