"""Tests for the LadyBanditGuard game preset."""

from __future__ import annotations

from orbital_game.config import ScenarioConfig
from orbital_game.games import LadyBanditGuard, make_lady_bandit_guard
from orbital_game.games.base import NoGame
from orbital_game.registry import GameKey, resolve_game


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
