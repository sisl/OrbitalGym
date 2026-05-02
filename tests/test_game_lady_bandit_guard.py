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
    assert cfg.game.breach_distance_m == 10.0


def test_lady_bandit_guard_breach_distance_threads_through_termination():
    """The breach_distance_m knob is wired into the termination function."""
    cfg = make_lady_bandit_guard(breach_distance_m=42.0)
    assert isinstance(cfg.game, LadyBanditGuard)
    assert cfg.game.breach_distance_m == 42.0
    assert cfg.termination_fn.breach_distance_m == 42.0


def test_lady_bandit_guard_serialize_roundtrip():
    cfg = make_lady_bandit_guard(breach_distance_m=15.0)
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert isinstance(cfg2.game, LadyBanditGuard)
    assert cfg2.game.breach_distance_m == 15.0


def test_lady_bandit_guard_not_no_game():
    cfg = make_lady_bandit_guard()
    assert not isinstance(cfg.game, NoGame)
