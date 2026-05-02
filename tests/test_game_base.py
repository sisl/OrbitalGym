"""Tests for the Game catalog base + NoGame default."""

from __future__ import annotations

from examples.reference_scenario import build_config
from orbital_game.config import ScenarioConfig
from orbital_game.games.base import Game, NoGame
from orbital_game.games.lady_bandit_guard import LadyBanditGuard
from orbital_game.registry import GameKey, resolve_game, resolve_game_class_to_key


def test_reference_scenario_game_is_lbg():
    """build_config() now returns an LBG scenario (migrated from NoGame)."""
    cfg = build_config()
    assert isinstance(cfg.game, LadyBanditGuard)


def test_no_game_is_a_game():
    assert isinstance(NoGame(), Game)


def test_no_game_registered_under_key_none():
    assert resolve_game(GameKey.NONE) is NoGame
    assert resolve_game_class_to_key(NoGame) is GameKey.NONE


def test_config_serializes_game_field():
    """After migration, build_config round-trips as LBG."""
    cfg = build_config()
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert isinstance(cfg2.game, LadyBanditGuard)
