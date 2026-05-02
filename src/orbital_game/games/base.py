"""Game base + NoGame default.

A Game is a typed knob bundle attached to ScenarioConfig.game. Game-specific
reward/termination implementations read knobs off cfg.game after an
isinstance check. NoGame is the default — for custom/bootstrap scenarios
where no preset semantics apply.
"""

from __future__ import annotations

from dataclasses import dataclass

from orbital_game.registry import GameKey, register_game


@dataclass(frozen=True)
class Game:
    """Marker base — typed game-specific knob bundle."""

    pass


@register_game(GameKey.NONE)
@dataclass(frozen=True)
class NoGame(Game):
    """Default — for users running custom scenarios with no preset semantics."""

    pass
