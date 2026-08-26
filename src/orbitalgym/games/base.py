"""Game base + NoGame default.

A Game is a typed knob bundle attached to ScenarioConfig.game. Game-specific
reward/termination implementations read knobs off cfg.game after an
isinstance check. Each Game subclass declares its default reward and
termination via `default_reward_fn()` / `default_termination_fn()` —
ScenarioConfig.__post_init__ calls these to populate `cfg.reward_fn` /
`cfg.termination_fn` when the user didn't pass them as constructor kwargs.

NoGame is the default for users running custom scenarios with no preset
semantics: it pairs ZeroReward (no-op) with MaxStepsOnly (step cap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbitalgym.registry import GameKey, register_game


@dataclass(frozen=True)
class Game:
    """Marker base — subclasses MUST override default_reward_fn/termination_fn."""

    def default_reward_fn(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must override default_reward_fn")

    def default_termination_fn(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must override default_termination_fn")


@register_game(GameKey.NONE)
@dataclass(frozen=True)
class NoGame(Game):
    """Default — for users running custom scenarios with no preset semantics."""

    def default_reward_fn(self) -> Any:
        from orbitalgym.rewards.reference import ZeroReward

        return ZeroReward()

    def default_termination_fn(self) -> Any:
        from orbitalgym.termination.reference import MaxStepsOnly

        return MaxStepsOnly()
