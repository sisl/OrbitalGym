"""Game base + NoGame default.

A Game is a typed knob bundle attached to ScenarioConfig.game. Game-specific
reward/termination implementations read knobs off cfg.game after an
isinstance check. Each Game subclass declares its default reward and
termination via `default_reward_fn()` / `default_termination_fn()` —
ScenarioConfig.__post_init__ calls these to populate `cfg.reward_fn` /
`cfg.termination_fn` when the user didn't pass them as constructor kwargs.
It also calls `validate(cfg)`, where a game rejects knob settings the
configured state layout cannot support.

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

    def advance_state(self, prev_state: Any, next_state: Any, cfg: Any) -> Any:
        """Game-owned bookkeeping on the state leaving a step.

        ``OrbitalGymEnv.step`` calls this once the dynamics have produced
        ``next_state`` and before the observations, rewards and termination
        read it, so a game that needs history across steps — how long a
        vehicle has held a radius, say — can keep it on the state rather
        than recomputing it. The default returns the state untouched.
        """
        del prev_state, cfg
        return next_state

    def validate(self, cfg: Any) -> None:
        """Check the cfg supports this game's knobs. Called by ScenarioConfig.__post_init__.

        Games whose knobs depend on the state layout (a propellant gate needs a
        mass component, say) reject the incoherent combination here, so the
        failure lands at config construction rather than mid-rollout.
        """


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
