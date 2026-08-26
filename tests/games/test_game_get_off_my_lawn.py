"""Tests for the GetOffMyLawn game preset."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.config import ScenarioConfig
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.games import (
    GetOffMyLawn,
    GetOffMyLawnReward,
    GetOffMyLawnTermination,
    make_get_off_my_lawn,
)
from orbitalgym.games.base import NoGame
from orbitalgym.registry import GameKey, resolve_game


def test_get_off_my_lawn_registered():
    assert resolve_game(GameKey.GET_OFF_MY_LAWN) is GetOffMyLawn


def test_make_default_returns_gol_game():
    cfg = make_get_off_my_lawn()
    assert isinstance(cfg.game, GetOffMyLawn)
    assert cfg.game.r_min_keep_m == 100.0
    assert cfg.game.r_max_keep_m == 500.0
    assert cfg.game.catch_radius_m == 50.0
    assert cfg.game.keep_out_radius_m == 1500.0


def test_knobs_thread_through_reward_and_termination():
    """Game knobs propagate to the reward and termination via game defaults."""
    cfg = make_get_off_my_lawn(
        r_min_keep_m=80.0,
        r_max_keep_m=300.0,
        catch_radius_m=25.0,
        keep_out_radius_m=900.0,
    )
    assert isinstance(cfg.game, GetOffMyLawn)
    assert isinstance(cfg.reward_fn, GetOffMyLawnReward)
    assert isinstance(cfg.termination_fn, GetOffMyLawnTermination)
    assert cfg.reward_fn.r_min_keep_m == 80.0
    assert cfg.reward_fn.r_max_keep_m == 300.0
    assert cfg.reward_fn.catch_radius_m == 25.0
    assert cfg.reward_fn.keep_out_radius_m == 900.0
    assert cfg.termination_fn.catch_radius_m == 25.0
    assert cfg.termination_fn.keep_out_radius_m == 900.0


def test_keep_out_must_exceed_keep_band():
    """keep_out_radius_m must be strictly larger than r_max_keep_m."""
    with pytest.raises(ValueError, match="keep_out_radius_m"):
        GetOffMyLawn(r_min_keep_m=10.0, r_max_keep_m=500.0, keep_out_radius_m=400.0)


def test_keep_band_min_must_be_below_max():
    with pytest.raises(ValueError, match="r_min_keep_m"):
        GetOffMyLawn(r_min_keep_m=500.0, r_max_keep_m=100.0, keep_out_radius_m=1500.0)


def test_reward_weights_thread_through_to_reward_fn():
    """Custom reward weights on the builder propagate to cfg.reward_fn."""
    cfg = make_get_off_my_lawn(
        r_loiter=2.5,
        alpha=5e-3,
        r_catch=42.0,
        r_pushout=17.0,
        alpha_chase=2e-3,
    )
    assert cfg.game.r_loiter == 2.5
    assert cfg.game.alpha == 5e-3
    assert cfg.game.r_catch == 42.0
    assert cfg.game.r_pushout == 17.0
    assert cfg.game.alpha_chase == 2e-3
    # default_reward_fn must thread these onto the constructed reward.
    assert cfg.reward_fn.r_loiter == 2.5
    assert cfg.reward_fn.alpha == 5e-3
    assert cfg.reward_fn.r_catch == 42.0
    assert cfg.reward_fn.r_pushout == 17.0
    assert cfg.reward_fn.alpha_chase == 2e-3


def test_dominant_catch_penalty_overrides_loiter_bonus():
    """With r_catch raised, a catch event utterly dominates a loiter bonus."""
    # Loiter bonus could theoretically fire if bandit-to-lady is in band,
    # but a co-located catch with r_catch=10000 should still drown it out.
    cfg = make_get_off_my_lawn(
        r_min_keep_m=100.0,
        r_max_keep_m=500.0,
        catch_radius_m=20.0,
        keep_out_radius_m=1500.0,
        r_loiter=1.0,
        r_catch=10_000.0,
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    # Co-locate guard and bandit inside the loiter band (HCW R = 300).
    bandit_rtn = jnp.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    state = state.replace(
        bandits=state.bandits.replace(rtn=bandit_rtn),
        guards=state.guards.replace(rtn=bandit_rtn),
    )
    actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )
    out = env.step(jax.random.PRNGKey(0), state, actions)
    # +1 loiter, -10000 catch → bandit ≈ -9999. Loiter bonus is dwarfed.
    assert float(out.outputs.bandit.reward) < -5000.0
    assert float(out.outputs.guard.reward) > 5000.0


def test_serialize_roundtrip():
    cfg = make_get_off_my_lawn(
        r_min_keep_m=120.0,
        r_max_keep_m=400.0,
        catch_radius_m=30.0,
        keep_out_radius_m=800.0,
        r_loiter=3.0,
        alpha=2e-3,
        r_catch=5000.0,
        r_pushout=250.0,
        alpha_chase=4e-4,
    )
    cfg2 = ScenarioConfig.from_json(cfg.to_json())
    assert isinstance(cfg2.game, GetOffMyLawn)
    assert cfg2.game.r_min_keep_m == 120.0
    assert cfg2.game.r_max_keep_m == 400.0
    assert cfg2.game.catch_radius_m == 30.0
    assert cfg2.game.keep_out_radius_m == 800.0
    assert cfg2.game.r_loiter == 3.0
    assert cfg2.game.alpha == 2e-3
    assert cfg2.game.r_catch == 5000.0
    assert cfg2.game.r_pushout == 250.0
    assert cfg2.game.alpha_chase == 4e-4
    # And the threaded reward fn agrees.
    assert cfg2.reward_fn.r_catch == 5000.0
    assert cfg2.reward_fn.alpha_chase == 4e-4


def test_not_no_game():
    cfg = make_get_off_my_lawn()
    assert not isinstance(cfg.game, NoGame)


def test_reward_loiter_event_fires_in_band():
    """Bandit at d_lady inside [r_min, r_max] earns the loiter bonus."""
    cfg = make_get_off_my_lawn(
        r_min_keep_m=100.0,
        r_max_keep_m=500.0,
        catch_radius_m=10.0,
        keep_out_radius_m=1500.0,
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    # Place the bandit cleanly in the middle of the band, guard far away.
    bandit_rtn = jnp.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    guard_rtn = jnp.array([[5000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    state = state.replace(
        bandits=state.bandits.replace(rtn=bandit_rtn),
        guards=state.guards.replace(rtn=guard_rtn),
    )
    actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )
    out = env.step(jax.random.PRNGKey(0), state, actions)
    # Bandit reward = +r_loiter (1.0) - alpha * 0 = 1.0; tolerance for HCW drift.
    assert float(out.outputs.bandit.reward) > 0.0
    # Guard reward = -1.0 - alpha_chase * |g - b|; clearly negative.
    assert float(out.outputs.guard.reward) < 0.0


def test_reward_uses_hcw_natural_standoff():
    """HCW-natural standoff R = sqrt(x_r² + (x_t/2)² + x_n²), not Euclidean.

    A bandit at along-track 200 m has Euclidean ||r|| = 200 m but HCW
    standoff R = 100 m (because along-track amplitude is 2x radial on
    bounded relative orbits). With a band tightened to [80, 120] m the
    bandit is in-band under HCW (loiter fires) but would be out-of-band
    under Euclidean.
    """
    cfg = make_get_off_my_lawn(
        r_min_keep_m=80.0,
        r_max_keep_m=120.0,
        catch_radius_m=10.0,
        keep_out_radius_m=1500.0,
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    # Bandit at along-track 200, radial 0: Euclidean = 200, HCW = 100.
    bandit_rtn = jnp.array([[0.0, 200.0, 0.0, 0.0, 0.0, 0.0]])
    guard_rtn = jnp.array([[5000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    state = state.replace(
        bandits=state.bandits.replace(rtn=bandit_rtn),
        guards=state.guards.replace(rtn=guard_rtn),
    )
    actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )
    out = env.step(jax.random.PRNGKey(0), state, actions)
    # HCW R=100 ∈ [80, 120] → loiter fires → bandit reward > 0.
    # Under Euclidean (||r||=200, outside [80, 120]) the loiter would NOT fire.
    assert float(out.outputs.bandit.reward) > 0.0


def test_termination_uses_hcw_metric_for_pushout():
    """Pushout shell is in HCW metric — pure along-track 2*keep_out triggers."""
    cfg = make_get_off_my_lawn(
        r_min_keep_m=50.0,
        r_max_keep_m=200.0,
        keep_out_radius_m=400.0,
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    # Along-track 1000 m: Euclidean = 1000, HCW = 500 — still > keep_out=400.
    bandit_rtn = jnp.array([[0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]])
    state = state.replace(bandits=state.bandits.replace(rtn=bandit_rtn))
    done = cfg.termination_fn(state, cfg, state.t)
    assert bool(done) is True
    # Along-track 700 m: Euclidean = 700 (would push out under Euclidean) but
    # HCW = 350 < keep_out=400, so under HCW the bandit is NOT pushed out.
    bandit_rtn = jnp.array([[0.0, 700.0, 0.0, 0.0, 0.0, 0.0]])
    state = state.replace(bandits=state.bandits.replace(rtn=bandit_rtn))
    done = cfg.termination_fn(state, cfg, state.t)
    assert bool(done) is False


def test_reward_catch_event_dominates():
    """Guard within catch_radius of bandit → bandit takes -r_catch hit."""
    cfg = make_get_off_my_lawn(
        r_min_keep_m=100.0,
        r_max_keep_m=500.0,
        catch_radius_m=20.0,
        keep_out_radius_m=1500.0,
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    # Co-locate guard and bandit far from the lady.
    bandit_rtn = jnp.array([[800.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    state = state.replace(
        bandits=state.bandits.replace(rtn=bandit_rtn),
        guards=state.guards.replace(rtn=bandit_rtn),
    )
    actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )
    out = env.step(jax.random.PRNGKey(0), state, actions)
    # Bandit eats r_catch=1000; very negative.
    assert float(out.outputs.bandit.reward) < -100.0
    # Guard wins big.
    assert float(out.outputs.guard.reward) > 100.0


def test_termination_triggers_on_pushout():
    """All bandits beyond keep_out_radius → episode ends."""
    cfg = make_get_off_my_lawn(
        r_min_keep_m=50.0,
        r_max_keep_m=200.0,
        keep_out_radius_m=400.0,
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    bandit_rtn = jnp.array([[1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    state = state.replace(bandits=state.bandits.replace(rtn=bandit_rtn))
    done = cfg.termination_fn(state, cfg, state.t)
    assert bool(done) is True


def test_termination_triggers_on_capture():
    """Guard within catch_radius of any bandit → episode ends."""
    cfg = make_get_off_my_lawn(catch_radius_m=100.0)
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    state = state.replace(bandits=state.bandits.replace(rtn=state.guards.rtn))
    done = cfg.termination_fn(state, cfg, state.t)
    assert bool(done) is True


def test_reward_rejects_wrong_game():
    class _Cfg:
        game = NoGame()

    fn = GetOffMyLawnReward()
    with pytest.raises(TypeError, match="GetOffMyLawn"):
        fn(None, None, None, Side.GUARD, _Cfg(), None)


def test_termination_rejects_wrong_game():
    class _Cfg:
        game = NoGame()
        max_steps = 100

    fn = GetOffMyLawnTermination(catch_radius_m=10.0, keep_out_radius_m=1000.0)

    class _Guards:
        rtn = jnp.zeros((1, 6))

    class _Bandits:
        rtn = jnp.zeros((1, 6))

    class _State:
        step = jnp.asarray(0)
        guards = _Guards()
        bandits = _Bandits()

    with pytest.raises(TypeError, match="GetOffMyLawn"):
        fn(_State(), _Cfg(), None)


def test_make_game_dispatcher():
    from orbitalgym.games import make_game

    cfg = make_game(GameKey.GET_OFF_MY_LAWN, r_min_keep_m=75.0, r_max_keep_m=200.0)
    assert isinstance(cfg.game, GetOffMyLawn)
    assert cfg.game.r_min_keep_m == 75.0
    assert cfg.game.r_max_keep_m == 200.0
