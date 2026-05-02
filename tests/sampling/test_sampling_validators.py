"""Tests for IC validators."""

from __future__ import annotations

import jax.numpy as jnp
from flax.struct import dataclass as fdataclass

from orbital_game.sampling.validators import (
    MaxRange,
    MinSeparation,
    SeparationScope,
    SideSelector,
)


@fdataclass
class _FakeSide:
    rtn: jnp.ndarray  # (N, 6)


@fdataclass
class _FakeConfig:
    pass


def _make(positions_guard, positions_bandit):
    """Build fake side states with given (N, 3) positions and zero velocity."""

    def _state(p):
        n = p.shape[0]
        rtn = jnp.zeros((n, 6))
        rtn = rtn.at[:, :3].set(p)
        return _FakeSide(rtn=rtn)

    return _state(positions_guard), _state(positions_bandit)


def test_min_separation_passes_when_all_pairs_far():
    guards, bandits = _make(jnp.array([[0.0, 0, 0]]), jnp.array([[100.0, 0, 0]]))
    v = MinSeparation(distance_m=10.0, scope=SeparationScope.ALL)
    assert bool(v(_FakeConfig(), guards, bandits))


def test_min_separation_fails_when_cross_pair_too_close():
    guards, bandits = _make(jnp.array([[0.0, 0, 0]]), jnp.array([[3.0, 0, 0]]))
    v = MinSeparation(distance_m=5.0, scope=SeparationScope.ALL)
    assert not bool(v(_FakeConfig(), guards, bandits))


def test_min_separation_within_guards_only():
    guards, bandits = _make(jnp.array([[0.0, 0, 0], [4.0, 0, 0]]), jnp.array([[1.0, 0, 0]]))
    # Guard-guard distance is 4m; cross distance is 1m.
    v_within = MinSeparation(distance_m=5.0, scope=SeparationScope.WITHIN_GUARDS)
    v_cross = MinSeparation(distance_m=5.0, scope=SeparationScope.CROSS)
    assert not bool(v_within(_FakeConfig(), guards, bandits))  # 4 < 5 — fail
    assert not bool(v_cross(_FakeConfig(), guards, bandits))  # 1 < 5 — fail
    # Same setup, only bandits: there's only one bandit so distance is +inf.
    v_within_bandit = MinSeparation(distance_m=5.0, scope=SeparationScope.WITHIN_BANDITS)
    assert bool(v_within_bandit(_FakeConfig(), guards, bandits))


def test_max_range_passes_within_bound():
    guards, bandits = _make(jnp.array([[1.0, 0, 0]]), jnp.array([[0.0, 2.0, 0]]))
    v = MaxRange(distance_m=5.0, side=SideSelector.ALL)
    assert bool(v(_FakeConfig(), guards, bandits))


def test_max_range_fails_when_bandit_too_far():
    guards, bandits = _make(jnp.array([[1.0, 0, 0]]), jnp.array([[0.0, 100.0, 0]]))
    v = MaxRange(distance_m=5.0, side=SideSelector.ALL)
    assert not bool(v(_FakeConfig(), guards, bandits))


def test_max_range_guards_only_ignores_bandits():
    guards, bandits = _make(jnp.array([[1.0, 0, 0]]), jnp.array([[0.0, 1000.0, 0]]))
    v = MaxRange(distance_m=5.0, side=SideSelector.GUARDS)
    assert bool(v(_FakeConfig(), guards, bandits))
