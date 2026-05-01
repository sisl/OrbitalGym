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


def _make(positions_def, positions_int):
    """Build fake side states with given (N, 3) positions and zero velocity."""
    def _state(p):
        n = p.shape[0]
        rtn = jnp.zeros((n, 6))
        rtn = rtn.at[:, :3].set(p)
        return _FakeSide(rtn=rtn)
    return _state(positions_def), _state(positions_int)


def test_min_separation_passes_when_all_pairs_far():
    defs, ints = _make(jnp.array([[0., 0, 0]]), jnp.array([[100., 0, 0]]))
    v = MinSeparation(distance_m=10.0, scope=SeparationScope.ALL)
    assert bool(v(_FakeConfig(), defs, ints))


def test_min_separation_fails_when_cross_pair_too_close():
    defs, ints = _make(jnp.array([[0., 0, 0]]), jnp.array([[3., 0, 0]]))
    v = MinSeparation(distance_m=5.0, scope=SeparationScope.ALL)
    assert not bool(v(_FakeConfig(), defs, ints))


def test_min_separation_within_defenders_only():
    defs, ints = _make(jnp.array([[0., 0, 0], [4., 0, 0]]), jnp.array([[1., 0, 0]]))
    # Defender-defender distance is 4m; cross distance is 1m.
    v_within = MinSeparation(distance_m=5.0, scope=SeparationScope.WITHIN_DEFENDERS)
    v_cross = MinSeparation(distance_m=5.0, scope=SeparationScope.CROSS)
    assert not bool(v_within(_FakeConfig(), defs, ints))  # 4 < 5 — fail
    assert not bool(v_cross(_FakeConfig(), defs, ints))   # 1 < 5 — fail
    # Same setup, only intruders: there's only one intruder so distance is +inf.
    v_within_int = MinSeparation(distance_m=5.0, scope=SeparationScope.WITHIN_INTRUDERS)
    assert bool(v_within_int(_FakeConfig(), defs, ints))


def test_max_range_passes_within_bound():
    defs, ints = _make(jnp.array([[1., 0, 0]]), jnp.array([[0., 2., 0]]))
    v = MaxRange(distance_m=5.0, side=SideSelector.ALL)
    assert bool(v(_FakeConfig(), defs, ints))


def test_max_range_fails_when_intruder_too_far():
    defs, ints = _make(jnp.array([[1., 0, 0]]), jnp.array([[0., 100., 0]]))
    v = MaxRange(distance_m=5.0, side=SideSelector.ALL)
    assert not bool(v(_FakeConfig(), defs, ints))


def test_max_range_defenders_only_ignores_intruders():
    defs, ints = _make(jnp.array([[1., 0, 0]]), jnp.array([[0., 1000., 0]]))
    v = MaxRange(distance_m=5.0, side=SideSelector.DEFENDERS)
    assert bool(v(_FakeConfig(), defs, ints))
