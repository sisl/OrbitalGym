"""Distance contact cannot pretend a global fusion mask represents a graph."""

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import orbitalgym.links as links
from orbitalgym.env.types import Side
from orbitalgym.registry import LinkKey
from orbitalgym.sampling.serialize import serializable_from_primitive, serializable_to_primitive


def _state(positions, dim):
    positions = jnp.asarray(positions).reshape((-1, dim))
    state = jnp.concatenate([positions, jnp.ones_like(positions) * 1.0e6], axis=-1)
    side = SimpleNamespace(**{"rtn" if dim == 3 else "rt": state})
    return SimpleNamespace(guards=side, bandits=side)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("separation,expected", [(0.0, True), (10.0, True), (10.01, False)])
def test_distance_contact_is_symmetric_inclusive_and_jittable(dim, separation, expected):
    link = links.DistanceLink(max_range_m=10.0)
    positions = np.zeros((2, dim))
    positions[1, -1] = separation
    state = _state(positions, dim)
    for side in [Side.GUARD, Side.BANDIT]:
        result = jax.jit(lambda side=side: link(state, side, 0.0))()
        np.testing.assert_array_equal(result, [expected] * 2)


@pytest.mark.parametrize("n", [0, 1])
def test_no_teammates_never_linked(n):
    result = links.DistanceLink(10.0)(_state(np.zeros((n, 3)), 3), Side.GUARD, 0.0)
    np.testing.assert_array_equal(result, np.zeros(n, dtype=bool))


def test_disconnected_pairs_and_three_agent_teams_fail_instead_of_global_fusion():
    for n in [3, 4]:
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0], [101.0, 0.0, 0.0]]
        )[:n]
        with pytest.raises(ValueError, match="two|2|graph"):
            jax.jit(
                lambda positions=positions: links.DistanceLink(5.0)(
                    _state(positions, 3), Side.GUARD, 0.0
                )
            )()


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_distance_link_requires_finite_positive_metres(radius):
    with pytest.raises(ValueError, match="max_range_m"):
        links.DistanceLink(radius)


def test_distance_link_registered_json_roundtrip():
    payload = json.loads(json.dumps(serializable_to_primitive(links.DistanceLink(321.0))))
    restored = serializable_from_primitive(payload, LinkKey)
    assert isinstance(restored, links.DistanceLink)
    assert restored.max_range_m == 321.0
    payload["max_range_m"] = -2.0
    with pytest.raises(ValueError, match="max_range_m"):
        serializable_from_primitive(payload, LinkKey)
