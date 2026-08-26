from collections.abc import Mapping
from typing import ClassVar

import jax
import jax.numpy as jnp

from orbitalgym.actions.components import ActionComponent


class _DummyComp:
    name: ClassVar[str] = "dummy"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"x": (2,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"x": jnp.zeros((n, 2))}

    def apply(self, command, side_state, side_params, dt, ref_eci6, key):
        return side_state


def test_dummy_satisfies_protocol():
    assert isinstance(_DummyComp(), ActionComponent)


def test_dummy_zeros_shape():
    z = _DummyComp.zeros(3)
    assert z["x"].shape == (3, 2)
