"""Tests for runtime assembly of per-scenario state classes."""

import jax
import jax.numpy as jnp

from orbital_game.state.assemble import build_state_class
from orbital_game.state.components import Mass, RTNState, RTState


def test_build_single_component_state_class():
    GuardState = build_state_class([RTNState], n_vehicles=3, class_name="GuardState")  # noqa: N806
    s = GuardState.zeros(3)
    assert s.rtn.shape == (3, 6)


def test_build_multi_component_state_class_has_all_fields():
    GuardState = build_state_class([RTNState, Mass], n_vehicles=4, class_name="GuardState")  # noqa: N806
    s = GuardState.zeros(4)
    assert s.rtn.shape == (4, 6)
    assert s.propellant_mass.shape == (4,)


def test_built_class_is_pytree_and_roundtrips():
    Cls = build_state_class([RTState, Mass], n_vehicles=2, class_name="Vs")  # noqa: N806
    s = Cls.zeros(2).replace(
        rt=jnp.ones((2, 4)),
        propellant_mass=jnp.array([10.0, 20.0]),
    )
    flat, treedef = jax.tree_util.tree_flatten(s)
    restored = jax.tree_util.tree_unflatten(treedef, flat)
    assert jnp.allclose(restored.rt, s.rt)
    assert jnp.allclose(restored.propellant_mass, s.propellant_mass)


def test_built_class_name_is_passed_through():
    Cls = build_state_class([RTNState], n_vehicles=1, class_name="MySpecialName")  # noqa: N806
    assert Cls.__name__ == "MySpecialName"


def test_field_order_is_deterministic_across_builds():
    """Same inputs must produce same field ordering (load-bearing for flatten layout)."""
    A = build_state_class([RTNState, Mass], n_vehicles=2, class_name="A")  # noqa: N806
    B = build_state_class([RTNState, Mass], n_vehicles=2, class_name="B")  # noqa: N806
    fa = [f.name for f in A.__dataclass_fields__.values()]
    fb = [f.name for f in B.__dataclass_fields__.values()]
    assert fa == fb
