"""Tests for HVAState."""

import jax
import jax.numpy as jnp

from orbital_game.hva import HVAState


def test_hva_state_construction_has_expected_fields():
    state = HVAState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
    )
    assert state.position_eci.shape == (3,)
    assert state.velocity_eci.shape == (3,)


def test_hva_state_is_pytree():
    """Must flatten/unflatten as a pytree so jax.tree_util treats it as a leaf container."""
    state = HVAState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
    )
    flat, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, flat)
    assert jnp.allclose(restored.position_eci, state.position_eci)
    assert jnp.allclose(restored.velocity_eci, state.velocity_eci)
