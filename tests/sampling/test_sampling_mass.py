"""Tests for mass sub-samplers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbital_game.sampling.mass import ConstantMass, UniformMass


def test_constant_mass_fills_all_vehicles_with_same_value():
    sampler = ConstantMass(propellant_mass_kg=12.5)
    out = sampler(n_vehicles=4, key=jax.random.PRNGKey(0))
    assert out.shape == (4,)
    assert jnp.allclose(out, 12.5)


def test_constant_mass_independent_of_key():
    sampler = ConstantMass(propellant_mass_kg=3.0)
    a = sampler(n_vehicles=2, key=jax.random.PRNGKey(0))
    b = sampler(n_vehicles=2, key=jax.random.PRNGKey(99))
    assert jnp.allclose(a, b)


def test_uniform_mass_lies_in_range():
    sampler = UniformMass(low_kg=1.0, high_kg=10.0)
    out = sampler(n_vehicles=128, key=jax.random.PRNGKey(0))
    assert out.shape == (128,)
    assert jnp.all(out >= 1.0)
    assert jnp.all(out <= 10.0)


def test_uniform_mass_deterministic_under_same_key():
    sampler = UniformMass(low_kg=0.0, high_kg=5.0)
    a = sampler(n_vehicles=4, key=jax.random.PRNGKey(42))
    b = sampler(n_vehicles=4, key=jax.random.PRNGKey(42))
    assert jnp.allclose(a, b)


def test_uniform_mass_distinct_under_different_keys():
    sampler = UniformMass(low_kg=0.0, high_kg=5.0)
    a = sampler(n_vehicles=8, key=jax.random.PRNGKey(0))
    b = sampler(n_vehicles=8, key=jax.random.PRNGKey(1))
    assert not jnp.allclose(a, b)


def test_uniform_mass_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="low_kg.*high_kg"):
        UniformMass(low_kg=10.0, high_kg=1.0)
