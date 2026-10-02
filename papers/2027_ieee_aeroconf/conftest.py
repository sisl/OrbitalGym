"""Tests run in double precision, as the experiments do."""

import jax.numpy as jnp

import orbitalgym

orbitalgym.set_precision(jnp.float64)
