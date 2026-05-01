"""Shared pytest fixtures for orbital-game tests."""

import jax
import pytest


@pytest.fixture
def key():
    """Deterministic master PRNGKey for reproducible tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def keys(key):
    """A batch of 4 split keys for multi-step tests."""
    return jax.random.split(key, 4)


@pytest.fixture
def rng_float64():
    """Enable float64 precision for tests that need it (e.g., reference orbit epoch, HCW)."""
    from jax import config

    config.update("jax_enable_x64", True)
    yield
