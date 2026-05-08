"""Shared pytest fixtures for orbital-game tests.

The package now defaults to float32 (so MPS users don't trip on float64
device-puts). Tests still want orbit-grade precision and explicit
``dtype=jnp.float64`` literals to round-trip without silent downcasts, so
we re-enable ``jax_enable_x64`` for the whole test session via an autouse
session-scoped fixture below.

Order of imports matters: we must pin ``JAX_PLATFORMS=cpu`` before any
``import jax`` runs, otherwise JAX latches MPS (when ``jax-mps`` is
installed) as the default backend and float64 device-puts crash. Pytest
imports this conftest before collecting any test module, so the env-var
setdefault here is effective for the whole session.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

import orbital_game  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _enable_x64_for_tests():
    """Force float64 precision throughout the test session.

    The package import sets float32 as the default for backend portability;
    tests assert ``dtype == jnp.float64`` and exercise astrojax KOE/ECI
    conversions where the float32 precision floor (~7 m) would cause flaky
    failures. Flip via ``orbital_game.set_precision`` so astrojax's internal
    dtype stays in sync with ``jax_enable_x64``.
    """
    orbital_game.set_precision(jnp.float64)
    yield


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
    """Backwards-compat alias.

    Tests that historically requested ``rng_float64`` already get x64 from
    the autouse ``_enable_x64_for_tests`` fixture above; keep this name so
    existing tests continue to import cleanly.
    """
    yield
