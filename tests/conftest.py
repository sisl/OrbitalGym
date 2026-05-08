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


@pytest.fixture
def make_minimal_lbg_config():
    """Returns a callable that builds a minimal LBG ScenarioConfig with kwargs forwarded."""

    def _make(**kwargs):
        from orbital_game.games.lady_bandit_guard import make_lady_bandit_guard

        return make_lady_bandit_guard(**kwargs)

    return _make


@pytest.fixture
def make_minimal_lbg_env_with_kf():
    """Build a minimal LBG env with KF beliefs and ZeroControl policies on both sides.

    Returns ``(env, belief_initializers, belief_updaters, init_policy_state_fns,
    policies)`` — exactly the bundle ``belief_rollout`` expects (minus the key
    and n_steps).
    """

    def _make(**kwargs):
        import dataclasses

        from orbital_game.belief.kf import KFBeliefUpdater, KFFromTruthInitializer
        from orbital_game.env.core import OrbitalGameEnv
        from orbital_game.env.types import BySide
        from orbital_game.games.lady_bandit_guard import make_lady_bandit_guard
        from orbital_game.policies.zero import ZeroControl

        cfg = make_lady_bandit_guard(**kwargs)
        env = OrbitalGameEnv(cfg)
        layout = env.layout
        d = layout.dynamics_state_dim

        belief_init = BySide(
            guard=KFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(d)),
            bandit=KFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(d)),
        )
        # KFBeliefUpdater needs uniform-across-tracked stm/B/Q. Identity stm
        # and zero B keep the test simple — the integration test only cares
        # that the rollout shape threads correctly, not filter accuracy.
        kf_upd = KFBeliefUpdater(
            stm=jnp.eye(d),
            control_matrix=jnp.zeros((d, 3)),
            process_noise=jnp.eye(d) * 0.01,
        )
        belief_upd = BySide(guard=kf_upd, bandit=kf_upd)

        guard_policy = dataclasses.replace(
            ZeroControl(),
            command_cls=env.guard_command_cls,
            n_vehicles=cfg.n_guards,
        )
        bandit_policy = dataclasses.replace(
            ZeroControl(),
            command_cls=env.bandit_command_cls,
            n_vehicles=cfg.n_bandits,
        )
        policies = BySide(guard=guard_policy, bandit=bandit_policy)

        # ZeroControl is stateless — return None for both sides.
        init_ps_fns = BySide(
            guard=lambda config, env_state, key: None,
            bandit=lambda config, env_state, key: None,
        )
        return env, belief_init, belief_upd, init_ps_fns, policies

    return _make
