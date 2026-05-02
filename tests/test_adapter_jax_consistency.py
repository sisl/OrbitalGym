"""Cross-adapter JAX-consistency tests.

Each adapter is supposed to be a pure projection of the JAX core — calling
adapter operations should produce the same numerical result as calling
env.step / env.reset directly (modulo numpy/jax type conversion). These
tests run both paths with the same key + same actions and assert
array-equality on the projected fields.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.reference_scenario import build_config
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")


def test_gymnasium_adapter_consistency_with_direct_env():
    """GymnasiumAdapter.reset() obs equals env.reset projected to controlled side."""
    from orbital_game.adapters.gymnasium import GymnasiumAdapter

    cfg = build_config()

    # Direct env path: same key, get reset outputs.
    env_direct = OrbitalGameEnv(cfg)
    state_direct, outputs_direct = env_direct.reset(jax.random.PRNGKey(0))

    # Adapter path: deterministic seeding via SingleAgentView (which splits
    # the key for opponent obs + env.step internally — but at reset time, the
    # adapter calls view.reset(key) which calls env.reset(key) directly.
    # The reset key is the SAME PRNGKey(0) passed to env_direct → byte-equal obs.
    env_adapter = OrbitalGameEnv(cfg)
    adapter = GymnasiumAdapter(env_adapter, seed=0)
    obs_adapter, _ = adapter.reset(seed=0)

    # GymnasiumAdapter splits its rng_key once before calling view.reset, so
    # the env.reset key inside the adapter is jax.random.split(PRNGKey(0))[1],
    # NOT PRNGKey(0). So byte-equality is only expected if we use the same
    # transformation. Replicate the adapter's key handling here:
    expected_rng = jax.random.PRNGKey(0)
    _, expected_k_reset = jax.random.split(expected_rng, 2)
    state_expected, outputs_expected = env_direct.reset(expected_k_reset)
    obs_expected = outputs_expected.guard.obs  # GUARD is the default controlled side
    np.testing.assert_array_equal(obs_adapter, np.asarray(obs_expected, dtype=np.float32))


def test_pettingzoo_adapter_consistency_with_direct_env():
    """PettingZooAdapter.reset() obs dict equals env.reset per-side outputs."""
    from orbital_game.adapters.pettingzoo import PettingZooAdapter

    cfg = build_config()

    env = OrbitalGameEnv(cfg)
    adapter = PettingZooAdapter(env, seed=0)
    obs_dict, _info = adapter.reset(seed=0)

    # Replicate the adapter's key handling.
    expected_rng = jax.random.PRNGKey(0)
    _, expected_k_reset = jax.random.split(expected_rng, 2)
    state_direct, outputs_direct = env.reset(expected_k_reset)

    np.testing.assert_array_equal(
        obs_dict["guard_0"], np.asarray(outputs_direct.guard.obs, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        obs_dict["bandit_0"], np.asarray(outputs_direct.bandit.obs, dtype=np.float32)
    )


def test_pomdp_adapter_consistency_with_direct_env():
    """POMDPAdapter.transition produces the same flat state as
    layout.flatten(env.step(...).state)."""
    from orbital_game.adapters.pomdp import POMDPAdapter

    cfg = build_config()

    # Same env instance for both paths so layout/state shapes match.
    env = OrbitalGameEnv(cfg)

    # Adapter path.
    adapter = POMDPAdapter(env)
    s0 = adapter.initialstate(jax.random.PRNGKey(0))  # uses PRNGKey(0) directly

    # Direct path: replicate adapter's reset (PRNGKey(0)).
    state_direct, _outs = env.reset(jax.random.PRNGKey(0))
    s0_expected = env.layout.flatten(state_direct.guards, state_direct.bandits)
    np.testing.assert_array_equal(np.asarray(s0), np.asarray(s0_expected))

    # Now step once.
    n_g = cfg.n_guards
    n_b = cfg.n_bandits
    d = adapter.action_dim_per_side
    a = jnp.zeros(n_g * d + n_b * d)
    actions_direct = Actions(
        sides=BySide(
            guard=a[: n_g * d].reshape(n_g, d),
            bandit=a[n_g * d : n_g * d + n_b * d].reshape(n_b, d),
        )
    )
    step_out_direct = env.step(jax.random.PRNGKey(1), state_direct, actions_direct)
    s_next_expected = env.layout.flatten(
        step_out_direct.state.guards, step_out_direct.state.bandits
    )

    s_next_adapter = adapter.transition(s0, a, jax.random.PRNGKey(1))
    np.testing.assert_array_equal(np.asarray(s_next_adapter), np.asarray(s_next_expected))
