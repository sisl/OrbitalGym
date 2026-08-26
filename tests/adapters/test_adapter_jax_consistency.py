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

from examples.reference_scenario import build_config
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide


def test_gymnasium_adapter_consistency_with_direct_env():
    """GymnasiumAdapter.reset() obs equals env.reset projected to controlled side."""
    from orbitalgym.adapters.gymnasium import GymnasiumAdapter

    cfg = build_config()

    # Direct env path: same key, get reset outputs.
    env_direct = OrbitalGymEnv(cfg)
    state_direct, outputs_direct = env_direct.reset(jax.random.PRNGKey(0))

    # Adapter path: deterministic seeding via SingleAgentView (which splits
    # the key for opponent obs + env.step internally — but at reset time, the
    # adapter calls view.reset(key) which calls env.reset(key) directly.
    # The reset key is the SAME PRNGKey(0) passed to env_direct → byte-equal obs.
    env_adapter = OrbitalGymEnv(cfg)
    adapter = GymnasiumAdapter(env_adapter, seed=0)
    obs_adapter, _ = adapter.reset(seed=0)

    # GymnasiumAdapter splits its rng_key once before calling view.reset, so
    # the env.reset key inside the adapter is jax.random.split(PRNGKey(0))[1],
    # NOT PRNGKey(0). So byte-equality is only expected if we use the same
    # transformation. Replicate the adapter's key handling here:
    from orbitalgym.observations.types import flatten_observations

    expected_rng = jax.random.PRNGKey(0)
    _, expected_k_reset = jax.random.split(expected_rng, 2)
    state_expected, outputs_expected = env_direct.reset(expected_k_reset)
    obs_expected = flatten_observations(
        outputs_expected.guard.obs
    )  # GUARD is the default controlled side
    np.testing.assert_array_equal(obs_adapter, np.asarray(obs_expected, dtype=np.float32))


def test_pettingzoo_adapter_consistency_with_direct_env():
    """PettingZooAdapter.reset() obs dict equals env.reset per-side outputs."""
    from orbitalgym.adapters.pettingzoo import PettingZooAdapter

    cfg = build_config()

    from orbitalgym.observations.types import flatten_observations

    env = OrbitalGymEnv(cfg)
    adapter = PettingZooAdapter(env, seed=0)
    obs_dict, _info = adapter.reset(seed=0)

    # Replicate the adapter's key handling.
    expected_rng = jax.random.PRNGKey(0)
    _, expected_k_reset = jax.random.split(expected_rng, 2)
    state_direct, outputs_direct = env.reset(expected_k_reset)

    np.testing.assert_array_equal(
        obs_dict["guard_0"],
        np.asarray(flatten_observations(outputs_direct.guard.obs), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        obs_dict["bandit_0"],
        np.asarray(flatten_observations(outputs_direct.bandit.obs), dtype=np.float32),
    )


def test_pomdp_adapter_consistency_with_direct_env():
    """POMDPAdapter.transition produces the same flat state as
    layout.flatten(env.step(...).state)."""
    from orbitalgym.adapters._command_flatten import unflatten_command
    from orbitalgym.adapters.pomdp import POMDPAdapter

    cfg = build_config()

    # Same env instance for both paths so layout/state shapes match.
    env = OrbitalGymEnv(cfg)

    # Adapter path.
    adapter = POMDPAdapter(env)
    s0 = adapter.initialstate(jax.random.PRNGKey(0))  # uses PRNGKey(0) directly

    # Direct path: replicate adapter's reset (PRNGKey(0)).
    state_direct, _outs = env.reset(jax.random.PRNGKey(0))
    # Adapter packs (guards, bandits) followed by (t, step) as a float tail.
    s0_xy_expected = env.layout.flatten(state_direct.guards, state_direct.bandits)
    np.testing.assert_array_equal(np.asarray(s0[:-2]), np.asarray(s0_xy_expected))
    np.testing.assert_array_equal(np.asarray(s0[-2]), np.asarray(state_direct.t))
    np.testing.assert_array_equal(
        np.asarray(s0[-1]), np.asarray(state_direct.step).astype(s0.dtype)
    )

    # Now step once. Build the flat action vector and unflatten into the
    # per-side Command pytrees so the direct path matches the adapter contract.
    a = jnp.zeros(adapter.guard_action_flat_dim + adapter.bandit_action_flat_dim)
    guard_command = unflatten_command(env.guard_command_cls, a[: adapter.guard_action_flat_dim])
    bandit_command = unflatten_command(
        env.bandit_command_cls,
        a[
            adapter.guard_action_flat_dim : adapter.guard_action_flat_dim
            + adapter.bandit_action_flat_dim
        ],
    )
    actions_direct = Actions(sides=BySide(guard=guard_command, bandit=bandit_command))
    step_out_direct = env.step(jax.random.PRNGKey(1), state_direct, actions_direct)
    s_next_xy_expected = env.layout.flatten(
        step_out_direct.state.guards, step_out_direct.state.bandits
    )

    s_next_adapter = adapter.transition(s0, a, jax.random.PRNGKey(1))
    np.testing.assert_array_equal(np.asarray(s_next_adapter[:-2]), np.asarray(s_next_xy_expected))
    np.testing.assert_array_equal(
        np.asarray(s_next_adapter[-2]), np.asarray(step_out_direct.state.t)
    )
