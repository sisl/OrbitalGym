"""UniformRandomDiscretePolicy — uniform sampling from a discrete Δv grid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from orbital_game.policies import UniformRandomDiscretePolicy
from tests.policies._helpers import make_impulsive_maneuver_command_cls


def _grid_8_dirs(dv_max: float = 1.0) -> jax.Array:
    """8 unit-length Δv directions in 2-D + a no-op."""
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    vecs = dv_max * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return jnp.asarray(np.concatenate([vecs, np.zeros((1, 2))], axis=0), dtype=jnp.float32)


def test_uniform_random_emits_command_pytree():
    cmd_cls = make_impulsive_maneuver_command_cls(2)  # 3-D dv (RTN)
    grid = _grid_8_dirs()
    policy = UniformRandomDiscretePolicy(action_grid=grid, n_vehicles=2, command_cls=cmd_cls)
    cmd, _ = policy(None, None, jax.random.PRNGKey(0), jnp.asarray(0.0))
    assert cmd.dv.shape == (2, 3)
    # Padded the 2-D grid Δv into the 3-D command.
    assert jnp.allclose(cmd.dv[:, 2], 0.0)


def test_uniform_random_samples_distribution_is_uniform():
    cmd_cls = make_impulsive_maneuver_command_cls(1)
    grid = _grid_8_dirs()
    policy = UniformRandomDiscretePolicy(action_grid=grid, n_vehicles=1, command_cls=cmd_cls)
    counts = np.zeros(grid.shape[0], dtype=int)
    n = 4096
    for i in range(n):
        cmd, _ = policy(None, None, jax.random.PRNGKey(i), jnp.asarray(0.0))
        # Recover the action-grid index by matching dv (with the no-op as a special case).
        chosen = np.asarray(cmd.dv[0, :2])
        # For each grid row, see if it matches.
        for j, row in enumerate(np.asarray(grid)):
            if np.allclose(chosen, row, atol=1e-5):
                counts[j] += 1
                break
    expected = n / grid.shape[0]
    # Each bin should be within ~6× std (very loose) of expected.
    std = np.sqrt(expected * (1 - 1 / grid.shape[0]))
    assert np.all(np.abs(counts - expected) < 6 * std), counts


def test_uniform_random_without_command_cls_raises():
    grid = _grid_8_dirs()
    policy = UniformRandomDiscretePolicy(action_grid=grid, n_vehicles=1)
    import pytest

    with pytest.raises(ValueError, match="UniformRandomDiscretePolicy"):
        policy(None, None, jax.random.PRNGKey(0), jnp.asarray(0.0))
