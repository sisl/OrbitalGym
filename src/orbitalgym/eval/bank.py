"""Stored initial conditions so every solver evaluates on identical starts.

A bank is an ``EnvState`` whose leaves carry a leading episode axis. It is
sampled once from a named seed, saved to HDF5 next to the scenario config,
and replayed with ``env.reset_from_state`` on any machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.config import ScenarioConfig
from orbitalgym.env.core import OrbitalGymEnv

SCHEMA_VERSION = "bank-1"


def sample_bank(env: OrbitalGymEnv, n_episodes: int, seed: int) -> Any:
    """Sample ``n_episodes`` initial states from ``seed`` with ``jax.vmap(env.reset)``."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    states, _ = jax.vmap(env.reset)(keys)
    return states


def bank_episode(states: Any, index: int) -> Any:
    """The ``index``-th initial state of a bank."""
    return jax.tree_util.tree_map(lambda x: x[index], states)


def _leaf_name(path) -> str:
    return jax.tree_util.keystr(path).lstrip(".").replace("/", ".")


def save_bank(path: str | Path, cfg: ScenarioConfig, states: Any, seed: int) -> None:
    """Write a bank and its scenario config to ``path``."""
    leaves = jax.tree_util.tree_flatten_with_path(states)[0]
    n_episodes = int(states.ic_valid.shape[0])
    with h5py.File(Path(path), "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["config"] = cfg.to_json()
        f.attrs["n_episodes"] = n_episodes
        f.attrs["seed"] = int(seed)
        grp = f.create_group("states")
        for key_path, leaf in leaves:
            grp.create_dataset(_leaf_name(key_path), data=np.asarray(leaf))


def load_bank(path: str | Path) -> tuple[ScenarioConfig, Any]:
    """Read a bank; the returned ``EnvState`` has the bank's leading episode axis."""
    with h5py.File(Path(path), "r") as f:
        if f.attrs["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"bank schema {f.attrs['schema_version']!r} != {SCHEMA_VERSION!r}: {path}"
            )
        cfg = ScenarioConfig.from_json(str(f.attrs["config"]))
        env = OrbitalGymEnv(cfg)
        template, _ = env.reset(jax.random.PRNGKey(0))
        leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
        loaded = [jnp.asarray(f["states"][_leaf_name(key_path)][...]) for key_path, _ in leaves]
    return cfg, jax.tree_util.tree_unflatten(treedef, loaded)
