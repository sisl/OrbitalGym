"""HDF5 writer for rollout trajectories + scenario config.

Writes:
  root attrs: config (JSON-encoded), schema_version
  /trajectory/<field>  each dataset gzip-compressed

Trajectory fields are flattened using jax.tree_util.tree_flatten_with_path so
nested pytrees (e.g., env_state.guards.rtn) become dotted dataset names.

Limitation: pytree field names or dict keys containing `/` are not supported —
HDF5 treats `/` as a group separator so such names would create nested groups
rather than a single dataset with the literal name. Standard `Trajectory`
pytrees (flax dataclasses with simple attribute names) don't trigger this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import jax
import numpy as np

from orbitalgym.config import ScenarioConfig

# v2: defenders → guards, intruders → bandits, hva → reference_orbit (Phase 0 rename)
SCHEMA_VERSION = "2"


def save_run(path: Path | str, config: ScenarioConfig, trajectory: Any) -> None:
    """Save (config, trajectory) to a single HDF5 file.

    Args:
      path: destination file path (created or overwritten)
      config: the ScenarioConfig used to produce the trajectory (round-tripped via JSON)
      trajectory: any pytree (typically a Trajectory) whose leaves are jax.Array

    The config is stored as a JSON string in a root-level attribute so an
    inspector (e.g., `h5py.File(path).attrs['config']`) can see the scenario
    without loading the arrays.
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["config"] = config.to_json()
        f.attrs["schema_version"] = SCHEMA_VERSION
        grp = f.create_group("trajectory")
        _write_pytree(grp, trajectory)


def _write_pytree(grp: h5py.Group, node: Any) -> None:
    """Flatten a pytree into datasets under `grp` using dotted-path names."""
    leaves_with_paths, _treedef = jax.tree_util.tree_flatten_with_path(node)
    for path, leaf in leaves_with_paths:
        name = ".".join(_key_to_str(k) for k in path) if path else "value"
        arr = np.asarray(leaf)
        grp.create_dataset(name, data=arr, compression="gzip")


def _key_to_str(key_entry) -> str:
    """Convert a JAX KeyEntry to a readable path segment."""
    if hasattr(key_entry, "name"):
        return str(key_entry.name)
    if hasattr(key_entry, "key"):
        return str(key_entry.key)
    if hasattr(key_entry, "idx"):
        return f"[{key_entry.idx}]"
    return str(key_entry)
