"""HDF5 reader — inverse of writer.save_run.

Returns (ScenarioConfig, trajectory_dict). The trajectory comes back as a flat
{dotted_path: jax.Array} dict rather than the original pytree: reconstructing
the pytree structurally requires knowing the concrete class shape, which is
scenario-specific. For bootstrap consumers this is enough — downstream analysis
code indexes by dotted name. A richer structural recovery is future work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig

_EXPECTED_SCHEMA_VERSION = "2"


def load_run(path: Path | str) -> tuple[ScenarioConfig, dict[str, Any]]:
    """Load a saved run from HDF5.

    Returns:
      (config, trajectory_dict) where trajectory_dict maps dotted-path strings
      to jax.Arrays (e.g., "reward", "action", "env_state.guards.rtn").

    Raises ValueError if the file is missing required structure (config attr,
    schema_version attr, /trajectory group) or if schema_version doesn't match
    what this reader understands.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "schema_version" not in f.attrs:
            raise ValueError(f"File {path} is missing required 'schema_version' attribute")
        schema_version = f.attrs["schema_version"]
        if schema_version != _EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unknown schema_version {schema_version!r}; expected {_EXPECTED_SCHEMA_VERSION!r}"
            )
        if "config" not in f.attrs:
            raise ValueError(f"File {path} is missing required 'config' attribute")
        config = ScenarioConfig.from_json(f.attrs["config"])
        if "trajectory" not in f:
            raise ValueError(f"File {path} is missing required '/trajectory' group")
        grp = f["trajectory"]
        traj: dict[str, Any] = {}
        for name in grp:
            traj[name] = jnp.asarray(grp[name][()])
    return config, traj
