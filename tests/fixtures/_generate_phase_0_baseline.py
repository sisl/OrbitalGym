"""One-shot generator for tests/fixtures/phase_0_baseline.npz.

Run BEFORE Phase 0 renames to capture the reference scenario's numerical
output. The output dict keys use POST-rename clean names. The attribute
accesses use whatever names exist at script-execution time; this script
gets updated bucket-by-bucket alongside the rename.

After Phase 0 (all renames applied), this script produces a byte-identical
.npz to the one captured pre-rename — that's the regression check.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.reference_scenario import build_config
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.rollout import rollout


def main():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)

    def policy(ps, obs, key, t):
        del obs, key, t
        # Bucket B: DONE
        return jnp.zeros((cfg.n_guards, 3)), ps

    def init_ps(config, env_state, key):
        del config, env_state, key
        return None

    traj = rollout(env, policy, init_ps, jax.random.PRNGKey(cfg.seed), n_steps=cfg.max_steps)

    # Output keys use POST-rename clean names. Attribute access uses CURRENT names.
    # Bucket A: DONE (traj.env_state.hva → traj.env_state.reference_orbit)
    # Bucket B: DONE (traj.env_state.defenders → traj.env_state.guards)
    # Bucket C: DONE (traj.env_state.intruders → traj.env_state.bandits)
    out = {
        "reward": np.asarray(traj.reward),
        "done": np.asarray(traj.done),
        "action": np.asarray(traj.action),
        "obs": np.asarray(traj.obs),
        "guards_rtn": np.asarray(traj.env_state.guards.rtn),
        "bandits_rtn": np.asarray(traj.env_state.bandits.rtn),
        "guards_propellant_mass": np.asarray(traj.env_state.guards.propellant_mass),
        "reference_orbit_position_eci": np.asarray(traj.env_state.reference_orbit.position_eci),
        "reference_orbit_velocity_eci": np.asarray(traj.env_state.reference_orbit.velocity_eci),
        "t": np.asarray(traj.env_state.t),
        "step": np.asarray(traj.env_state.step),
    }
    fixture_path = Path(__file__).parent / "phase_0_baseline.npz"
    np.savez_compressed(fixture_path, **out)
    print(f"wrote {fixture_path}")


if __name__ == "__main__":
    main()
