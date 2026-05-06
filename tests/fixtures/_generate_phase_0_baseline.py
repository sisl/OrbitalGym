"""One-shot generator for tests/fixtures/phase_0_baseline.npz."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from examples.reference_scenario import build_config
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import BySide
from orbital_game.policies import ZeroControl
from orbital_game.rollout import rollout


def main():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)

    guard_policy = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
    bandit_policy = ZeroControl(n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls)

    def init_none(c, s, k):
        return None

    traj = rollout(
        env,
        BySide(guard=guard_policy, bandit=bandit_policy),
        BySide(guard=init_none, bandit=init_none),
        jax.random.PRNGKey(cfg.seed),
        n_steps=cfg.max_steps,
    )

    out = {
        "reward": np.asarray(traj.sides.guard.reward),
        "done": np.asarray(traj.episode_done),
        "action": np.asarray(traj.sides.guard.action.dv),
        "obs": np.asarray(traj.sides.guard.obs),
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
