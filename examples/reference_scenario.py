"""End-to-end reference scenario: 1 defender + 1 intruder + HCW-RTN rollout.

Run as a script:   uv run python -m examples.reference_scenario
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.hva import HVAState
from orbital_game.logging.writer import save_run
from orbital_game.registry import StateComponentKey
from orbital_game.rollout import rollout
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec
from orbital_game.viz.summaries import plot_mass_curve, plot_reward_curve
from orbital_game.viz.trajectories import plot_rtn_3d


def build_config() -> ScenarioConfig:
    """Construct the canonical bootstrap scenario.

    1 defender with RTN+Mass, 1 intruder with RTN, HCW-RTN dynamics, impulsive
    actuator. Both vehicles share a bounded 1 km radial-ellipse relative orbit
    (defender at phase 0, intruder at phase pi) with sigma=10 m extent jitter.
    200 steps at 10 s each.
    """
    return ScenarioConfig(
        n_defenders=1,
        n_intruders=1,
        epoch_mjd_utc=60067.0,
        hva=HVAState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        defender_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        intruder_components=(StateComponentKey.RTN,),
        defender_params=VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        intruder_params=VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0),
        ic_sampler=ICSpec(
            defender_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            intruder_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=jnp.pi,
                sigma_radial_ellipse_m=10.0,
            ),
            validators=(),
            max_attempts=100,
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )


def run(hdf5_path: Path, plots_dir: Path) -> None:
    """Execute the scenario and write HDF5 + PNG outputs."""
    cfg = build_config()
    env = OrbitalGameEnv(cfg)

    # Zero-control defender; null placeholder policy state.
    def policy(ps, obs, key, t):
        del obs, key, t
        return jnp.zeros((cfg.n_defenders, 3)), ps

    def init_ps(config, env_state, key):
        del config, env_state, key
        return None

    traj = rollout(env, policy, init_ps, jax.random.PRNGKey(cfg.seed), n_steps=cfg.max_steps)

    save_run(hdf5_path, cfg, traj)

    # Defender RTN trajectory: shape (T, N_defenders, 6)
    defender_rtn = traj.env_state.defenders.rtn

    ax3d = plot_rtn_3d(defender_rtn)
    ax3d.figure.savefig(plots_dir / "defender_rtn_3d.png", dpi=120)
    plt.close(ax3d.figure)

    fig, ax = plt.subplots()
    plot_reward_curve(traj.reward, ax=ax)
    fig.savefig(plots_dir / "reward.png", dpi=120)
    plt.close(fig)

    if hasattr(traj.env_state.defenders, "propellant_mass"):
        fig, ax = plt.subplots()
        plot_mass_curve(traj.env_state.defenders.propellant_mass, ax=ax)
        fig.savefig(plots_dir / "defender_mass.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    run(tmp / "run.h5", tmp)
    print(f"wrote rollout + plots to {tmp}")
