"""End-to-end reference scenario: 1 guard + 1 bandit + HCW-RTN rollout.

Run as a script:   uv run python -m examples.reference_scenario
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.single_agent import SingleAgentView
from orbital_game.logging.writer import save_run
from orbital_game.policies.library import ZeroControl
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import StateComponentKey
from orbital_game.rollout import rollout_single_agent
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec
from orbital_game.viz.summaries import plot_mass_curve, plot_reward_curve
from orbital_game.viz.trajectories import plot_rtn_3d


def build_config() -> ScenarioConfig:
    """Construct the canonical bootstrap scenario.

    1 guard with RTN+Mass, 1 bandit with RTN, HCW-RTN dynamics, impulsive
    actuator. Both vehicles share a bounded 1 km radial-ellipse relative orbit
    (guard at phase 0, bandit at phase pi) with sigma=10 m extent jitter.
    200 steps at 10 s each.
    """
    return ScenarioConfig(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        bandit_params=VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
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
    view = SingleAgentView(env)

    # Controlled side's policy (zero-control guard for this reference scenario)
    controlled_policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)

    def init_none(c, s, k):
        del c, s, k
        return None

    traj = rollout_single_agent(
        view, controlled_policy, init_none,
        jax.random.PRNGKey(cfg.seed), n_steps=cfg.max_steps,
    )

    save_run(hdf5_path, cfg, traj)

    # Defender RTN trajectory: shape (T, N_guards, 6)
    guard_rtn = traj.env_state.guards.rtn

    ax3d = plot_rtn_3d(guard_rtn)
    ax3d.figure.savefig(plots_dir / "guard_rtn_3d.png", dpi=120)
    plt.close(ax3d.figure)

    fig, ax = plt.subplots()
    plot_reward_curve(traj.sides.guard.reward, ax=ax)
    fig.savefig(plots_dir / "reward.png", dpi=120)
    plt.close(fig)

    if hasattr(traj.env_state.guards, "propellant_mass"):
        fig, ax = plt.subplots()
        plot_mass_curve(traj.env_state.guards.propellant_mass, ax=ax)
        fig.savefig(plots_dir / "guard_mass.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    run(tmp / "run.h5", tmp)
    print(f"wrote rollout + plots to {tmp}")
