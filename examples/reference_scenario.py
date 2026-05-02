"""End-to-end reference scenario: 1 guard + 1 bandit + HCW-RTN rollout.

Run as a script:   uv run python -m examples.reference_scenario
"""

from __future__ import annotations

from pathlib import Path

import jax
import matplotlib.pyplot as plt

from orbital_game.config import ScenarioConfig
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.single_agent import SingleAgentView
from orbital_game.logging.writer import save_run
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout_single_agent
from orbital_game.viz.summaries import plot_mass_curve, plot_reward_curve
from orbital_game.viz.trajectories import plot_rtn_3d


def build_config() -> ScenarioConfig:
    """Construct the canonical bootstrap scenario via the LBG builder.

    1 guard with RTN+Mass, 1 bandit with RTN, HCW-RTN dynamics, impulsive
    actuator. Both vehicles share a bounded 1 km radial-ellipse relative
    orbit (guard at phase 0, bandit at phase pi) with sigma=10 m extent
    jitter. 200 steps at 10 s each.
    """
    from orbital_game.games import make_lady_bandit_guard
    return make_lady_bandit_guard(
        n_guards=1,
        n_bandits=1,
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
