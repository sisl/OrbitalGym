"""T2 — Build an RTN scenario with viz. Source-of-truth for snippets in
docs/tutorials/t2-rtn-scenario.md.

Plot rendering is gated behind a separate test that creates a tempdir;
the snippet regions stay focused on the data flow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")  # tests headless


def test_t2_rtn_scenario():
    # --8<-- [start:imports]
    import jax
    import jax.numpy as jnp

    from orbitalgym import OrbitalGymEnv, make_pursuit_evasion
    from orbitalgym.env.types import BySide
    from orbitalgym.policies import ZeroControl
    from orbitalgym.rollout import episode_mask, rollout
    # --8<-- [end:imports]

    # --8<-- [start:build-config]
    cfg = make_pursuit_evasion(
        n_guards=2,
        n_bandits=1,
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )
    # --8<-- [end:build-config]

    # --8<-- [start:run-rollout]
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits),
    )
    init_ps = BySide(
        guard=lambda c, s, k: None,
        bandit=lambda c, s, k: None,
    )
    traj = rollout(
        env,
        policies,
        init_ps,
        jax.random.PRNGKey(cfg.seed),
        n_steps=cfg.max_steps,
    )
    # --8<-- [end:run-rollout]

    # --8<-- [start:shapes]
    guard_rtn = traj.env_state.guards.rtn  # (T, N_g, 6)
    bandit_rtn = traj.env_state.bandits.rtn  # (T, N_b, 6)
    valid = episode_mask(traj)  # (T,) bool
    # --8<-- [end:shapes]

    # --8<-- [start:capture-distance]
    # Pairwise distance: guard 0 vs. bandit 0 over time.
    pos_guard0 = guard_rtn[:, 0, :3]  # (T, 3)
    pos_bandit0 = bandit_rtn[:, 0, :3]  # (T, 3)
    dist_g0_b0 = jnp.linalg.norm(pos_guard0 - pos_bandit0, axis=-1)
    # --8<-- [end:capture-distance]

    assert guard_rtn.shape == (cfg.max_steps, cfg.n_guards, 6)
    assert bandit_rtn.shape == (cfg.max_steps, cfg.n_bandits, 6)
    assert valid.shape == (cfg.max_steps,)
    assert bool(valid[0])  # step 0 is always valid
    assert dist_g0_b0.shape == (cfg.max_steps,)


def test_t2_plots_render():
    """The plot calls in the snippet block actually produce non-empty figures."""
    import matplotlib.pyplot as plt

    from orbitalgym import OrbitalGymEnv, make_pursuit_evasion
    from orbitalgym.env.types import BySide
    from orbitalgym.policies import ZeroControl
    from orbitalgym.rollout import rollout
    from orbitalgym.viz.summaries import plot_reward_curve
    from orbitalgym.viz.trajectories import plot_rtn_3d

    cfg = make_pursuit_evasion(n_guards=2, n_bandits=1, seed=0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits),
    )
    init_ps = BySide(
        guard=lambda c, s, k: None,
        bandit=lambda c, s, k: None,
    )
    traj = rollout(
        env,
        policies,
        init_ps,
        jax.random.PRNGKey(cfg.seed),
        n_steps=cfg.max_steps,
    )

    tmp = Path(tempfile.mkdtemp())

    # --8<-- [start:plot-rtn]
    ax3d = plot_rtn_3d(traj.env_state.guards.rtn)
    ax3d.figure.savefig(tmp / "guards_rtn_3d.png", dpi=120)
    plt.close(ax3d.figure)
    # --8<-- [end:plot-rtn]

    # --8<-- [start:plot-reward]
    fig, ax = plt.subplots()
    plot_reward_curve(traj.sides.guard.reward, ax=ax)
    fig.savefig(tmp / "guard_reward.png", dpi=120)
    plt.close(fig)
    # --8<-- [end:plot-reward]

    assert (tmp / "guards_rtn_3d.png").stat().st_size > 0
    assert (tmp / "guard_reward.png").stat().st_size > 0
