"""Smoke tests for the new RolloutScene knobs (show_contact_state).

These exercise the render path with a synthetic Trajectory whose
policy_state has the PlanCachePolicy shape. We only check that the
render call returns without error — visual correctness is verified
in the notebook.
"""

import flax.struct
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbitalgym.env.types import BySide, SideTrajectory, Trajectory
from orbitalgym.viz.animation import RolloutScene, render_frame


@flax.struct.dataclass
class _StubPolicyState:
    """Minimal stub mimicking PlanCacheState shape — only the fields the
    render path reads."""

    in_contact_prev: jnp.ndarray


@flax.struct.dataclass
class _StubGuardState:
    rt: jnp.ndarray  # (T, N, 4) — RT positions+velocities


@flax.struct.dataclass
class _StubBanditState:
    rt: jnp.ndarray


@flax.struct.dataclass
class _StubEnvState:
    t: jnp.ndarray
    step: jnp.ndarray
    guards: _StubGuardState
    bandits: _StubBanditState


def _build_synthetic_trajectory(T: int = 10, n_g: int = 1, n_b: int = 1):  # noqa: N803 - T is time-axis math convention
    # ---- env state ----
    guard_rt = jnp.zeros((T, n_g, 4))
    bandit_rt = jnp.zeros((T, n_b, 4))
    es = _StubEnvState(
        t=jnp.arange(T) * 10.0,
        step=jnp.arange(T),
        guards=_StubGuardState(rt=guard_rt),
        bandits=_StubBanditState(rt=bandit_rt),
    )
    # ---- per-side policy state (PlanCacheState-shaped) ----
    in_contact = jnp.array([False] * (T // 2) + [True] * (T - T // 2))
    guard_ps = _StubPolicyState(in_contact_prev=in_contact)
    bandit_ps = _StubPolicyState(
        in_contact_prev=jnp.zeros((T,), dtype=bool),
    )
    # ---- side trajectories ----
    # Need an action with `dv` field — the scene reads it. Build the minimal
    # Command class via build_command_class to satisfy the validator (which
    # checks for ImpulsiveManeuver in _orbitalgym_action_components).
    from orbitalgym.actions.assemble import build_command_class
    from orbitalgym.actions.components import ImpulsiveManeuver
    from orbitalgym.registry import Frame

    impulsive = ImpulsiveManeuver(action_frame=Frame.RT, truth_frame=Frame.RT, track_mass=False)
    GuardCmd = build_command_class((impulsive,), n_g, "GuardCmd")  # noqa: N806 - dynamic class
    BanditCmd = build_command_class((impulsive,), n_b, "BanditCmd")  # noqa: N806 - dynamic class
    # Time-stack the commands.
    g_action = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (T,) + x.shape), GuardCmd.zeros(n_g)
    )
    b_action = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (T,) + x.shape), BanditCmd.zeros(n_b)
    )

    sides = BySide(
        guard=SideTrajectory(
            obs=jnp.zeros((T, 4)),
            action=g_action,
            reward=jnp.zeros((T,)),
            done=jnp.zeros((T,), dtype=bool),
            policy_state=guard_ps,
        ),
        bandit=SideTrajectory(
            obs=jnp.zeros((T, 4)),
            action=b_action,
            reward=jnp.zeros((T,)),
            done=jnp.zeros((T,), dtype=bool),
            policy_state=bandit_ps,
        ),
    )
    return Trajectory(env_state=es, sides=sides, episode_done=jnp.zeros((T,), dtype=bool))


def test_scene_show_contact_state_smoke():
    traj = _build_synthetic_trajectory()
    scene = RolloutScene(traj=traj, dt=10.0, show_contact_state=True, mode="2d")
    fig, ax = plt.subplots()
    render_frame(scene, ax, frame=5)  # in second half — guard is in contact
    plt.close(fig)


def test_scene_toggles_default_to_false_no_op():
    """With toggles off (default), render_frame must work even when
    policy_state has no `in_contact_prev` field."""
    traj = _build_synthetic_trajectory()
    # Replace policy_state with None (typical for stateless policies).
    sides = BySide(
        guard=traj.sides.guard.replace(policy_state=None),
        bandit=traj.sides.bandit.replace(policy_state=None),
    )
    traj_stateless = traj.replace(sides=sides)
    scene = RolloutScene(traj=traj_stateless, dt=10.0, mode="2d")
    fig, ax = plt.subplots()
    render_frame(scene, ax, frame=5)
    plt.close(fig)
