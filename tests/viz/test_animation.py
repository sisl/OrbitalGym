"""Smoke test for show_link_cones — single-frame render asserts one cone
artist per vehicle per side (3D mode) and that the solid/translucent alpha
switches with the per-frame link mask."""

import math

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from orbitalgym.config import ScenarioConfig
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide, Side
from orbitalgym.links.predicates import PointingConeLink
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from orbitalgym.rollout import rollout
from orbitalgym.viz.animation import RolloutScene, render_frame
from tests.helpers.minimal_scenario import minimal_scenario_kwargs


def _zero_init(c, s, k):
    return None


def _build_scene(rng_float64, link_mask):
    cfg_kwargs = minimal_scenario_kwargs(
        n_guards=2,
        truth_dynamics=DynamicsKey.HCW_RTN,
        action_frame=Frame.RTN,
        guard_components=(
            StateComponentKey.RTN,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        bandit_components=(
            StateComponentKey.RTN,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        guard_action_components=(ActionComponentKey.IMPULSIVE_MANEUVER,),
        bandit_action_components=(ActionComponentKey.IMPULSIVE_MANEUVER,),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=AttitudeParams(
            inertia_diag=jnp.ones(3, dtype=jnp.float32),
            omega_max=jnp.full(3, jnp.float32(0.5)),
        ),
        bandit_attitude_params=AttitudeParams(
            inertia_diag=jnp.ones(3, dtype=jnp.float32),
            omega_max=jnp.full(3, jnp.float32(0.5)),
        ),
    )
    cfg = ScenarioConfig(**cfg_kwargs)
    env = OrbitalGymEnv(cfg)

    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, jax.random.PRNGKey(0), n_steps=3)

    link = PointingConeLink(half_angle_rad=math.radians(30.0))
    return RolloutScene(
        traj=traj,
        cfg=cfg,
        mode="3d",
        guard_link=link,
        link_mask={Side.GUARD: link_mask},
        show_link_cones=True,
        link_cone_length_m=10.0,
        show_cubes=False,
        show_sensor_cones=False,
        show_trail=False,
        show_thrust=False,
        show_reference_marker=False,
    )


def _link_cones(ax):
    return [c for c in ax.collections if isinstance(c, Poly3DCollection)]


def test_show_link_cones_renders_one_cone_per_guard_and_toggles_alpha(rng_float64):
    n_guards = 2
    n_frames = 4
    # Guard 0 linked at frame 0, unlinked elsewhere; guard 1 never linked.
    mask = np.zeros((n_frames, n_guards), dtype=bool)
    mask[0, 0] = True

    scene = _build_scene(rng_float64, mask)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    render_frame(scene, ax, frame=0)
    cones_linked = _link_cones(ax)
    assert len(cones_linked) == n_guards
    alphas_linked = sorted(c.get_alpha() for c in cones_linked)
    assert alphas_linked == sorted([scene.link_cone_alpha_closed, scene.link_cone_alpha_open])

    render_frame(scene, ax, frame=1)
    cones_unlinked = _link_cones(ax)
    assert len(cones_unlinked) == n_guards
    alphas_unlinked = [c.get_alpha() for c in cones_unlinked]
    assert all(a == scene.link_cone_alpha_open for a in alphas_unlinked)

    plt.close(fig)
