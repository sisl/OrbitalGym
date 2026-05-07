"""Smoke test for show_sensor_cones — single-frame render asserts a Wedge
patch is added per guard per sensor (2D mode)."""

import math

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge

from orbital_game.config import ScenarioConfig
from orbital_game.dynamics.attitude import AttitudeParams
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import BySide
from orbital_game.observations.conical import ConicalObservation
from orbital_game.policies.zero import ZeroControl
from orbital_game.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from orbital_game.rollout import rollout
from orbital_game.viz.animation import RolloutScene, render_frame
from tests.helpers.minimal_scenario import minimal_scenario_kwargs


def _zero_init(c, s, k):
    return None


def test_show_sensor_cones_renders_one_wedge_per_sensor_per_guard(rng_float64):
    boresights = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    n_sensors = boresights.shape[0]

    # Build a config with attitude state + impulsive maneuver (required by
    # RolloutScene) on both sides. RT dynamics so the scene resolves to 2D
    # and we can count Wedge patches.
    cfg_kwargs = minimal_scenario_kwargs(
        truth_dynamics=DynamicsKey.HCW_RT,
        action_frame=Frame.RT,
        guard_components=(
            StateComponentKey.RT,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        bandit_components=(
            StateComponentKey.RT,
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
    layout = OrbitalGameEnv(ScenarioConfig(**cfg_kwargs)).layout

    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=boresights,
        half_angle_rad=math.radians(30.0),
        sigma=0.0,
    )
    cfg = ScenarioConfig(
        **{
            **cfg_kwargs,
            "guard_observation_fn": obs_fn,
        }
    )
    env = OrbitalGameEnv(cfg)

    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, jax.random.PRNGKey(0), n_steps=3)

    scene = RolloutScene(
        traj=traj,
        cfg=cfg,
        mode="2d",
        show_sensor_cones=True,
        cone_length_m=10.0,
        show_cubes=False,
        show_trail=False,
        show_thrust=False,
        show_reference_marker=False,
    )

    fig, ax = plt.subplots()
    render_frame(scene, ax, frame=0)

    wedges = [p for p in ax.patches if isinstance(p, Wedge)]
    assert len(wedges) == cfg.n_guards * n_sensors
    plt.close(fig)
