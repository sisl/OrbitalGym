"""Tests for viz/animation.py — RolloutScene + render_frame.

These confirm:
- RolloutScene precomputes axis_limit / cube_scale / thrust_scale.
- render_frame works for every toggle combination on a real rollout.
- Sensor range is auto-inferred from a RangeLimitedObservation in cfg.
- Belief ellipsoids render when belief_history is provided.

MP4 export is skipped if ffmpeg isn't available.
"""

import shutil
from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from orbital_game.belief import (  # noqa: E402
    BeliefRollout,
    EKFBeliefUpdater,
    EKFUniformDefaultInitializer,
    run_belief_rollout,
)
from orbital_game.dynamics.hcw import hcw_rtn_step  # noqa: E402
from orbital_game.env.core import OrbitalGameEnv  # noqa: E402
from orbital_game.env.types import BySide, Side  # noqa: E402
from orbital_game.games.lady_bandit_guard import make_lady_bandit_guard  # noqa: E402
from orbital_game.observations.range_limited import RangeLimitedObservation  # noqa: E402
from orbital_game.policies import ZeroControl  # noqa: E402
from orbital_game.reference_orbit import mean_motion as ref_mean_motion  # noqa: E402
from orbital_game.rollout import rollout  # noqa: E402
from orbital_game.viz.animation import (  # noqa: E402
    RolloutScene,
    render_frame,
    save_animation,
)


def _zero_init(c, s, k):
    return None


@pytest.fixture
def basic_cfg_and_traj(key):
    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1, dt=10.0, max_horizon_s=200.0)
    env = OrbitalGameEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=12)
    return cfg, traj


@pytest.fixture
def range_limited_cfg_traj_belief(key):
    proto = make_lady_bandit_guard(n_guards=1, n_bandits=1, dt=10.0, max_horizon_s=200.0)
    layout = OrbitalGameEnv(proto).layout
    obs_g = RangeLimitedObservation(layout=layout, sensor_range_m=2000.0, sigma_range=1.0)
    obs_b = RangeLimitedObservation(layout=layout, sensor_range_m=2000.0, sigma_range=1.0)
    cfg = make_lady_bandit_guard(
        n_guards=1,
        n_bandits=1,
        dt=10.0,
        max_horizon_s=200.0,
        guard_observation_fn=obs_g,
        bandit_observation_fn=obs_b,
    )
    env = OrbitalGameEnv(cfg)

    n_motion = float(ref_mean_motion(cfg.reference_orbit))
    hcw_params = SimpleNamespace(mean_motion=n_motion)

    def per_vehicle_dyn(x, u, dt):
        return hcw_rtn_step(x[None, :], u[None, :], hcw_params, dt)[0]

    updater = EKFBeliefUpdater(
        dynamics_fn=per_vehicle_dyn,
        process_noise=jnp.eye(6) * 0.01,
        dt=10.0,
    )
    init = EKFUniformDefaultInitializer(
        layout=layout,
        default_mean=jnp.zeros(6),
        variance_diag=jnp.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
    )

    br = BeliefRollout(env, init, updater, init, updater)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj, belief_history = run_belief_rollout(br, policies, init_ps, key, n_steps=12)
    return cfg, traj, belief_history


def test_scene_precomputes_axis_and_scales(basic_cfg_and_traj):
    cfg, traj = basic_cfg_and_traj
    scene = RolloutScene(traj=traj, dt=10.0, show_thrust=True)
    assert scene.axis_limit_m > 0
    assert scene.n_frames == 12
    assert scene._cube_scale > 0
    # No thrust in zero-control rollout → scale is 0.
    assert scene._thrust_scale == 0.0


def test_scene_axis_limit_override(basic_cfg_and_traj):
    _, traj = basic_cfg_and_traj
    scene = RolloutScene(traj=traj, axis_limit=5000.0)
    assert scene.axis_limit_m == 5000.0


def test_render_frame_no_toggles(basic_cfg_and_traj):
    _, traj = basic_cfg_and_traj
    scene = RolloutScene(
        traj=traj,
        show_cubes=False,
        show_trail=False,
        show_sensor_range=False,
        show_thrust=False,
        show_belief=False,
        show_reference_marker=False,
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    render_frame(scene, ax, 0)
    plt.close(fig)


def test_render_frame_all_toggles_no_belief(basic_cfg_and_traj):
    cfg, traj = basic_cfg_and_traj
    scene = RolloutScene(
        traj=traj,
        cfg=cfg,
        dt=10.0,
        show_cubes=True,
        show_trail=True,
        show_sensor_range=True,
        show_thrust=True,
        show_belief=False,
    )
    # cfg has FullObservation → sensor_range layer becomes a no-op.
    assert scene._sensor_ranges[Side.GUARD] is None
    assert scene._sensor_ranges[Side.BANDIT] is None
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    render_frame(scene, ax, scene.n_frames - 1)
    plt.close(fig)


def test_scene_infers_sensor_range_from_cfg(range_limited_cfg_traj_belief):
    cfg, traj, _ = range_limited_cfg_traj_belief
    scene = RolloutScene(traj=traj, cfg=cfg, show_sensor_range=True)
    assert scene._sensor_ranges[Side.GUARD] == 2000.0
    assert scene._sensor_ranges[Side.BANDIT] == 2000.0


def test_render_frame_with_belief(range_limited_cfg_traj_belief):
    cfg, traj, belief_history = range_limited_cfg_traj_belief
    scene = RolloutScene(
        traj=traj,
        cfg=cfg,
        dt=10.0,
        show_cubes=True,
        show_sensor_range=True,
        show_belief=True,
        belief_history=belief_history,
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for f in (0, scene.n_frames // 2, scene.n_frames - 1):
        render_frame(scene, ax, f)
    plt.close(fig)


def test_explicit_sensor_range_overrides_cfg(range_limited_cfg_traj_belief):
    cfg, traj, _ = range_limited_cfg_traj_belief
    scene = RolloutScene(
        traj=traj,
        cfg=cfg,
        show_sensor_range=True,
        sensor_range_m={Side.GUARD: 100.0, Side.BANDIT: 200.0},
    )
    assert scene._sensor_ranges[Side.GUARD] == 100.0
    assert scene._sensor_ranges[Side.BANDIT] == 200.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_save_animation_writes_file(basic_cfg_and_traj, tmp_path):
    _, traj = basic_cfg_and_traj
    scene = RolloutScene(traj=traj, dt=10.0)
    out = tmp_path / "rollout.mp4"
    path = save_animation(scene, str(out), fps=5, dpi=72)
    assert out.exists()
    assert out.stat().st_size > 0
    assert path == str(out)


def test_run_belief_rollout_shapes(range_limited_cfg_traj_belief):
    _, traj, belief_history = range_limited_cfg_traj_belief
    n_frames = traj.env_state.guards.rtn.shape[0]
    assert belief_history.guard.mean.shape == (n_frames, 1, 2, 6)
    assert belief_history.guard.cov.shape == (n_frames, 1, 2, 6, 6)
    assert belief_history.bandit.mean.shape == (n_frames, 1, 2, 6)


def test_scene_auto_resolves_3d_for_rtn_traj(basic_cfg_and_traj):
    _, traj = basic_cfg_and_traj
    scene = RolloutScene(traj=traj)
    assert scene.resolved_mode == "3d"


def test_scene_explicit_mode_override(basic_cfg_and_traj):
    _, traj = basic_cfg_and_traj
    scene = RolloutScene(traj=traj, mode="2d")
    assert scene.resolved_mode == "2d"
    # 2D positions are (T, R) → shape (T_steps, N_actors, 2)
    assert scene._g_xyz.shape[-1] == 2


def test_scene_invalid_mode_raises(basic_cfg_and_traj):
    _, traj = basic_cfg_and_traj
    with pytest.raises(ValueError, match="mode must be"):
        RolloutScene(traj=traj, mode="bogus")


def test_render_frame_2d_smoke(basic_cfg_and_traj):
    """Render a 2D frame onto a non-3D axes and confirm no errors."""
    cfg, traj = basic_cfg_and_traj
    scene = RolloutScene(
        traj=traj,
        cfg=cfg,
        mode="2d",
        dt=10.0,
        show_cubes=True,
        show_trail=True,
        show_thrust=False,
        show_belief=False,
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)  # plain 2D axes — no projection="3d"
    from orbital_game.viz.animation import render_frame

    for f in (0, scene.n_frames - 1):
        render_frame(scene, ax, f)
    assert ax.get_xlabel() == "T (m)"
    assert ax.get_ylabel() == "R (m)"
    plt.close(fig)


def test_render_frame_2d_with_belief(range_limited_cfg_traj_belief):
    """2D rendering with belief ellipses + sigma parameter."""
    cfg, traj, belief_history = range_limited_cfg_traj_belief
    scene = RolloutScene(
        traj=traj,
        cfg=cfg,
        mode="2d",
        dt=10.0,
        show_belief=True,
        belief_history=belief_history,
        belief_sigma=2.0,
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    from orbital_game.viz.animation import render_frame

    render_frame(scene, ax, scene.n_frames - 1)
    plt.close(fig)


def test_belief_sigma_is_one_by_default(basic_cfg_and_traj):
    _, traj = basic_cfg_and_traj
    scene = RolloutScene(traj=traj)
    assert scene.belief_sigma == 1.0
