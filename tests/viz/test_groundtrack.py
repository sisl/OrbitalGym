import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbitalgym.groundstations import (
    ContactSchedule,
    GroundStation,
    GroundStationNetwork,
)
from orbitalgym.viz.groundtrack import _split_antimeridian, plot_groundtrack


def _network():
    sta = GroundStation(
        name="A",
        lat_deg=jnp.asarray(64.8),
        lon_deg=jnp.asarray(-147.7),
        altitude_m=jnp.asarray(150.0),
        elevation_mask_deg=jnp.asarray(5.0),
    )
    sch = ContactSchedule(
        windows=jnp.asarray([[100.0, 300.0]] + [[-1.0, -1.0]] * 7, dtype=jnp.float32),
        n_valid=jnp.asarray(1),
        station_ix=jnp.asarray([0] + [-1] * 7),
    )
    return GroundStationNetwork(stations=(sta,), schedule=sch)


def test_plot_groundtrack_smoke(tmp_path):
    """Render a minimal ground-track without raising."""
    T = 60  # noqa: N806 - T is time-axis math convention
    ref_eci = jnp.zeros((T, 6))
    ref_eci = ref_eci.at[:, 0].set(7000e3)
    ref_eci = ref_eci.at[:, 1].set(jnp.linspace(0.0, 7.5e3 * 600.0, T))
    guard_eci = jnp.broadcast_to(ref_eci[:, None, :], (T, 1, 6))
    bandit_eci = guard_eci

    fig = plot_groundtrack(
        epoch_mjd_utc=60067.0,
        time_s=jnp.arange(T) * 10.0,
        reference_orbit_eci=ref_eci,
        guard_eci=guard_eci,
        bandit_eci=bandit_eci,
        guard_network=_network(),
        bandit_network=None,
    )
    out = tmp_path / "groundtrack.png"
    fig.savefig(out)
    assert out.exists()
    plt.close(fig)


def test_plot_groundtrack_no_networks_smoke(tmp_path):
    """Both networks None — just ground tracks, no stations."""
    T = 30  # noqa: N806 - T is time-axis math convention
    ref_eci = jnp.zeros((T, 6))
    ref_eci = ref_eci.at[:, 0].set(7000e3)
    ref_eci = ref_eci.at[:, 1].set(jnp.linspace(0.0, 7.5e3 * 300.0, T))
    guard_eci = jnp.broadcast_to(ref_eci[:, None, :], (T, 1, 6))
    bandit_eci = guard_eci

    fig = plot_groundtrack(
        epoch_mjd_utc=60067.0,
        time_s=jnp.arange(T) * 10.0,
        reference_orbit_eci=ref_eci,
        guard_eci=guard_eci,
        bandit_eci=bandit_eci,
        guard_network=None,
        bandit_network=None,
    )
    out = tmp_path / "groundtrack_no_nets.png"
    fig.savefig(out)
    assert out.exists()
    plt.close(fig)


def test_split_antimeridian_default_threshold_at_90_deg():
    """A 100° lon jump (unphysical between consecutive ticks) MUST be split.

    With the old 180° threshold this jump silently painted a horizontal line
    across the map (the Australia↔Africa artifact). With the new 90° default
    a NaN row is inserted between the two segments so ``ax.plot`` breaks
    the line.
    """
    lonlat = np.array(
        [
            [110.0, -30.0],  # Australia-ish
            [130.0, -30.0],
            [30.0, -30.0],  # Africa-ish — 100° backward jump
            [40.0, -30.0],
        ]
    )
    out = _split_antimeridian(lonlat)
    # NaN row inserted between index 1 and index 2.
    assert out.shape[0] == lonlat.shape[0] + 1
    assert np.isnan(out[2, 0])
    assert np.isnan(out[2, 1])
    # Real samples preserved.
    assert out[0, 0] == 110.0
    assert out[1, 0] == 130.0
    assert out[3, 0] == 30.0
    assert out[4, 0] == 40.0


def test_split_antimeridian_no_split_below_threshold():
    """Small lon increments (real LEO motion) MUST NOT be split."""
    lon = np.linspace(0.0, 80.0, 41)  # 2°/step — well under any reasonable threshold
    lat = np.linspace(-30.0, 30.0, 41)
    lonlat = np.stack([lon, lat], axis=1)
    out = _split_antimeridian(lonlat)
    assert out.shape == lonlat.shape
    assert not np.any(np.isnan(out))


def test_split_antimeridian_handles_real_wrap():
    """A genuine 180°→-180° wrap is split (this was already covered)."""
    lonlat = np.array(
        [
            [178.0, 0.0],
            [179.5, 0.0],
            [-179.5, 0.0],  # antimeridian wrap (~359° apparent jump)
            [-178.0, 0.0],
        ]
    )
    out = _split_antimeridian(lonlat)
    # NaN inserted between index 1 and 2.
    assert out.shape[0] == lonlat.shape[0] + 1
    assert np.isnan(out[2, 0])


def test_compound_figure_smoke(tmp_path):
    """Build a compound figure with a stub scene + minimal inputs."""
    # Build a minimal scene reusing fixtures from the animation extension test.
    # Easiest path: reuse the synthetic trajectory builder pattern, run it
    # in 2D mode to avoid 3D axes complications.
    import flax.struct
    import jax

    from orbitalgym.actions.assemble import build_command_class
    from orbitalgym.actions.components import ImpulsiveManeuver
    from orbitalgym.env.types import BySide, SideTrajectory, Trajectory
    from orbitalgym.registry import Frame
    from orbitalgym.viz.animation import RolloutScene
    from orbitalgym.viz.groundstation_compound import make_compound_figure

    @flax.struct.dataclass
    class _G:
        rt: jnp.ndarray

    @flax.struct.dataclass
    class _B:
        rt: jnp.ndarray

    @flax.struct.dataclass
    class _ES:
        t: jnp.ndarray
        step: jnp.ndarray
        guards: _G
        bandits: _B

    T = 5  # noqa: N806 - T is time-axis math convention
    n_g, n_b = 1, 1
    es = _ES(
        t=jnp.arange(T) * 10.0,
        step=jnp.arange(T),
        guards=_G(rt=jnp.zeros((T, n_g, 4))),
        bandits=_B(rt=jnp.zeros((T, n_b, 4))),
    )
    impulsive = ImpulsiveManeuver(action_frame=Frame.RT, truth_frame=Frame.RT, track_mass=False)
    GuardCmd = build_command_class((impulsive,), n_g, "GuardCmd")  # noqa: N806 - dynamic class
    BanditCmd = build_command_class((impulsive,), n_b, "BanditCmd")  # noqa: N806 - dynamic class
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
            policy_state=None,
        ),
        bandit=SideTrajectory(
            obs=jnp.zeros((T, 4)),
            action=b_action,
            reward=jnp.zeros((T,)),
            done=jnp.zeros((T,), dtype=bool),
            policy_state=None,
        ),
    )
    traj = Trajectory(env_state=es, sides=sides, episode_done=jnp.zeros((T,), dtype=bool))
    scene = RolloutScene(traj=traj, dt=10.0, mode="2d")

    ref_eci = jnp.zeros((T, 6))
    ref_eci = ref_eci.at[:, 0].set(7000e3)
    ref_eci = ref_eci.at[:, 1].set(jnp.linspace(0.0, 7.5e3 * 50.0, T))
    guard_eci = jnp.broadcast_to(ref_eci[:, None, :], (T, n_g, 6))
    bandit_eci = jnp.broadcast_to(ref_eci[:, None, :], (T, n_b, 6))

    fig, ax_scene, ax_map = make_compound_figure(
        scene,
        epoch_mjd_utc=60067.0,
        reference_orbit_eci=ref_eci,
        guard_eci=guard_eci,
        bandit_eci=bandit_eci,
        guard_network=None,
        bandit_network=None,
    )
    out = tmp_path / "compound.png"
    fig.savefig(out)
    assert out.exists()
    plt.close(fig)
