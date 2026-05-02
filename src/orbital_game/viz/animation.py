"""3D animation system for rollouts: interactive viewer + MP4 export.

The shared core is :func:`render_frame`, which draws one frame onto an
existing 3D ``Axes``. The same renderer is consumed by:

- :func:`interactive_viewer` — ``ipywidgets`` Play + IntSlider in a Jupyter
  notebook (mirrors ``refs/visualization.py``)
- :func:`save_animation` — ``matplotlib.animation.FuncAnimation`` exported
  via ``ffmpeg`` to MP4

A :class:`RolloutScene` bundles the trajectory plus toggleable layers:
trails, agent cubes, sensor-range spheres, thrust arrows, and EKF belief
ellipsoids. Heavy data-dependent quantities (axis limit, max thrust
magnitude) are precomputed once on the scene so per-frame work stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from orbital_game.env.types import BySide, Side, Trajectory
from orbital_game.viz.glyphs import (
    draw_belief_ellipsoid,
    draw_cube,
    draw_sphere,
    draw_thrust_arrow,
    quat_to_rotation_matrix,
)


def _positions_xyz(side_state: Any) -> np.ndarray:
    """Extract a ``(T, N, 3)`` position array from a guard/bandit state.

    Handles both ``rtn`` (6-dim) and ``rt`` (4-dim, padded with zero N).
    """
    if hasattr(side_state, "rtn"):
        return np.asarray(side_state.rtn[..., :3])
    if hasattr(side_state, "rt"):
        rt = np.asarray(side_state.rt)
        rn = np.zeros(rt.shape[:-1] + (1,), dtype=rt.dtype)
        return np.concatenate([rt[..., :2], rn], axis=-1)
    raise AttributeError("Side state has no `rtn` or `rt` field.")


def _quat_history(side_state: Any) -> np.ndarray | None:
    """Return ``(T, N, 4)`` quaternion history if Attitude is present, else None."""
    if hasattr(side_state, "quat"):
        return np.asarray(side_state.quat)
    return None


def _infer_sensor_ranges(cfg: Any) -> dict[Side, float | None]:
    """Read ``sensor_range_m`` from a config if its observation fn is range-limited."""
    from orbital_game.observations.range_limited import RangeLimitedObservation

    out: dict[Side, float | None] = {Side.GUARD: None, Side.BANDIT: None}
    if cfg is None:
        return out
    g = getattr(cfg, "guard_observation_fn", None)
    b = getattr(cfg, "bandit_observation_fn", None)
    if isinstance(g, RangeLimitedObservation):
        out[Side.GUARD] = float(g.sensor_range_m)
    if isinstance(b, RangeLimitedObservation):
        out[Side.BANDIT] = float(b.sensor_range_m)
    return out


@dataclass
class RolloutScene:
    """Bundle of trajectory + render configuration.

    Pass the same ``RolloutScene`` to :func:`interactive_viewer` and
    :func:`save_animation` to render the same scene through both paths.

    All ``show_*`` flags are independent — flipping ``show_sensor_range``
    on with ``cfg`` set will read ``sensor_range_m`` from
    ``cfg.guard_observation_fn`` / ``cfg.bandit_observation_fn`` if they
    are ``RangeLimitedObservation``; otherwise the layer is silently
    skipped for that side.
    """

    traj: Trajectory
    cfg: Any | None = None
    dt: float | None = None

    # ---- layer toggles ----
    show_trail: bool = True
    show_cubes: bool = True
    show_sensor_range: bool = False
    show_thrust: bool = False
    show_belief: bool = False
    show_reference_marker: bool = True

    # ---- layer params ----
    cube_scale: float | None = None
    sensor_range_m: dict[Side, float] | None = None
    sensor_range_color: str = "gray"
    sensor_range_alpha: float = 0.08
    thrust_max_fraction: float = 1.0 / 20.0
    thrust_color: str = "orange"
    belief_history: BySide | None = None
    belief_alpha: float = 0.15
    trail_alpha: float = 0.6

    # ---- view params ----
    axis_limit: float | None = None
    title_prefix: str = ""
    elev: float = 25.0
    azim: float = -60.0
    figsize: tuple[float, float] = (8.0, 7.0)

    # ---- precomputed (filled by __post_init__) ----
    _g_xyz: np.ndarray = field(init=False, repr=False)
    _b_xyz: np.ndarray = field(init=False, repr=False)
    _g_quat: np.ndarray | None = field(init=False, repr=False)
    _b_quat: np.ndarray | None = field(init=False, repr=False)
    _g_action: np.ndarray = field(init=False, repr=False)
    _b_action: np.ndarray = field(init=False, repr=False)
    _axis_limit: float = field(init=False, repr=False)
    _cube_scale: float = field(init=False, repr=False)
    _thrust_scale: float = field(init=False, repr=False)
    _sensor_ranges: dict[Side, float | None] = field(init=False, repr=False)
    _n_frames: int = field(init=False, repr=False)

    def __post_init__(self):
        self._g_xyz = _positions_xyz(self.traj.env_state.guards)
        self._b_xyz = _positions_xyz(self.traj.env_state.bandits)
        self._g_quat = _quat_history(self.traj.env_state.guards)
        self._b_quat = _quat_history(self.traj.env_state.bandits)
        self._g_action = np.asarray(self.traj.sides.guard.action)
        self._b_action = np.asarray(self.traj.sides.bandit.action)
        self._n_frames = self._g_xyz.shape[0]

        # Auto axis limit: max |coord| across all guards + bandits over all time.
        if self.axis_limit is None:
            both = np.concatenate([self._g_xyz.reshape(-1, 3), self._b_xyz.reshape(-1, 3)], axis=0)
            extent = float(np.max(np.abs(both))) if both.size else 1.0
            self._axis_limit = max(extent * 1.1, 1.0)
        else:
            self._axis_limit = float(self.axis_limit)

        # Auto cube scale: 1/40 of the largest axis extent (full extent = 2*limit).
        if self.cube_scale is None:
            self._cube_scale = (2.0 * self._axis_limit) / 40.0
        else:
            self._cube_scale = float(self.cube_scale)

        # Auto thrust scaling: max Δv occupies thrust_max_fraction of the axis
        # extent (full extent = 2*axis_limit). Smaller maneuvers shrink linearly.
        if self.show_thrust:
            max_dv_g = (
                float(np.max(np.linalg.norm(self._g_action[..., :3], axis=-1)))
                if self._g_action.size
                else 0.0
            )
            max_dv_b = (
                float(np.max(np.linalg.norm(self._b_action[..., :3], axis=-1)))
                if self._b_action.size
                else 0.0
            )
            max_dv = max(max_dv_g, max_dv_b)
            if max_dv > 1e-12:
                self._thrust_scale = (2.0 * self._axis_limit) * self.thrust_max_fraction / max_dv
            else:
                self._thrust_scale = 0.0
        else:
            self._thrust_scale = 0.0

        # Sensor range: explicit dict wins; otherwise infer from cfg.
        if self.sensor_range_m is None:
            self._sensor_ranges = _infer_sensor_ranges(self.cfg)
        else:
            self._sensor_ranges = {
                Side.GUARD: self.sensor_range_m.get(Side.GUARD),
                Side.BANDIT: self.sensor_range_m.get(Side.BANDIT),
            }

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def axis_limit_m(self) -> float:
        return self._axis_limit


# ---- per-frame rendering ----

_GUARD_COLORS = (0.10, 0.35, 0.75)
_BANDIT_COLORS = (0.75, 0.20, 0.20)


def _draw_trail(ax, xyz: np.ndarray, color, alpha: float, frame: int) -> None:
    if frame < 1:
        return
    ax.plot(
        xyz[: frame + 1, 0],
        xyz[: frame + 1, 1],
        xyz[: frame + 1, 2],
        color=color,
        alpha=alpha,
        linewidth=1.2,
    )


def _belief_position_block(belief: Any, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Pull (means, covs_3x3) for *opposing* targets at ``frame`` from a belief history.

    Belief shape: ``mean (T, N_obs, N_total, d)``, ``cov (T, N_obs, N_total, d, d)``.
    We only render belief about the *other* side (own-side belief mirrors truth).
    Returns position-only slices: ``(N_obs, N_opp, 3)`` and ``(N_obs, N_opp, 3, 3)``.
    """
    if belief is None:
        return None
    mean = np.asarray(belief.mean[frame])  # (N_obs, N_total, d)
    cov = np.asarray(belief.cov[frame])  # (N_obs, N_total, d, d)
    n_obs, n_total, _ = mean.shape
    n_opp = n_total - n_obs
    if n_opp <= 0:
        return None
    opp_mean = mean[:, n_obs:, :3]
    opp_cov = cov[:, n_obs:, :3, :3]
    return opp_mean, opp_cov


def render_frame(scene: RolloutScene, ax: Any, frame: int) -> None:
    """Render one frame onto an existing 3D axes. Clears the axes first."""
    ax.cla()
    lim = scene.axis_limit_m
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("T (m)")
    ax.set_zlabel("N (m)")
    ax.view_init(elev=scene.elev, azim=scene.azim)

    if scene.show_reference_marker:
        ax.scatter([0], [0], [0], marker="*", color="black", s=60, label="reference")

    # Trails first so cubes overlay cleanly.
    if scene.show_trail:
        for i in range(scene._g_xyz.shape[1]):
            _draw_trail(ax, scene._g_xyz[:, i, :], _GUARD_COLORS, scene.trail_alpha, frame)
        for j in range(scene._b_xyz.shape[1]):
            _draw_trail(ax, scene._b_xyz[:, j, :], _BANDIT_COLORS, scene.trail_alpha, frame)

    _render_side(scene, ax, frame, Side.GUARD)
    _render_side(scene, ax, frame, Side.BANDIT)

    if scene.show_belief and scene.belief_history is not None:
        _render_belief(scene, ax, frame)

    if scene.dt is not None:
        elapsed = frame * scene.dt
        title = f"{scene.title_prefix}t = {elapsed:.0f}s  (frame {frame}/{scene.n_frames - 1})"
    else:
        title = f"{scene.title_prefix}frame {frame}/{scene.n_frames - 1}"
    ax.set_title(title)


def _render_side(scene: RolloutScene, ax: Any, frame: int, side: Side) -> None:
    if side is Side.GUARD:
        xyz = scene._g_xyz
        action = scene._g_action
        quat = scene._g_quat
        color = _GUARD_COLORS
    else:
        xyz = scene._b_xyz
        action = scene._b_action
        quat = scene._b_quat
        color = _BANDIT_COLORS

    sensor_range = scene._sensor_ranges.get(side)

    for i in range(xyz.shape[1]):
        center = xyz[frame, i, :]

        if scene.show_sensor_range and sensor_range is not None:
            draw_sphere(
                ax,
                center,
                sensor_range,
                color=scene.sensor_range_color,
                alpha=scene.sensor_range_alpha,
            )

        if scene.show_cubes:
            R = quat_to_rotation_matrix(quat[frame, i]) if quat is not None else None  # noqa: N806
            draw_cube(
                ax,
                center,
                rotation=R,
                scale=scene._cube_scale,
                face_color=(*color, 0.5),
                edge_color="black",
                linewidth=0.4,
            )
        else:
            ax.scatter(*center, color=color, s=30)

        if scene.show_thrust and scene._thrust_scale > 0.0:
            dv = action[frame, i, :3] if action.shape[-1] >= 3 else np.append(action[frame, i], 0.0)
            draw_thrust_arrow(
                ax,
                center,
                dv,
                scale=scene._thrust_scale,
                color=scene.thrust_color,
            )


def _render_belief(scene: RolloutScene, ax: Any, frame: int) -> None:
    bh = scene.belief_history
    assert bh is not None
    for side, color in ((Side.GUARD, _GUARD_COLORS), (Side.BANDIT, _BANDIT_COLORS)):
        block = _belief_position_block(bh.get(side), frame)
        if block is None:
            continue
        means, covs = block
        for i in range(means.shape[0]):
            for k in range(means.shape[1]):
                draw_belief_ellipsoid(
                    ax,
                    means[i, k],
                    covs[i, k],
                    color=color,
                    alpha=scene.belief_alpha,
                )


# ---- interactive Jupyter viewer (ipywidgets) ----


def interactive_viewer(scene: RolloutScene, fig: Any | None = None) -> Any:
    """Open an interactive 3D viewer in a Jupyter notebook.

    Mirrors ``refs/visualization.py``'s ``animate_pass_3d``: a ``Play``
    widget jslinked to an ``IntSlider``, with frame redraws via ``observe``.
    Returns the figure; the widget control bar is displayed via
    ``IPython.display.display``.

    Outside Jupyter (no ``ipywidgets`` / no display), use
    :func:`save_animation` instead.
    """
    import ipywidgets as widgets  # pyrefly: ignore[missing-import]
    from IPython.display import display  # pyrefly: ignore[missing-import]

    if fig is None:
        fig = plt.figure(figsize=scene.figsize)
    ax = fig.add_subplot(111, projection="3d")
    plt.tight_layout()

    render_frame(scene, ax, 0)

    def _on_change(change):
        render_frame(scene, ax, int(change["new"]))
        fig.canvas.draw_idle()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=scene.n_frames - 1,
        step=1,
        description="Frame:",
        continuous_update=False,
        layout=widgets.Layout(width="80%"),
    )
    slider.observe(_on_change, names="value")

    play = widgets.Play(
        value=0, min=0, max=scene.n_frames - 1, step=1, interval=200, description="Play"
    )
    widgets.jslink((play, "value"), (slider, "value"))

    display(widgets.HBox([play, slider]))
    return fig


# ---- MP4 export (matplotlib.animation + ffmpeg) ----


def save_animation(
    scene: RolloutScene,
    output_path: str,
    fps: int = 10,
    dpi: int = 100,
    fig: Any | None = None,
) -> str:
    """Render the scene to a video file (requires ``ffmpeg``).

    Returns the output path on success.
    """
    import os

    import matplotlib.animation as mpl_animation

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if fig is None:
        fig = plt.figure(figsize=scene.figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.92)

    def _update(frame: int):
        render_frame(scene, ax, frame)

    # FuncAnimation's stub types `func` as returning `Iterable[Artist]`, which is
    # only required when `blit=True`. We don't use blitting (we cla()/redraw),
    # so returning None is fine at runtime — suppress the stub-driven type error.
    anim = mpl_animation.FuncAnimation(
        fig,
        _update,  # pyrefly: ignore[bad-argument-type]
        frames=scene.n_frames,
        interval=1000 // max(fps, 1),
    )
    anim.save(output_path, writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
    return output_path
