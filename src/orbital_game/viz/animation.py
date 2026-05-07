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
    draw_belief_ellipse_2d,
    draw_belief_ellipsoid,
    draw_circle,
    draw_cone_3d,
    draw_cube,
    draw_sphere,
    draw_thrust_arrow,
    draw_wedge_2d,
    quat_to_rotation_matrix,
)

# Permutation matrix mapping the data's RTN axis order to matplotlib's plot
# axis order (x=T, y=N, z=R). For positions and Δv vectors we use
# ``arr[..., [1, 2, 0]]``; for covariance matrices we use ``P @ cov @ P.T``.
_RTN_TO_PLOT = np.array(
    [
        [0.0, 1.0, 0.0],  # plot x = T
        [0.0, 0.0, 1.0],  # plot y = N
        [1.0, 0.0, 0.0],  # plot z = R
    ]
)


def _positions_xyz(side_state: Any) -> np.ndarray:
    """Extract a ``(T_steps, N_actors, 3)`` position array in *plot* axis order.

    The underlying state stores positions in RTN order (``[R, T, N]``); this
    helper permutes them so matplotlib's ``(x, y, z)`` axes correspond to
    ``(T, N, R)`` — i.e. R is vertical, the orbital-mechanics convention.
    Handles both ``rtn`` (6-dim) and ``rt`` (4-dim, N filled with zero).
    """
    if hasattr(side_state, "rtn"):
        rtn = np.asarray(side_state.rtn[..., :3])
        return rtn[..., [1, 2, 0]]  # [R, T, N] → [T, N, R]
    if hasattr(side_state, "rt"):
        rt = np.asarray(side_state.rt)  # [R, T] (last axis dim 2+)
        t_col = rt[..., 1:2]
        r_col = rt[..., 0:1]
        zeros = np.zeros_like(t_col)
        return np.concatenate([t_col, zeros, r_col], axis=-1)  # [T, 0, R]
    raise AttributeError("Side state has no `rtn` or `rt` field.")


def _action_to_plot(action: np.ndarray) -> np.ndarray:
    """Permute an action array's last axis from RTN-Δv to 3D plot order.

    For 3-dim actions (``[dvR, dvT, dvN]``) returns ``[dvT, dvN, dvR]``.
    For 2-dim RT actions (``[dvR, dvT]``) returns ``[dvT, 0, dvR]``.
    """
    if action.shape[-1] >= 3:
        return action[..., [1, 2, 0]]
    # 2-dim RT case
    t_col = action[..., 1:2]
    r_col = action[..., 0:1]
    zeros = np.zeros_like(t_col)
    return np.concatenate([t_col, zeros, r_col], axis=-1)


def _positions_xy(side_state: Any) -> np.ndarray:
    """Extract a ``(T_steps, N_actors, 2)`` position array in 2D plot order ``[T, R]``.

    Works for both ``rt`` (4-dim, ``[R, T, ...]``) and ``rtn`` (6-dim,
    ``[R, T, N, ...]``) by dropping the N component for the 2D projection.
    """
    if hasattr(side_state, "rt"):
        rt = np.asarray(side_state.rt)
        return rt[..., [1, 0]]  # [R, T] → [T, R]
    if hasattr(side_state, "rtn"):
        rtn = np.asarray(side_state.rtn)
        return rtn[..., [1, 0]]  # drop N; [R, T, N] → [T, R]
    raise AttributeError("Side state has no `rtn` or `rt` field.")


def _action_to_plot_2d(action: np.ndarray) -> np.ndarray:
    """Permute an action's last axis from RTN-Δv to 2D plot order ``[dvT, dvR]``."""
    return action[..., [1, 0]]


def _quat_history(side_state: Any) -> np.ndarray | None:
    """Return ``(T, N, 4)`` quaternion history if Attitude is present, else None."""
    if hasattr(side_state, "quat"):
        return np.asarray(side_state.quat)
    return None


def _walk_for_conical(obs_fn: Any) -> list:
    """Recurse through CompositeObservation; return all ConicalObservation instances."""
    from orbital_game.observations.composite import CompositeObservation
    from orbital_game.observations.conical import ConicalObservation

    if isinstance(obs_fn, ConicalObservation):
        return [obs_fn]
    if isinstance(obs_fn, CompositeObservation):
        out = []
        for c in obs_fn.constituents:
            out.extend(_walk_for_conical(c))
        return out
    return []


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

    Limitation: every side must have ``ImpulsiveManeuver`` in its action
    components (the scene reads ``.dv`` for thrust quivers). For attitude-
    only scenarios, configure ``ImpulsiveManeuver`` with ``track_mass=False``
    and emit zero Δv from the policy — this satisfies the constructor
    without changing dynamics.
    """

    traj: Trajectory
    cfg: Any | None = None
    dt: float | None = None

    # ---- mode ----
    # ``"3d"`` (full RTN), ``"2d"`` (RT projection), or ``"auto"`` which picks
    # 2D when both sides only carry the ``rt`` (4-dim) state component, else
    # 3D. The two modes use different matplotlib axes types — ``interactive_viewer``
    # and ``save_animation`` create the correct kind based on this resolved mode.
    mode: str = "auto"

    # ---- layer toggles ----
    show_trail: bool = True
    show_cubes: bool = True
    show_sensor_range: bool = False
    show_sensor_cones: bool = False
    show_thrust: bool = False
    show_belief: bool = False
    show_reference_marker: bool = True

    # ---- layer params ----
    cube_scale: float | None = None
    sensor_range_m: dict[Side, float] | None = None
    sensor_range_color: str = "gray"
    sensor_range_alpha: float = 0.08
    cone_length_m: float | None = None  # None ⇒ render to plot edge (cones are unbounded);
    # set to a float to truncate (cosmetic only).
    cone_alpha: float = 0.15
    thrust_max_fraction: float = 1.0 / 20.0
    thrust_color: str = "orange"
    belief_history: BySide | None = None
    belief_sigma: float = 1.0
    belief_alpha: float = 0.15
    trail_alpha: float = 0.6

    # ---- view params ----
    axis_limit: float | None = None
    title_prefix: str = ""
    elev: float = 25.0
    azim: float = -60.0
    figsize: tuple[float, float] = (8.0, 7.0)

    # ---- precomputed (filled by __post_init__) ----
    # ``_g_xyz`` / ``_b_xyz`` carry the per-mode plot-order positions: shape
    # ``(T, N, 3)`` in 3D mode, ``(T, N, 2)`` in 2D mode. Same for actions.
    _mode: str = field(init=False, repr=False)
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
        # Resolve mode.
        if self.mode == "auto":
            g = self.traj.env_state.guards
            b = self.traj.env_state.bandits
            self._mode = "3d" if (hasattr(g, "rtn") and hasattr(b, "rtn")) else "2d"
        elif self.mode in ("2d", "3d"):
            self._mode = self.mode
        else:
            raise ValueError(f"mode must be 'auto', '2d', or '3d'; got {self.mode!r}")

        # Fail fast: the renderer reads `action.dv` for the thrust-quiver
        # overlays, so both sides' Command pytrees must carry the
        # ImpulsiveManeuver component (which contributes the `dv` field).
        # A comms-only side would otherwise raise an opaque AttributeError
        # below; this check names the offending side up front.
        from orbital_game.actions.components import ImpulsiveManeuver

        for side_name in ("guard", "bandit"):
            action = getattr(self.traj.sides, side_name).action
            comps = getattr(type(action), "_orbital_game_action_components", ())
            # `comps` may hold either component classes (legacy) or instances
            # (post-frame-aware refactor). Accept both forms.
            has_impulsive = any(
                c is ImpulsiveManeuver or isinstance(c, ImpulsiveManeuver) for c in comps
            )
            if not has_impulsive:
                raise ValueError(
                    f"RolloutScene requires IMPULSIVE_MANEUVER in "
                    f"{side_name}_action_components (it reads "
                    f"traj.sides.{side_name}.action.dv to draw thrust arrows). "
                    f"Got components={comps!r}"
                )

        # Trajectories store actions as Command pytrees; pull the ImpulsiveManeuver
        # component's `dv` field for the Δv quiver overlays.
        guard_dv = np.asarray(self.traj.sides.guard.action.dv)
        bandit_dv = np.asarray(self.traj.sides.bandit.action.dv)
        if self._mode == "3d":
            self._g_xyz = _positions_xyz(self.traj.env_state.guards)
            self._b_xyz = _positions_xyz(self.traj.env_state.bandits)
            self._g_action = _action_to_plot(guard_dv)
            self._b_action = _action_to_plot(bandit_dv)
        else:
            self._g_xyz = _positions_xy(self.traj.env_state.guards)
            self._b_xyz = _positions_xy(self.traj.env_state.bandits)
            self._g_action = _action_to_plot_2d(guard_dv)
            self._b_action = _action_to_plot_2d(bandit_dv)

        self._g_quat = _quat_history(self.traj.env_state.guards)
        self._b_quat = _quat_history(self.traj.env_state.bandits)
        self._n_frames = self._g_xyz.shape[0]

        # Auto axis limit: max |coord| across all guards + bandits over all time.
        coord_dim = self._g_xyz.shape[-1]
        if self.axis_limit is None:
            both = np.concatenate(
                [self._g_xyz.reshape(-1, coord_dim), self._b_xyz.reshape(-1, coord_dim)], axis=0
            )
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
                float(np.max(np.linalg.norm(self._g_action, axis=-1)))
                if self._g_action.size
                else 0.0
            )
            max_dv_b = (
                float(np.max(np.linalg.norm(self._b_action, axis=-1)))
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
    def resolved_mode(self) -> str:
        """``"2d"`` or ``"3d"`` — the mode actually used for rendering."""
        return self._mode

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
    """Draw a fading trail of past positions. Works for 2D and 3D — passes
    each coordinate column as a separate positional arg to ``ax.plot``."""
    if frame < 1:
        return
    cols = [xyz[: frame + 1, k] for k in range(xyz.shape[-1])]
    ax.plot(*cols, color=color, alpha=alpha, linewidth=1.2)


_RT_TO_PLOT_2D = np.array([[0.0, 1.0], [1.0, 0.0]])  # [R, T] → [T, R]


def _belief_position_block(belief: Any, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Pull (means, covs_3x3) for *opposing* targets at ``frame`` from a belief history.

    Belief shape: ``mean (T, N_obs, N_total, d)``, ``cov (T, N_obs, N_total, d, d)``.
    We only render belief about the *other* side (own-side belief mirrors truth).
    Returns position-only slices in 3D *plot* axis order: ``(N_obs, N_opp, 3)``
    for means (RTN→[T,N,R] permuted) and ``(N_obs, N_opp, 3, 3)`` for covariance
    (similarity transform ``P @ cov[:3,:3] @ P.T``). Used for ``mode="3d"``.
    """
    if belief is None:
        return None
    mean = np.asarray(belief.mean[frame])  # (N_obs, N_total, d) in RTN
    cov = np.asarray(belief.cov[frame])  # (N_obs, N_total, d, d) in RTN
    n_obs, n_total, _ = mean.shape
    n_opp = n_total - n_obs
    if n_opp <= 0:
        return None
    # Permute mean: position columns [R, T, N] → [T, N, R].
    opp_mean = mean[:, n_obs:, :3][..., [1, 2, 0]]
    # Permute cov: P @ cov[:3,:3] @ P.T over the per-pair leading axes.
    opp_cov_rtn = cov[:, n_obs:, :3, :3]
    p = _RTN_TO_PLOT
    opp_cov = np.einsum("ij,...jk,lk->...il", p, opp_cov_rtn, p)
    return opp_mean, opp_cov


def _belief_position_block_2d(belief: Any, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    """2D analogue of :func:`_belief_position_block`.

    Slices the position-only ``[R, T]`` 2x2 block (state dims 0..1) and
    permutes to plot order ``[T, R]`` via the swap matrix
    ``P_2d = [[0,1],[1,0]]``. Works for both 4-dim RT belief and 6-dim RTN
    belief (where the first two state dims are still R and T).
    """
    if belief is None:
        return None
    mean = np.asarray(belief.mean[frame])  # (N_obs, N_total, d)
    cov = np.asarray(belief.cov[frame])
    n_obs, n_total, _ = mean.shape
    n_opp = n_total - n_obs
    if n_opp <= 0:
        return None
    # Permute mean: [R, T] → [T, R].
    opp_mean = mean[:, n_obs:, :2][..., [1, 0]]
    # Permute cov: P_2d @ cov[:2,:2] @ P_2d.T over the per-pair leading axes.
    opp_cov_rt = cov[:, n_obs:, :2, :2]
    p = _RT_TO_PLOT_2D
    opp_cov = np.einsum("ij,...jk,lk->...il", p, opp_cov_rt, p)
    return opp_mean, opp_cov


def render_frame(scene: RolloutScene, ax: Any, frame: int) -> None:
    """Render one frame onto an existing axes. Dispatches to the 2D or 3D
    path based on ``scene.resolved_mode``. Clears the axes first.

    The caller is responsible for creating the right axes type:
    ``add_subplot(111, projection="3d")`` for 3D mode, plain ``add_subplot(111)``
    for 2D. :func:`interactive_viewer` and :func:`save_animation` do this.
    """
    if scene.resolved_mode == "3d":
        _render_frame_3d(scene, ax, frame)
    else:
        _render_frame_2d(scene, ax, frame)


def _set_title(scene: RolloutScene, ax: Any, frame: int) -> None:
    if scene.dt is not None:
        elapsed = frame * scene.dt
        title = f"{scene.title_prefix}t = {elapsed:.0f}s  (frame {frame}/{scene.n_frames - 1})"
    else:
        title = f"{scene.title_prefix}frame {frame}/{scene.n_frames - 1}"
    ax.set_title(title)


def _render_frame_3d(scene: RolloutScene, ax: Any, frame: int) -> None:
    ax.cla()
    lim = scene.axis_limit_m
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("T (m)")
    ax.set_ylabel("N (m)")
    ax.set_zlabel("R (m)")
    ax.view_init(elev=scene.elev, azim=scene.azim)

    if scene.show_reference_marker:
        ax.scatter([0], [0], [0], marker="*", color="black", s=60, label="reference")

    if scene.show_trail:
        for i in range(scene._g_xyz.shape[1]):
            _draw_trail(ax, scene._g_xyz[:, i, :], _GUARD_COLORS, scene.trail_alpha, frame)
        for j in range(scene._b_xyz.shape[1]):
            _draw_trail(ax, scene._b_xyz[:, j, :], _BANDIT_COLORS, scene.trail_alpha, frame)

    _render_side_3d(scene, ax, frame, Side.GUARD)
    _render_side_3d(scene, ax, frame, Side.BANDIT)

    if scene.show_belief and scene.belief_history is not None:
        _render_belief_3d(scene, ax, frame)

    _set_title(scene, ax, frame)


def _render_frame_2d(scene: RolloutScene, ax: Any, frame: int) -> None:
    ax.cla()
    lim = scene.axis_limit_m
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("T (m)")
    ax.set_ylabel("R (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if scene.show_reference_marker:
        ax.scatter([0], [0], marker="*", color="black", s=80, zorder=5, label="reference")

    if scene.show_trail:
        for i in range(scene._g_xyz.shape[1]):
            _draw_trail(ax, scene._g_xyz[:, i, :], _GUARD_COLORS, scene.trail_alpha, frame)
        for j in range(scene._b_xyz.shape[1]):
            _draw_trail(ax, scene._b_xyz[:, j, :], _BANDIT_COLORS, scene.trail_alpha, frame)

    _render_side_2d(scene, ax, frame, Side.GUARD)
    _render_side_2d(scene, ax, frame, Side.BANDIT)

    if scene.show_belief and scene.belief_history is not None:
        _render_belief_2d(scene, ax, frame)

    _set_title(scene, ax, frame)


def _render_side_3d(scene: RolloutScene, ax: Any, frame: int, side: Side) -> None:
    if side is Side.GUARD:
        xyz = scene._g_xyz
        action = scene._g_action
        quat = scene._g_quat
        color = _GUARD_COLORS
        obs_fn = getattr(scene.cfg, "guard_observation_fn", None) if scene.cfg is not None else None
    else:
        xyz = scene._b_xyz
        action = scene._b_action
        quat = scene._b_quat
        color = _BANDIT_COLORS
        obs_fn = (
            getattr(scene.cfg, "bandit_observation_fn", None) if scene.cfg is not None else None
        )

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

        if scene.show_sensor_cones:
            eff_length = (
                scene.cone_length_m if scene.cone_length_m is not None else 2.5 * scene._axis_limit
            )
            for cone_obs in _walk_for_conical(obs_fn):
                R_body = (  # noqa: N806
                    _RTN_TO_PLOT @ quat_to_rotation_matrix(quat[frame, i])
                    if quat is not None
                    else np.eye(3)
                )
                boresights = np.asarray(cone_obs.sensor_boresights_body)  # (k, 3)
                if isinstance(cone_obs.half_angle_rad, (int, float)):
                    half_angles = [float(cone_obs.half_angle_rad)] * boresights.shape[0]
                else:
                    half_angles = [float(a) for a in np.asarray(cone_obs.half_angle_rad).ravel()]
                for s in range(boresights.shape[0]):
                    b_world = R_body @ boresights[s]
                    draw_cone_3d(
                        ax,
                        center,
                        b_world,
                        half_angles[s],
                        eff_length,
                        color=color,
                        alpha=scene.cone_alpha,
                    )

        if scene.show_cubes:
            # Quat is body→RTN; the plot frame is permuted from RTN, so
            # body→plot = P @ R_rtn.
            R = (  # noqa: N806
                _RTN_TO_PLOT @ quat_to_rotation_matrix(quat[frame, i]) if quat is not None else None
            )
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


def _render_side_2d(scene: RolloutScene, ax: Any, frame: int, side: Side) -> None:
    if side is Side.GUARD:
        xyz = scene._g_xyz
        action = scene._g_action
        quat = scene._g_quat
        color = _GUARD_COLORS
        obs_fn = getattr(scene.cfg, "guard_observation_fn", None) if scene.cfg is not None else None
    else:
        xyz = scene._b_xyz
        action = scene._b_action
        quat = scene._b_quat
        color = _BANDIT_COLORS
        obs_fn = (
            getattr(scene.cfg, "bandit_observation_fn", None) if scene.cfg is not None else None
        )

    sensor_range = scene._sensor_ranges.get(side)

    for i in range(xyz.shape[1]):
        cx, cy = float(xyz[frame, i, 0]), float(xyz[frame, i, 1])

        if scene.show_sensor_range and sensor_range is not None:
            draw_circle(
                ax,
                np.array([cx, cy]),
                sensor_range,
                color=scene.sensor_range_color,
                alpha=scene.sensor_range_alpha,
            )

        if scene.show_sensor_cones:
            eff_length = (
                scene.cone_length_m if scene.cone_length_m is not None else 2.5 * scene._axis_limit
            )
            for cone_obs in _walk_for_conical(obs_fn):
                R_rtn = (  # noqa: N806
                    quat_to_rotation_matrix(quat[frame, i]) if quat is not None else np.eye(3)
                )
                boresights = np.asarray(cone_obs.sensor_boresights_body)  # (k, 3)
                if isinstance(cone_obs.half_angle_rad, (int, float)):
                    half_angles = [float(cone_obs.half_angle_rad)] * boresights.shape[0]
                else:
                    half_angles = [float(a) for a in np.asarray(cone_obs.half_angle_rad).ravel()]
                for s in range(boresights.shape[0]):
                    b_rtn = R_rtn @ boresights[s]
                    # Project to 2D plot coords [T, R]: b_rtn is [R, T, N] order
                    b_plot_xy = np.array([b_rtn[1], b_rtn[0]])  # [T, R]
                    draw_wedge_2d(
                        ax,
                        np.array([cx, cy]),
                        b_plot_xy,
                        half_angles[s],
                        eff_length,
                        color=color,
                        alpha=scene.cone_alpha,
                    )

        if scene.show_cubes:
            # No 3D orientation in 2D — render the agent as a filled square
            # marker. Quat-derived attitude is ignored for 2D scenes; project
            # to a yaw glyph if/when an Attitude component is added.
            ax.scatter(
                cx, cy, marker="s", s=80, c=[color], edgecolors="black", linewidths=0.6, zorder=4
            )
        else:
            ax.scatter(cx, cy, color=color, s=30, zorder=4)

        if scene.show_thrust and scene._thrust_scale > 0.0:
            dv_t, dv_r = float(action[frame, i, 0]), float(action[frame, i, 1])
            length = (dv_t**2 + dv_r**2) ** 0.5
            if length > 0.0:
                ax.quiver(
                    cx,
                    cy,
                    dv_t * scene._thrust_scale,
                    dv_r * scene._thrust_scale,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color=scene.thrust_color,
                    width=0.004,
                    zorder=4,
                )


def _render_belief_3d(scene: RolloutScene, ax: Any, frame: int) -> None:
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
                    sigma=scene.belief_sigma,
                )


def _render_belief_2d(scene: RolloutScene, ax: Any, frame: int) -> None:
    bh = scene.belief_history
    assert bh is not None
    for side, color in ((Side.GUARD, _GUARD_COLORS), (Side.BANDIT, _BANDIT_COLORS)):
        block = _belief_position_block_2d(bh.get(side), frame)
        if block is None:
            continue
        means, covs = block
        for i in range(means.shape[0]):
            for k in range(means.shape[1]):
                draw_belief_ellipse_2d(
                    ax,
                    means[i, k],
                    covs[i, k],
                    color=color,
                    alpha=scene.belief_alpha,
                    sigma=scene.belief_sigma,
                )


# ---- interactive Jupyter viewer (ipywidgets) ----


def interactive_viewer(scene: RolloutScene, fig: Any | None = None) -> Any:
    """Open an interactive 3D viewer in a Jupyter notebook.

    Mirrors ``refs/visualization.py``'s ``animate_pass_3d``: a ``Play``
    widget jslinked to an ``IntSlider``, with frame redraws via ``observe``.
    Returns the figure; the widget control bar is displayed via
    ``IPython.display.display``.

    .. warning::
        Requires a *live* matplotlib backend such as ``%matplotlib widget``
        (provided by ``ipympl``) or ``%matplotlib notebook``. With the
        default ``%matplotlib inline`` backend the figure is rendered as a
        static PNG and the slider will appear to advance the frame counter
        without redrawing the figure. This function emits a warning if the
        active backend doesn't support live updates.

    Outside Jupyter (no ``ipywidgets`` / no display), use
    :func:`save_animation` instead.
    """
    import warnings

    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "ipympl" not in backend and "nbagg" not in backend and "widget" not in backend:
        warnings.warn(
            f"interactive_viewer needs a live matplotlib backend (e.g. "
            f"'%matplotlib widget'), but the active backend is "
            f"'{matplotlib.get_backend()}'. The slider may advance without "
            f"redrawing the figure. Install ipympl and run "
            f"'%matplotlib widget' before constructing the viewer.",
            stacklevel=2,
        )

    import ipywidgets as widgets  # pyrefly: ignore[missing-import]
    from IPython.display import display  # pyrefly: ignore[missing-import]

    if fig is None:
        fig = plt.figure(figsize=scene.figsize)
    if scene.resolved_mode == "3d":
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
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
    """Render the scene to a video or GIF file.

    The writer is chosen from the file extension:
      - ``.mp4`` / ``.mov`` / ``.mkv`` → ffmpeg (must be on PATH).
      - ``.gif`` → PillowWriter (bundled with matplotlib; no external dep).

    Returns the output path on success.
    """
    import os
    import shutil

    import matplotlib.animation as mpl_animation

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        writer: Any = mpl_animation.PillowWriter(fps=fps)
    elif ext in (".mp4", ".mov", ".mkv"):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                f"save_animation({ext}) requires the 'ffmpeg' binary on PATH, "
                "but it was not found. Install it (e.g. `brew install ffmpeg` "
                "on macOS, `apt-get install ffmpeg` on Debian/Ubuntu), or save "
                "as .gif instead (no external dependency)."
            )
        writer = "ffmpeg"
    else:
        raise ValueError(
            f"save_animation: unsupported extension {ext!r} for {output_path!r}. "
            "Use .mp4/.mov/.mkv (ffmpeg) or .gif (PillowWriter)."
        )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if fig is None:
        fig = plt.figure(figsize=scene.figsize)
    if scene.resolved_mode == "3d":
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
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
    # When `writer` is an already-instantiated MovieWriter (e.g. PillowWriter),
    # matplotlib raises if fps/codec/etc. are also passed to anim.save — those
    # must be supplied only at writer construction time.  When it is a string
    # (e.g. "ffmpeg"), anim.save forwards them to the writer it creates.
    if isinstance(writer, str):
        anim.save(output_path, writer=writer, fps=fps, dpi=dpi)
    else:
        anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path
