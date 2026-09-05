"""3D animation system for rollouts: interactive viewer + MP4 export.

The shared core is :func:`render_frame`, which draws one frame onto an
existing 3D ``Axes``. The same renderer is consumed by:

- :func:`interactive_viewer` — ``ipywidgets`` Play + IntSlider in a Jupyter
  notebook (mirrors ``refs/visualization.py``)
- :func:`save_animation` — ``matplotlib.animation.FuncAnimation`` exported
  via ``ffmpeg`` to MP4

A :class:`RolloutScene` bundles the trajectory plus toggleable layers:
trails, agent cubes, sensor-range spheres, sensor cones, communication-link
cones, thrust arrows, and EKF belief ellipsoids. Heavy data-dependent
quantities (axis limit, max thrust magnitude) are precomputed once on the
scene so per-frame work stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from orbitalgym.env.types import BySide, Side, Trajectory
from orbitalgym.viz.glyphs import (
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
    from orbitalgym.observations.composite import CompositeObservation
    from orbitalgym.observations.conical import ConicalObservation

    if isinstance(obs_fn, ConicalObservation):
        return [obs_fn]
    if isinstance(obs_fn, CompositeObservation):
        out = []
        for c in obs_fn.constituents:
            out.extend(_walk_for_conical(c))
        return out
    return []


def _link_cone_params(link: Any) -> tuple[np.ndarray, float] | None:
    """Return ``(boresight_body_unit, half_angle_rad)`` for a ``PointingConeLink``.

    Returns ``None`` for other link kinds (e.g. ``AlwaysLinked``,
    ``GroundNetworkLink``) which carry no cone geometry to draw.
    """
    from orbitalgym.links.predicates import PointingConeLink

    if not isinstance(link, PointingConeLink):
        return None
    boresight = np.asarray(link.boresight_body, dtype=float)
    boresight = boresight / np.linalg.norm(boresight)
    return boresight, float(link.half_angle_rad)


def _resolve_link_masks(link_mask: Any, contact: Any) -> dict[Side, np.ndarray | None]:
    """Resolve the per-side link mask source: explicit ``link_mask`` wins,
    else the trajectory's logged ``contact`` field, else unknown (``None``).

    Accepts a ``BySide`` or a plain ``dict`` keyed by ``Side`` — both support
    ``.get(side)`` with a ``Side`` key.
    """
    source = link_mask if link_mask is not None else contact
    if source is None:
        return {Side.GUARD: None, Side.BANDIT: None}
    out: dict[Side, np.ndarray | None] = {}
    for side in (Side.GUARD, Side.BANDIT):
        mask = source.get(side)
        out[side] = np.asarray(mask) if mask is not None else None
    return out


def _infer_sensor_ranges(cfg: Any) -> dict[Side, float | None]:
    """Read ``sensor_range_m`` from a config if its observation fn is range-limited."""
    from orbitalgym.observations.range_limited import RangeLimitedObservation

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

    Thrust quivers draw ``traj.applied_dv``, the Δv the env actually
    imparted, whenever the trajectory carries it. A trajectory without that
    field falls back to the commanded ``traj.sides.<side>.action.dv``, which
    overstates a burn that thrust limits or an empty tank clipped.

    Limitation: a side that falls back to the commanded Δv must have
    ``ImpulsiveManeuver`` in its action components. For attitude-only
    scenarios, configure ``ImpulsiveManeuver`` with ``track_mass=False``
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
    show_contact_state: bool = False
    # When True, overlay each vehicle's communication cone from its side's
    # link predicate (``guard_link`` / ``bandit_link``), solid when
    # ``link_mask`` (or, absent that, ``traj.contact``) marks the link
    # closed at that frame, translucent otherwise. Only ``PointingConeLink``
    # carries cone geometry; other link kinds (e.g. ``AlwaysLinked``,
    # ``GroundNetworkLink``) are silently skipped.
    show_link_cones: bool = False
    # When True, render the cached LQR plan as a faint dashed polyline
    # forward from each agent's current position. Reads
    # ``traj.sides.<side>.policy_state.{plan, step_in_plan}`` and
    # propagates the remaining Δv commands forward from the agent's
    # CURRENT (pos, vel) state — read from
    # ``traj.env_state.<side>.rtn[frame, v, :6]`` — through ``plan_propagator``
    # if set, or a velocity-aware Euler fallback otherwise.
    #
    # ``plan_propagator`` signature: ``(state_rtn_6, dv_rtn_3, dt) ->
    # next_state_rtn_6``. The function should apply the impulsive Δv to
    # the velocity components and propagate one tick through the
    # scenario's dynamics. For HCW scenarios pass a function built from
    # the closed-form HCW STM. If left ``None``, the renderer uses a
    # straight-line Euler approximation that includes the agent's
    # initial velocity but NOT Coriolis / gravity-gradient terms — the
    # resulting polyline diverges from the true HCW response after a
    # quarter-orbit or so. Set this to get a curve that matches the
    # spacecraft's actual response.
    show_planned_trajectory: bool = False
    plan_trajectory_color: str = "magenta"
    plan_trajectory_alpha: float = 0.6
    plan_trajectory_linewidth: float = 1.2
    plan_propagator: Any = None

    # ---- layer params ----
    cube_scale: float | None = None
    sensor_range_m: dict[Side, float] | None = None
    sensor_range_color: str = "gray"
    sensor_range_alpha: float = 0.08
    cone_length_m: float | None = None  # None ⇒ render to plot edge (cones are unbounded);
    # set to a float to truncate (cosmetic only).
    cone_alpha: float = 0.15
    guard_link: Any = None  # link predicate (e.g. PointingConeLink) for comm-cone geometry
    bandit_link: Any = None
    link_mask: Any = None  # BySide or dict[Side, (T, n)] bool; defaults to traj.contact
    link_cone_length_m: float | None = None  # None ⇒ render to plot edge, like cone_length_m
    link_cone_color: str = "limegreen"
    link_cone_alpha_closed: float = 0.35
    link_cone_alpha_open: float = 0.08
    thrust_max_fraction: float = 1.0 / 20.0
    thrust_color: str = "orange"
    belief_history: BySide | None = None
    belief_sigma: float = 1.0
    belief_alpha: float = 0.15
    # Particle-filter-belief layer params. The renderer detects a particle-filter
    # belief by duck-typing (presence of ``particles`` + ``log_weights``) and
    # dispatches to a scatter-cloud overlay instead of the Gaussian ellipse.
    # ``belief_particle_size``: scatter marker size (matplotlib points).
    # ``belief_particle_alpha_{min,max}``: alpha mapped from normalized weight;
    #   the highest-weight particle gets ``alpha_max``, the lowest ``alpha_min``,
    #   intermediate weights interpolate linearly. Keeping a non-zero floor
    #   makes the cloud visible even after heavy collapse.
    # ``belief_particle_max_drawn``: per-pair cap. If ``K`` exceeds this, a
    #   weight-proportional resample picks ``belief_particle_max_drawn`` of them
    #   for display only — useful for large-K runs that would saturate the canvas.
    belief_particle_size: float = 8.0
    belief_particle_alpha_min: float = 0.05
    belief_particle_alpha_max: float = 0.7
    belief_particle_max_drawn: int | None = None
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
    _link_predicates: dict[Side, Any] = field(init=False, repr=False)
    _link_masks: dict[Side, np.ndarray | None] = field(init=False, repr=False)
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

        # Thrust quivers show the Δv the env imparted, so prefer
        # `traj.applied_dv`. Falling back to the commanded `action.dv`
        # requires the ImpulsiveManeuver component; a comms-only side would
        # otherwise raise an opaque AttributeError below, so name it up front.
        from orbitalgym.actions.components import ImpulsiveManeuver

        applied = getattr(self.traj, "applied_dv", None)
        side_dv: dict[str, np.ndarray] = {}
        for side_name in ("guard", "bandit"):
            side_applied = None if applied is None else getattr(applied, side_name)
            if side_applied is not None:
                side_dv[side_name] = np.asarray(side_applied)
                continue
            action = getattr(self.traj.sides, side_name).action
            comps = getattr(type(action), "_orbitalgym_action_components", ())
            # `comps` may hold either component classes or instances. Accept both.
            has_impulsive = any(
                c is ImpulsiveManeuver or isinstance(c, ImpulsiveManeuver) for c in comps
            )
            if not has_impulsive:
                raise ValueError(
                    f"RolloutScene requires IMPULSIVE_MANEUVER in "
                    f"{side_name}_action_components (without traj.applied_dv it "
                    f"reads traj.sides.{side_name}.action.dv to draw thrust "
                    f"arrows). Got components={comps!r}"
                )
            side_dv[side_name] = np.asarray(action.dv)

        guard_dv = side_dv["guard"]
        bandit_dv = side_dv["bandit"]
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

        self._link_predicates = {Side.GUARD: self.guard_link, Side.BANDIT: self.bandit_link}
        self._link_masks = _resolve_link_masks(self.link_mask, self.traj.contact)

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


def _is_particle_filter_belief(belief: Any) -> bool:
    """Duck-type check: a belief carrying ``particles`` and ``log_weights`` is a PF."""
    return belief is not None and hasattr(belief, "particles") and hasattr(belief, "log_weights")


def _pf_opposing_block_2d(belief: Any, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Pull (particles, log_weights) for *opposing* targets at ``frame``, in 2D plot order.

    Returns ``(particles_xy, log_weights)`` with shapes
    ``(N_obs, N_opp, K, 2)`` and ``(N_obs, N_opp, K)`` — already permuted from
    state order ``[R, T]`` to plot order ``[T, R]``. ``None`` for own-side-only beliefs.
    """
    if belief is None:
        return None
    particles = np.asarray(belief.particles[frame])  # (N_obs, N_total, K, d)
    log_w = np.asarray(belief.log_weights[frame])  # (N_obs, N_total, K)
    n_obs, n_total, _k, _d = particles.shape
    n_opp = n_total - n_obs
    if n_opp <= 0:
        return None
    # Position dims for both RT (d=4) and RTN (d=6) live in [0:2] = [R, T].
    opp_rt = particles[:, n_obs:, :, :2]
    opp_xy = opp_rt[..., [1, 0]]  # [R, T] → [T, R]
    return opp_xy, log_w[:, n_obs:, :]


def _pf_opposing_block_3d(belief: Any, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    """3D analogue of :func:`_pf_opposing_block_2d`.

    Returns ``(particles_xyz, log_weights)`` with positions permuted from state
    order ``[R, T, N]`` to plot order ``[T, N, R]`` for d>=6 RTN beliefs. For
    d=4 RT beliefs the N coordinate is filled with zero so PF beliefs from
    a planar scenario still render in 3D mode (defensive — RT beliefs only
    surface in 3D mode if mixed-frame scenes are constructed manually).
    """
    if belief is None:
        return None
    particles = np.asarray(belief.particles[frame])
    log_w = np.asarray(belief.log_weights[frame])
    n_obs, n_total, _k, d = particles.shape
    n_opp = n_total - n_obs
    if n_opp <= 0:
        return None
    if d >= 6:
        opp_rtn = particles[:, n_obs:, :, :3]
        opp_xyz = opp_rtn[..., [1, 2, 0]]  # [R, T, N] → [T, N, R]
    else:
        opp_rt = particles[:, n_obs:, :, :2]
        t_col = opp_rt[..., 1:2]
        r_col = opp_rt[..., 0:1]
        zeros = np.zeros_like(t_col)
        opp_xyz = np.concatenate([t_col, zeros, r_col], axis=-1)
    return opp_xyz, log_w[:, n_obs:, :]


def _alpha_per_particle(
    log_weights_pair: np.ndarray, alpha_min: float, alpha_max: float
) -> np.ndarray:
    """Map a per-pair ``(K,)`` log-weight vector to ``(K,)`` alpha values.

    Highest-weight particle gets ``alpha_max``; the rest scale linearly down
    toward ``alpha_min``. The mapping uses the normalized weight, so a sharply
    peaked posterior renders as one bright particle in a faint sea of others
    (which is exactly the read we want post-collapse).
    """
    log_w = log_weights_pair - log_weights_pair.max()
    w = np.exp(log_w)
    s = w.sum()
    if s > 0:
        w = w / s
    rel = w / (w.max() + 1e-12)
    return alpha_min + (alpha_max - alpha_min) * rel


def _maybe_subsample_particles(
    particles: np.ndarray, log_weights: np.ndarray, max_drawn: int | None, key_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weight-proportional resample to at most ``max_drawn`` particles for display.

    Pure visual subsampling — does not touch the actual belief. Uses
    ``numpy.random.default_rng`` with a frame-derived seed so the same frame
    redraws the same subsample (stable interactive scrubbing).
    """
    if max_drawn is None or particles.shape[0] <= max_drawn:
        return particles, log_weights
    rng = np.random.default_rng(key_seed)
    log_w = log_weights - log_weights.max()
    w = np.exp(log_w)
    w_sum = w.sum()
    probs = (w / w_sum) if w_sum > 0 else np.full(particles.shape[0], 1.0 / particles.shape[0])
    idx = rng.choice(particles.shape[0], size=max_drawn, replace=True, p=probs)
    return particles[idx], log_weights[idx]


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

    if scene.show_contact_state:
        _render_contact_state_3d(scene, ax, frame)

    if scene.show_planned_trajectory:
        _render_planned_trajectory_3d(scene, ax, frame)

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

    if scene.show_contact_state:
        _render_contact_state_2d(scene, ax, frame)

    if scene.show_planned_trajectory:
        _render_planned_trajectory_2d(scene, ax, frame)

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

        if scene.show_link_cones:
            cone_params = _link_cone_params(scene._link_predicates.get(side))
            if cone_params is not None:
                boresight_body, half_angle = cone_params
                mask = scene._link_masks.get(side)
                eff_length = (
                    scene.link_cone_length_m
                    if scene.link_cone_length_m is not None
                    else 2.5 * scene._axis_limit
                )
                closed = bool(mask[frame, i]) if mask is not None else False
                alpha = scene.link_cone_alpha_closed if closed else scene.link_cone_alpha_open
                R_body = (  # noqa: N806
                    _RTN_TO_PLOT @ quat_to_rotation_matrix(quat[frame, i])
                    if quat is not None
                    else np.eye(3)
                )
                b_world = R_body @ boresight_body
                draw_cone_3d(
                    ax,
                    center,
                    b_world,
                    half_angle,
                    eff_length,
                    color=scene.link_cone_color,
                    alpha=alpha,
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

        if scene.show_link_cones:
            cone_params = _link_cone_params(scene._link_predicates.get(side))
            if cone_params is not None:
                boresight_body, half_angle = cone_params
                mask = scene._link_masks.get(side)
                eff_length = (
                    scene.link_cone_length_m
                    if scene.link_cone_length_m is not None
                    else 2.5 * scene._axis_limit
                )
                closed = bool(mask[frame, i]) if mask is not None else False
                alpha = scene.link_cone_alpha_closed if closed else scene.link_cone_alpha_open
                R_rtn = (  # noqa: N806
                    quat_to_rotation_matrix(quat[frame, i]) if quat is not None else np.eye(3)
                )
                b_rtn = R_rtn @ boresight_body
                b_plot_xy = np.array([b_rtn[1], b_rtn[0]])  # [R, T, N] → plot [T, R]
                draw_wedge_2d(
                    ax,
                    np.array([cx, cy]),
                    b_plot_xy,
                    half_angle,
                    eff_length,
                    color=scene.link_cone_color,
                    alpha=alpha,
                )

        if scene.show_cubes:
            # 2D yaw glyph: rotated square + heading-indicator triangle
            # pointing along the body +x axis. When `quat` is None
            # (scenarios without an Attitude component) we fall back to
            # an unrotated square — visually identical to the previous
            # scatter-marker rendering.
            if quat is not None:
                R_rtn = quat_to_rotation_matrix(quat[frame, i])  # noqa: N806
                # Body +x in RTN -> [R, T, N] components. Plot uses [T, R].
                bx_t, bx_r = float(R_rtn[1, 0]), float(R_rtn[0, 0])
                yaw = np.arctan2(bx_r, bx_t)
            else:
                yaw = 0.0
            edge = scene._axis_limit / 50.0
            half = edge / 2.0
            cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
            R2 = np.array([[cos_y, -sin_y], [sin_y, cos_y]])  # noqa: N806
            corners_body = np.array(
                [
                    [+half, +half],
                    [-half, +half],
                    [-half, -half],
                    [+half, -half],
                ]
            )
            corners = corners_body @ R2.T + np.array([cx, cy])
            ax.add_patch(
                mpatches.Polygon(
                    corners,
                    closed=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.85,
                    zorder=4,
                )
            )
            # Heading-indicator triangle: tip at +x body face, base inside.
            tri_body = np.array(
                [
                    [+half * 1.7, 0.0],
                    [+half * 0.7, +half * 0.5],
                    [+half * 0.7, -half * 0.5],
                ]
            )
            tri = tri_body @ R2.T + np.array([cx, cy])
            ax.add_patch(
                mpatches.Polygon(
                    tri,
                    closed=True,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=5,
                )
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
        belief = bh.get(side)
        if belief is None:
            continue
        if _is_particle_filter_belief(belief):
            _render_pf_belief_3d(scene, ax, frame, belief, color)
            continue
        block = _belief_position_block(belief, frame)
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
        belief = bh.get(side)
        if belief is None:
            continue
        if _is_particle_filter_belief(belief):
            _render_pf_belief_2d(scene, ax, frame, belief, color)
            continue
        block = _belief_position_block_2d(belief, frame)
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


def _render_pf_belief_2d(
    scene: RolloutScene, ax: Any, frame: int, belief: Any, color: tuple[float, float, float]
) -> None:
    block = _pf_opposing_block_2d(belief, frame)
    if block is None:
        return
    particles, log_w = block  # (N_obs, N_opp, K, 2), (N_obs, N_opp, K)
    n_obs, n_opp, _k, _ = particles.shape
    for i in range(n_obs):
        for k in range(n_opp):
            pts = particles[i, k]
            lw = log_w[i, k]
            pts, lw = _maybe_subsample_particles(
                pts, lw, scene.belief_particle_max_drawn, key_seed=frame * 7919 + i * 31 + k
            )
            alphas = _alpha_per_particle(
                lw, scene.belief_particle_alpha_min, scene.belief_particle_alpha_max
            )
            rgba = np.zeros((pts.shape[0], 4))
            rgba[:, :3] = color
            rgba[:, 3] = alphas
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=scene.belief_particle_size,
                c=rgba,
                edgecolors="none",
                zorder=3,
            )
    # Also draw the weighted-mean point per opposing pair so a fully collapsed
    # cloud (one alpha-bright particle in a faint sea) still has a clear focus.
    mean_block = np.asarray(belief.mean[frame])  # (N_obs, N_total, d)
    n_total = mean_block.shape[1]
    n_obs_total = n_total - n_opp
    opp_means = mean_block[:, n_obs_total:, :2][..., [1, 0]]  # (N_obs, N_opp, 2)
    for i in range(n_obs):
        for k in range(n_opp):
            ax.scatter(
                opp_means[i, k, 0],
                opp_means[i, k, 1],
                s=scene.belief_particle_size * 4,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                zorder=4,
            )


def _render_pf_belief_3d(
    scene: RolloutScene, ax: Any, frame: int, belief: Any, color: tuple[float, float, float]
) -> None:
    block = _pf_opposing_block_3d(belief, frame)
    if block is None:
        return
    particles, log_w = block  # (N_obs, N_opp, K, 3), (N_obs, N_opp, K)
    n_obs, n_opp, _k, _ = particles.shape
    for i in range(n_obs):
        for k in range(n_opp):
            pts = particles[i, k]
            lw = log_w[i, k]
            pts, lw = _maybe_subsample_particles(
                pts, lw, scene.belief_particle_max_drawn, key_seed=frame * 7919 + i * 31 + k
            )
            alphas = _alpha_per_particle(
                lw, scene.belief_particle_alpha_min, scene.belief_particle_alpha_max
            )
            rgba = np.zeros((pts.shape[0], 4))
            rgba[:, :3] = color
            rgba[:, 3] = alphas
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=scene.belief_particle_size,
                c=rgba,
                edgecolors="none",
            )
    # Weighted-mean overlay (3D version).
    mean_block = np.asarray(belief.mean[frame])
    n_total = mean_block.shape[1]
    n_obs_total = n_total - n_opp
    if mean_block.shape[-1] >= 6:
        opp_means = mean_block[:, n_obs_total:, :3][..., [1, 2, 0]]
    else:
        opp_means_rt = mean_block[:, n_obs_total:, :2]
        t = opp_means_rt[..., 1:2]
        r = opp_means_rt[..., 0:1]
        zeros = np.zeros_like(t)
        opp_means = np.concatenate([t, zeros, r], axis=-1)
    for i in range(n_obs):
        for k in range(n_opp):
            ax.scatter(
                opp_means[i, k, 0],
                opp_means[i, k, 1],
                opp_means[i, k, 2],
                s=scene.belief_particle_size * 4,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
            )


def _render_contact_state_3d(scene: RolloutScene, ax: Any, frame: int) -> None:
    """Draw a bright outline ring around in-contact agents (3D scatter).

    Skipped silently for sides whose policy_state lacks an ``in_contact_prev``
    field, or when the flag is False at the current frame.
    """
    for side, color, positions in (
        (Side.GUARD, _GUARD_COLORS, scene._g_xyz),
        (Side.BANDIT, _BANDIT_COLORS, scene._b_xyz),
    ):
        policy_state = getattr(scene.traj.sides, side.value).policy_state
        in_contact_prev = (
            getattr(policy_state, "in_contact_prev", None) if policy_state is not None else None
        )
        if in_contact_prev is None:
            continue
        if not bool(np.asarray(in_contact_prev[frame])):
            continue
        for v in range(positions.shape[1]):
            pt = positions[frame, v]
            ax.scatter(
                *pt,
                marker="o",
                facecolors="none",
                edgecolors=color,
                s=200,
                linewidths=1.5,
            )


def _render_contact_state_2d(scene: RolloutScene, ax: Any, frame: int) -> None:
    """2D analogue of :func:`_render_contact_state_3d`."""
    for side, color, positions in (
        (Side.GUARD, _GUARD_COLORS, scene._g_xyz),
        (Side.BANDIT, _BANDIT_COLORS, scene._b_xyz),
    ):
        policy_state = getattr(scene.traj.sides, side.value).policy_state
        in_contact_prev = (
            getattr(policy_state, "in_contact_prev", None) if policy_state is not None else None
        )
        if in_contact_prev is None:
            continue
        if not bool(np.asarray(in_contact_prev[frame])):
            continue
        for v in range(positions.shape[1]):
            pt = positions[frame, v]
            ax.scatter(
                pt[0],
                pt[1],
                marker="o",
                facecolors="none",
                edgecolors=color,
                s=200,
                linewidths=1.5,
            )


_RTN_POS_TO_PLOT_3D = np.array([1, 2, 0])  # [R, T, N] → [T, N, R]
_RT_POS_TO_PLOT_2D = np.array([1, 0])  # [R, T] → [T, R]


def _planned_trajectory_path(
    plan_remaining: np.ndarray,  # (T_remaining, dv_dim) in [R, T, N] order
    cur_state_rtn: np.ndarray,  # (6,) [R, T, N, dR, dT, dN] — full state in RTN
    dt: float,
    plot_dim: int,
    propagator: Any = None,  # optional callable (state, dv, dt) -> next_state
) -> np.ndarray:
    """Propagate the remaining plan forward from the agent's CURRENT
    (pos, vel) state to produce a polyline of predicted positions.

    `cur_state_rtn` is the full RTN state ``[R, T, N, dR, dT, dN]`` read
    from the recorded trajectory. The agent's velocity at the current
    frame is what makes the predicted curve diverge from a straight
    line (orbital drift dominates over the LQR Δv at HCW scales).

    If ``propagator`` is provided, it propagates ``state`` through the
    scenario's true dynamics each tick (so the curve matches what the
    spacecraft actually does). Signature:
    ``propagator(state_rtn_6, dv_rtn_3, dt) -> next_state_rtn_6``.

    Without a propagator, the helper uses a velocity-aware Euler
    fallback that produces a straight line through the agent's
    instantaneous velocity. This shows direction and magnitude but does
    NOT match the true HCW curve.

    Returns a polyline of shape ``(T_remaining + 1, plot_dim)`` in plot
    axis order — [T, R] for 2D or [T, N, R] for 3D.
    """
    n_remaining = plan_remaining.shape[0]
    cur_pos_plot = (
        cur_state_rtn[:3][_RTN_POS_TO_PLOT_3D]
        if plot_dim == 3
        else cur_state_rtn[:2][_RT_POS_TO_PLOT_2D]
    )
    if n_remaining == 0:
        return cur_pos_plot.reshape(1, -1)

    # Pad Δv to width 3 (R, T, N) — natural form the propagator expects.
    dv_dim = plan_remaining.shape[-1]
    if dv_dim < 3:
        pad = np.zeros((n_remaining, 3 - dv_dim))
        plan_rtn = np.concatenate([plan_remaining, pad], axis=-1)
    else:
        plan_rtn = plan_remaining[:, :3]

    path = np.empty((n_remaining + 1, plot_dim))
    path[0] = cur_pos_plot

    if propagator is not None:
        # True-dynamics path: propagator handles impulse application AND
        # one-tick free-drift propagation.
        state = np.asarray(cur_state_rtn, dtype=np.float64).copy()
        for h in range(n_remaining):
            state = np.asarray(propagator(state, plan_rtn[h], dt))
            if plot_dim == 3:
                path[h + 1] = state[:3][_RTN_POS_TO_PLOT_3D]
            else:
                path[h + 1] = state[:2][_RT_POS_TO_PLOT_2D]
        return path

    # Fallback: velocity-aware Euler. Uses the agent's CURRENT velocity
    # (so the line is tangent to the true orbit at frame=now) but
    # straight (no Coriolis / gravity-gradient curving).
    pos_rtn = cur_state_rtn[:3].copy()
    vel_rtn = cur_state_rtn[3:6].copy() if cur_state_rtn.shape[0] >= 6 else np.zeros(3)
    for h in range(n_remaining):
        vel_rtn = vel_rtn + plan_rtn[h]
        pos_rtn = pos_rtn + vel_rtn * dt
        if plot_dim == 3:
            path[h + 1] = pos_rtn[_RTN_POS_TO_PLOT_3D]
        else:
            path[h + 1] = pos_rtn[:2][_RT_POS_TO_PLOT_2D]
    return path


def _full_state_rtn(side_state: Any, frame: int, v: int) -> np.ndarray:
    """Read the full ``(6,)`` RTN state ``[R, T, N, dR, dT, dN]`` for one
    vehicle at one frame. Falls back to ``[R, T, dR, dT, 0, 0]`` for
    rt-only states (planar HCW)."""
    if hasattr(side_state, "rtn"):
        return np.asarray(side_state.rtn[frame, v, :6], dtype=np.float64)
    if hasattr(side_state, "rt"):
        rt = np.asarray(side_state.rt[frame, v], dtype=np.float64)
        # rt layout: [R, T, dR, dT]. Lift to 6-D with zero N-components.
        return np.array([rt[0], rt[1], 0.0, rt[2], rt[3], 0.0])
    raise AttributeError("side_state has no rtn or rt field")


def _render_planned_trajectory_3d(scene: RolloutScene, ax: Any, frame: int) -> None:
    """Render the cached plan as a faint dashed polyline forward from
    each agent's current position.

    Uses ``scene.plan_propagator`` (if set) to propagate through real
    dynamics — produces the curved HCW trajectory the spacecraft will
    actually follow. Falls back to a velocity-aware straight Euler
    (which matches the spacecraft's velocity vector but not its curving
    response) when no propagator is provided.

    Reads ``traj.sides.<side>.policy_state.{plan, step_in_plan}`` for
    the cached plan and ``traj.env_state.<side>.rtn[frame, v, :6]`` for
    the agent's current full state. Skips silently when the side's
    policy_state isn't `PlanCacheState`-shaped.
    """
    dt = scene.dt if scene.dt is not None else 1.0
    for side, side_state in (
        (Side.GUARD, scene.traj.env_state.guards),
        (Side.BANDIT, scene.traj.env_state.bandits),
    ):
        policy_state = getattr(scene.traj.sides, side.value).policy_state
        if policy_state is None:
            continue
        plan = getattr(policy_state, "plan", None)
        step_in_plan = getattr(policy_state, "step_in_plan", None)
        if plan is None or step_in_plan is None:
            continue
        plan_arr = np.asarray(plan[frame])  # (H, n_v, dv_dim)
        H, n_v, _ = plan_arr.shape  # noqa: N806 — H is the plan-horizon ticks, matches the spec
        step = int(np.asarray(step_in_plan[frame]))
        if step >= H:
            # Plan expired — nothing to render.
            continue
        plan_remaining = plan_arr[step:]  # (T_remaining, n_v, dv_dim)
        for v in range(n_v):
            state_rtn = _full_state_rtn(side_state, frame, v)
            path = _planned_trajectory_path(
                plan_remaining[:, v, :],
                state_rtn,
                dt,
                plot_dim=3,
                propagator=scene.plan_propagator,
            )
            ax.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                color=scene.plan_trajectory_color,
                linestyle="--",
                alpha=scene.plan_trajectory_alpha,
                linewidth=scene.plan_trajectory_linewidth,
            )


def _render_planned_trajectory_2d(scene: RolloutScene, ax: Any, frame: int) -> None:
    """2D analogue of :func:`_render_planned_trajectory_3d`."""
    dt = scene.dt if scene.dt is not None else 1.0
    for side, side_state in (
        (Side.GUARD, scene.traj.env_state.guards),
        (Side.BANDIT, scene.traj.env_state.bandits),
    ):
        policy_state = getattr(scene.traj.sides, side.value).policy_state
        if policy_state is None:
            continue
        plan = getattr(policy_state, "plan", None)
        step_in_plan = getattr(policy_state, "step_in_plan", None)
        if plan is None or step_in_plan is None:
            continue
        plan_arr = np.asarray(plan[frame])
        H, n_v, _ = plan_arr.shape  # noqa: N806 — H is the plan-horizon ticks, matches the spec
        step = int(np.asarray(step_in_plan[frame]))
        if step >= H:
            continue
        plan_remaining = plan_arr[step:]
        for v in range(n_v):
            state_rtn = _full_state_rtn(side_state, frame, v)
            path = _planned_trajectory_path(
                plan_remaining[:, v, :],
                state_rtn,
                dt,
                plot_dim=2,
                propagator=scene.plan_propagator,
            )
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=scene.plan_trajectory_color,
                linestyle="--",
                alpha=scene.plan_trajectory_alpha,
                linewidth=scene.plan_trajectory_linewidth,
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
