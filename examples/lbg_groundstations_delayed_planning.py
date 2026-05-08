# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: orbital-game (3.13.1)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LBG with ground-station-gated planning (delayed LQR / delayed MCTS)
#
# This notebook demonstrates the operational picture for Phase B of the
# ground-station-comms workstream:
#
# - **Two guards** (defenders, RTN frame) protect the lady (RTN origin)
#   from **one bandit**. Both sides share the **same ground-station
#   network** (Alaska + Australia) — both teams' planners are gated by
#   the same set of contact windows.
# - **Guards** carry a **Kalman-filter belief** + a **belief-aware LQR**
#   inner planner, wrapped in a `PlanCachePolicy` so the cached single-
#   step LQR command only refreshes on contact (default
#   `replan_contacts_lag=1`). The guards are the *operationally
#   constrained* side: their thrust commands are gated by the comms
#   schedule — between uploads the most recently uploaded Δv is replayed
#   verbatim every tick.
# - **Bandit** is the *studied* agent — full sensor processing and
#   continuous replanning. It carries a **particle-filter belief** +
#   a `BeliefAdaptedMCTSPolicy` that runs **every tick** (no
#   `PlanCachePolicy` wrapper). Inside the tree search the bandit models
#   the guards as **delayed-LQR**: the guards' LQR command is recomputed
#   only when the simulated tick falls inside a contact window, otherwise
#   the cached command from the most recent contact tick is replayed.
#   This is wired via `MCTSPolicy(opponent_schedule=schedule)`, which
#   threads a cached opponent Δv through the MCTS embedding so it
#   persists across tree-edge advances.
# - **Lag sweep** at the end varies the **guards'**
#   `replan_contacts_lag in {0, 1, 2, 3}` over `N_SEEDS=16` paired seeds
#   and reports outcome counts plus mean ± std of the bandit's closest
#   approach. The bandit's policy is unchanged across the sweep — we are
#   measuring how guard delay affects bandit success.

# %%
# Pin CPU as default device before importing JAX.
import os

os.environ["JAX_DEFAULT_DEVICE"] = "cpu"

import sys
from pathlib import Path

_here = Path.cwd()
if (_here / "src" / "orbital_game").is_dir():
    _repo_root = _here
elif (_here.parent / "src" / "orbital_game").is_dir():
    _repo_root = _here.parent
else:
    raise RuntimeError(f"Could not locate orbital-game repo root from cwd={_here}")
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# The imports below intentionally follow the sys.path mutation above so that
# running this notebook from the examples/ directory (without an editable
# install) still resolves the orbital_game package. Each import carries a
# per-line ruff suppression for the E-four-zero-two rule that the path-
# injection idiom necessarily triggers.
import dataclasses  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from orbital_game import OrbitalGameEnv, Side  # noqa: E402
from orbital_game.adapters.pomdp.adapter import POMDPAdapter  # noqa: E402
from orbital_game.belief.kf import KFBeliefUpdater, KFFromTruthInitializer  # noqa: E402
from orbital_game.belief.pf import (  # noqa: E402
    ParticleFilterBeliefUpdater,
    ParticleFilterFromTruthInitializer,
)
from orbital_game.config import VehicleParamsSpec  # noqa: E402
from orbital_game.dynamics.hcw import hcw_rt_step  # noqa: E402
from orbital_game.env.types import BySide  # noqa: E402
from orbital_game.games.lady_bandit_guard import make_lady_bandit_guard  # noqa: E402
from orbital_game.groundstations import GroundStation, GroundStationNetwork  # noqa: E402
from orbital_game.groundstations.contacts import precompute_contact_schedule  # noqa: E402
from orbital_game.observations.range_limited import RangeLimitedObservation  # noqa: E402
from orbital_game.policies.mcts import BeliefAdaptedMCTSPolicy, MCTSPolicy  # noqa: E402
from orbital_game.policies.plan_cache import PlanCachePolicy  # noqa: E402
from orbital_game.reference_orbit import ReferenceOrbitState, mean_motion  # noqa: E402
from orbital_game.registry import StateComponentKey  # noqa: E402
from orbital_game.rollout import belief_rollout  # noqa: E402
from orbital_game.sampling.mass import ConstantMass  # noqa: E402
from orbital_game.sampling.side import RelativeEllipse  # noqa: E402
from orbital_game.sampling.spec import ICSpec  # noqa: E402
from orbital_game.viz import RolloutScene, save_animation  # noqa: E402
from orbital_game.viz.groundtrack import plot_groundtrack  # noqa: E402

OUTPUT_DIR = _repo_root / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Reference orbit
#
# 500 km circular LEO at 51.6° inclination — orbital period ≈ 95 min.

# %%
EPOCH_MJD_UTC = 60067.5
HORIZON_S = 43200.0  # 12 hours. At MJD 60067.5 the orbit's geometry
# gives ~12 contact windows across the 5-station
# polar-coverage network, with the first window
# opening at t≈912s — early enough that the lag
# knob actually bites within the rollout horizon.

reference = ReferenceOrbitState.from_keplerian(
    semi_major_axis_m=6878e3,
    eccentricity=0.0,
    inclination=51.6,
    raan=0.0,
    argument_of_perigee=0.0,
    mean_anomaly=0.0,
)
period_s = 2.0 * np.pi * np.sqrt(6878e3**3 / 3.986004418e14)
n_motion = float(mean_motion(reference))
print(f"Reference orbit ECI position: {np.asarray(reference.position_eci)}")
print(f"Mean motion: {n_motion:.4e} rad/s   period: {period_s:.0f} s ({period_s / 60:.1f} min)")
print(f"Horizon: {HORIZON_S:.0f} s ≈ {HORIZON_S / period_s:.2f} orbits")

# %% [markdown]
# ## 2. Shared ground-station network
#
# Both the guards and the bandit are gated by the **same** network at
# 5° elevation mask. We pull station coordinates from
# `brahe.datasets.groundstations_load_all()` and pick five sites that
# give near-continuous global coverage from a polar-orbiting platform:
# Svalbard (78.2 °N), Athens, Hartebeesthoek (–25.9 °), Awarua
# (–46.5 °), and Punta Arenas (–53.0 °). Sharing a single network
# across both sides keeps the lag sweep apples-to-apples (the lag
# effect is isolated to the contact schedule, not the geometry).

# %%
import brahe  # noqa: E402

_brahe_stations = brahe.datasets.groundstations_load_all()


def _find_station(targets: tuple[str, ...]):
    """Return the first brahe PointLocation whose name matches any target (case-insensitive)."""
    for s in _brahe_stations:
        n = (s.get_name() or "").lower()
        for t in targets:
            if t.lower() in n:
                return s
    raise LookupError(f"None of {targets!r} found in brahe groundstations dataset")


def _from_brahe(point_location, *, mask_deg: float = 5.0) -> GroundStation:
    return GroundStation(
        name=point_location.get_name(),
        lat_deg=jnp.asarray(float(point_location.lat)),
        lon_deg=jnp.asarray(float(point_location.lon)),
        altitude_m=jnp.asarray(float(point_location.alt)),
        elevation_mask_deg=jnp.asarray(mask_deg),
    )


stations = tuple(
    _from_brahe(_find_station((name,)))
    for name in ("Punta Arenas", "Athens", "Hartebeesthoek", "Awarua", "Svalbard")
)

schedule = precompute_contact_schedule(
    stations=stations,
    reference_orbit=reference,
    epoch_mjd_utc=EPOCH_MJD_UTC,
    horizon_s=HORIZON_S,
)
network = GroundStationNetwork(stations=stations, schedule=schedule)

print(f"Shared network: {int(schedule.n_valid)} contact windows over {HORIZON_S / 3600:.2f} h.")
for i in range(int(schedule.n_valid)):
    t0 = float(schedule.windows[i, 0])
    t1 = float(schedule.windows[i, 1])
    sx = int(schedule.station_ix[i])
    print(f"  {stations[sx].name:10s}: t=[{t0:7.1f}, {t1:7.1f}] s   ({t1 - t0:5.1f} s long)")

# %% [markdown]
# ## 3. Gantt chart of contact windows

# %%
fig, ax = plt.subplots(figsize=(11, 2.2), constrained_layout=True)
n_windows = int(schedule.n_valid)
station_colors = ["tab:blue", "tab:orange"]
for i in range(n_windows):
    t0 = float(schedule.windows[i, 0])
    t1 = float(schedule.windows[i, 1])
    sx = int(schedule.station_ix[i])
    ax.barh(
        sx,
        t1 - t0,
        left=t0,
        color=station_colors[sx % len(station_colors)],
        height=0.6,
        edgecolor="black",
        linewidth=0.4,
    )
ax.set_yticks(range(len(stations)))
ax.set_yticklabels([s.name for s in stations])
ax.set_xlabel("Episode time (s)")
ax.set_xlim(0, HORIZON_S)
ax.set_title(f"Shared contact schedule ({n_windows} windows over {HORIZON_S / 3600:.2f} h)")
ax.grid(axis="x", alpha=0.3)
plt.show()

# %% [markdown]
# ## 4. Build the LBG environment
#
# 2 guards vs 1 bandit on RTN dynamics. Range-limited observations on
# both sides (10 km), the same shared ground-station network on both sides.

# %%
DT = 30.0
N_GUARDS = 2
N_BANDITS = 1
BREACH_RADIUS_M = 50.0
CATCH_RADIUS_M = 200.0

GUARD_PARAMS = VehicleParamsSpec(dry_mass_kg=12.0, isp_s=65.0, max_thrust_n=3.6)
BANDIT_PARAMS = VehicleParamsSpec(dry_mass_kg=12.0, isp_s=65.0, max_thrust_n=3.6)
PROPELLANT_KG = 5.0

# Geometry: guards on an RT-plane relative orbit (cross_track=0 → motion
# only in radial + along-track), bandit on an RN-dominant relative orbit
# (small radial coupling so cross-track dominates → bandit's natural
# motion is largely orthogonal to the guards' plane). This gives the
# bandit room to approach the lady (RTN origin) by maneuvering through
# cross-track that the guards' RT orbit doesn't cover, while the lag
# windows give it gaps where the guards' chase commands are stale.
guard_phases = jnp.linspace(0.0, 2.0 * jnp.pi, N_GUARDS, endpoint=False)
ic_sampler = ICSpec(
    # Guards: RT plane only — natural HCW radial-along-track ellipse.
    # Bumped radial amplitude to 2000m so guards start farther from the
    # lady, giving the bandit more approach options.
    guard_sampler=RelativeEllipse(
        radial_ellipse_m=2000.0,
        cross_track_m=0.0,  # RT plane (no cross-track motion)
        along_track_offset_m=0.0,
        phase_rad=guard_phases,
        mass_sampler=ConstantMass(propellant_mass_kg=PROPELLANT_KG),
    ),
    # Bandit: RN-dominant — small radial-coupled motion, large cross-track.
    # phase_rad=π/2 puts it at (r=0, t=0, n=8000) initially: directly
    # above the lady in the cross-track direction, 8 km away. Starting
    # further out gives the guards time to react before any meaningful
    # close-approach geometry develops.
    bandit_sampler=RelativeEllipse(
        radial_ellipse_m=500.0,  # small radial coupling
        cross_track_m=8000.0,  # bandit on RN plane (cross-track dominant)
        along_track_offset_m=0.0,
        phase_rad=jnp.asarray([jnp.pi / 2.0]),
        mass_sampler=ConstantMass(propellant_mass_kg=PROPELLANT_KG),
    ),
    validators=(),
    max_attempts=10,
)


def build_env(n_guards: int = N_GUARDS, n_bandits: int = N_BANDITS):
    cfg = make_lady_bandit_guard(
        n_guards=n_guards,
        n_bandits=n_bandits,
        breach_radius_m=BREACH_RADIUS_M,
        catch_radius_m=CATCH_RADIUS_M,
        dt=DT,
        max_horizon_s=HORIZON_S,
        seed=0,
        epoch_mjd_utc=EPOCH_MJD_UTC,
        reference_orbit=reference,
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        guard_params=GUARD_PARAMS,
        bandit_params=BANDIT_PARAMS,
        ic_sampler=ic_sampler,
        guard_ground_station_network=network,
        bandit_ground_station_network=network,
    )
    env = OrbitalGameEnv(cfg)
    layout = env.layout
    obs_fn = RangeLimitedObservation(
        layout=layout,
        sensor_range_m=10_000.0,
        sigma_range=5.0,
    )
    cfg = dataclasses.replace(
        cfg,
        guard_observation_fn=obs_fn,
        bandit_observation_fn=obs_fn,
        guard_ground_station_network=network,
        bandit_ground_station_network=network,
    )
    env = OrbitalGameEnv(cfg)
    return env, cfg


env, cfg = build_env()

# Compose the default LBG event termination (catch / breach / max-steps) with
# a runaway-drift guard that ends the episode if any vehicle gets more than
# MAX_DRIFT_M from the reference origin. Without this, an under-actuated
# controller can let HCW secular drift carry a satellite tens of km away
# before max_steps fires — which makes the demo plots unreadable.
from orbital_game.termination.max_distance import (  # noqa: E402
    AnyOfTermination,
    MaxDistanceTermination,
)

MAX_DRIFT_M = 20000.0
cfg = dataclasses.replace(
    cfg,
    termination_fn=AnyOfTermination(
        (cfg.termination_fn, MaxDistanceTermination(max_distance_m=MAX_DRIFT_M)),
    ),
)
env = OrbitalGameEnv(cfg)
adapter = POMDPAdapter(env)
print(
    f"Env: n_guards={cfg.n_guards}, n_bandits={cfg.n_bandits}, dt={cfg.dt}s, "
    f"truth dim={env.layout.dynamics_state_dim}"
)
print(f"Termination: AnyOf(LbgEvents, MaxDistance @ {MAX_DRIFT_M:.0f} m)")
print(f"Total ticks for {HORIZON_S:.0f}s @ dt={DT:.0f}s: {int(HORIZON_S / DT)}")

# %% [markdown]
# ## 5. Belief setup
#
# Architectural picture per side:
#
# - **Guards (operationally delayed):** Kalman filter + belief-aware LQR
#   inner, wrapped in `PlanCachePolicy`. Only refreshes plans on contact.
# - **Bandit (studied agent, full processing):** Particle filter +
#   `BeliefAdaptedMCTSPolicy` running *every tick* (no `PlanCachePolicy`
#   wrapper). Inside the tree the bandit predicts the guards as a
#   **delayed-LQR** controller: the LQR command is recomputed only on
#   contact ticks; otherwise the cached command from the most recent
#   contact tick is replayed. This is encoded by passing
#   `opponent_schedule=schedule` to `MCTSPolicy`.
#
# **Guards: Kalman filter** with a proper HCW-RTN STM. Six-state ``[R, T,
# N, Rdot, Tdot, Ndot]``. KF predict propagates beliefs correctly through
# the cache windows so the LQR gets a sensible state to act on.
#
# **Bandit: Particle filter** initialized from truth + tiny jitter (the
# bandit knows where it is at t=0). Process noise + range-limited
# updates produce belief shrinkage during contacts.

# %%
layout = env.layout
d = layout.dynamics_state_dim  # 6 for RTN


def _hcw_rtn_stm(mean_motion_rad_s: float, dt: float) -> jnp.ndarray:
    n = mean_motion_rad_s
    s = jnp.sin(n * dt)
    c = jnp.cos(n * dt)
    phi_in = jnp.array(
        [
            [4 - 3 * c, 0.0, s / n, 2 * (1 - c) / n],
            [6 * (s - n * dt), 1.0, -2 * (1 - c) / n, (4 * s - 3 * n * dt) / n],
            [3 * n * s, 0.0, c, 2 * s],
            [-6 * n * (1 - c), 0.0, -2 * s, 4 * c - 3],
        ]
    )
    phi_out = jnp.array(
        [
            [c, s / n],
            [-n * s, c],
        ]
    )
    Phi = jnp.zeros((6, 6))  # noqa: N806 - state transition matrix (math convention)
    in_axes = jnp.asarray([0, 1, 3, 4])
    Phi = Phi.at[jnp.ix_(in_axes, in_axes)].set(phi_in)  # noqa: N806
    out_axes = jnp.asarray([2, 5])
    Phi = Phi.at[jnp.ix_(out_axes, out_axes)].set(phi_out)  # noqa: N806
    return Phi


stm = _hcw_rtn_stm(env.mean_motion, DT)
control_matrix = stm[:, jnp.asarray([3, 4, 5])]
process_noise_kf = jnp.eye(d) * 0.5


# Bandit-side dynamics for the PF: HCW propagator over each particle.
def per_vehicle_hcw_rtn(x, u, dt):
    """Per-particle HCW propagator for the PF.
    Wraps `hcw_rt_step` for in-plane components and stitches the
    cross-track component via the closed-form 2×2 STM. Adding a
    standalone HCW-RTN propagator wrapper is left as a follow-up; this
    inline version is sufficient for a 6-D PF predict step."""
    n = n_motion
    s = jnp.sin(n * dt)
    c = jnp.cos(n * dt)
    # In-plane: reuse hcw_rt_step.
    x_in = jnp.array([x[0], x[1], x[3], x[4]])
    u_in = jnp.array([u[0], u[1]])
    x_in_next = hcw_rt_step(x_in[None, :], u_in[None, :], SimpleNamespace(mean_motion=n), dt)[0]
    # Cross-track 2x2 STM applied to (N, Ndot); the dv along N adds to Ndot.
    n_dot_after_dv = x[5] + u[2]
    x_n = c * x[2] + (s / n) * n_dot_after_dv
    x_n_dot = -n * s * x[2] + c * n_dot_after_dv
    return jnp.array([x_in_next[0], x_in_next[1], x_n, x_in_next[2], x_in_next[3], x_n_dot])


process_noise_pf = jnp.diag(jnp.asarray([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3]))

belief_init = BySide(
    guard=KFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(d) * 5.0),
    bandit=ParticleFilterFromTruthInitializer(
        layout=layout,
        n_particles=64,
        jitter_scale=1e-2,
    ),
)
belief_upd = BySide(
    guard=KFBeliefUpdater(
        stm=stm,
        control_matrix=control_matrix,
        process_noise=process_noise_kf,
    ),
    bandit=ParticleFilterBeliefUpdater(
        dynamics_fn=per_vehicle_hcw_rtn,
        process_noise=process_noise_pf,
        dt=DT,
        n_eff_threshold=0.5,
        resample_jitter_scale=0.05,
    ),
)
print("Belief: guards=KF (RTN STM), bandit=PF (K=64 particles)")

# %% [markdown]
# ## 6. Action grids + LQR gains
#
# 27-action 3D grid (6 axis-aligned + 1 zero, plus 20 mid-magnitude
# diagonals). Sized as a balance between MCTS expressivity and joint-
# space cardinality (with ``coordination='joint'`` and n_self=1 the
# bandit's joint cardinality is just A=27).

# %%
DV_MAX = float(GUARD_PARAMS.max_thrust_n * DT / (GUARD_PARAMS.dry_mass_kg + PROPELLANT_KG))
# Per-tick Δv caps. The 30-s tick is long: even a small impulse imparts
# secular HCW drift over a 12-h horizon if applied continuously. We size
# Δv per tick so the controllers can actually navigate the ~3 km initial
# bandit→lady gap (and the ~few-km guard formation) within the episode
# rather than drift secularly. With BANDIT_DV ≈ 0.15 m/s a single tick
# imparts up to ~5 m of position change in the next tick (HCW kinematics
# at dt=30 s) — enough that MCTS can find directions that close on the
# lady. Earlier 0.05 / 0.02 settings were too dampened: the bandit
# couldn't overcome HCW secular drift toward the +T direction and ended
# up at ~30 km from the lady by t = 14000 s.
GUARD_DV = 0.2
BANDIT_DV = 0.15
print(
    f"DV_MAX (saturated)={DV_MAX:.3f} m/s   "
    f"guard dv={GUARD_DV:.3f} m/s   bandit dv={BANDIT_DV:.3f} m/s"
)


# 27-direction 3D action grid (incl. zero) at fixed magnitude per side.
def build_action_grid_3d(dv: float) -> jax.Array:
    pts = []
    for r in (-1, 0, 1):
        for t in (-1, 0, 1):
            for n in (-1, 0, 1):
                if r == 0 and t == 0 and n == 0:
                    pts.append((0.0, 0.0, 0.0))
                else:
                    norm = (r * r + t * t + n * n) ** 0.5
                    pts.append((dv * r / norm, dv * t / norm, dv * n / norm))
    return jnp.asarray(pts, dtype=jnp.float32)


guard_action_grid = build_action_grid_3d(GUARD_DV)
bandit_action_grid = build_action_grid_3d(BANDIT_DV)
print(f"Action grids: guard={guard_action_grid.shape}, bandit={bandit_action_grid.shape}")


# Build LQR gains for guards (chasing the bandit) and the bandit
# (chasing the lady). The LQR kernel is the standard finite-horizon
# closed-form solve adapted to our HCW-RTN 6-D state.


def _build_lqr_gain(
    A: jax.Array,  # noqa: N803 - LQR state matrix (control-theory math convention)
    B: jax.Array,  # noqa: N803 - LQR control matrix (control-theory math convention)
    *,
    horizon: int,
    control_cost: float,
) -> jax.Array:
    """Returns gain matrix G such that u_0 = -G @ x_0 minimizes
    ``||C x_H||^2 + lambda ||U||^2`` where C selects positions [R,T,N]."""
    powers = [jnp.eye(A.shape[0])]
    for _ in range(horizon):
        powers.append(A @ powers[-1])
    A_powH = powers[horizon]  # noqa: N806 - A^H power (math convention)
    M = jnp.concatenate([powers[horizon - 1 - k] @ B for k in range(horizon)], axis=1)  # noqa: N806
    C = jnp.zeros((3, A.shape[0]))  # noqa: N806 - LQR output-selection matrix
    C = C.at[0, 0].set(1.0)  # noqa: N806
    C = C.at[1, 1].set(1.0)  # noqa: N806
    C = C.at[2, 2].set(1.0)  # noqa: N806
    H_lam = M.T @ C.T @ C @ M + control_cost * jnp.eye(M.shape[1])  # noqa: N806 - Hessian
    K = jnp.linalg.solve(H_lam, M.T @ C.T @ C @ A_powH)  # noqa: N806 - LQR gain
    u_dim = B.shape[1]
    return K[:u_dim, :]


A_rtn, B_rtn = stm, control_matrix
LQR_HORIZON = 8
LQR_LAMBDA = 1.0  # Reduced from 10.0 so the LQR doesn't over-dampen the
# guards' chase response. With the larger Δv cap above,
# an unclipped LQR command in the natural-motion regime
# is still small enough not to saturate; lowering λ
# makes the controller actually pursue the bandit rather
# than tracking a slow, smoothed trajectory that lets
# secular drift dominate the geometry.
lqr_gain = _build_lqr_gain(A_rtn, B_rtn, horizon=LQR_HORIZON, control_cost=LQR_LAMBDA)
print(f"LQR gain shape: {tuple(lqr_gain.shape)}, lambda={LQR_LAMBDA}")


# %% [markdown]
# ## 7. Belief-aware LQR policy (works for both sides)
#
# Generic per-side LQR that consumes a `Belief`-shaped `agent_view` (or a
# `_LaggedView` from PlanCachePolicy whose `.mean` is `(N_obs, N_total,
# d)`). Per-vehicle state is observer ``i``'s estimate of itself
# (``mean[i, i, :]``) — the LQR gain steers the vehicle toward the
# tracked target's position (``mean[i, target, :3]`` plus optional
# velocity matching). Wraps `LQRBanditPolicy`'s solve-and-clip strategy
# in a Belief-shaped contract.


# %%
@dataclass(frozen=True)
class BeliefAwareLQR:
    """Belief-aware closed-form LQR for either side.

    `target_index` selects which entity in observer i's tracked-entities
    block to chase. For guards ``target_index = n_self`` (first opposing
    bandit). For the bandit chasing the lady (origin), target is
    irrelevant — set ``chase_origin=True`` and the controller drives
    toward zero displacement.
    """

    n_self: int
    gain: jax.Array  # (3, 6): u = -gain @ (x_self - x_target)
    dv_max: float
    command_cls: Any
    target_index: int = 0
    chase_origin: bool = False

    def __call__(self, policy_state, agent_view, key, t):
        del key, t
        # Accept ContactAwareBelief, _LaggedView, or any Belief: read .mean.
        m = agent_view.inner.mean if hasattr(agent_view, "inner") else agent_view.mean
        # m shape: (n_obs, n_total, d=6).
        # Each observer i tracks itself at index i; the per-pair self-
        # belief is m[i, i, :].
        own = jnp.diagonal(m[:, : self.n_self, :], axis1=0, axis2=1).T  # (n_self, 6)
        target = jnp.zeros_like(own) if self.chase_origin else m[:, self.target_index, :]
        x_err = own - target  # (n_self, 6)
        dv_unclipped = -x_err @ self.gain.T  # (n_self, 3)
        dv = jnp.clip(dv_unclipped, -self.dv_max, self.dv_max)
        cmd = self.command_cls.zeros(self.n_self)
        return cmd.replace(dv=dv.astype(cmd.dv.dtype)), policy_state


guard_inner_lqr = BeliefAwareLQR(
    n_self=N_GUARDS,
    gain=lqr_gain,
    dv_max=GUARD_DV,
    command_cls=env.guard_command_cls,
    target_index=N_GUARDS,  # first bandit slot in the (own, opp) layout
    chase_origin=False,
)
print(
    f"Guard inner: BeliefAwareLQR "
    f"(chase bandit, target_index={guard_inner_lqr.target_index}, dv_max={GUARD_DV} m/s)"
)


# %% [markdown]
# ## 8. Bandit MCTS with **delayed-LQR** opponent model
#
# Inside the MCTS tree the bandit needs to predict how the **delayed**
# guards will behave so it can plan around the comms gaps. We build the
# opponent_model as a plain `FlatStateLQR` and wire the schedule into
# MCTS via `opponent_schedule=schedule`: at each tree-edge advance the
# bandit's MCTS recomputes the guards' LQR command only when the
# simulated tick falls inside a contact window; otherwise the *cached*
# command from the most recent contact tick is replayed. The cached Δv
# lives in the MCTS embedding alongside the flat state, so it persists
# across tree-edge advances correctly. This is the faithful delayed-LQR
# model — between contacts the guards keep thrusting with their last
# uplinked command, not zero.


# %%
@dataclass(frozen=True)
class FlatStateLQR:
    """Flat-position-obs LQR adapter. Used inside MCTS as the opponent_model.

    Inside MCTS the opponent's observation comes from
    `env.guard_observation_fn` (RangeLimitedObservation), whose obs is
    per-pair positions only — shape ``(n_self, n_total, 3)`` flattened.
    This adapter does a position-only LQR: it reads each observer's
    self-position estimate and the target position, and applies the
    first three columns of the 6-D LQR gain (velocity columns dropped).
    Velocity matching is unavailable in this view; the controller is
    therefore P-only relative to the full-state LQR, which is fine as a
    *model* of guard behaviour inside the bandit's tree search.
    """

    n_self: int
    n_opp: int
    obs_dim: int  # m=3 for RangeLimitedObservation positions
    gain_pos: jax.Array  # (3, 3) — position columns of the full LQR gain
    dv_max: float
    command_cls: Any
    target_index: int = 0
    chase_origin: bool = False

    def __call__(self, policy_state, agent_view, key, t):
        del key, t
        n_self = self.n_self
        n_total = n_self + self.n_opp
        obs_3d = agent_view.reshape((n_self, n_total, self.obs_dim))
        own = jnp.diagonal(obs_3d[:, :n_self, :], axis1=0, axis2=1).T  # (n_self, 3)
        target = jnp.zeros_like(own) if self.chase_origin else obs_3d[:, self.target_index, :]
        x_err = own - target  # (n_self, 3)
        dv = jnp.clip(-x_err @ self.gain_pos.T, -self.dv_max, self.dv_max)
        cmd = self.command_cls.zeros(n_self)
        return cmd.replace(dv=dv.astype(cmd.dv.dtype)), policy_state


# Opponent model the bandit's MCTS uses: a plain flat-LQR. Delay
# semantics are introduced by passing `opponent_schedule=schedule` to
# `MCTSPolicy` below — the LQR is recomputed on contact ticks and the
# cached Δv is replayed otherwise. The full-state LQR gain is (3, 6);
# drop velocity columns to get a position-only gain (3, 3) compatible
# with the position-only obs.
lqr_gain_pos = lqr_gain[:, :3]
guard_flat_lqr_for_mcts = FlatStateLQR(
    n_self=N_GUARDS,
    n_opp=N_BANDITS,
    obs_dim=3,
    gain_pos=lqr_gain_pos,
    dv_max=GUARD_DV,
    command_cls=env.guard_command_cls,
    target_index=N_GUARDS,
    chase_origin=False,
)


# Leaf-value heuristic: bias the search toward states where the bandit is
# closer to the lady (RTN origin). The flat-state layout puts the bandit's
# RTN position right after the guards' state — read those directly via the
# adapter so the index into the flat vector is layout-correct.
def _bandit_close_to_lady_value(s_flat: jax.Array) -> jax.Array:
    state = adapter.unpack(s_flat)
    bandit_pos = state.bandits.rtn[:, :3]  # (n_bandits, 3)
    dist = jnp.min(jnp.linalg.norm(bandit_pos, axis=-1))
    # Negative scaled distance → closer = higher value. We scale by 300m
    # (vs 1000m initially) so the gradient meaningfully exceeds the
    # ±1 terminal-reward signal: at 3000m, value = -10 (10x larger
    # magnitude than the catch penalty), so the bandit prefers
    # closing-toward-lady trajectories even at risk of guard catch.
    return -dist / 300.0


# Build the MCTS searching for the bandit. `opponent_schedule=schedule`
# turns on delayed-LQR semantics inside the tree: the guards' LQR is
# recomputed on contact ticks; otherwise the cached Δv is replayed.
mcts_bandit = MCTSPolicy(
    env_model=adapter,
    side=Side.BANDIT,
    action_grid=bandit_action_grid,
    opponent_model=guard_flat_lqr_for_mcts,
    opponent_action_grid=guard_action_grid,
    opponent_schedule=schedule,
    num_simulations=64,  # bumped from 16 so MCTS can actually find
    # close-to-lady directions in the 27-action grid.
    leaf_value_fn=_bandit_close_to_lady_value,
    n_vehicles=N_BANDITS,
    command_cls=env.bandit_command_cls,
    coordination="joint",
)

# Wrap MCTS to consume Belief-shaped agent_view (PF mean) — the
# wrapper bridges (N_obs, N_total, d) → flat state via the
# template_env_state and the env adapter.
template_state, _ = env.reset(jax.random.key(0))
bandit_inner_mcts = BeliefAdaptedMCTSPolicy(
    inner_mcts=mcts_bandit,
    template_env_state=template_state,
)
print(
    f"Bandit MCTS: A={bandit_action_grid.shape[0]} actions, "
    f"num_simulations={mcts_bandit.num_simulations}, "
    f"opponent_model=delayed-FlatStateLQR (via opponent_schedule)"
)


# %% [markdown]
# ## 9. Policy wrappers
#
# **Guards** wrap their LQR inner in `PlanCachePolicy` with a
# configurable `replan_contacts_lag`. Default lag=1: plan uplinked this
# contact was computed from the belief synced last contact.
#
# **Bandit** runs `BeliefAdaptedMCTSPolicy` directly — no
# `PlanCachePolicy` wrapper. The bandit replans every tick; the
# delayed-comms constraint is encoded *inside* MCTS via
# `opponent_schedule=schedule`, which models the guards as a delayed-LQR
# controller (LQR command frozen between contacts, recomputed on contact).


# %%
# Plan horizon: tuned to span the longest expected inter-contact gap so
# the cached LQR plan stays active through every off-contact period.
# At MJD 60067.5 with our 5-station network the longest gap in the 12-h
# horizon is ~5500s (~1 orbit). H=250 ticks @ dt=30s = 7500s ≈ 1.3
# orbits, so the plan covers slightly more than the longest gap. If the
# next upload doesn't arrive within H ticks the policy emits zero Δv
# (vehicle goes safe-mode).
GUARD_PLAN_HORIZON = 250
print(
    f"Guard plan horizon: H={GUARD_PLAN_HORIZON} ticks "
    f"= {GUARD_PLAN_HORIZON * DT:.0f}s ({GUARD_PLAN_HORIZON * DT / period_s:.2f} orbits)"
)


def make_guard_policy(replan_contacts_lag: int) -> PlanCachePolicy:
    return PlanCachePolicy(
        inner=guard_inner_lqr,
        plan_horizon=GUARD_PLAN_HORIZON,
        schedule=schedule,
        dt=DT,
        replan_contacts_lag=replan_contacts_lag,
        upload_delay_s=0.0,
        n_vehicles=N_GUARDS,
        command_cls=env.guard_command_cls,
    )


# Bandit policy is fixed — replans every tick regardless of guard lag.
bandit_policy = bandit_inner_mcts


print("Default policies (guard lag=1):")
print("  Guards:  PlanCachePolicy[BeliefAwareLQR] (H-tick plan, zero-fallback after expiry)")
print(
    "  Bandit:  BeliefAdaptedMCTSPolicy (per-tick), "
    "opp_model=FlatStateLQR + opponent_schedule (delayed-LQR)"
)


# %% [markdown]
# ## 10. Run a single rollout (lag=1)

# %%
N_STEPS = int(HORIZON_S / DT)
print(f"Rollout horizon: {HORIZON_S:.0f}s @ dt={DT:.0f}s = {N_STEPS} ticks")


def make_guard_init_ps_fn(lag: int):
    """Return a guard `(cfg, env_state, key) -> PlanCacheState` initializer.

    The PlanCacheState carries a belief_history ring buffer shaped
    against the KF belief mean. The guards' LQR inner is stateless under
    our protocol — pass ``inner_init_state=None``. We pass ``key`` so
    `init_state` can compute the initial plan against the seed belief
    (LQR is deterministic, but the API requires a key for non-
    deterministic inner planners).
    """

    def _init(_cfg, env_state, k):
        seed_belief = belief_init.guard(env_state, Side.GUARD, k)
        return make_guard_policy(lag).init_state(seed_belief, inner_init_state=None, key=k)

    return _init


def bandit_init_ps_fn(_cfg, _env_state, _k):
    """Bandit MCTS is stateless under the policy protocol — return None."""
    return None


def run_rollout(replan_contacts_lag: int, key: jax.Array):
    """Run one rollout where ``replan_contacts_lag`` controls the GUARDS' delay.

    The bandit policy is invariant across the lag sweep — it always
    replans every tick. The lag knob only affects the guards.
    """
    g_pol = make_guard_policy(replan_contacts_lag)
    policies = BySide(guard=g_pol, bandit=bandit_policy)
    init_ps_fns = BySide(
        guard=make_guard_init_ps_fn(replan_contacts_lag),
        bandit=bandit_init_ps_fn,
    )
    return belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
        key=key,
        n_steps=N_STEPS,
        guard_ground_station_network=network,
        bandit_ground_station_network=network,
    )


print("Running default rollout (guard lag=1, seed=0)...")
import time as _time  # noqa: E402

_t0 = _time.perf_counter()
traj_default, beliefs_default = run_rollout(replan_contacts_lag=1, key=jax.random.PRNGKey(0))
_elapsed = _time.perf_counter() - _t0
print(f"Done in {_elapsed:.1f}s ({_elapsed / N_STEPS * 1000:.1f} ms/tick)")

done = np.asarray(traj_default.episode_done)
n_real = int(np.argmax(done)) + 1 if done.any() else int(N_STEPS)
print(f"Episode terminated at tick {n_real} of {N_STEPS}")
final_state = jax.tree_util.tree_map(lambda x: x[n_real - 1], traj_default.env_state)
last_b = np.asarray(final_state.bandits.rtn[:, :3])
last_g = np.asarray(final_state.guards.rtn[:, :3])
d_lady = float(np.linalg.norm(last_b, axis=-1).min())
d_gb = float(np.linalg.norm(last_g[:, None] - last_b, axis=-1).min())
print(f"Final bandit→lady distance: {d_lady:.1f} m   (breach @ {BREACH_RADIUS_M:.0f} m)")
print(f"Final guard→bandit distance: {d_gb:.1f} m   (catch @ {CATCH_RADIUS_M:.0f} m)")

# %% [markdown]
# ## 11. Distance traces with contact windows shaded

# %%
times = np.asarray(traj_default.env_state.t)
bandit_positions = np.asarray(traj_default.env_state.bandits.rtn[:, 0, :3])
guard_positions = np.asarray(traj_default.env_state.guards.rtn[:, :, :3])
d_b_lady = np.linalg.norm(bandit_positions, axis=-1)
d_g_b = np.linalg.norm(guard_positions - bandit_positions[:, None, :], axis=-1)

fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
ax.plot(times, d_b_lady, color="tab:red", linewidth=1.5, label="bandit → lady")
for gi in range(N_GUARDS):
    ax.plot(
        times,
        d_g_b[:, gi],
        color="tab:blue",
        linewidth=1.0,
        alpha=0.7,
        label=f"guard{gi} → bandit" if gi == 0 else None,
    )

for i in range(int(schedule.n_valid)):
    t0 = float(schedule.windows[i, 0])
    t1 = float(schedule.windows[i, 1])
    ax.axvspan(t0, t1, color="gray", alpha=0.15, lw=0)

ax.axhline(
    BREACH_RADIUS_M,
    color="red",
    linestyle=":",
    linewidth=1.0,
    label=f"breach radius ({BREACH_RADIUS_M:.0f} m)",
)
ax.axhline(
    CATCH_RADIUS_M,
    color="green",
    linestyle=":",
    linewidth=1.0,
    label=f"catch radius ({CATCH_RADIUS_M:.0f} m)",
)
ax.set_xlabel("Episode time (s)")
ax.set_ylabel("Range (m)")
ax.set_title("Distance traces with contact windows shaded")
ax.set_yscale("log")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 12. Ground tracks + station footprints
#
# Cartopy equirectangular projection with the Earth basemap. Reference
# orbit traces in red; station triangles + elevation-mask footprints
# show where the comms windows land on the planet. HCW formation members
# (guards / bandit) are within km of the reference, which is below the
# resolution of a global map — they would draw exactly on top of the
# reference, so the per-side overlays are skipped automatically.


# %%
def _propagate_reference_eci(reference: ReferenceOrbitState, times_s: np.ndarray) -> np.ndarray:
    import brahe as _brahe

    _brahe.set_global_eop_provider_from_static_provider(_brahe.StaticEOPProvider.from_zero())
    epc0 = _brahe.Epoch.from_mjd(EPOCH_MJD_UTC, _brahe.TimeSystem.UTC)
    ref6 = np.concatenate(
        [np.asarray(reference.position_eci), np.asarray(reference.velocity_eci)]
    ).astype(np.float64)
    prop = _brahe.KeplerianPropagator.from_eci(epc0, ref6, 60.0)
    out = np.empty((times_s.shape[0], 6))
    for i, t in enumerate(times_s):
        epc = epc0 + float(t)
        out[i] = np.asarray(prop.state_eci(epc))
    return out


ref_eci = _propagate_reference_eci(reference, np.asarray(times))
fig_track = plot_groundtrack(
    epoch_mjd_utc=EPOCH_MJD_UTC,
    time_s=times,
    reference_orbit_eci=ref_eci,
    guard_eci=None,
    bandit_eci=None,
    guard_network=network,
    bandit_network=None,  # shared network — render once
    figsize=(13, 6.5),
)
fig_track.suptitle("Ground track with shared ground-station network", fontsize=13)
groundtrack_png = OUTPUT_DIR / "lbg_delayed_planning_groundtrack.png"
fig_track.savefig(groundtrack_png, dpi=130, bbox_inches="tight")
print(f"Saved ground-track image to {groundtrack_png}")
plt.show()

# %% [markdown]
# ## 13. Animation cell — RolloutScene → GIF
#
# `RolloutScene` with `show_contact_state=True` would surface the per-
# tick contact mask if the trajectory carried that state. v1 leaves the
# layer off (the PlanCacheState is in `traj.sides.guard.policy_state` but
# the renderer would need extra plumbing); we still get a clean RTN-frame
# animation of the chase.


# %%
# Truncate the trajectory to its actual termination tick — the rollout
# pads to N_STEPS with frozen-on-done state, which would otherwise show
# 90% of the video as a static frame after the episode ends.
def _truncate_trajectory(traj, n_real_ticks: int):
    """Slice every leaf along the leading time axis."""
    return jax.tree_util.tree_map(lambda x: x[:n_real_ticks], traj)


traj_for_scene = _truncate_trajectory(traj_default, n_real)
print(f"Animation: trimmed {N_STEPS} ticks → {n_real} actual episode ticks")


# HCW propagator for the planned-trajectory overlay — applies the impulsive
# Δv to velocity, then propagates one tick through the closed-form HCW
# STM. Without this the overlay falls back to a velocity-aware Euler
# straight line; with this the overlay curves to match the spacecraft's
# true HCW response, including Coriolis and gravity-gradient effects.
def _hcw_plan_propagator(state_rtn_6: np.ndarray, dv_rtn_3: np.ndarray, dt_s: float) -> np.ndarray:
    """`(state, dv, dt) -> next_state` using the closed-form HCW-RTN STM.

    `state_rtn_6` is ``[R, T, N, dR, dT, dN]``. Δv is applied
    impulsively to the velocity components; the STM then propagates the
    full state forward one tick.
    """
    state = np.asarray(state_rtn_6, dtype=np.float64).copy()
    state[3:6] += np.asarray(dv_rtn_3, dtype=np.float64)
    # Reuse the notebook's existing HCW STM (computed at the top of the
    # belief-setup cell as `stm = _hcw_rtn_stm(env.mean_motion, DT)`).
    return np.asarray(stm) @ state


scene = RolloutScene(
    traj=traj_for_scene,
    cfg=cfg,
    dt=DT,
    show_planned_trajectory=True,  # show guards' cached LQR plan as dashed magenta line
    show_contact_state=True,  # bright outline ring on agents in contact
    show_thrust=True,
    show_reference_marker=True,
    title_prefix="LBG delayed planning — ",
    figsize=(7.5, 7.5),
    plan_propagator=_hcw_plan_propagator,
)
print(
    f"RolloutScene: mode={scene.resolved_mode}, frames={scene.n_frames}, "
    f"axis_limit={scene.axis_limit_m:.0f} m"
)

gif_path = OUTPUT_DIR / "lbg_delayed_planning.gif"
save_animation(scene, str(gif_path), fps=10, dpi=90)
print(f"Saved animation to {gif_path}")

# Also export an mp4 when ffmpeg is available — gives the reader pause /
# scrub controls (HTML5 <video controls>) which a GIF can't provide. Falls
# back gracefully if ffmpeg isn't installed.
import shutil as _shutil  # noqa: E402

mp4_path = OUTPUT_DIR / "lbg_delayed_planning.mp4"
_have_ffmpeg = _shutil.which("ffmpeg") is not None
if _have_ffmpeg:
    save_animation(scene, str(mp4_path), fps=15, dpi=90)
    print(f"Saved animation to {mp4_path}")
else:
    print(
        "ffmpeg not on PATH — skipping mp4 export. Install ffmpeg "
        "(`brew install ffmpeg` / `apt-get install ffmpeg`) to enable."
    )

# %% [markdown]
# ### Inline playback
#
# Display both the mp4 (with HTML5 controls — pause / scrub / loop) and
# the GIF inline so the notebook stands alone as a demo. The saved files
# at `examples/output/lbg_delayed_planning.{gif,mp4}` are also durable
# artifacts you can share independently.

# %%
from IPython.display import Image, Video, display  # noqa: E402

if _have_ffmpeg and mp4_path.exists():
    display(Video(filename=str(mp4_path), embed=True, html_attributes="controls loop"))
display(Image(filename=str(gif_path), embed=True))

# %% [markdown]
# ## 14. Lag sweep — varies the **guards'** delay
#
# Vary the guards' `replan_contacts_lag in {0, 1, 2, 3}` over
# **N_SEEDS=16** paired seeds. The bandit policy stays unchanged across
# the sweep — we are isolating the operational effect of guard delay on
# the bandit's success.
#
# We report **two** per-side metrics — the natural reward / cost signals
# for each team's planner:
#
# - **bandit→lady closest** — the bandit's metric. Lower is better
#   *for the bandit* (closing on the lady is its objective).
# - **guard→bandit closest** — the guards' metric. Lower is better
#   *for the guards* (closing on the bandit is their objective).
#
# Higher guard lag → staler guard plans → guards should close on the
# bandit *less* effectively (guard→bandit closest grows), and the
# bandit should close on the lady *more* effectively (bandit→lady
# closest shrinks).

# %%
LAGS = (0, 1, 2, 3)
N_SEEDS = 16


def _trajectory_metrics(traj):
    """Return ``(bandit_to_lady_closest, guard_to_bandit_closest)`` for one rollout.

    Truncates to the actual termination tick (the rollout pads the rest
    with frozen-on-done state, which would otherwise drag both metrics
    toward whatever the final tick happens to be).
    """
    bandit_pos = np.asarray(traj.env_state.bandits.rtn[:, 0, :3])  # (T, 3)
    guard_pos = np.asarray(traj.env_state.guards.rtn[:, :, :3])  # (T, n_g, 3)
    done = np.asarray(traj.episode_done)
    n_real = int(np.argmax(done)) + 1 if done.any() else len(done)
    bandit_pos = bandit_pos[:n_real]
    guard_pos = guard_pos[:n_real]
    # Bandit's metric: closest the bandit got to the lady (origin).
    d_bandit_to_lady = np.linalg.norm(bandit_pos, axis=-1)  # (T,)
    bandit_closest = float(d_bandit_to_lady.min())
    # Guards' metric: closest any guard got to the bandit.
    d_guard_to_bandit = np.linalg.norm(guard_pos - bandit_pos[:, None, :], axis=-1)  # (T, n_g)
    guard_closest = float(d_guard_to_bandit.min())
    return bandit_closest, guard_closest


def evaluate_lag(lag: int, n_seeds: int = N_SEEDS):
    bandit_closests = []
    guard_closests = []
    breaches = 0
    catches = 0
    for s in range(n_seeds):
        traj, _ = run_rollout(replan_contacts_lag=lag, key=jax.random.PRNGKey(s))
        b_closest, g_closest = _trajectory_metrics(traj)
        bandit_closests.append(b_closest)
        guard_closests.append(g_closest)
        if b_closest < BREACH_RADIUS_M:
            breaches += 1
        if g_closest < CATCH_RADIUS_M:
            catches += 1
    return np.asarray(bandit_closests), np.asarray(guard_closests), breaches, catches


print(f"Running lag sweep: lags={LAGS}, seeds/lag={N_SEEDS}, ticks/rollout={N_STEPS}")
results = {}
for lag in LAGS:
    print(f"  lag={lag}...", end=" ", flush=True)
    _t0 = _time.perf_counter()
    bandit_closest, guard_closest, breaches, catches = evaluate_lag(lag)
    _el = _time.perf_counter() - _t0
    results[lag] = (bandit_closest, guard_closest, breaches, catches)
    print(
        f"bandit→lady closest mean={bandit_closest.mean():.0f}m "
        f"(min {bandit_closest.min():.0f}, max {bandit_closest.max():.0f}); "
        f"guard→bandit closest mean={guard_closest.mean():.0f}m "
        f"(min {guard_closest.min():.0f}, max {guard_closest.max():.0f}); "
        f"breach/catch={breaches}/{catches}  ({_el:.1f}s)"
    )

# %% [markdown]
# ## 15. Lag-sweep plots
#
# Three panels:
#
# - **Left:** bandit→lady closest distance vs lag (red — *the bandit's*
#   reward/cost metric; lower means the bandit closed on the lady).
# - **Middle:** guard→bandit closest distance vs lag (blue — *the guards'*
#   reward/cost metric; lower means the guards closed on the bandit).
# - **Right:** outcome counts (breach / catch / neither).

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)

lags_arr = np.asarray(list(LAGS))
mean_b = np.asarray([results[lag][0].mean() for lag in LAGS])
std_b = np.asarray([results[lag][0].std() for lag in LAGS])
mean_g = np.asarray([results[lag][1].mean() for lag in LAGS])
std_g = np.asarray([results[lag][1].std() for lag in LAGS])

axes[0].errorbar(
    lags_arr,
    mean_b,
    yerr=std_b,
    fmt="o-",
    color="tab:red",
    capsize=5,
    linewidth=1.5,
    markersize=8,
    label="bandit→lady (lower = better for bandit)",
)
axes[0].axhline(
    BREACH_RADIUS_M,
    color="red",
    linestyle=":",
    linewidth=1.0,
    label=f"breach radius ({BREACH_RADIUS_M:.0f} m)",
)
axes[0].set_xlabel("replan_contacts_lag")
axes[0].set_ylabel("Closest bandit→lady distance (m)")
axes[0].set_title(f"Bandit metric vs lag (mean ± std, {N_SEEDS} seeds)")
axes[0].set_xticks(lags_arr)
axes[0].grid(alpha=0.3)
axes[0].legend(loc="best", fontsize=9)

axes[1].errorbar(
    lags_arr,
    mean_g,
    yerr=std_g,
    fmt="o-",
    color="tab:blue",
    capsize=5,
    linewidth=1.5,
    markersize=8,
    label="guard→bandit (lower = better for guards)",
)
axes[1].axhline(
    CATCH_RADIUS_M,
    color="green",
    linestyle=":",
    linewidth=1.0,
    label=f"catch radius ({CATCH_RADIUS_M:.0f} m)",
)
axes[1].set_xlabel("replan_contacts_lag")
axes[1].set_ylabel("Closest guard→bandit distance (m)")
axes[1].set_title(f"Guard metric vs lag (mean ± std, {N_SEEDS} seeds)")
axes[1].set_xticks(lags_arr)
axes[1].grid(alpha=0.3)
axes[1].legend(loc="best", fontsize=9)

breach_counts = np.asarray([results[lag][2] for lag in LAGS])
catch_counts = np.asarray([results[lag][3] for lag in LAGS])
neither_counts = N_SEEDS - breach_counts - catch_counts
axes[2].bar(
    lags_arr,
    breach_counts,
    color="tab:red",
    edgecolor="black",
    linewidth=0.6,
    label="breach (bandit win)",
)
axes[2].bar(
    lags_arr,
    catch_counts,
    bottom=breach_counts,
    color="tab:green",
    edgecolor="black",
    linewidth=0.6,
    label="catch (guard win)",
)
axes[2].bar(
    lags_arr,
    neither_counts,
    bottom=breach_counts + catch_counts,
    color="lightgray",
    edgecolor="black",
    linewidth=0.6,
    label="neither",
)
axes[2].set_xticks(lags_arr)
axes[2].set_xlabel("replan_contacts_lag")
axes[2].set_ylabel(f"count (out of {N_SEEDS})")
axes[2].set_ylim(0, N_SEEDS)
axes[2].set_title("Outcomes vs lag")
axes[2].legend(loc="best", fontsize=9)
axes[2].grid(axis="y", alpha=0.3)

plt.show()

# %% [markdown]
# ## 16. Discussion
#
# **The mental model.** The bandit is the *studied* agent — full sensor
# processing, continuous replanning. The operational constraint is on
# the guards (real defense-system reality): their thrust commands are
# gated by the comms schedule via `PlanCachePolicy`. The bandit's task
# is to exploit those gaps. Inside MCTS the bandit predicts how the
# delayed-LQR guards will behave by passing
# `opponent_schedule=schedule` to `MCTSPolicy`: the guards' LQR command
# is recomputed on contact ticks, and the cached command is replayed
# between contacts. The cached Δv is carried as part of the MCTS
# embedding so it persists across tree-edge advances.
#
# **As guard lag grows**, the guards' uplinked plans go increasingly
# stale: the LQR is solving against a belief from `lag` contacts ago,
# which mismatches the bandit's actual trajectory more and more. The
# bandit, planning every tick against fresh beliefs, finds it easier to
# slip past the guards' chase.
#
# **At lag=0** plans refresh on every contact's freshest belief — closest
# to a continuous-replanning baseline for the guards, modulated by the
# `PlanCachePolicy` horizon `H=4` and the gaps between contacts.
#
# **What's not shown / FUTURE WORK.**
# - **Bandit-lag sweep**: in this version the bandit always replans
#   every tick. A future variant could put the bandit under
#   `PlanCachePolicy` too (e.g. a tactically delayed adversary) and
#   sweep its lag — the infrastructure is there, just wrap
#   `bandit_inner_mcts` in `PlanCachePolicy` analogously.
# - **Belief-staleness inside the tree**: the current
#   `opponent_schedule` machinery threads a *cached Δv* between
#   contacts, which captures the operational essence of delayed-LQR
#   (the guards keep thrusting with their last uplinked command).
#   A more faithful version would thread the full `PlanCacheState`
#   (including a stale belief) through `recurrent_fn` so MCTS also
#   simulates the lagged-belief input to the LQR solve directly.
# - **Animation overlays**: `RolloutScene(show_contact_state=True)`
#   requires the renderer to read `traj.sides.guard.policy_state.in_contact_prev`.
#   The guard `PlanCacheState` is logged but not yet plumbed into the
#   animation layer.
# - **Per-vehicle ground tracks**: at the global-map scale our HCW-
#   formation members are within km of the reference and not visually
#   distinguishable; the new `plot_groundtrack` skips them automatically.
#   Converting RTN→ECI per tick would only matter at much wider
#   formation spreads.
