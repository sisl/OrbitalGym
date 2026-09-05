# Attitude dynamics & conical sensors

## Overview

A scenario can model rigid-body attitude alongside translational state.
With the `Attitude` and `BodyRates` state components, a torque-commanding
`AttitudeControl` action component, and the registered
`rigid_body_attitude_step` dynamics, an agent rotates its body in 2D
(N-axis only) or 3D (full SO(3)). Combined with a body-fixed
`ConicalObservation`, attitude becomes decision-relevant: an agent must
*point* its sensor cones at a target to receive belief updates.

Both sides in a scenario need not have attitude. The env's per-step loop
is guarded per-side, so asymmetric setups — guards with attitude, bandits
without (or vice versa) — work cleanly.

## State components

Four components participate in attitude-augmented scenarios:

- **`Attitude`** — quaternion `(n, 4)` in wxyz convention. Initialized to the
  identity quaternion `(1, 0, 0, 0)` by default (`Attitude.zeros` sets
  `quat[:, 0] = 1`). Represents body-to-reference-frame rotation.
- **`BodyRates`** — body-frame angular rate `(n, 3)` in rad/s. Zero-initialized.
- **`AppliedDV`** — transient translational Δv `(n, 3)` in m/s. Written by
  `ImpulsiveManeuver.apply`, consumed and zeroed by `env.step`.
- **`AppliedTorque`** — transient body-frame torque `(n, 3)` in N·m. Written by
  `AttitudeControl.apply`, consumed and zeroed by `env.step`.

`AppliedDV` and `AppliedTorque` are **transient** — they carry the "control
input for this tick" from action components to the dynamics block. Both are
zeroed after the dynamics consumes them, so they always read zero outside the
dynamics block. Users do not list them in `guard_components` or
`bandit_components`; `ScenarioConfig.__post_init__` auto-extends the
components tuples based on which spatial / attitude components are present:

- Any side with a spatial component (`RT`, `RTN`, `ECI`) gets `APPLIED_DV`
  added automatically.
- Any side with both `ATTITUDE` and `BODY_RATES` gets `APPLIED_TORQUE` added
  automatically.

`ATTITUDE` and `BODY_RATES` must appear together or both be absent — mixing
them is a fail-fast error raised in `ScenarioConfig.__post_init__`.

## The unified action-to-dynamics pattern

`ImpulsiveManeuver` and `AttitudeControl` follow the same pattern:

1. **Action fold** — `env.step` calls each action component's `apply` in
   registration order. Components write control inputs onto transient state
   fields (`applied_dv`, `applied_torque`) and return the updated side state.
   They do **not** invoke dynamics.
2. **Translational dynamics block** — always runs for any side with spatial
   state. Reads `applied_dv`, propagates the orbital state, zeros
   `applied_dv`.
3. **Attitude dynamics block** — runs iff `cfg.attitude_dynamics_key` is set
   for a side. Reads `applied_torque`, integrates `(quat, omega)`, zeros
   `applied_torque`.

A per-step ASCII diagram:

```
actions (per side)
  │
  ▼
┌─────────────────────────────────────────┐
│  Action fold  (in registration order)   │
│  ImpulsiveManeuver.apply → applied_dv   │
│  AttitudeControl.apply  → applied_torque│
└─────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────┐
│  Translational dynamics      │
│  truth_dynamics(state, dv)   │
│  applied_dv ← zeros          │
└──────────────────────────────┘
  │
  ▼
┌──────────────────────────────┐
│  Attitude dynamics           │  (only if attitude_dynamics_key set)
│  rigid_body_attitude_step(   │
│    quat, omega, torque, ...) │
│  applied_torque ← zeros      │
└──────────────────────────────┘
  │
  ▼
derived-frame views → observations → reward
```

The translational block always runs, even with zero `applied_dv` — the body
coasts. The attitude block always runs when configured, even with zero
`applied_torque` — the body precesses freely under Euler dynamics.

Prior to this refactor, `ImpulsiveManeuver.apply` called `truth_dynamics`
internally. It now only writes `applied_dv`; the env's explicit dynamics block
replaced the inline call. This makes the two dynamics layers orthogonal and
consistently owned by `env.step`.

## Rigid-body attitude dynamics

`rigid_body_attitude_step` is registered under `AttitudeDynamicsKey.RIGID_BODY`.
It integrates the coupled `(quat, omega)` ODE under applied body-frame torque.

**Equations of motion:**

```
ω̇ = I⁻¹(τ − ω × (I·ω))    # Euler's rotation equation (diagonal inertia)
q̇ = ½ Ω(ω) ⊗ q             # quaternion kinematics
```

where `I = diag(I_xx, I_yy, I_zz)` is the principal-axis inertia tensor and
`Ω(ω)` is the quaternion multiplication matrix for `ω = (ωx, ωy, ωz)`.

**Integration:** RK4 over the joint `(quat, omega)` state, with the torque
held constant across sub-steps. After each full step:

1. The quaternion is renormalized: `q ← q / ||q||`. RK4 drifts off the
   unit sphere by O(dt⁴) per step; renormalization is the standard fix.
2. Angular rates are clipped per axis: `ω ← clip(ω, −ω_max, +ω_max)`.

**Reaction-wheel saturation model:** `omega_max` lives in `AttitudeParams`,
not in the controller. When the body reaches `omega_max` on any axis,
commanding additional torque in that direction produces no further angular
acceleration — the reaction wheel is saturated. This is the physical model:
the controller can request unlimited torque; the dynamics enforces the
physical limit.

`AttitudeParams` carries:

- `inertia_diag` — `(3,)` array of principal inertias in kg·m² (`I_xx,
  I_yy, I_zz`).
- `omega_max` — `(3,)` array of per-axis saturation rates in rad/s.

`AttitudeDynamicsKey` is a **separate enum** from `DynamicsKey`. The
spatial-dynamics registration enforces `frame=` and `kind=` metadata that
attitude dynamics does not have (it operates on body-frame state, not on
any relative-orbit frame). Keeping them as separate enums means the
spatial-dynamics validator remains unaware of attitude propagators.

## Action: torque producer

`AttitudeControl` is the bundled torque-commanding action component:

```python
from orbitalgym.actions.components import AttitudeControl

component = AttitudeControl(
    rotation_dim=3,          # 1 for RT (N-axis only), 3 for RTN
    torque_max=(0.5, 0.5, 0.5),   # optional per-axis actuator clip (N·m)
)
```

`rotation_dim=1` configures the 2D (RT-plane) case. The policy emits a
scalar torque about the body-z / N axis; `AttitudeControl.apply` lifts it
to a `(n, 3)` vector `[0, 0, τz]` before writing `applied_torque`.
`rotation_dim=3` configures full 3D attitude control; the policy emits a
3-vector.

`torque_max` is an optional **actuator-side** clip applied before writing
`applied_torque`. It models the maximum torque the actuator can produce
(e.g. thruster authority), independently of the reaction-wheel saturation
cap (`omega_max`) in the dynamics. Either or both limits may be set; setting
only `omega_max` (no `torque_max`) is valid and common.

## Conical observation

`ConicalObservation` is a body-fixed-cone-gated full-state measurement
channel. It is configured with:

- `sensor_boresights_body` — `(k, 3)` array of unit vectors in the
  observer's body frame. Each row is one sensor's pointing direction.
- `half_angle_rad` — half-angle of each cone. Either a scalar (same for all
  sensors) or a `(k,)` array / tuple for per-sensor half-angles.
- `sigma_floor` — measurement-noise standard deviation at zero range.
- `sigma_range_frac` — growth of that std per metre of observer-target
  range. Leaving it at `0.0` reproduces a constant per-pair sigma equal to
  `sigma_floor`. A zero std produces a near-noiseless `R = 1e-12 I` instead
  of `R = 0` so Kalman updates remain numerically valid.

**Visibility logic:** For each (observer, target) pair, `ConicalObservation`
rotates every body-fixed boresight into the world frame using the observer's
quaternion, then checks whether the line-of-sight to the target falls within
any cone. A pair is visible iff it passes the cone test AND the target is on
the opposing side (own-side pairs are always masked out).

**Measurement model:** When visible, the measurement is the target's full
dynamics state plus additive Gaussian noise with per-pair standard
deviation `σ_ij = sigma_floor + sigma_range_frac * range_ij`. `H = I_d` and
`R[i, j] = σ_ij² I_d`, so `obs_noise` carries the per-pair shape
`(N_obs, N_total, m, m)`. The same `σ_ij` scales every measurement row, so
velocity rows are noised in proportion to range exactly as position rows
are. The `range_ij` driving `σ_ij` is the **true** range read from
ground-truth state — a benchmark convention that avoids modelling a range
estimator. The filter is handed only the resulting `R`, never the range.
This is the full-state measurement model used by the bundled belief
updater — no bearing-only or range-only reduction.

**Multi-sensor vmap:** Multiple sensors per agent are handled entirely inside
`ConicalObservation` via `jnp.any` over the sensor axis. A single
`Observation` channel is returned regardless of how many sensors are
configured, so the existing belief updater consumes it unchanged.

**2D/3D handling:** `ConicalObservation` works in both RT (2D) and RTN (3D)
scenarios. For 2D, position vectors are zero-padded to 3D internally before
the cone math. The `layout.dynamics_state_dim` attribute drives this
automatically; no separate 2D class is needed.

**Common multi-sensor configurations:**

```python
import jax.numpy as jnp

# Single forward-facing sensor
boresights_1 = jnp.array([[0.0, 1.0, 0.0]])   # +T body axis

# Two opposing sensors (fore/aft)
boresights_2 = jnp.array([
    [ 0.0,  1.0, 0.0],   # +T
    [ 0.0, -1.0, 0.0],   # -T
])

# Four-face configuration (standard in demos)
boresights_4 = jnp.array([
    [ 1.0,  0.0, 0.0],   # +R
    [-1.0,  0.0, 0.0],   # -R
    [ 0.0,  1.0, 0.0],   # +T
    [ 0.0, -1.0, 0.0],   # -T
])

from orbitalgym.observations.conical import ConicalObservation

obs_fn = ConicalObservation(
    layout=cfg.layout,
    sensor_boresights_body=boresights_4,
    half_angle_rad=0.5236,      # 30 degrees
    sigma_floor=10.0,           # 10 m / (m/s) noise std at zero range
    sigma_range_frac=0.01,      # + 1 cm of std per metre of range
)
```

## Configuration knobs

A complete worked example for a 3D RTN scenario where guards have full
attitude control and bandits do not:

```python
import jax.numpy as jnp
from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    DynamicsKey,
    StateComponentKey,
)
from orbitalgym.observations.conical import ConicalObservation

boresights = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

cfg = ScenarioConfig(
    n_guards=1,
    n_bandits=1,
    # ...other required fields...
    truth_dynamics=DynamicsKey.HCW_RTN,
    guard_components=(
        StateComponentKey.RTN,
        StateComponentKey.MASS,
        StateComponentKey.ATTITUDE,
        StateComponentKey.BODY_RATES,
    ),
    bandit_components=(
        StateComponentKey.RTN,
        StateComponentKey.MASS,
    ),
    guard_action_components=(
        ActionComponentKey.IMPULSIVE_MANEUVER,
        ActionComponentKey.ATTITUDE_CONTROL,
    ),
    bandit_action_components=(
        ActionComponentKey.IMPULSIVE_MANEUVER,
    ),
    attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
    guard_attitude_params=AttitudeParams(
        inertia_diag=jnp.array([10.0, 10.0, 5.0]),   # kg m²
        omega_max=jnp.array([0.5, 0.5, 1.0]),         # rad/s
    ),
    bandit_attitude_params=None,   # bandits have no attitude
    attitude_control_torque_max=(0.5, 0.5, 0.5),
    guard_observation_fn=ConicalObservation(
        layout=None,   # filled by __post_init__; pass cfg.layout after creation
        sensor_boresights_body=boresights,
        half_angle_rad=0.5236,
        sigma_floor=10.0,
        sigma_range_frac=0.01,
    ),
)
```

`attitude_control_torque_max` is threaded into `AttitudeControl` instances at
env construction time and clips commanded torque before writing `applied_torque`.
It is independent of `omega_max` in `AttitudeParams`.

## IC sampling for attitude

The new `AttitudeSampler` family plugs into `RelativeKeplerian` and
`RelativeEllipse` via their `attitude_sampler` field. When `attitude_sampler`
is `None` (the default), the initial quaternion and body rates come from
`Attitude.zeros()` and `BodyRates.zeros()` — identity quaternion, zero rates.

See [Extending → Customize IC sampling](../extending/customize-ic-sampling.md#attitude-samplers)
for the full family description and worked examples.

## Validation rules (fail-fast)

`ScenarioConfig.__post_init__` enforces these checks and raises `ValueError`
on any violation — no silent defaulting:

- `ATTITUDE` and `BODY_RATES` must appear together or both be absent on each
  side. Having one without the other is invalid.
- If any side has `ATTITUDE+BODY_RATES`, `attitude_dynamics_key` must be set.
- If `attitude_dynamics_key` is set, at least one side must have
  `ATTITUDE+BODY_RATES`.
- If `attitude_dynamics_key` is set and a side has `ATTITUDE+BODY_RATES`,
  that side's `attitude_params` must be non-`None`.
- If `ATTITUDE_CONTROL` is in a side's action components, that side must have
  `ATTITUDE+BODY_RATES`.

## Pointing policies

`PointingPolicy` wraps a delta-v policy and overwrites the `PointAt`
command's `target_dir` from the wrapped agent's belief, so pointing is
driven by what the agent thinks it knows rather than by ground truth. The
belief mean has shape `(N_obs, N_total, d)` — rows are this side's
observers, columns are own-side vehicles first and then the opposing
side, and observer `i` reads its own position from cell `(i, i)`.

`PointingTarget` selects the rule:

- `HOLD` — zero vector, which `PointAt` reads as "keep the current attitude".
- `LADY` — point at the origin.
- `TEAMMATE` — point at the nearest own-side vehicle.
- `OPPONENT_BELIEF` — point at the nearest opposing vehicle's belief mean.
- `OPPONENT_SCAN` — sweep until the opponent belief is tight enough to track.

### The scan rule

`OPPONENT_BELIEF` fails badly when the prior is multi-modal. A ring prior
over a bandit somewhere on a 3 km natural-motion ellipse has its *mean* at
the lady, so a guard that points at the mean holds its cone on empty space
and never gets the detection that would collapse the prior.

`OPPONENT_SCAN` breaks that deadlock. Per observer it measures the
positional spread of the nearest opponent's belief:

- a particle belief reports the RMS distance of that opponent's particle
  cloud from its mean position;
- a Gaussian belief reports the square root of the trace of the position
  block of its covariance;
- a belief carrying only a mean reports zero.

While the spread exceeds `scan_spread_m` the observer sweeps its boresight
around the radial/along-track plane, `(cos(az), sin(az), 0)` with
`az = scan_rate_rad_s * t + 2*pi*i/n_self`. The per-observer phase term
staggers teammates so an `n`-guard team covers `n` bearings at once. Once
the spread drops below `scan_spread_m` — which a single in-cone detection
is normally enough to do — the observer reverts to `OPPONENT_BELIEF` and
tracks.

Two knobs on `PointingPolicy` control it:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `scan_spread_m` | `200.0` | Spread above which the observer scans instead of tracking |
| `scan_rate_rad_s` | `0.0175` | Sweep rate, about 1 degree per second |

Set `scan_rate_rad_s` so a full sweep is slower than the cone crossing
time: with a half-angle `h` the target stays in the cone for about
`2h / scan_rate_rad_s` seconds, which must cover at least one step of
`dt`. Set `scan_spread_m` above the tracking accuracy the filter reaches
after a detection and below the width of the prior, so the rule latches.

```python
from orbitalgym.policies.pointing import PointingPolicy, PointingTarget

guard_policy = PointingPolicy(
    inner=guard_dv_policy,
    target=PointingTarget.OPPONENT_SCAN,
    scan_spread_m=200.0,
    scan_rate_rad_s=0.0175,
)
```

## Visualizing cones

Set `RolloutScene.show_sensor_cones=True` when animating a rollout to overlay
the sensor cone visualization on each vehicle. The `cone_length_m` parameter
controls the drawn cone length; when `None` it auto-extends to the edge of
the plot bounding box.

Two underlying glyphs are used:

- `draw_wedge_2d` — 2D RT scenarios. Draws a filled wedge in the RT plane
  using the body's current quaternion to orient the boresight.
- `draw_cone_3d` — 3D RTN scenarios. Draws a cone surface in 3D using
  `mpl_toolkits.mplot3d`.

## Visualizing communication-link cones

Set `RolloutScene.show_link_cones=True` and pass `guard_link` / `bandit_link`
(a `PointingConeLink` instance) to overlay each vehicle's communication cone
in a second colour (`link_cone_color`, default `"limegreen"`). The cone is
drawn solid (`link_cone_alpha_closed`) on frames where the link is closed and
translucent (`link_cone_alpha_open`) otherwise. The per-frame open/closed
state comes from `link_mask` (a `BySide` or `dict[Side, ...]` of `(T, n)`
bool arrays) if given, else from the trajectory's logged `contact` field
(populated by `belief_rollout` when a link predicate is passed). Link kinds
other than `PointingConeLink` carry no cone geometry and are skipped.

## Pointers

- API reference: [attitude dynamics and action components](../api/components.md#attitude-dynamics)
- Demo notebooks: `examples/workflow_2d_rt_attitude.ipynb`, `examples/workflow_3d_rtn_attitude.ipynb`
- Customize IC sampling for attitude: [extending/customize-ic-sampling.md](../extending/customize-ic-sampling.md#attitude-samplers)
- Dynamics layer overview: [in-depth/dynamics.md](dynamics.md)
- Action components overview: [in-depth/action-components.md](action-components.md)
