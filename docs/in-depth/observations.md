# Observations

Observations are how the env exposes state to a policy. Four channels
ship in the box; you can compose them or write your own.

## The protocol

```
observation_fn: (env_state, actions, side, params, key, t) → tuple[Observation, ...]
```

A function returns a **tuple** of `Observation` channels. Each
channel carries `(obs, visible, obs_matrix, obs_noise[, obs_fn])`:

- `obs` — per-pair measurement payload, shape `(N_self, N_total, m)`.
- `visible` — boolean mask, shape `(N_self, N_total)`. False entries
  are not measured this step.
- `obs_matrix` (`H`) — linear measurement matrix `(m, d)`.
- `obs_noise` (`R`) — measurement covariance, either `(m, m)` shared by
  every pair or `(N_self, N_total, m, m)` conditioned on the pair. Call
  `channel.noise_for(i, j)` to read one pair's `(m, m)` block whichever
  form the channel supplies.
- `obs_fn` — optional nonlinear measurement function (EKF only).

Each side has its own observation function; `OrbitalGymEnv` exposes
them as `env.guard_observation_fn` and `env.bandit_observation_fn`.
You can also call any channel directly as a plain callable — useful
for unit tests and for swapping in alternative sensor models.

## Per-pair shape conventions

Every bundled channel uses the same `(N_self, N_total, m)` layout:
the first axis is the **observer** (a vehicle on the requesting
side); the second axis runs over **all tracked entities**, with
own-side first and opposite-side second; the third axis is the
measurement payload of width `m`.

The `visible` mask is the contract for "which pairs were actually
measured this step." Downstream belief updaters consume it to decide
which corrections to apply — masked entries are skipped, not zeroed.

## The four bundled channels

### `FullObservation`

Reveals the full per-pair concatenation `[own_truth, opp_truth]` to
the requesting side, with all-visible mask and a near-zero noise
floor. The reference baseline.

```python
--8<-- "tests/docs/test_indepth_observations.py:full-observation"
```

The second axis is **own-side first, opposite-side second** — a
policy can always read its own state from the leading slice without
knowing which side it's wired into.

Use `FullObservation` when you want a fully-observable baseline (e.g.
for sanity checks of policy / planner code before adding partial
observability).

### `OnboardGPSObservation`

Each vehicle observes its own absolute state with isotropic Gaussian
noise. Models a real onboard GPS receiver: `visible[i, k]` is True
iff `k == i` (the observer's own slot); every other entry is masked.

### `RangeLimitedObservation`

Per-pair distance-gated 3-D position measurement. Each observer
measures the position of every opposing entity within
`sensor_range_m`; pairs beyond range, and own-side pairs, are masked.

```python
--8<-- "tests/docs/test_indepth_observations.py:range-limited"
```

`channel.obs` has shape `(N_self, N_total, 3)`; `channel.visible`
has shape `(N_self, N_total)`. The downstream belief updater uses
`visible` to decide which corrections to apply.

### `ConicalObservation`

Per-pair full-state measurement gated by body-fixed sensor cones. Designed
for scenarios where agents must point their sensors at a target to receive
belief updates.

```python
import jax.numpy as jnp
from orbitalgym.observations.conical import ConicalObservation

boresights = jnp.array([
    [ 1.0,  0.0, 0.0],   # +R body axis
    [-1.0,  0.0, 0.0],   # -R
    [ 0.0,  1.0, 0.0],   # +T
    [ 0.0, -1.0, 0.0],   # -T
])

obs_fn = ConicalObservation(
    layout=cfg.layout,
    sensor_boresights_body=boresights,
    half_angle_rad=0.5236,      # 30-degree half-angle
    sigma_floor=10.0,           # noise std at zero range (m or m/s)
    sigma_range_frac=0.01,      # + 1 cm of std per metre of range
)
```

**Body-fixed boresights:** `sensor_boresights_body` is a `(k, 3)` array of
unit vectors in the observer's body frame. One row per sensor. `ConicalObservation`
rotates them into the world frame using the observer's quaternion (from
`env_state.guards.quat` / `env_state.bandits.quat`) before testing visibility.

**Visibility logic:** A (observer, target) pair is visible iff the
line-of-sight vector to the target falls inside **any** of the observer's
cones (angle to boresight < `half_angle_rad`) AND the target is on the
opposing side. Own-side pairs are always masked out.

**Multi-sensor handling:** Visibility across all `k` sensors is OR-reduced
(`jnp.any` over the sensor axis) inside `ConicalObservation`. A single
`Observation` channel is returned regardless of sensor count — the belief
updater sees the same interface as any other channel.

**Measurement model:** When visible, `obs[i, j, :]` is the target's full
dynamics state plus additive Gaussian noise, independent per entry. The
noise std is conditioned on the observer-target range:

```
sigma_ij = sigma_floor + sigma_range_frac * range_ij
```

`H = I_d` and `R[i, j] = sigma_ij² I_d`, so this channel's `obs_noise` has
the per-pair shape `(N_self, N_total, m, m)`. The same `sigma_ij` scales
every measurement row, so velocity rows are noised in proportion to range
exactly as position rows are. A zero std produces `R = 1e-12 I`
(near-noiseless) rather than `R = 0` so that Kalman updates remain
numerically stable and correctly weight near-perfect measurements.

**2D / 3D handling:** Works in both RT and RTN scenarios. For RT (2D),
positions are zero-padded to 3D internally before the cone math. The
`layout.dynamics_state_dim` attribute drives the padding automatically.

**Per-sensor half-angles:** Pass a `(k,)` tuple to `half_angle_rad` for
sensors with different FOVs (e.g., a wide fore sensor paired with narrow
aft sensors).

### `CompositeObservation`

Concatenates multiple channels into one tuple. Use when an observer
has both onboard GPS *and* a range-limited radar — wrap them with
`CompositeObservation` and downstream consumers see both. Each
constituent receives a fresh PRNG subkey so noise samples are
independent across channels.

## Decision tree

| You want | Use |
|---|---|
| Full state visible to both sides | `FullObservation` |
| Each vehicle sees only its own state (with noise) | `OnboardGPSObservation` |
| Distance-gated measurements of opponents | `RangeLimitedObservation` |
| Body-frame cone-gated measurements (attitude-dependent) | `ConicalObservation` |
| Multiple of the above on the same observer | `CompositeObservation` |

## How adapters consume observations

Two helpers in `orbitalgym.observations.types` collapse the per-pair
tensors into the flat shapes that adapters publish:

- `flatten_observations(channels)` returns a single 1-D vector for the
  side. Used by `GymnasiumAdapter` (single-agent view of the controlled
  side) and `POMDPAdapter` (per-side observation).
- `flatten_observations_per_agent(channels)` returns shape
  `(N_self, total_per_agent_dim)` — one row per observer, ready to be
  indexed by agent. Used by `PettingZooAdapter`, where agent
  `<side>_i` receives row `i`. Per-vehicle channels (GPS, range-limited)
  give every agent a distinct view; broadcast channels like
  `FullObservation` give every agent on the side the same view because
  the channel writes the same row for every observer.

## Writing a custom channel

The practical "how to write one" walkthrough lives in the Extending
section. See [Extending → Customize observations](../extending/customize-observations.md)
for the protocol, two worked examples, and per-game applicability.

## Where to next

- [Belief](belief.md) — what consumes `Observation` tuples to maintain
  estimates over time.
- [API reference → Components](../api/components.md) — the
  `ObservationFn` protocol and bundled channels.
