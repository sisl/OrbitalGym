# Customize IC sampling

`cfg.ic_sampler` is an `ICSpec` composing two per-side samplers and a
tuple of validators. The env runs validators in `jax.lax.while_loop` and
sets `EnvState.ic_valid=False` if `max_attempts` is exhausted.

## The protocol

::: orbitalgym.sampling.spec.ICSpec

::: orbitalgym.sampling.spec.SideSampler

::: orbitalgym.sampling.spec.Validator

## The bundled samplers

::: orbitalgym.sampling.side.RelativeKeplerian

::: orbitalgym.sampling.side.RelativeEllipse

## The bundled validators

::: orbitalgym.sampling.validators.MinSeparation

::: orbitalgym.sampling.validators.MaxRange

## Worked example: tighter Gaussian for evaluation

For evaluation, you usually want a tighter spread around a fixed
reference configuration so the metric isn't dominated by IC variance:

```python
--8<-- "tests/docs/test_extending_customize_ic_sampling.py:imports"
```

```python
--8<-- "tests/docs/test_extending_customize_ic_sampling.py:tighter-gaussian"
```

```python
--8<-- "tests/docs/test_extending_customize_ic_sampling.py:wire-it-up"
```

## Per-game applicability

| Game | Default IC | What's reasonable to swap |
|---|---|---|
| Lady-Bandit-Guard | 1km radial-ellipse co-orbit, guard@0, bandit@π | Tighter sigma for eval; Gaussian-around-RT for 2D bench |
| Pursuit-Evasion | Same shape, asymmetric phase | Wider initial separation for harder problems |
| Sun-Blocking | Sun-illuminated geometry | Twilight-band initial states for edge-case study |
| Observation-Blocking | Off-target initial geometry | Pre-aligned ICs for the simplest variant |

## Attitude samplers

Translational samplers (`RelativeKeplerian`, `RelativeEllipse`) accept an
optional `attitude_sampler` field that initializes the per-vehicle quaternion
and body rates at episode reset. When `attitude_sampler=None` (the default),
initial attitude comes from `Attitude.zeros()` — identity quaternion
`(1, 0, 0, 0)` and zero body rates.

Five samplers ship in `orbitalgym.sampling.attitude`:

- **`IdentityAttitude()`** — identity quaternion `(1, 0, 0, 0)`, zero rates.
  The default when no sampler is configured. Use this explicitly if you want
  to document intent.
- **`FixedAttitude(quat_wxyz=(w,x,y,z), omega_rad_s=(wx,wy,wz))`** — pinned
  quaternion and body rates broadcast to all N vehicles. Useful for
  deterministic demos and unit tests where every spacecraft starts with the
  same known orientation and spin rate.
- **`UniformAttitude()`** — uniformly random quaternion (Marsaglia method:
  sample from N(0, I_4), normalize, canonicalize to w >= 0 hemisphere); zero
  body rates. Use when initial pointing should be isotropically random.
- **`UniformBodyRates(omega_max_rad_s=(wx,wy,wz))`** — identity quaternion;
  uniform random body rates in `[-omega_max, +omega_max]` per axis. Use
  when you want a known initial orientation with a random spin.
- **`UniformAttitudeAndRates(omega_max_rad_s=(wx,wy,wz))`** — both quaternion
  (Marsaglia) and body rates (uniform per axis) randomized. Use for the most
  adversarial initial conditions.

Plug them into `RelativeKeplerian` or `RelativeEllipse` via the
`attitude_sampler` field:

```python
from orbitalgym.sampling.attitude import FixedAttitude, UniformAttitude
from orbitalgym.sampling.side import RelativeKeplerian

# Guards start with a known slow spin (0.1 deg/s about body-z)
guard_sampler = RelativeKeplerian(
    sigma_delta_mean_anomaly_rad=0.1,
    attitude_sampler=FixedAttitude(
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        omega_rad_s=(0.0, 0.0, 0.001745),   # 0.1 deg/s in rad/s
    ),
)

# Bandits start with a random orientation, zero spin
bandit_sampler = RelativeKeplerian(
    sigma_delta_mean_anomaly_rad=0.1,
    attitude_sampler=UniformAttitude(),
)
```

The attitude sampler is called only when `ATTITUDE` and `BODY_RATES` are in
the side's components list. When they are absent, the sampler is ignored.

### Worked example: asymmetric attitude ICs

Guards with a known initial slow spin, bandits tumbling randomly:

```python
import math
import jax.numpy as jnp
from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    DynamicsKey,
    StateComponentKey,
)
from orbitalgym.sampling.attitude import FixedAttitude, UniformAttitudeAndRates
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeKeplerian
from orbitalgym.sampling.spec import ICSpec
from orbitalgym.sampling.validators import MinSeparation

guard_sampler = RelativeKeplerian(
    sigma_delta_mean_anomaly_rad=0.1,
    mass_sampler=ConstantMass(propellant_mass_kg=50.0),
    attitude_sampler=FixedAttitude(
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        omega_rad_s=(0.0, 0.0, 0.001745),   # 0.1 deg/s about body-z
    ),
)

bandit_sampler = RelativeKeplerian(
    sigma_delta_mean_anomaly_rad=0.5,
    mass_sampler=ConstantMass(propellant_mass_kg=30.0),
    attitude_sampler=UniformAttitudeAndRates(
        omega_max_rad_s=(0.05, 0.05, 0.1),  # up to ~3–6 deg/s tumble
    ),
)

boresights = jnp.array([
    [ 1.0,  0.0, 0.0],
    [-1.0,  0.0, 0.0],
    [ 0.0,  1.0, 0.0],
    [ 0.0, -1.0, 0.0],
])

cfg = ScenarioConfig(
    n_guards=1,
    n_bandits=1,
    # ... (epoch, reference_orbit, dt, max_horizon_s, seed, guard_params, bandit_params)
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
        StateComponentKey.ATTITUDE,
        StateComponentKey.BODY_RATES,
    ),
    guard_action_components=(
        ActionComponentKey.IMPULSIVE_MANEUVER,
        ActionComponentKey.ATTITUDE_CONTROL,
    ),
    bandit_action_components=(
        ActionComponentKey.IMPULSIVE_MANEUVER,
        ActionComponentKey.ATTITUDE_CONTROL,
    ),
    attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
    guard_attitude_params=AttitudeParams(
        inertia_diag=jnp.array([10.0, 10.0, 5.0]),
        omega_max=jnp.array([0.5, 0.5, 1.0]),
    ),
    bandit_attitude_params=AttitudeParams(
        inertia_diag=jnp.array([8.0, 8.0, 4.0]),
        omega_max=jnp.array([0.5, 0.5, 0.5]),
    ),
    ic_sampler=ICSpec(
        guard_sampler=guard_sampler,
        bandit_sampler=bandit_sampler,
        validators=(MinSeparation(min_separation_m=50.0),),
    ),
)
```

The IC sampler is called at every `env.reset`. Each episode the guards start
with the fixed orientation + spin while the bandits start tumbling randomly —
creating an asymmetric observation challenge where the guard must point a
moving sensor at an unpredictably-oriented bandit.

## See also

- [How-to → Round-trip config to JSON](../how-to/json-roundtrip-config.md) — IC samplers serialize cleanly.
- [API → Components → IC sampling](../api/components.md#ic-sampling)
- [In-depth → Attitude dynamics & conical sensors](../in-depth/attitude.md) — the full attitude system description.
