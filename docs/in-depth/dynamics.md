# Dynamics

`OrbitalGameEnv` composes per-side **action components** (commands →
post-component side state, including any state propagation) with the
configured **dynamics**. The bundled `ImpulsiveManeuver` action
component applies an impulsive `Δv` and then invokes `truth_dynamics` for the
remainder of the step. Both action components and dynamics are
swappable.

## What ships

| Key | What | Per-vehicle state shape | Form |
|---|---|---|---|
| `DynamicsKey.HCW_RT` | 2D planar Clohessy–Wiltshire | `[r, θ, ṙ, θ̇]` (4) | step function |
| `DynamicsKey.HCW_RTN` | 3D Clohessy–Wiltshire | `[r, θ, n, ṙ, θ̇, ṅ]` (6) | step function |
| `DynamicsKey.KEPLERIAN_ECI` | Point-mass two-body in ECI | `[rx, ry, rz, vx, vy, vz]` (6) | typed-instance class |
| `DynamicsKey.J2_ECI` | Two-body + J2 oblateness in ECI | `[rx, ry, rz, vx, vy, vz]` (6) | step function |
| `DynamicsKey.ASTROJAX_ORBIT` | Configurable astrojax force model | `[rx, ry, rz, vx, vy, vz]` (6) | typed-instance class |
| `ActionComponentKey.IMPULSIVE_MANEUVER` | Frame-aware Δv impulse + propagation | n/a (Δv is `(N_side, 2)` for RT, `(N_side, 3)` for RTN/ECI) | action component |

Both HCW dynamics ignore J2, drag, and finite-burn duration. They are
exact for instantaneous impulses applied to a circular reference orbit.

### Step functions vs typed-instance dynamics

Two dynamics shapes ship side by side:

- **Step functions** (`HCW_RT`, `HCW_RTN`, `J2_ECI`) are stateless
  callables registered against their key. The env resolves the key to
  the function and calls it directly each step. No per-instance state.
- **Typed-instance classes** (`KeplerianEciDynamics`,
  `AstrojaxOrbitDynamics`) have per-instance fields (e.g.
  `epoch_mjd_utc`, `force_model`) that drive how the dynamics is
  configured. Their Epoch and integrator RHS are built once in
  `__post_init__` (Python time, before any JIT trace) and stored on the
  instance — this avoids module-level state and lets each scenario
  carry its own reference epoch. When the env is given a bare
  `DynamicsKey.KEPLERIAN_ECI`, it instantiates the class with defaults
  (J2000 epoch); to bind your scenario's epoch, pass the typed instance
  directly: `truth_dynamics=KeplerianEciDynamics(epoch_mjd_utc=58849.0)`.
  `ScenarioConfig.__post_init__` already does this for you for
  `reference_orbit_dynamics`, threading `cfg.epoch_mjd_utc` through.

## The dynamics protocol

```
dynamics: (state, dv, params, dt) → next_state
```

- `state` — raw per-side dynamics array, shape `(N_side, 4)` for
  `HCW_RT` or `(N_side, 6)` for `HCW_RTN`. One row per vehicle.
- `dv` — applied impulse, shape `(N_side, action_dim)`. Sourced
  from the per-side `ImpulsiveManeuver` action component's command.
- `params` — `VehicleParams` (mass, max-thrust, dry-mass, mean motion).
- `dt` — scalar timestep in seconds.

`OrbitalGameEnv` exposes the configured dynamics callables as
`env.truth_dynamics`, `env.policy_dynamics`, and `env.belief_dynamics`.
You can call them directly:

```python
--8<-- "tests/docs/test_indepth_dynamics.py:hcw-rtn-step"
```

The dynamics callable operates on the raw `rtn` array (or `rt` for the
2D variant), not the assembled side-state pytree. The env's `step`
method handles the wrapping/unwrapping; calling the dynamics directly
gives you the same physics with no observation, reward, or
termination machinery in the way.

## Composition with the actuator

`OrbitalGameEnv.step` runs the actuator first, then the dynamics:

```python
--8<-- "tests/docs/test_indepth_dynamics.py:actuator-composition"
```

The actuator owns the action shape semantics (e.g. "action is a
desired `Δv`, clipped to `max_dv_mps`"). The dynamics owns the
physics (e.g. "given a `Δv`, propagate the HCW state by `dt`"). Swap
either independently.

## Truth, policy, and belief dynamics

`ScenarioConfig` carries three dynamics roles:

- `truth_dynamics` — what the env actually integrates each step. The
  ground truth.
- `policy_dynamics` — what `POMDPAdapter`'s `transition` uses for
  the model-based solver. May differ from truth (e.g. simpler model
  for a faster planner; or pretend-J2 for robustness studies).
- `belief_dynamics` — what the belief updater uses to propagate the
  belief between observations. Defaults to `policy_dynamics` when
  unset (read the resolved value as `cfg.belief_dynamics_resolved` /
  `env.belief_dynamics`).

Bundled games default all three to the same key.

## Writing a custom dynamics module

See [Extending → Customize dynamics](../extending/customize-dynamics.md)
for the full extension surface — registered vs typed-instance
patterns, `frame`/`kind` metadata, and the headline mixed-frame recipe
(ECI truth + HCW belief/policy via `AstrojaxOrbitDynamics`).

## What's not yet swappable

The actuator surface assumes impulsive dynamics (one `Δv` per
timestep). Finite-burn or thrust-vs-time actuators would need a
richer protocol — out of scope for the bundled framework, addressable
by extending the `Actuator` protocol.

## Precision and dtype

The package configures astrojax for `float64` at import time, which
also sets JAX's `jax_enable_x64=True`. This is required for sub-mm
KOE↔ECI round-trip residuals on Earth-orbit scenarios. Some hardware
(notably Apple Metal / MPS) is float32-only; flip with the package's
helper before constructing any env:

```python
import jax.numpy as jnp
import orbital_game

orbital_game.set_precision(jnp.float32)   # for MPS / GPU throughput
# ... build ScenarioConfig and OrbitalGameEnv after this ...
```

`set_precision` flips both astrojax's internal dtype *and*
`jax_enable_x64` together — calling `astrojax.config.set_dtype(jnp.float32)`
alone does not toggle `jax_enable_x64` back to `False` and leaves the
package in an inconsistent state. Existing typed-instance dynamics
captured the old dtype in their `__post_init__`; build new envs after
the precision flip to pick up the new dtype.

## Where to next

- [State layout & adding a Power component](state-layout.md) — what
  the dynamics step reads and writes.
- [API reference → Components](../api/components.md) — `Dynamics` and
  `Actuator` protocols.
