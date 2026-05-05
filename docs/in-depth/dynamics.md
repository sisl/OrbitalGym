# Dynamics

`OrbitalGameEnv` composes per-side **action components** (commands →
post-component side state, including any state propagation) with the
configured **dynamics**. The bundled `ImpulsiveManeuver` action
component applies an impulsive `Δv` and then invokes `truth_dynamics` for the
remainder of the step. Both action components and dynamics are
swappable.

## What ships

| Key | What | Per-vehicle state shape |
|---|---|---|
| `DynamicsKey.HCW_RT` | 2D planar Clohessy–Wiltshire | `[r, θ, ṙ, θ̇]` (4) |
| `DynamicsKey.HCW_RTN` | 3D Clohessy–Wiltshire | `[r, θ, n, ṙ, θ̇, ṅ]` (6) |
| `ActionComponentKey.IMPULSIVE_MANEUVER` | Discrete `Δv` impulse + propagation | n/a |

Both HCW dynamics ignore J2, drag, and finite-burn duration. They are
exact for instantaneous impulses applied to a circular reference orbit.

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

## Where to next

- [State layout & adding a Power component](state-layout.md) — what
  the dynamics step reads and writes.
- [API reference → Components](../api/components.md) — `Dynamics` and
  `Actuator` protocols.
