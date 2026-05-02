# Dynamics

`OrbitalGameEnv` composes an **actuator** (action → applied control)
with a **dynamics** step (state + control → next state) once per
timestep, per side. Both are pluggable: pick a different combination
to model different physics.

## What ships

| Key | What | Per-vehicle state shape |
|---|---|---|
| `DynamicsKey.HCW_RT` | 2D planar Clohessy–Wiltshire | `[r, θ, ṙ, θ̇]` (4) |
| `DynamicsKey.HCW_RTN` | 3D Clohessy–Wiltshire | `[r, θ, n, ṙ, θ̇, ṅ]` (6) |
| `ActuatorKey.IMPULSIVE` | Discrete `Δv` impulse | n/a |

Both HCW dynamics ignore J2, drag, and finite-burn duration. They are
exact for instantaneous impulses applied to a circular reference orbit.

## The dynamics protocol

```
dynamics: (state, dv, params, dt) → next_state
```

- `state` — raw per-side dynamics array, shape `(N_side, 4)` for
  `HCW_RT` or `(N_side, 6)` for `HCW_RTN`. One row per vehicle.
- `dv` — applied impulse, shape `(N_side, action_dim)`. Already
  produced by the actuator.
- `params` — `VehicleParams` (mass, max-thrust, dry-mass, mean motion).
- `dt` — scalar timestep in seconds.

`OrbitalGameEnv` exposes the configured dynamics callables as
`env.truth_dynamics` and `env.planning_dynamics`. You can call them
directly:

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

## Truth vs planning dynamics

`ScenarioConfig` carries two dynamics keys:

- `truth_dynamics` — what the env actually integrates each step. The
  ground truth.
- `planning_dynamics` — what `POMDPAdapter`'s `transition` uses for
  the model-based solver. May differ from truth (e.g. simpler model
  for a faster planner; or pretend-J2 for robustness studies).

Bundled games default both to the same key.

## Writing a custom dynamics module

Frozen dataclass + `@register(DynamicsKey.MY_DYNAMICS)`:

```python
from dataclasses import dataclass
from orbital_game.registry import DynamicsKey, register

@register(DynamicsKey.J2_HCW_RTN)   # add the enum member first
@dataclass(frozen=True)
class J2HCWRTNStep:
    """HCW-RTN with a first-order J2 secular drift correction."""

    j2_coefficient: float = 1.082626e-3

    def __call__(self, state, dv, params, dt):
        # 1. Apply HCW propagation as in hcw_rtn_step.
        # 2. Add a J2 secular drift term to (theta_dot, n_dot).
        # 3. Return the corrected state.
        ...
```

Register the new key under `DynamicsKey` (in `orbital_game.registry`),
then point a scenario at it via `truth_dynamics=DynamicsKey.J2_HCW_RTN`.

## What's not yet pluggable

The actuator surface assumes impulsive dynamics (one `Δv` per
timestep). Finite-burn or thrust-vs-time actuators would need a
richer protocol — out of scope for the bundled framework, addressable
by extending the `Actuator` protocol.

## Where to next

- [State layout & adding a Power component](state-layout.md) — what
  the dynamics step reads and writes.
- [API reference → Pluggables](../api/pluggables.md) — `Dynamics` and
  `Actuator` protocols.
