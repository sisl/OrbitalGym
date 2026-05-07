# State layout & adding a Power component

`EnvState` carries one *side-state* per side. Each side-state is a
flax pytree composed of **state components** — `RTN` for relative-orbital
state, `Mass` for propellant tracking, and so on. State components are
swappable: pick the set you want when building the side-state class.

This page documents how to read the existing fields, then walks
through adding a new `Power` component end-to-end.

## Reading the bundled components

```python
--8<-- "tests/docs/test_indepth_state_layout.py:basic-rtn-fields"
```

The columns of `state.guards.rtn` are
`[r, θ, n, ṙ, θ̇, ṅ]` in the rotating reference-orbit frame. For
HCW-RT (planar), the analogous attribute is `state.guards.rt` with 4
columns `[r, θ, ṙ, θ̇]`.

If the side has the `Mass` component, propellant mass is exposed
directly:

```python
--8<-- "tests/docs/test_indepth_state_layout.py:propellant-mass"
```

## Flat layout

`POMDPAdapter` and any other consumer that wants a 1-D state vector
uses `env.layout.flatten` and `env.layout.unflatten`:

```python
--8<-- "tests/docs/test_indepth_state_layout.py:flat-flatten"
```

The `StateLayout`-level flat vector excludes the scalar time / step
counter and the reference orbit. `POMDPAdapter` packs `(t, step)`
onto the tail of its own flat vector so `transition` can stay pure
(see [Use the POMDP adapter](../how-to/use-pomdp-adapter.md)); the
reference orbit is treated as a per-scenario constant.

## Worked example: adding a `Power` component

Suppose your scenario needs a per-vehicle battery state-of-charge
that decreases proportionally to commanded thrust magnitude. End-to-end:

### 1. The component dataclass

```python
import flax.struct
import jax

@flax.struct.dataclass
class Power:
    soc: jax.Array      # (N_side,) state of charge in [0, 1]
```

### 2. Register it with the side-state assembler

Side-state classes are built dynamically via `build_state_class` (see
`orbital_game.state.assemble`). Register `Power` so the assembler
knows how to allocate space for it in the flat layout.

### 3. Sampler hook

Initial SoC is sampled at episode reset. Add a sampler that draws
`soc ~ Uniform(0.7, 1.0)` per vehicle and writes it into the
`Power` component of the freshly-sampled side-state.

### 4. Dynamics-side update

Each step, drain SoC by `α · |dv|` for each vehicle, where `dv` is
the realised impulse magnitude that step. The dynamics step gets
`(state, dv, params, dt)`; you read `dv`, decrement `state.power.soc`,
and return the new state.

### 5. Plot hook (optional)

Add `viz.summaries.plot_power_curve(state.guards.power.soc)` analogous
to the existing `plot_mass_curve`. Same `(T, N)` → matplotlib axis
shape.

### What this is *not*

The Power component is illustrative — it is not part of the merged
library. The point is the pattern: a frozen dataclass + the
assembler hook + a sampler + a dynamics-side update + an optional
viz hook. Apply the same template to sensor health or any other
per-vehicle quantity you want side-states to carry.

## Attitude-related components

Two state components track rigid-body rotation:

- **`Attitude`** — quaternion `(n, 4)` in wxyz convention. Initialized to
  the identity quaternion `(1, 0, 0, 0)`. Represents body-to-reference
  rotation.
- **`BodyRates`** — body-frame angular rate `(n, 3)` in rad/s. Zero-initialized.

Add them to `guard_components` or `bandit_components` together (they must
appear as a pair; having one without the other raises in `__post_init__`):

```python
from orbital_game.registry import StateComponentKey

guard_components=(
    StateComponentKey.RTN,
    StateComponentKey.MASS,
    StateComponentKey.ATTITUDE,
    StateComponentKey.BODY_RATES,
)
```

## Transient control components

Two additional components carry "control input for this tick" from action
components to the dynamics block. Both are **auto-extended** onto the per-side
components by `ScenarioConfig.__post_init__` — users do not list them
explicitly:

- **`AppliedDV`** — translational Δv `(n, 3)` in m/s (truth frame). Added
  automatically whenever the side has any spatial component (`RT`, `RTN`, or
  `ECI`). Written by `ImpulsiveManeuver.apply`; read and zeroed by
  `env.step`'s translational dynamics block.
- **`AppliedTorque`** — body-frame torque `(n, 3)` in N·m. Added
  automatically whenever the side has both `ATTITUDE` and `BODY_RATES`.
  Written by `AttitudeControl.apply`; read and zeroed by `env.step`'s
  attitude dynamics block.

Both are zeroed each step after dynamics consumes them, so they always read
zero outside the dynamics block. They appear in the flat state layout (since
the flat layout covers the full assembled pytree) but are not meaningful as
persistent quantities — treat them as implementation detail of the
action-to-dynamics handoff.

`AppliedDV` is always width 3 for shape uniformity. The dynamics block slices
`[:, :2]` for `Frame.RT` and `[:, :3]` for `RTN`/`ECI`.

## Where to next

- [Dynamics](dynamics.md) — how the dynamics step consumes
  components and writes the next state.
- [API reference → Components → State](../api/components.md) —
  `StateLayout`, `build_state_class`.
