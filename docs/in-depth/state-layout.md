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
viz hook. Apply the same template to attitude state, sensor health,
or anything else you want side-states to carry.

## Where to next

- [Dynamics](dynamics.md) — how the dynamics step consumes
  components and writes the next state.
- [API reference → Components → State](../api/components.md) —
  `StateLayout`, `build_state_class`.
