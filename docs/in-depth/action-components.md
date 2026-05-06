# Action components

`OrbitalGameEnv` composes per-side **action components** the same way it
composes state components. Each action component declares a static
field schema (`fields()`, `zeros(n)`) plus an instance method
`apply(...)` that mutates the side state. The env folds the configured
tuple of components per side, in registration order, every step.

This page covers the protocol, the default `(IMPULSIVE_MANEUVER,)`
setup, the LBG-with-comms composition, and the worked example for
adding a second component to a side.

## What ships

| Key | What it carries | Per-agent fields |
|---|---|---|
| `ActionComponentKey.IMPULSIVE_MANEUVER` | Impulsive Δv + truth-dynamics propagation | `dv` (`action_frame.dim`,) — 2 for RT, 3 for RTN/ECI |
| `ActionComponentKey.COMMUNICATE` | Per-step broadcast flag + payload | `active` (), `payload` (6,) |

`ImpulsiveManeuver` is **frame-aware**: its `dv` field shape is read
from its `action_frame` setting (which the env wires from
`cfg.action_frame`, defaulting to the truth-dynamics frame). RT
scenarios get a 2-D Δv; RTN and ECI scenarios get 3-D. This means
`fields()` and `zeros()` are *instance* methods (not static) on
`ImpulsiveManeuver` and `Communicate` — they read `self`. The
`build_command_class` helper takes component **instances**, not
classes, so the assembled Command pytree carries the correct shape.

`ImpulsiveManeuver` invokes the configured truth dynamics inside its `apply` —
the env loop does not call dynamics on its own. `Communicate` is
state-identity; its only effect on rewards/observations flows through
the `Actions` pytree that downstream consumers receive.

## The component protocol

```
fields() -> Mapping[str, tuple[int, ...]]      # per-agent field shapes
zeros(n) -> Mapping[str, jax.Array]            # identity command (n agents)
apply(command, side_state, side_params, dt, ref_eci6, key) -> side_state
```

`fields` and `zeros` are **static** — they describe the schema and the
"do nothing" command. `apply` is the **instance** method — it consumes
one component's slice of the per-side `Command` pytree and returns the
post-component side state. Components are responsible for any state
mutation (including invoking dynamics); the env loop only folds.

`OrbitalGameEnv` exposes the per-side Command class as
`env.guard_command_cls` and `env.bandit_command_cls`. Both are
flax-struct dataclasses (JAX pytrees) with one attribute per field
contributed by any registered component.

## The `(IMPULSIVE_MANEUVER,)` default

Every bundled scenario starts both sides with a single
IMPULSIVE_MANEUVER component:

```python
--8<-- "tests/docs/test_indepth_action_components.py:default-impulsive-maneuver"
```

`ScenarioConfig.guard_action_components` and
`ScenarioConfig.bandit_action_components` both default to
`(ActionComponentKey.IMPULSIVE_MANEUVER,)`. Under that default the
per-side Command pytree carries exactly one field, `dv`, with shape
`(N_side, action_frame.dim)` — which is `(N_side, 3)` for RTN/ECI and
`(N_side, 2)` for the 2-D RT case.

## Composing components

Set `guard_action_components` (or `bandit_action_components`) to a
longer tuple to add components. The LBG-with-comms factory does this
to add a `Communicate` component to the guard side:

```python
--8<-- "tests/docs/test_indepth_action_components.py:compose-communicate"
```

The guard's Command class now carries `dv` *and* `active` *and*
`payload`. The bandit's Command class is unchanged — sides compose
independently.

!!! note "Registration order = apply order"
    Components in `guard_action_components` are applied in tuple order
    each step. With `(IMPULSIVE_MANEUVER, COMMUNICATE)` the guard's
    IMPULSIVE_MANEUVER runs first (propagating the dynamics), then
    COMMUNICATE runs on the post-maneuver state. Reorder the tuple to
    reorder the apply sequence.

## Worked example: LBG with communication

`make_lady_bandit_guard(with_communication=True, comm_cost=5.0)`
swaps in:

1. `guard_action_components = (IMPULSIVE_MANEUVER, COMMUNICATE)` — the
   guard's Command pytree gains the `active` / `payload` fields.
2. `LbgWithCommsReward(comm_cost=5.0)` — charges `comm_cost` per
   active broadcast on top of the distance-to-reference baseline.

The bandit side is unchanged. Step the env from the same starting
state with comms on vs off, and the reward delta isolates the comm
cost:

```python
--8<-- "tests/docs/test_indepth_action_components.py:reward-delta"
```

`silent_reward - comms_reward` is exactly `comm_cost`: the
distance-to-reference baseline cancels (same starting state, zero Δv
on both branches), the only difference is the `active` flag the
reward function reads off the guard's command.

## Building a Command class directly

You usually don't construct a Command class by hand — the env builds
one per side from the configured tuple at `OrbitalGameEnv.__init__`
time. But the same `build_command_class` helper is available if you
need it (for unit tests, custom adapters, or documentation that
constructs a policy in isolation):

```python
--8<-- "tests/docs/test_indepth_action_components.py:build-command-class"
```

`build_command_class` takes a tuple of component **instances**, the
per-side agent count, and a class name. The returned class is a
`flax.struct.dataclass` with a `zeros(n)` classmethod that produces the
identity command — exactly what `ZeroControl` returns.

!!! note "Instances, not classes"
    Earlier versions accepted component *classes* with `@staticmethod`
    `fields()` / `zeros()`. The frame-aware `ImpulsiveManeuver` reads
    `self.action_frame` to size its `dv` field, so callers now pass
    instances (e.g. `(ImpulsiveManeuver(truth_dynamics=...,
    action_frame=Frame.RT, truth_frame=Frame.RT, track_mass=True),)`)
    and the env's `_build_component_instances` helper does this for
    you when you configure via the registry key.

## Where it lives

- `src/orbital_game/actions/components.py` — the `ActionComponent`
  protocol plus `ImpulsiveManeuver` and `Communicate` as canonical examples.
- `src/orbital_game/actions/assemble.py` — `build_command_class`, the
  per-side Command-class builder. Caches by component-tuple identity
  so `jax.lax.while_loop` carry-pytree-structure equality holds across
  resets.
- `src/orbital_game/env/core.py` — wires `guard_action_components` /
  `bandit_action_components` from the config into per-side
  `_action_component_instances` tuples and `guard_command_cls` /
  `bandit_command_cls`.

## Writing a new action component

The pattern mirrors the [State layout walkthrough](state-layout.md):

1. **Component dataclass.** A frozen dataclass with a `name` ClassVar
   keying it into `ActionComponentKey`. Implement `fields()`,
   `zeros(n)`, and `apply(...)` per the protocol.
2. **Register it.** `@register(ActionComponentKey.YOUR_KEY)` so the
   env can resolve the key from a `ScenarioConfig`.
3. **Wire the side.** Add the key to `guard_action_components` or
   `bandit_action_components` on the scenario config (typically inside
   a `make_<game>` builder, behind a kwarg like `with_communication`).
4. **Reward / observation hook (optional).** If your component should
   cost something or be visible to the opponent, write a custom
   reward / observation that reads the new command field off the
   `Actions` pytree — exactly how `LbgWithCommsReward` reads
   `action.sides.guard.active`.

The env loop and the rollout machinery require no changes — they fold
whatever tuple you configure, in tuple order.

## Where to next

- [Dynamics](dynamics.md) — what `ImpulsiveManeuver.apply` invokes
  after the Δv lands. Action components and dynamics are independently
  swappable.
- [State layout & adding a Power component](state-layout.md) — the
  state-side analogue. State components carry data; action components
  carry commands and apply them.
- [API reference → Components](../api/components.md) — the
  `ActionComponent` protocol signature.
