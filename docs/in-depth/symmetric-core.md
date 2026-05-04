# Symmetric core & data shapes

The `OrbitalGameEnv` core is **symmetric**: the same step function and
the same data shapes work whether you're learning the guard, learning
the bandit, training both simultaneously, or planning over beliefs.
This page is the spine of the documentation — every other in-depth
page assumes you've internalised the types and indexing patterns
described here.

## Sides

```python
from orbital_game import Side
Side.GUARD       # "guard"
Side.BANDIT      # "bandit"
Side.GUARD.opposite()   # Side.BANDIT
```

`Side` is a Python-level `StrEnum`, **never** a traced JAX array.
That distinction matters: side identity is a static structural
property of the computation graph, not data flowing through it. You
can `if side is Side.GUARD: ...` inside a `jit`-traced function and
JAX will specialise the trace per-side.

## `BySide[T]` — the universal one-per-side container

Anywhere a value is naturally per-side — actions, observations,
rewards, beliefs, scripted policies — the type is `BySide[T]`.

```python
--8<-- "tests/docs/test_indepth_symmetric_core.py:byside"
```

`BySide` is registered as a JAX pytree, so `vmap`, `lax.scan`, and
`tree.map` traverse it transparently:

```python
--8<-- "tests/docs/test_indepth_symmetric_core.py:byside-pytree"
```

## The big four step types

`env.step` takes `Actions(sides=BySide(guard, bandit))` and returns
`StepOutput(state, outputs, episode_done, info)`, where:

- `state` — the next `EnvState`.
- `outputs.guard`, `outputs.bandit` — `SideOutput(obs, reward, done)`.
- `episode_done` — scalar bool. Per-side `done` fields are this
  scalar broadcast for shape uniformity across sides with different
  vehicle counts.

## Axis ordering: `BATCH → TIME → VEHICLE → FEATURE`

Outer-to-inner. Read every shape spec on this principle:

| Axis | When present |
|---|---|
| `B` (batch) | Only under `jax.vmap` (e.g. seed batch). |
| `T` (time) | Only on `Trajectory` leaves (added by `lax.scan` in `rollout`). |
| `N_side` (vehicle) | Always present per side, even if `N=1`. |
| feature | Innermost. `action_dim=3` for HCW-RTN, `6` for RTN state. |

```python
--8<-- "tests/docs/test_indepth_symmetric_core.py:rollout-shapes"
```

## Indexing recipes

```python
--8<-- "tests/docs/test_indepth_symmetric_core.py:indexing-recipes"
```

The recipe pattern: pick the axes you want as colons, slice or index
the rest.

## Episode termination and the mask

`Trajectory.episode_done` is `(T,)` bool, **latched** after
termination — once True, all subsequent entries are True. To restrict
analysis to valid steps:

```python
--8<-- "tests/docs/test_indepth_symmetric_core.py:episode-mask"
```

## Under vmap

Wrap a rollout in `jax.vmap` and every leaf gains a leading batch
dimension. The full `B → T → N → feature` order applies:

```python
--8<-- "tests/docs/test_indepth_symmetric_core.py:vmap-shape"
```

This is the workhorse pattern for seed-parallel evaluation; see
[How-to → vmap rollouts](../how-to/vmap-rollouts.md) for a more
detailed recipe and [T5 — GPU / MPS](../tutorials/t5-acceleration.md)
for the acceleration story.

## Single-agent vs multi-agent framing

The same env supports both. In multi-agent framing, the caller passes
both sides' actions and reads both sides' outputs (this is what the
PettingZoo adapter and the symmetric `rollout` use).

In single-agent framing, `SingleAgentView` reads
`cfg.controlled_side` (default `Side.GUARD`) and runs the opposite
side's `cfg.<side>_policy` internally. The view's `step`
takes only the controlled side's action and returns only its reward
and observation. The Gymnasium adapter wraps this view.

## Why per-side policies live on the config

`ScenarioConfig.guard_policy` and `bandit_policy` are populated by the
make_<game> builders to `ZeroControl` by default. `SingleAgentView`
reads the *opposite* side's policy at construction. Multi-agent adapters
ignore both fields. This means a swap from "guard learning vs heuristic
bandit" to "bandit learning vs heuristic guard" is a single
`dataclasses.replace(cfg, controlled_side=Side.BANDIT)` call — see
[How-to → Switch the controlled side](../how-to/switch-controlled-side.md).

## Where to next

- [State layout & adding a Power component](state-layout.md) — what
  lives inside `traj.env_state.guards.<...>` and how to add new
  fields.
- [Observations](observations.md) — what flows into the `obs` axis.
- [API reference → Types](../api/types.md) — the dataclass field lists
  for `BySide`, `Actions`, `StepOutput`, `Trajectory`.
