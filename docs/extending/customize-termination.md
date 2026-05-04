# Customize termination

`cfg.termination_fn` is called every step with the new state and returns
a scalar `bool` array. Once it fires, the episode ends and per-side
`done` fields broadcast that scalar (see [In depth → Symmetric core](../in-depth/symmetric-core.md)).

## The protocol

::: orbital_game.termination.base.TerminationFn

## The bundled implementation

::: orbital_game.termination.reference.MaxStepsOrBreach

Pursuit-Evasion ships `PursuitEvasionTermination` that fires on
max_steps OR capture (relative distance below `cfg.game.capture_distance_m`).

## Worked example A: max-steps only (no breach)

```python
--8<-- "tests/docs/test_extending_customize_termination.py:imports"
```

```python
--8<-- "tests/docs/test_extending_customize_termination.py:max-steps-only"
```

## Worked example B: AND-composite of two existing conditions

```python
--8<-- "tests/docs/test_extending_customize_termination.py:and-composite"
```

`AndComposite` accepts any two callables of the right shape and ANDs
their results. Use it sparingly — most game-design reasons want OR
composition, which you can write the same way with `jnp.logical_or`.

Wiring is the usual `dataclasses.replace`:

```python
--8<-- "tests/docs/test_extending_customize_termination.py:wire-it-up"
```

## Per-game applicability

| Game | Default termination | What's reasonable to swap | Worth varying? |
|---|---|---|---|
| Lady-Bandit-Guard | Max steps OR breach | Max-steps-only for unbiased eval | Yes — eliminates breach-survivor bias |
| Pursuit-Evasion | Max steps OR capture | Add fuel-out condition | Sometimes — depends on study design |
| Sun-Blocking | Max steps OR breach | Time-window-only termination | Yes — for fixed-window benchmarks |
| Observation-Blocking | Max steps OR breach | Visibility-window-only | Yes — matches the game's gating logic |

## See also

- [API → Components → Termination](../api/components.md#termination)
- [In depth → Symmetric core](../in-depth/symmetric-core.md) — how
  termination interacts with the latched `episode_done` mask.
