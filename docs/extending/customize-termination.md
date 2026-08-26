# Customize termination

`cfg.termination_fn` is called every step with the new state and returns
a scalar `bool` array. Once it fires, the episode ends and per-side
`done` fields broadcast that scalar (see [In depth → Symmetric core](../in-depth/symmetric-core.md)).

Two paths to swap the termination:

1. **Pass it as a constructor kwarg** — `ScenarioConfig(..., termination_fn=MyTermination())`. `__post_init__` only consults `cfg.game.default_termination_fn()` when `termination_fn` is `None`.
2. **`dataclasses.replace`** — `cfg = replace(cfg, termination_fn=MyTermination())`.

## Where defaults come from

Every `Game` subclass declares its default termination via `default_termination_fn`. `ScenarioConfig.__post_init__` calls it when `cfg.termination_fn` is `None`. The mapping is:

| Game | Default termination |
|---|---|
| `NoGame` | `MaxStepsOnly` (step-cap only, no spatial event) |
| `LadyBanditGuard` | `LbgEventTermination` (max-steps OR bandit breach OR guard catch) |
| `PursuitEvasion` | `PursuitEvasionTermination` (max-steps OR capture) |
| `SunBlocking` | `MaxStepsOnly` |
| `ObservationBlocking` | `MaxStepsOnly` |

## The protocol

::: orbitalgym.termination.base.TerminationFn

## Bundled implementations

`MaxStepsOnly` is the universal step-cap termination — it ends the episode when `state.step >= params.max_steps`. It reads `max_steps` from the cfg at call time, so it can be constructed without a cfg in scope. `NoGame`, `SunBlocking`, and `ObservationBlocking` use it as their default:

::: orbitalgym.termination.reference.MaxStepsOnly

`LbgEventTermination` adds the bandit-breach and guard-catch events on top of the step cap; it's `LadyBanditGuard`'s default and reads `breach_radius_m` / `catch_radius_m` from its own fields (which `LadyBanditGuard.default_termination_fn` threads from the game knobs):

::: orbitalgym.termination.lbg_events.LbgEventTermination

`PursuitEvasion` ships `PursuitEvasionTermination` that fires on
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
| Lady-Bandit-Guard | `LbgEventTermination` (max-steps + breach + catch) | `MaxStepsOnly` for unbiased eval; AND-composites for stricter end conditions | Yes — eliminates terminal-event survivor bias |
| Pursuit-Evasion | `PursuitEvasionTermination` (max-steps + capture) | Add fuel-out condition | Sometimes — depends on study design |
| Sun-Blocking | `MaxStepsOnly` | Time-window plus geometry-gated termination | Yes — for fixed-window benchmarks |
| Observation-Blocking | `MaxStepsOnly` | Visibility-window-only | Yes — matches the game's gating logic |

## See also

- [API → Components → Termination](../api/components.md#termination)
- [In depth → Symmetric core](../in-depth/symmetric-core.md) — how
  termination interacts with the latched `episode_done` mask.
