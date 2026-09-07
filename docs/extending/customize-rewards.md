# Customize rewards

`cfg.reward_fn` is the function the env calls every step to score the new
state for both sides. Two paths to swap it:

1. **Pass it as a constructor kwarg** — `ScenarioConfig(..., reward_fn=MyReward())`. This is the canonical override path; `__post_init__` only consults `cfg.game.default_reward_fn()` when `reward_fn` is `None`.
2. **`dataclasses.replace`** — `cfg = replace(cfg, reward_fn=MyReward())`. Useful when you already have a cfg in hand and want to swap one field without re-listing the rest.

## Where defaults come from

Every `Game` subclass declares its default reward (and termination) via `default_reward_fn` / `default_termination_fn`. `ScenarioConfig.__post_init__` calls those when `cfg.reward_fn` is `None`. The mapping is:

| Game | Default reward |
|---|---|
| `NoGame` | `ZeroReward` (no-op; you'll want to override) |
| `LadyBanditGuard` | `LbgZeroSumReward` (parameterized by `breach_radius_m`, `catch_radius_m`) |
| `PursuitEvasion` | `PursuitEvasionReward` (reads `cfg.game.capture_distance_m`) |
| `SunBlocking` | `SunBlockingReward` |
| `ObservationBlocking` | `ObservationBlockingReward` |

## The protocol

::: orbitalgym.rewards.base.RewardFn

::: orbitalgym.rewards.base.RewardScope

A `RewardFn` returns a `jax.Array` whose shape depends on `scope`:

- `PER_SIDE` — scalar (`()`).
- `PER_VEHICLE` — `(N_side,)`.

It receives both sides' actions in one call (so multi-side rewards like
zero-sum penalties are natural), the previous and next env states, the
`Side` it's scoring, the cfg via `params`, and the step counter `t`.

## Bundled implementations

`ZeroReward` is the no-op reward (`NoGame`'s default). It always returns 0; useful as an explicit "I will supply my own reward" placeholder:

::: orbitalgym.rewards.reference.ZeroReward

`DistanceToReferenceOrbit` is a reference single-agent reward — negative
sum of guard distances to the reference origin, zero for the bandit. Useful
when you want a guard-only signal without LBG event semantics:

::: orbitalgym.rewards.reference.DistanceToReferenceOrbit

`LbgZeroSumReward` is the LBG default — each side gets potential-based
shaping over the distance it is trying to close, plus mirrored terminal
events on catch / breach, and the guard side pays a separation charge. Reads
`catch_radius_m` and `breach_radius_m`, and the dwell each event requires,
from its own fields (the
`LadyBanditGuard.default_reward_fn` builder threads those from the game
knobs):

::: orbitalgym.rewards.lbg_zero_sum.LbgZeroSumReward

The potentials and the separation charge are module-level functions, so a
custom reward or a value estimate can reuse the same geometry:

::: orbitalgym.rewards.lbg_zero_sum.lbg_distances

::: orbitalgym.rewards.lbg_zero_sum.lbg_potential_from_distances

::: orbitalgym.rewards.lbg_zero_sum.lbg_potential

::: orbitalgym.rewards.lbg_zero_sum.guard_separation_cost

Pursuit-Evasion ships its own zero-sum reward (`PursuitEvasionReward`)
that reads `cfg.game.capture_distance_m`. Sun-Blocking ships
`SunBlockingReward`, and Observation-Blocking ships
`ObservationBlockingReward`.

## Worked example: a sparse "breach happened" reward

Replace the shaped default with a sparse penalty that fires only on
the breach step:

```python
--8<-- "tests/docs/test_extending_customize_rewards.py:imports"
```

```python
--8<-- "tests/docs/test_extending_customize_rewards.py:sparse-breach-reward"
```

Wire it onto the cfg:

```python
--8<-- "tests/docs/test_extending_customize_rewards.py:wire-it-up"
```

## Per-game applicability

| Game | Default reward | What's reasonable to swap | Worth varying? |
|---|---|---|---|
| Lady-Bandit-Guard | `LbgZeroSumReward` (per-side potential shaping + zero-sum terminal events) | Sparse breach-only reward, distance + control-effort, single-agent `DistanceToReferenceOrbit` | Yes — sparse vs dense changes learning dynamics fundamentally |
| Pursuit-Evasion | Zero-sum on relative distance | Time-discounted distance, capture-bonus | Yes — capture-bonus speeds up learning |
| Sun-Blocking | Sun-line geometry score | Add control effort penalty | Less so — geometry term is the load-bearing signal |
| Observation-Blocking | Visibility-gated geometry | Multi-target reward composition | Yes — multi-target is research-relevant |

## See also

- [In depth → Symmetric core & data shapes](../in-depth/symmetric-core.md) — what's in `prev_state` / `next_state`.
- [API → Components → Rewards](../api/components.md#rewards) — auto-generated reference.
- [Game guides](../games/index.md) — each game's reward in context.
