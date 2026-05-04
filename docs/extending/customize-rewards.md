# Customize rewards

`cfg.reward_fn` is the function the env calls every step to score the new
state for both sides. Swap it via `dataclasses.replace` to reshape what
the controlled side optimizes for.

## The protocol

::: orbital_game.rewards.base.RewardFn

::: orbital_game.rewards.base.RewardScope

A `RewardFn` returns a `jax.Array` whose shape depends on `scope`:

- `PER_SIDE` — scalar (`()`).
- `PER_VEHICLE` — `(N_side,)`.

It receives both sides' actions in one call (so multi-side rewards like
zero-sum penalties are natural), the previous and next env states, the
`Side` it's scoring, the cfg via `params`, and the step counter `t`.

## The bundled implementation

`DistanceToReferenceOrbit` is the reference reward for Lady-Bandit-Guard:

::: orbital_game.rewards.reference.DistanceToReferenceOrbit

Pursuit-Evasion ships its own zero-sum reward (`PursuitEvasionReward`)
that reads `cfg.game.capture_distance_m`. Sun-Blocking ships
`SunBlockingReward`, and Observation-Blocking ships
`ObservationBlockingReward`.

## Worked example: a sparse "breach happened" reward

Replace dense distance shaping with a sparse penalty that fires only on
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
| Lady-Bandit-Guard | Negative guard distance to reference orbit | Sparse breach, distance + control-effort | Yes — sparse vs dense changes learning dynamics fundamentally |
| Pursuit-Evasion | Zero-sum on relative distance | Time-discounted distance, capture-bonus | Yes — capture-bonus speeds up learning |
| Sun-Blocking | Sun-line geometry score | Add control effort penalty | Less so — geometry term is the load-bearing signal |
| Observation-Blocking | Visibility-gated geometry | Multi-target reward composition | Yes — multi-target is research-relevant |

## See also

- [In depth → Symmetric core & data shapes](../in-depth/symmetric-core.md) — what's in `prev_state` / `next_state`.
- [API → Pluggables → Rewards](../api/pluggables.md#rewards) — auto-generated reference.
- [Game guides](../games/index.md) — each game's reward in context.
