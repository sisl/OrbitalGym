# Customize observations

`cfg.guard_observation_fn` and `cfg.bandit_observation_fn` are per-side
`ObservationFn` callables. Each returns a tuple of `Observation`
channels — one per sensor modality. Belief updaters fold each channel
sequentially.

For the conceptual story (what's in a channel, scope rules, the four
bundled channels), read [In depth → Observations](../in-depth/observations.md).
This page is the practical "how to write one" track.

## The protocol

::: orbital_game.observations.base.ObservationFn

::: orbital_game.observations.types.Observation

## The four bundled channels as a vocabulary

| Channel | Module | Use when |
|---|---|---|
| `FullObservation` | `observations/reference.py` | Bootstrap; observer sees full truth (own + opp). |
| `OnboardGPSObservation` | `observations/onboard_gps.py` | Realistic single-side GPS-style noise on own state. |
| `RangeLimitedObservation` | `observations/range_limited.py` | Pair visibility gated on range; multi-vehicle realism. |
| `CompositeObservation` | `observations/composite.py` | Glue: combine several channels into one tuple. |

## Worked example: a position-only channel

A custom channel that drops velocity and exposes only position:

```python
--8<-- "tests/docs/test_extending_customize_observations.py:imports"
```

```python
--8<-- "tests/docs/test_extending_customize_observations.py:position-only-channel"
```

Wire it onto the cfg:

```python
--8<-- "tests/docs/test_extending_customize_observations.py:wire-it-up"
```

The `StateLayout` is not a top-level `ScenarioConfig` field; it is held
by the default `FullObservation` that `ScenarioConfig.__post_init__`
constructs. Pull it off `cfg.guard_observation_fn.layout` (as above)
and pass it to the custom channel so `dynamics_state_dim`, `n_guards`,
and `n_bandits` line up with the rest of the scenario.

## Composing channels

To compose, return a length-2+ tuple from your `ObservationFn`, or use
`CompositeObservation` from the reference module. The belief updater
folds the channels in tuple order.

## See also

- [How-to → Pick an observation channel](../how-to/pick-observation-channel.md) — when you want one of the four shipped channels rather than a new one.
- [In depth → Observations](../in-depth/observations.md) — the full conceptual deep-dive.
- [In depth → Belief](../in-depth/belief.md) — how channels feed the belief updater.
