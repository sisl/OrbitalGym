# Customize observations

`cfg.guard_observation_fn` and `cfg.bandit_observation_fn` are per-side
`ObservationFn` callables. Each returns a tuple of `Observation`
channels — one per sensor modality. Belief updaters fold each channel
sequentially.

For the conceptual story (what's in a channel, scope rules, the four
bundled channels), read [In depth → Observations](../in-depth/observations.md).
This page is the practical "how to write one" track.

## The protocol

::: orbitalgym.observations.base.ObservationFn

::: orbitalgym.observations.types.Observation

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

`cfg.layout` is the canonical `StateLayout` for the scenario — a small
struct bundling the per-side state dataclass types, vehicle counts,
flat-vector dimension, and pytree↔flat conversion functions. Pass it
to your custom channel (as above) so `dynamics_state_dim`, `n_guards`,
and `n_bandits` line up with the rest of the scenario. See
[API → Components → State](../api/components.md#state) for the full
class reference.

## Composing channels

To compose, return a length-2+ tuple from your `ObservationFn`, or use
`CompositeObservation` from the reference module. The belief updater
folds the channels in tuple order.

## See also

- [How-to → Pick an observation channel](../how-to/pick-observation-channel.md) — when you want one of the four shipped channels rather than a new one.
- [In depth → Observations](../in-depth/observations.md) — the full conceptual deep-dive.
- [In depth → Belief](../in-depth/belief.md) — how channels feed the belief updater.
