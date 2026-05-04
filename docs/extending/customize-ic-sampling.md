# Customize IC sampling

`cfg.ic_sampler` is an `ICSpec` composing two per-side samplers and a
tuple of validators. The env runs validators in `jax.lax.while_loop` and
sets `EnvState.ic_valid=False` if `max_attempts` is exhausted.

## The protocol

::: orbital_game.sampling.spec.ICSpec

::: orbital_game.sampling.spec.SideSampler

::: orbital_game.sampling.spec.Validator

## The bundled samplers

::: orbital_game.sampling.side.RelativeKeplerian

::: orbital_game.sampling.side.RelativeEllipse

## The bundled validators

::: orbital_game.sampling.validators.MinSeparation

::: orbital_game.sampling.validators.MaxRange

## Worked example: tighter Gaussian for evaluation

For evaluation, you usually want a tighter spread around a fixed
reference configuration so the metric isn't dominated by IC variance:

```python
--8<-- "tests/docs/test_extending_customize_ic_sampling.py:imports"
```

```python
--8<-- "tests/docs/test_extending_customize_ic_sampling.py:tighter-gaussian"
```

```python
--8<-- "tests/docs/test_extending_customize_ic_sampling.py:wire-it-up"
```

## Per-game applicability

| Game | Default IC | What's reasonable to swap |
|---|---|---|
| Lady-Bandit-Guard | 1km radial-ellipse co-orbit, guard@0, bandit@π | Tighter sigma for eval; Gaussian-around-RT for 2D bench |
| Pursuit-Evasion | Same shape, asymmetric phase | Wider initial separation for harder problems |
| Sun-Blocking | Sun-illuminated geometry | Twilight-band initial states for edge-case study |
| Observation-Blocking | Off-target initial geometry | Pre-aligned ICs for the simplest variant |

## See also

- [How-to → Round-trip config to JSON](../how-to/json-roundtrip-config.md) — IC samplers serialize cleanly.
- [API → Components → IC sampling](../api/components.md#ic-sampling)
