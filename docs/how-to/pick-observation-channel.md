# Pick an observation channel

Decision tree for choosing among the four bundled observation channels.

## Quick decision tree

| Your problem | Use |
|---|---|
| Sanity check / debug; want full state visible | `FullObservation` |
| Each vehicle reports its own GPS-noised position | `OnboardGPSObservation` |
| Distance-gated detection of opposing vehicles | `RangeLimitedObservation` |
| Multiple sensors on one observer | `CompositeObservation(constituents=...)` |

## Setting one

`ScenarioConfig` carries observation functions per side —
`guard_observation_fn` and `bandit_observation_fn`. Set whichever
side(s) you need:

```python
import dataclasses
from orbital_game import OrbitalGameEnv, make_pursuit_evasion
from orbital_game.observations.range_limited import RangeLimitedObservation

cfg = make_pursuit_evasion()
# RangeLimited needs the layout to know N_self / N_target / state dim:
env_for_layout = OrbitalGameEnv(cfg)
rl = RangeLimitedObservation(
    layout=env_for_layout.layout,
    sensor_range_m=2000.0,
    sigma_range=1.0,
)
cfg = dataclasses.replace(
    cfg,
    guard_observation_fn=rl,
    bandit_observation_fn=rl,
)
```

The next `OrbitalGameEnv(cfg)` rebuilds with the new observation
function.

## Composing channels

```python
from orbital_game.observations.composite import CompositeObservation
from orbital_game.observations.onboard_gps import OnboardGPSObservation

composite = CompositeObservation(constituents=(
    OnboardGPSObservation(layout=env_for_layout.layout, sigma_gps=2.0),
    rl,
))
cfg = dataclasses.replace(
    cfg,
    guard_observation_fn=composite,
    bandit_observation_fn=composite,
)
```

## See also

- [In depth → Observations](../in-depth/observations.md) — the full
  channel walk-through with shape diagrams.
- [Extending → Customize observations](../extending/customize-observations.md)
  for *writing* a new observation function rather than choosing among
  the shipped ones.
