# Round-trip a config to JSON

`ScenarioConfig` and every registered component round-trip through
JSON.

## Recipe

```python
from orbitalgym import make_pursuit_evasion
from orbitalgym.config import ScenarioConfig

cfg = make_pursuit_evasion(capture_distance_m=20.0, seed=0)
text = cfg.to_json()             # str
cfg_loaded = ScenarioConfig.from_json(text)
assert cfg_loaded.game.capture_distance_m == 20.0
```

## What's serialized

Every field of `ScenarioConfig` plus every nested component. Each
component's enum key is the discriminator; `dataclasses.asdict`
walks the knob fields.

Derived fields like `ObservationBlocking.target_ecef_m` are recomputed
in `__post_init__` on load — they're `init=False` and not stored.

## When registration is required

Only if you need JSON round-trip. In-memory composition works without
`@register`.

## See also

- [API reference → Components](../api/components.md) — the registration
  convention.
