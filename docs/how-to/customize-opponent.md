# Customize the opponent

In single-agent framing, the *opposite* side runs a scripted policy.
Replace it with anything that satisfies the `Policy` protocol.

## Recipe

```python
import dataclasses
from orbital_game import make_pursuit_evasion

# Any class with the Policy __call__ shape works.
class MyPursuer:
    n_vehicles: int = 0
    action_dim: int = 0
    def __call__(self, ps, obs, key, t):
        ...

cfg = make_pursuit_evasion()
cfg = dataclasses.replace(cfg, bandit_policy=MyPursuer())
```

`ScenarioConfig` is frozen, hence `dataclasses.replace`. The Gymnasium
adapter (which wraps `SingleAgentView`) will pick up the change.

## Why `n_vehicles` and `action_dim` default to 0

The env populates these via `dataclasses.replace` at construction.
Hard-coding them defeats portability across scenario sizes.

## See also

- [T3 — Active-pursuer policy](../tutorials/t3-active-pursuer.md) for
  a worked example.
- [API reference → Components](../api/components.md).
