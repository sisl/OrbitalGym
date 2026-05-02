# Game guides

Four games ship with `orbital-game`. Each has a typed `Game` subtype attached to `cfg.game`, a matching reward and termination, and a `make_<game>(...)` builder that wires everything up.

| Game | Page | Summary |
|---|---|---|
| Lady-Bandit-Guard | [→](lady-bandit-guard.md) | Guard protects the reference orbit from the bandit |
| Pursuit-Evasion | [→](pursuit-evasion.md) | 1v1 — bandit pursues, guard evades |
| Sun-Blocking | [→](sun-blocking.md) | Bandit occludes the guard's view of the Sun |
| Observation-Blocking | [→](observation-blocking.md) | Bandit blocks the guard's view of an Earth target, gated on visibility |

Every builder defaults to a 1-guard / 1-bandit RTN scenario in a circular LEO orbit (~7000 km radius), with a 10-second timestep and a game-appropriate IC sampler. Override any field by keyword:

```python
cfg = make_pursuit_evasion(
    capture_distance_m=5.0,
    max_horizon_s=4000.0,
    seed=42,
)
```

A registry-keyed dispatch is also available for code that needs to choose a game by enum:

```python
from orbital_game import make_game, GameKey
cfg = make_game(GameKey.OBSERVATION_BLOCKING, target_lat_deg=37.4, target_lon_deg=-122.2)
```

`ScenarioConfig.to_json()` / `from_json()` round-trip every game, including derived fields like `ObservationBlocking.target_ecef_m` (recomputed in `__post_init__` on load).
