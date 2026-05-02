# Game Guides

`orbital-game` ships four bootstrap games. Each lives in its own module under `orbital_game.games` with a typed `Game` subtype, reward/termination implementations, and a `make_<game>(...)` builder.

| Game | Page | One-line summary |
|---|---|---|
| **LBG** | [Lady-Bandit-Guard](lady-bandit-guard.md) | Guard protects the reference orbit from the bandit |
| **PE** | [Pursuit-Evasion](pursuit-evasion.md) | 1v1; bandit pursues, guard evades |
| **SB** | [Sun-Blocking](sun-blocking.md) | Bandit occludes the guard's view of the Sun |
| **OB** | [Observation-Blocking](observation-blocking.md) | Bandit blocks the guard's view of an Earth target, gated on visibility |

## Common usage

Every game has a `make_<name>(...)` builder that returns a fully-wired `ScenarioConfig`:

```python
from orbital_game import (
    make_lady_bandit_guard,
    make_pursuit_evasion,
    make_sun_blocking,
    make_observation_blocking,
)

cfg = make_lady_bandit_guard(breach_distance_m=15.0)
cfg = make_pursuit_evasion(capture_distance_m=20.0)
cfg = make_sun_blocking(angle_sigma_deg=3.0)
cfg = make_observation_blocking(target_lat_deg=37.4, target_lon_deg=-122.2)
```

A registry-keyed dispatch is also available:

```python
from orbital_game import make_game, GameKey

cfg = make_game(GameKey.OBSERVATION_BLOCKING, target_lat_deg=37.4, target_lon_deg=-122.2)
```

## Builder defaults

Builders default to a 1-guard / 1-bandit RTN scenario in a circular LEO orbit (~7000 km radius), 10-second timestep, and game-appropriate IC samplers. Override any field via keyword:

```python
cfg = make_pursuit_evasion(
    capture_distance_m=5.0,
    n_guards=1,
    n_bandits=1,
    max_horizon_s=4000.0,
    dt=5.0,
    seed=42,
    # Plus any ScenarioConfig field via scenario_kwargs:
    epoch_mjd_utc=60100.0,
)
```

## Game serialization

Every game serializes through `ScenarioConfig.to_json()` / `from_json()`:

```python
import orbital_game as og

cfg = og.make_sun_blocking(angle_sigma_deg=2.5)
s = cfg.to_json()
cfg2 = og.ScenarioConfig.from_json(s)
assert isinstance(cfg2.game, og.SunBlocking)
assert cfg2.game.angle_sigma_deg == 2.5
```

For games with `init=False` derived fields (e.g. `ObservationBlocking.target_ecef_m`), the field is recomputed in `__post_init__` on deserialization — only user-provided knobs round-trip through JSON.
