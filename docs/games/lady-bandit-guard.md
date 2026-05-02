# Lady-Bandit-Guard

The Guard protects the reference orbit (the "Lady", currently virtual at the reference) from the Bandit.

## What this game models

A station-keeping/protection scenario. The Guard's job is to keep some asset — the reference orbit origin — safe from the Bandit. The Lady is virtual: the reference orbit's position/velocity is the protected anchor, and we never instantiate her as a separate agent.

The bootstrap implementation reuses the existing `DistanceToReferenceOrbit` reward and `MaxStepsOrBreach` termination — the LBG game is currently a typed *label* that wires the breach-distance knob into the existing termination. Future work may promote the Lady to a third controllable agent (see spec §10).

## Reward formula

`scope = PER_SIDE`

```text
guard_reward(s) = -∑_i ‖guard_i.position‖   # encourages staying near the reference orbit
bandit_reward(s) = 0                         # bootstrap-only — bandit reward is zero
```

For an N-guard fleet, `‖guard_i.position‖` is the radial distance of guard `i` from the reference-orbit origin in the RTN frame (3-norm for `RTN`, 2-norm for `RT`).

The bandit reward is currently fixed at 0.0; future LBG variants can introduce a zero-sum reward (bandit minimizes guard's reward).

## Termination

Episode ends on either:

- `max_steps` exhaustion — set by `cfg.max_horizon_s / cfg.dt`
- Any guard within `breach_distance_m` of the reference orbit origin

```python
cfg.termination_fn  # MaxStepsOrBreach(max_steps=..., breach_distance_m=10.0)
```

## Knobs

The `LadyBanditGuard` Game subtype carries:

| Knob | Default | Description |
|---|---|---|
| `breach_distance_m` | `10.0` | Termination triggers when any guard is closer than this distance to the reference-orbit origin |

## Builder

```python
from orbital_game import make_lady_bandit_guard

cfg = make_lady_bandit_guard(
    n_guards=1,
    n_bandits=1,
    breach_distance_m=10.0,
    max_horizon_s=2000.0,
    dt=10.0,
    seed=0,
)
```

Defaults match the bootstrap reference scenario: 1km radial-ellipse co-orbit, guard at phase 0, bandit at phase π, 200 steps at 10 seconds each. Override any field via the `**scenario_kwargs` passthrough (e.g. `reference_orbit=...`, `ic_sampler=...`, `guard_components=...`).

## Single-agent vs multi-agent

```python
# Single-agent (Gymnasium-style): guard is controlled, bandit is scripted
from orbital_game import GymnasiumAdapter
env = GymnasiumAdapter(OrbitalGameEnv(make_lady_bandit_guard()))

# Multi-agent (PettingZoo Parallel API): both guard and bandit are agents
from orbital_game import PettingZooAdapter
env = PettingZooAdapter(OrbitalGameEnv(make_lady_bandit_guard()))
```

## Suggested experiments

- **Sanity baseline**: zero-control on both sides → no breach, episode ends at `max_steps`. (This is what `examples/reference_scenario.py` runs.)
- **Bandit attack**: replace `cfg.bandit_scripted_policy` with a heuristic that maneuvers toward the origin; observe whether the guard's zero-control baseline still avoids breach (it shouldn't, given enough time).
- **Guard station-keeping**: train a guard policy to minimize the negative reward (i.e., stay close to the reference orbit) under perturbations.

## Source

- Game subtype + builder: [`src/orbital_game/games/lady_bandit_guard.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/games/lady_bandit_guard.py)
- Reward: [`src/orbital_game/rewards/reference.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/rewards/reference.py) (`DistanceToReferenceOrbit`)
- Termination: [`src/orbital_game/termination/reference.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/termination/reference.py) (`MaxStepsOrBreach`)
- Tests: [`tests/test_game_lady_bandit_guard.py`](https://github.com/duncaneddy/orbital-game/blob/main/tests/test_game_lady_bandit_guard.py)
