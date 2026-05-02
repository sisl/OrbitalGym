# Pursuit-Evasion

A 1v1 zero-sum game. The Bandit pursues the Guard; the Guard evades. The reference orbit is the dynamical anchor only — it is not a protected asset.

## Reward formula

`scope = PER_SIDE`, **zero-sum** on relative distance:

```text
bandit_reward(s) = -‖r_guard - r_bandit‖   # bandit wants to close
guard_reward(s)  = +‖r_guard - r_bandit‖   # guard wants to escape
```

Both vehicles share the RTN frame anchored on the reference orbit, so the relative distance is just the difference of their RTN positions.

## Termination

Episode ends on either:

- `max_steps` exhaustion
- **Capture**: relative distance falls below `capture_distance_m`

The `PursuitEvasionTermination` class checks `cfg.game` is a `PursuitEvasion` instance and reads `capture_distance_m` from there.

## Knobs

| Knob | Default | Description |
|---|---|---|
| `capture_distance_m` | `10.0` | Termination triggers when `‖r_guard - r_bandit‖ < capture_distance_m` |

## Builder

```python
from orbital_game import make_pursuit_evasion

cfg = make_pursuit_evasion(
    capture_distance_m=10.0,
    n_guards=1,
    n_bandits=1,
    max_horizon_s=2000.0,
    dt=10.0,
    seed=0,
)
```

Defaults place the guard on a 1 km radial ellipse at phase 0 and the bandit on a smaller (500 m) ellipse with cross-track + along-track offset at phase π/2 — set up so neither side is trivially captured nor trivially safe.

## Single-agent vs multi-agent

```python
# Train a guard evader against a scripted bandit pursuer:
cfg = make_pursuit_evasion(controlled_side=Side.GUARD, bandit_scripted_policy=MyPursuitPolicy(...))
env = GymnasiumAdapter(OrbitalGameEnv(cfg))

# Or train both sides simultaneously via PettingZoo Parallel API:
env = PettingZooAdapter(OrbitalGameEnv(make_pursuit_evasion()))
```

## Reward sanity check

Because `PE` is zero-sum, `guard_reward + bandit_reward == 0` at every step. The test suite includes this assertion (`tests/test_game_pursuit_evasion.py::test_pe_reward_is_zero_sum`).

## Suggested experiments

- **Optimal evasion**: train a guard policy under a fixed (e.g. proportional-navigation) bandit pursuer.
- **Self-play**: train both sides simultaneously via PettingZoo and observe equilibrium-ish capture rates.
- **Asymmetric capabilities**: vary `guard_params.max_thrust_n` vs `bandit_params.max_thrust_n` to study how thrust ratio affects capture probability.

## Source

- Game subtype + reward + termination + builder: [`src/orbital_game/games/pursuit_evasion.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/games/pursuit_evasion.py)
- Tests: [`tests/test_game_pursuit_evasion.py`](https://github.com/duncaneddy/orbital-game/blob/main/tests/test_game_pursuit_evasion.py)
