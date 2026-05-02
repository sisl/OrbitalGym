# Sun-Blocking

The Bandit is rewarded for occluding the Guard's view of the Sun — forming a Sun → Guard → Bandit collinear line with the Bandit on the far-from-Sun side of the Guard.

## What this game models

Adversarial dazzling / sun-shadowing in space proximity operations. The Bandit's mission is to position itself so that a sensor on the Guard pointed toward the Sun is occluded. The geometric condition is two-fold: the three bodies must be approximately collinear, AND the Bandit must be farther from the Sun than the Guard (so it actually blocks light, not the other way around).

## Reward formula

`scope = PER_SIDE`, zero-sum:

```text
angle = ∠(sun→guard, sun→bandit)             # in degrees
in_front = |bandit - sun| > |guard - sun|

bandit_reward = exp(-angle² / (2σ²))   if in_front else 0
guard_reward  = -bandit_reward
```

The Gaussian `exp(-angle² / 2σ²)` peaks at perfect collinearity (angle = 0°). The `in_front` gate enforces the geometric ordering condition — bandit-between-sun-and-guard does NOT block the guard's view of the sun, so reward is 0.

## Implementation

The reward looks up Sun ECI position via `astrojax.sun_position(epoch)` and converts vehicle RTN positions to ECI via the shared `games/_frames.py` helper (composes `astrojax.state_eci_to_koe`, `astrojax.state_koe_to_eci`, and `astrojax.rotation_rtn_to_eci`).

## Knobs

| Knob | Default | Description |
|---|---|---|
| `angle_sigma_deg` | `5.0` | Gaussian width on the collinearity angle. Smaller σ → sharper reward peak (harder to score) |

## Termination

`MaxStepsOrBreach` with `breach_distance_m=0.0` — termination on `max_steps` only. There is no capture concept in SB.

## Builder

```python
from orbital_game import make_sun_blocking

cfg = make_sun_blocking(
    angle_sigma_deg=5.0,
    max_horizon_s=5400.0,    # ~90 min, full LEO orbit
    dt=10.0,
    seed=0,
)
```

The default horizon is one full LEO orbit so the Sun-Guard-Bandit geometry varies meaningfully throughout the episode.

## Reward sanity check

Because SB is zero-sum, `guard_reward + bandit_reward == 0` at every step. With zero-control on both sides at the default IC, the bandit reward stays in `[0, 1]` (test: `test_sb_reward_in_unit_range_for_bandit`).

## Suggested experiments

- **Bandit positioning**: train a bandit policy to maximize cumulative reward — i.e., position itself for maximum sun-occlusion duration over the orbit.
- **Guard counter-positioning**: train a guard policy to actively maneuver away from the bandit's occlusion line.
- **σ sweep**: vary `angle_sigma_deg` from 1° to 30°. Smaller σ gives a harder problem (must be very precise); larger σ gives a smoother reward landscape.

## Source

- Game subtype + reward + builder: [`src/orbital_game/games/sun_blocking.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/games/sun_blocking.py)
- RTN→ECI helper: [`src/orbital_game/games/_frames.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/games/_frames.py)
- Tests: [`tests/test_game_sun_blocking.py`](https://github.com/duncaneddy/orbital-game/blob/main/tests/test_game_sun_blocking.py)
