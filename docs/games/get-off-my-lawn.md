# Get-Off-My-Lawn

## Quickstart

```python
import jax
from orbitalgym import OrbitalGymEnv, BySide, make_get_off_my_lawn
from orbitalgym.policies import ZeroControl
from orbitalgym.rollout import rollout

cfg = make_get_off_my_lawn()
env = OrbitalGymEnv(cfg)
guard = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
bandit = ZeroControl(n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls)
traj = rollout(env, BySide(guard=guard, bandit=bandit),
               BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None),
               jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

The bandit wants to **approach and loiter** near the lady — a virtual point at the reference-orbit origin — staying inside a desired bandit-to-lady standoff band `[r_min_keep_m, r_max_keep_m]`. The guard wants to **intercept** the bandit (within `catch_radius_m`) or **repel** it past `keep_out_radius_m`. The keep-out shell is tuned to be strictly larger than the bandit's loiter band so the two regions never overlap.

Inverting the LBG framing: where Lady-Bandit-Guard rewards the bandit for *reaching* the lady, Get-Off-My-Lawn rewards the bandit for *holding station* near her — closer to a real-world rendezvous-and-loiter task.

## HCW-natural standoff metric

Bounded relative orbits in the HCW (Hill–Clohessy–Wiltshire) frame are not constant-radius — they trace a 2:1 ellipse in the in-plane (radial, along-track) plane, with cross-track an independent 1:1 oscillator. So Get-Off-My-Lawn measures the bandit's "standoff" from the lady not as raw Euclidean distance, but as the radial-ellipse semi-major axis `R` of the *smallest* bounded relative orbit passing through the bandit's current position:

```
R_lady(x_r, x_t, x_n) = sqrt(x_r² + (x_t/2)² + x_n²)
```

This makes the loiter band an *elliptical annulus* aligned with the natural HCW geometry. A bandit station-keeping on a stable bounded relative orbit with semi-major axis `R ∈ [r_min_keep, r_max_keep]` earns the loiter bonus at *every* point on its orbit — not just at the radial apsides. The `keep_out_radius_m` shell is similarly defined in the `R_lady` metric.

The catch event stays Euclidean (it's a physical proximity event, not an orbit-shape event).

## What the reward shapes

`GetOffMyLawnReward` mirrors guard and bandit signals every step, with a small extra shaping term that rewards the guard for closing on the bandit (a chase prior).

- **Bandit**: `+r_loiter · 1[R_lady ∈ [r_min, r_max]] − α · d_band − R_catch · 1[catch]`
- **Guard** (mirror with chase shaping): `−r_loiter · 1[loiter] + α · d_band + R_catch · 1[catch] − α_chase · d_guard_bandit_min + R_pushout · 1[all bandits beyond keep_out]`

Where `d_band` is the bandit-to-band distance in the `R_lady` metric (zero inside, linear with the gap on either side), `R_lady` is the HCW-natural standoff above, and `d_guard_bandit_min` is the minimum guard-bandit *Euclidean* distance over all pairs.

Defaults: `α = 1e-3`, `α_chase = 1e-3`, `r_loiter = 1.0`, `R_catch = 1000.0`, `R_pushout = 500.0`. The chase shaping and pushout bonus are **not** zero-sum with the bandit; they're guard-only priors. To run a strictly zero-sum variant, override the reward with `GetOffMyLawnReward(alpha_chase=0.0, r_pushout=0.0)`.

## Termination

`GetOffMyLawnTermination` ends the episode as soon as any of:

1. `state.step >= cfg.max_steps` (horizon exhausted).
2. **Caught** — any guard within `catch_radius_m` of any bandit (guard win).
3. **Pushed out** — every bandit beyond `keep_out_radius_m` of the lady (guard win).

Reward and termination read the same `catch_radius_m` and `keep_out_radius_m` off `cfg.game`, so they always agree on the radii.

## Builder

```python
from orbitalgym import make_get_off_my_lawn

cfg = make_get_off_my_lawn(
    r_min_keep_m=100.0,
    r_max_keep_m=500.0,
    catch_radius_m=50.0,
    keep_out_radius_m=1500.0,
    max_horizon_s=3000.0,
    seed=0,
)
```

`GetOffMyLawn` validates `r_min_keep_m < r_max_keep_m < keep_out_radius_m` in `__post_init__` so misconfiguration fails fast at construction. The guard IC sits on a 300 m radial-ellipse co-orbit and the bandit on a 600 m ellipse on the opposite side (phase = π), placing the bandit clearly outside the loiter band but well inside the keep-out shell — so the game has nontrivial dynamics under both passive and active control.

To override the reward or termination, pass them directly to `ScenarioConfig`:

```python
from orbitalgym.config import ScenarioConfig
from orbitalgym.games import GetOffMyLawn, GetOffMyLawnReward

cfg = ScenarioConfig(
    ...,
    game=GetOffMyLawn(r_min_keep_m=100.0, r_max_keep_m=500.0, keep_out_radius_m=1500.0),
    reward_fn=GetOffMyLawnReward(alpha_chase=0.0, r_pushout=0.0),  # strict zero-sum
)
```

## Knobs at a glance

| Field | Type | Default | What it does |
|---|---|---|---|
| `n_guards` | `int` | `1` | Number of guard vehicles. |
| `n_bandits` | `int` | `1` | Number of bandit vehicles. |
| `r_min_keep_m` | `float` | `100.0` | Inner semi-major axis of the loiter band (HCW metric). |
| `r_max_keep_m` | `float` | `500.0` | Outer semi-major axis of the loiter band (HCW metric). |
| `catch_radius_m` | `float` | `50.0` | Guard wins when any guard-to-bandit *Euclidean* distance falls below this. |
| `keep_out_radius_m` | `float` | `1500.0` | Guard wins when every bandit's *HCW standoff* `R_lady` exceeds this. Must be `> r_max_keep_m`. |
| `r_loiter` | `float` | `1.0` | Per-step bonus when at least one bandit is in band. |
| `alpha` | `float` | `1e-3` | Dense band-distance shaping coefficient (per meter). |
| `r_catch` | `float` | `1000.0` | Terminal penalty on bandit / bonus on guard at catch. Raise to make catch dominate. |
| `r_pushout` | `float` | `500.0` | Bonus on guard when every bandit is beyond `keep_out_radius_m`. Set to `0.0` for strict zero-sum. |
| `alpha_chase` | `float` | `1e-3` | Guard-side chase shaping per meter of guard-bandit distance. Set to `0.0` for strict zero-sum. |
| `max_horizon_s` | `float` | `3000.0` | Total episode duration in seconds. |
| `dt` | `float` | `10.0` | Step size in seconds. |
| `seed` | `int` | `0` | PRNG seed for IC sampling. |

For the full field list see [API → `GetOffMyLawn`](../api/games.md).

## Suggested experiments

- **Sanity baseline.** Zero-control on both sides — neither side wins, the episode runs to `max_steps`, and the bandit accrues a small dense penalty for sitting outside the loiter band.
- **Bandit loiter.** Wire a bandit policy that targets a station-keeping point inside the keep band (`r_min_keep_m + r_max_keep_m) / 2` from the lady) and observe the loiter bonus accumulate when the guard is too far to intercept.
- **Guard repel.** Train a guard policy to maximize `R_pushout`-weighted return — the guard should learn to nudge the bandit beyond the keep-out shell rather than always chasing for a hard catch.
- **Asymmetric capability.** Vary `bandit_params.max_thrust_n` against `guard_params.max_thrust_n` to study how the thrust ratio sets the equilibrium between loiter and intercept.

## Variants

The four customization axes (reward, termination, IC, observation) all swap by passing a kwarg to `ScenarioConfig` (or via `dataclasses.replace`).

### Variant 1: strict zero-sum (no chase / pushout shaping)

Drop the guard-only chase shaping and pushout bonus so the guard and bandit rewards mirror exactly each step. Useful for self-play training where any non-zero-sum prior introduces solver bias.

```python
from orbitalgym import make_get_off_my_lawn

cfg = make_get_off_my_lawn(alpha_chase=0.0, r_pushout=0.0)
```

### Variant 1b: catch-dominated reward

Raise `r_catch` so a single catch event wipes out the cumulative loiter bonus from a long episode. Useful when training a guard whose *only* incentive should be intercepting (or a bandit whose *only* concern is being caught).

```python
from orbitalgym import make_get_off_my_lawn

cfg = make_get_off_my_lawn(r_catch=10_000.0, r_loiter=1.0)
```

### Variant 2: tight keep band

Shrink the keep band to a narrow annulus — the bandit must hit a precise standoff to score, making the loiter bonus harder to earn and the chase shaping more dominant.

```python
cfg = make_get_off_my_lawn(
    r_min_keep_m=180.0,
    r_max_keep_m=220.0,
    catch_radius_m=40.0,
    keep_out_radius_m=1500.0,
)
```

For more axes, see [Extending → Customize rewards](../extending/customize-rewards.md), […termination](../extending/customize-termination.md), […IC sampling](../extending/customize-ic-sampling.md), and […observations](../extending/customize-observations.md).

## Built-in policies and adversaries

Sensible gallery picks for Get-Off-My-Lawn:

- **Heuristic policies (opponent):** [`LeadInterceptPursuer`](../extending/gallery.md#leadinterceptpursuer) (guard chase), [`JitteredPolicy`](../extending/gallery.md#jitteredpolicy) (wrap any of the above).
- **Controlled side:** any class from [Controlled-side cookbook](../extending/controlled-policy-cookbook.md) ([`HeuristicWithFallbackPolicy`](../extending/gallery.md#heuristicwithfallbackpolicy), [`CompositeActionPolicy`](../extending/gallery.md#compositeactionpolicy), [`MCTSPolicy`](../extending/gallery.md#mctspolicy)).

## Sanity-check notebook

[`examples/games/get_off_my_lawn.ipynb`](https://github.com/sisl/OrbitalGym/blob/main/examples/games/get_off_my_lawn.ipynb) builds the scenario, runs a rollout, renders the trajectory, and plots the reward diagnostic with the keep band and keep-out shell drawn for reference.

## Where to next

- **API:** [`GetOffMyLawn`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [State layout](../in-depth/state-layout.md).
