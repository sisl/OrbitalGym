# Lady-Bandit-Guard

## Quickstart

```python
import jax
from orbital_game import OrbitalGameEnv, SingleAgentView, make_lady_bandit_guard
from orbital_game.policies import ZeroControl
from orbital_game.rollout import rollout_single_agent

cfg = make_lady_bandit_guard()
env = OrbitalGameEnv(cfg)
view = SingleAgentView(env)
guard = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
traj = rollout_single_agent(view, guard, lambda c, s, k: None,
                            jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

For a full walkthrough see [T1 — First rollout](../tutorials/t1-first-rollout.md).

The guard protects the reference orbit — the "Lady", currently virtual — from the bandit. This is a station-keeping / protection scenario: the guard's job is to stay near a protected asset while the bandit threatens to breach it.

## What the reward shapes

The guard's per-step reward is the negative sum of guard radial distances from the reference orbit origin (RTN frame). This pulls the guard toward the protected anchor; staying close earns less negative reward, drifting away costs more. The bandit's reward is fixed at zero in this game — Lady-Bandit-Guard is currently asymmetric, with the bandit acting as a scripted threat rather than an optimizing adversary. Future variants may promote the Lady to a third controllable agent or wire a zero-sum bandit reward.

## Termination

The episode ends on `max_steps` exhaustion or when any guard comes within `breach_distance_m` of the reference-orbit origin (the breach condition).

## Builder

```python
from orbital_game import make_lady_bandit_guard

cfg = make_lady_bandit_guard(
    breach_distance_m=10.0,
    max_horizon_s=2000.0,
    seed=0,
)
```

Defaults match the reference scenario: a 1 km radial-ellipse co-orbit with the guard at phase 0 and the bandit at phase π, 200 steps at 10 s each.

## Knobs at a glance

| Field | Type | Default | What it does |
|---|---|---|---|
| `n_guards` | `int` | `1` | Number of guard vehicles. |
| `n_bandits` | `int` | `1` | Number of bandit vehicles. |
| `breach_distance_m` | `float` | `10.0` | Episode terminates when any guard is within this radius of the reference origin. |
| `max_horizon_s` | `float` | `2000.0` | Total episode duration in seconds. |
| `dt` | `float` | `10.0` | Step size in seconds. |
| `seed` | `int` | `0` | PRNG seed for IC sampling. |

For the full field list see [API → `LadyBanditGuard`](../api/games.md).

## Suggested experiments

- **Sanity baseline.** Run with zero control on both sides — no breach occurs, the episode ends at `max_steps`. This is what `examples/reference_scenario.py` does.
- **Bandit attack.** Replace `cfg.bandit_policy` with a heuristic that maneuvers toward the reference origin; the zero-control guard should eventually lose.
- **Guard station-keeping.** Train a guard policy to minimize the negative reward (stay close to the reference orbit) under bandit perturbations.

## Variants

The four customization axes (reward, termination, IC, observation) all
swap by `dataclasses.replace`. Two LBG-flavored recipes:

### Variant 1: jittered chasing bandit

Wrap a `LeadInterceptPursuer` in `JitteredPolicy` to give the guard a
randomized chaser threat — the bandit closes on the reference origin but
with stochastic per-step deviation, preventing a guard from exploiting a
deterministic threat trajectory.

```python
--8<-- "tests/docs/test_games_lady_bandit_guard_variants.py:variant-jittered-bandit"
```

### Variant 2: tighter IC for evaluation

Shrink the IC sampler's `sigma_radial_ellipse_m` so evaluation rollouts
draw from a narrow band around the canonical phase-π geometry. Useful
when comparing learned guard policies head-to-head: lower IC variance
means lower episode-return variance, which means fewer rollouts needed
to separate two policies.

```python
--8<-- "tests/docs/test_games_lady_bandit_guard_variants.py:variant-tighter-ic"
```

For more axes, see [Extending → Customize rewards](../extending/customize-rewards.md),
[…termination](../extending/customize-termination.md),
[…IC sampling](../extending/customize-ic-sampling.md), and
[…observations](../extending/customize-observations.md).

## Built-in policies and adversaries

Sensible gallery picks for Lady-Bandit-Guard:

- **Heuristic policies (opponent):** [`LeadInterceptPursuer`](../extending/gallery.md#leadinterceptpursuer),
  [`JitteredPolicy`](../extending/gallery.md#jitteredpolicy) (wrap any of the above).
- **Controlled side:** any class from [Controlled-side cookbook](../extending/controlled-policy-cookbook.md)
  ([`HeuristicWithFallbackPolicy`](../extending/gallery.md#heuristicwithfallbackpolicy),
  [`CompositeActionPolicy`](../extending/gallery.md#compositeactionpolicy),
  [`MCTSPolicy`](../extending/gallery.md#mctspolicy)).

## Sanity-check notebook

[`examples/games/lady_bandit_guard.ipynb`](https://github.com/sisl/orbital-game/blob/main/examples/games/lady_bandit_guard.ipynb) is a full walkthrough that builds an LBG scenario, runs a rollout, and renders the rollout diagnostic plus the 2D guard-position reward surface.

## Where to next

- **Tutorial:** [T1 — First rollout](../tutorials/t1-first-rollout.md).
- **API:** [`LadyBanditGuard`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [State layout](../in-depth/state-layout.md).
