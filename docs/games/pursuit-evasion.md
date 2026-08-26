# Pursuit-Evasion

## Quickstart

```python
import jax
from orbitalgym import OrbitalGymEnv, BySide, make_pursuit_evasion
from orbitalgym.policies import ZeroControl
from orbitalgym.rollout import rollout

cfg = make_pursuit_evasion()
env = OrbitalGymEnv(cfg)
guard = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
bandit = ZeroControl(n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls)
traj = rollout(env, BySide(guard=guard, bandit=bandit),
               BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None),
               jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

For the active-pursuer walkthrough see [T3](../tutorials/t3-active-pursuer.md).

A 1v1 zero-sum game. The bandit pursues, the guard evades. The reference orbit is a dynamical anchor only — it has no protected-asset semantics here.

## What the reward shapes

Reward is the relative distance between guard and bandit, with opposite sign on each side: the bandit is rewarded for closing distance, the guard for opening it. Because both vehicles share the RTN frame anchored on the reference orbit, the relative distance is just the difference of their RTN positions.

Zero-sum is a hard constraint: `guard_reward + bandit_reward == 0` at every step, and the test suite asserts this.

## Termination

Either `max_steps` exhaustion, or **capture** when the relative distance drops below `capture_distance_m`.

## Builder

```python
from orbitalgym import make_pursuit_evasion

cfg = make_pursuit_evasion(
    capture_distance_m=10.0,
    max_horizon_s=2000.0,
    seed=0,
)
```

Defaults place the guard on a 1 km radial ellipse at phase 0 and the bandit on a smaller (500 m) ellipse with cross-track and along-track offset at phase π/2 — neither side is trivially captured nor trivially safe at IC.

## Knobs at a glance

| Field | Type | Default | What it does |
|---|---|---|---|
| `n_guards` | `int` | `1` | Evader count. |
| `n_bandits` | `int` | `1` | Pursuer count. |
| `capture_distance_m` | `float` | `10.0` | Episode terminates on relative distance below this. |
| `max_horizon_s` | `float` | `2000.0` | Total duration. |
| `dt` | `float` | `10.0` | Step size. |
| `seed` | `int` | `0` | PRNG seed. |

## Suggested experiments

- **Optimal evasion.** Train a guard policy under a fixed bandit pursuer (e.g. proportional navigation).
- **Self-play.** Train both sides simultaneously via the [PettingZoo adapter](../adapters/pettingzoo.md) and observe equilibrium-ish capture rates.
- **Asymmetric capabilities.** Vary `guard_params.max_thrust_n` against `bandit_params.max_thrust_n` to study how the thrust ratio affects capture probability.

## Variants

The four customization axes (reward, termination, IC, observation) all
swap by `dataclasses.replace`. Two PE-flavored recipes:

### Variant 1: jittered pursuer

Wrap a `LeadInterceptPursuer` in `JitteredPolicy` to randomize the
chaser. A stochastic bandit prevents the guard from exploiting a
deterministic pursuit law and gives a more robust evader benchmark.

```python
--8<-- "tests/docs/test_games_pursuit_evasion_variants.py:variant-jittered-pursuer"
```

### Variant 2: tighter IC for evaluation

Shrink `sigma_radial_ellipse_m` on both sides so evaluation rollouts
draw from a narrow band around the canonical phase-0/phase-π/2
configuration. Lower IC variance → lower return variance → fewer
episodes to separate two policies under the same compute budget.

```python
--8<-- "tests/docs/test_games_pursuit_evasion_variants.py:variant-tighter-ic"
```

For more axes, see [Extending → Customize rewards](../extending/customize-rewards.md),
[…termination](../extending/customize-termination.md),
[…IC sampling](../extending/customize-ic-sampling.md), and
[…observations](../extending/customize-observations.md).

## Built-in policies and adversaries

Sensible gallery picks for Pursuit-Evasion:

- **Heuristic policies (opponent):** [`LeadInterceptPursuer`](../extending/gallery.md#leadinterceptpursuer),
  [`OrthogonalEvader`](../extending/gallery.md#orthogonalevader),
  [`JitteredPolicy`](../extending/gallery.md#jitteredpolicy) (wrap any of the above).
- **Controlled side:** any class from [Controlled-side cookbook](../extending/controlled-policy-cookbook.md)
  ([`HeuristicWithFallbackPolicy`](../extending/gallery.md#heuristicwithfallbackpolicy),
  [`CompositeActionPolicy`](../extending/gallery.md#compositeactionpolicy),
  [`MCTSPolicy`](../extending/gallery.md#mctspolicy)).

## Sanity-check notebook

[`examples/games/pursuit_evasion.ipynb`](https://github.com/sisl/OrbitalGym/blob/main/examples/games/pursuit_evasion.ipynb) is a full walkthrough that builds a PE scenario, runs a rollout, and renders the rollout diagnostic plus the 2D guard-position reward surface.

## Where to next

- **Tutorials:** [T2 — RTN scenario](../tutorials/t2-rtn-scenario.md), [T3 — Active pursuer](../tutorials/t3-active-pursuer.md), [T4 — Short-horizon search](../tutorials/t4-short-horizon-search.md).
- **API:** [`PursuitEvasion`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [Observations](../in-depth/observations.md), [Belief](../in-depth/belief.md).
