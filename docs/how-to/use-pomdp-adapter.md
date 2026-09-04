# Use the POMDP adapter

Expose `OrbitalGymEnv` as a duck-typed POMDPPlanners-shape protocol —
flat-state-vector `transition` / `observation` / `reward` / `step` /
`discount` / `initialstate` over `StateLayout.flatten` / `unflatten`.

The adapter does not import POMDPPlanners. It conforms to the
*shape* of the protocol so consumers (POMCP, DESPOT, learned
belief-state planners) plug in without an external dependency.

## Install

No extra needed.

## Recipe

```python
import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter

cfg = make_lady_bandit_guard()
adapter = POMDPAdapter(OrbitalGymEnv(cfg))

s0 = adapter.initialstate(jax.random.PRNGKey(0))    # (states_dim,)
n_g, n_b = cfg.n_guards, cfg.n_bandits
d = adapter.action_dim_per_side
a = jnp.zeros((n_g + n_b) * d)                       # concatenated guard+bandit actions
s1 = adapter.transition(s0, a, jax.random.PRNGKey(1))
r_guard = adapter.reward(s0, a, s1, Side.GUARD)
r_bandit = adapter.reward(s0, a, s1, Side.BANDIT)
```

## Macro steps: `action_repeat`

A search that plans at the env's tick rate covers very little physical
time per tree edge, so an intercept that takes hundreds of ticks never
appears inside the horizon. `action_repeat` widens the planner's step:

```python
adapter = POMDPAdapter(OrbitalGymEnv(cfg), action_repeat=8, discount=0.99)

adapter.action_repeat   # 8
adapter.macro_dt        # cfg.dt * 8 — seconds per planner action
adapter.discount()      # 0.99 ** 8 — per-macro-step discount
```

With `action_repeat = k`, `transition` applies the same `a_flat` for k
consecutive env steps. Every substep goes through the env, so each one
takes the env's own per-step delta-v cap: a macro action is k capped
burns in one direction, exactly what the env executes when a policy
repeats the command. Once a substep reports `episode_done`, the state
freezes for the rest of the macro step.

`reward` returns the k per-substep rewards discounted by the
per-env-step `discount` and summed, with zero reward after a terminal
substep. The intermediate states are not available from the
`(s, a, s', side)` signature, so `reward` re-simulates them — exactly,
since the dynamics are deterministic given the state and action and no
observation-noise key enters the reward. Planners should prefer `step`,
which returns both results from one rollout:

```python
s1, r_guard = adapter.step(s0, a, jax.random.PRNGKey(1), Side.GUARD)
```

To execute a macro plan in the env, wrap the policy in
[`ActionRepeatPolicy`](../api/policies-action-repeat.md) with the same
`repeat`, so the flown trajectory matches the searched one.

## Action vector layout

Guards' actions first, bandits' second; total length
`(n_g + n_b) * action_dim_per_side`.

## State vector layout

`env.layout.flatten(state.guards, state.bandits)` followed by two
scalar tail entries `(t, step)` cast to the flat vector's float
dtype. The reference orbit is captured at `__init__` time as a
per-scenario constant. Total length: `env.layout.flat_dim + 2`.

## Per-side dispatch

Unlike Gymnasium and PettingZoo, this adapter does not pre-commit
to a controlled side. `observation` and `reward` take an explicit
`side` argument, so the same `(s, a, s')` triple yields both
perspectives.

## Pure under JAX transforms

`transition`, `observation`, and `reward` are pure functions of
their flat-vector arguments. They compose with `jax.vmap`,
`jax.lax.scan`, and `jax.jit` without leaking tracers — vectorised
search loops (e.g. random shooting over K candidates × H horizon)
compile to a single JIT call. See [T4 — Plan with short-horizon
search](../tutorials/t4-short-horizon-search.md) for an end-to-end
`vmap(scan)` planner.

## Discount

`discount()` returns `discount ** action_repeat`, the per-macro-step
discount, and defaults to `1.0` since all four bundled games are
episode-bounded. Pass `discount=γ` to the constructor when the search
horizon is shorter than the episode: a terminal reward then propagates
back to the root decision on the same scale as the dense shaping.

## See also

- [T4 — Plan with short-horizon search](../tutorials/t4-short-horizon-search.md)
  walks through a planner built on this adapter end-to-end.
- [API reference → Adapters](../api/adapters.md).
