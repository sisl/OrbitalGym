# Use the POMDP adapter

Expose `OrbitalGymEnv` as a duck-typed POMDPPlanners-shape protocol —
flat-state-vector `transition` / `observation` / `reward` / `discount` /
`initialstate` over `StateLayout.flatten` / `unflatten`.

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

`discount() == 1.0` by default. Override per-game if your solver
needs `γ < 1`. All four bundled games are episode-bounded.

## See also

- [T4 — Plan with short-horizon search](../tutorials/t4-short-horizon-search.md)
  walks through a planner built on this adapter end-to-end.
- [API reference → Adapters](../api/adapters.md).
