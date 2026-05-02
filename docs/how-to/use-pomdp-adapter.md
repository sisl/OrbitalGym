# Use the POMDP adapter

Expose `OrbitalGameEnv` as a duck-typed POMDPPlanners-shape protocol —
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

from orbital_game import OrbitalGameEnv, Side, make_lady_bandit_guard
from orbital_game.adapters.pomdp import POMDPAdapter

cfg = make_lady_bandit_guard()
adapter = POMDPAdapter(OrbitalGameEnv(cfg))

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

`env.layout.flatten(state.guards, state.bandits)`. Excludes the
scalar time / step counter and the reference orbit.

## Per-side dispatch

Unlike Gymnasium and PettingZoo, this adapter does not pre-commit
to a controlled side. `observation` and `reward` take an explicit
`side` argument, so the same `(s, a, s')` triple yields both
perspectives.

## Adapter is stateful

`POMDPAdapter._last_state` is mutated between calls to thread the
scalar time, step counter, and reference orbit. Calling `transition`
inside a JAX transform (`vmap`, `lax.scan`) leaks tracers; use
plain Python loops if you need to enumerate candidates. See [T4 —
Plan with short-horizon search](../tutorials/t4-short-horizon-search.md).

## Discount

`discount() == 1.0` by default. Override per-game if your solver
needs `γ < 1`. All four bundled games are episode-bounded.

## See also

- [T4 — Plan with short-horizon search](../tutorials/t4-short-horizon-search.md)
  walks through a planner built on this adapter end-to-end.
- [API reference → Adapters](../api/adapters.md).
