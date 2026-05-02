# POMDPPlanners-shape adapter

`POMDPAdapter` exposes `OrbitalGameEnv` as a duck-typed POMDPPlanners-shape protocol — flat-state-vector `transition` / `observation` / `reward` / `discount` / `initialstate` methods over the env's `StateLayout.flatten` and `unflatten`.

The adapter does not import POMDPPlanners; it conforms to the *shape* of the protocol so consumers (POMCP, DESPOT, learned belief-state planners, POMDPs.jl-style frameworks) can plug into the env without any required external dependency.

## Public surface

```python
class POMDPAdapter:
    states_dim: int                            # = env.layout.flat_dim
    action_dim_per_side: int                   # 2 for HCW_RT, 3 for HCW_RTN

    def discount(self) -> float                # episode-bounded → 1.0
    def initialstate(self, key) -> jax.Array   # samples flat state vector
    def transition(self, s_flat, a_flat, key)  # → s_next_flat
    def observation(self, s, a, s_next, side)  # → per-side observation
    def reward(self, s, a, s_next, side)       # → per-side scalar reward
```

## Minimal example

```python
import jax
import jax.numpy as jnp

from orbital_game import OrbitalGameEnv, make_lady_bandit_guard, Side
from orbital_game.adapters.pomdp import POMDPAdapter

cfg = make_lady_bandit_guard()
adapter = POMDPAdapter(OrbitalGameEnv(cfg))

s0 = adapter.initialstate(jax.random.PRNGKey(0))    # (states_dim,)

n_g, n_b = cfg.n_guards, cfg.n_bandits
d = adapter.action_dim_per_side
a = jnp.zeros(n_g * d + n_b * d)                    # concatenated guard+bandit actions
s1 = adapter.transition(s0, a, jax.random.PRNGKey(1))

r_guard = adapter.reward(s0, a, s1, Side.GUARD)
r_bandit = adapter.reward(s0, a, s1, Side.BANDIT)
```

## Flat layouts

The action vector concatenates the guard's flat action then the bandit's flat action. The state vector is `env.layout.flatten(state.guards, state.bandits)`; it deliberately excludes the scalar time / step counter and the reference orbit, since POMDPPlanners-shape consumers typically work with `(s, a) → s'` transitions where time bookkeeping is implicit.

## Per-side dispatch

Unlike Gymnasium and PettingZoo, this adapter does not pre-commit to a controlled side. `observation` and `reward` take an explicit `side` argument, so the same `(s, a, s')` triple can yield both perspectives — useful for centralized critics in MARL and for evaluating both sides' beliefs simultaneously.

## Discount

`discount() → 1.0` by default. All four bundled games are episode-bounded by `MaxStepsOrBreach`, so an undiscounted return is well-defined; override per-game if your solver needs `γ < 1`.
