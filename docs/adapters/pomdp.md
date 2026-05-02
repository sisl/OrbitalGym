# POMDPPlanners-shape Adapter

`POMDPAdapter` exposes `OrbitalGameEnv` as a duck-typed POMDPPlanners-shape protocol — flat-state-vector `transition` / `observation` / `reward` / `discount` / `initialstate` methods over the env's `StateLayout.flatten`/`unflatten`.

## No external dependency

The adapter does not import POMDPPlanners — it conforms to the *shape* of the protocol so consumers (POMCP, DESPOT, learned belief-state planners) can plug into the env. The shape is compatible with POMDPPlanners (a Python POMDPs.jl-style framework) and most generic POMDP solvers.

## Public API

```python
class POMDPAdapter:
    states_dim: int                            # = env.layout.flat_dim
    action_dim_per_side: int                   # 2 for HCW_RT, 3 for HCW_RTN

    def discount(self) -> float                # episode-bounded → 1.0
    def initialstate(self, key) -> jax.Array   # samples flat state vector
    def transition(self, s_flat, a_flat, key)  # → s_next_flat
    def observation(self, s, a, s', side)      # → per-side observation
    def reward(self, s, a, s', side)           # → per-side scalar reward
```

## Minimal example

```python
import jax
import jax.numpy as jnp

from orbital_game import OrbitalGameEnv, make_lady_bandit_guard, Side
from orbital_game.adapters.pomdp import POMDPAdapter

cfg = make_lady_bandit_guard()
adapter = POMDPAdapter(OrbitalGameEnv(cfg))

# Sample initial state.
s0 = adapter.initialstate(jax.random.PRNGKey(0))
# s0.shape == (adapter.states_dim,)

# Step.
n_g, n_b = cfg.n_guards, cfg.n_bandits
d = adapter.action_dim_per_side
a = jnp.zeros(n_g * d + n_b * d)        # concatenated guard+bandit actions
s1 = adapter.transition(s0, a, jax.random.PRNGKey(1))

# Per-side reward and observation:
r_guard = adapter.reward(s0, a, s1, Side.GUARD)
r_bandit = adapter.reward(s0, a, s1, Side.BANDIT)
obs_guard = adapter.observation(s0, a, s1, Side.GUARD)
obs_bandit = adapter.observation(s0, a, s1, Side.BANDIT)
```

## Action layout

`a_flat` is the concatenated per-side actions:

```text
a_flat[: n_g * d]                 = guard_actions.reshape(-1)
a_flat[n_g * d : (n_g + n_b) * d] = bandit_actions.reshape(-1)
```

## State layout

`s_flat` is the result of `env.layout.flatten(state.guards, state.bandits)`. The flat encoding excludes the scalar `t`/`step` and the `reference_orbit` — those are bookkeeping carried by the adapter internally. POMDPPlanners-shape consumers typically work with `(s, a) → s'` where the time/step bookkeeping is implicit; this matches that convention.

## Per-side observation/reward dispatch

Unlike Gymnasium and PettingZoo, the POMDP adapter doesn't pre-commit to a controlled side. Pass `side=Side.GUARD` or `side=Side.BANDIT` explicitly to `observation()` and `reward()`:

```python
# Same s/a/s' triple, two different perspectives:
guard_pov = adapter.observation(s, a, s_next, Side.GUARD)
bandit_pov = adapter.observation(s, a, s_next, Side.BANDIT)
```

This is useful for centralized critics in MARL or for evaluating both sides' beliefs simultaneously.

## Discount

`discount() → 1.0` by default. Episode-bounded games (which all four bootstrap games are) don't strictly need a discount factor, since the `MaxStepsOrBreach` termination prevents unbounded value functions. Override per-game if needed.

## Source

- [`src/orbital_game/adapters/pomdp/adapter.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/adapters/pomdp/adapter.py)
- Tests: [`tests/test_adapter_pomdp.py`](https://github.com/duncaneddy/orbital-game/blob/main/tests/test_adapter_pomdp.py)
