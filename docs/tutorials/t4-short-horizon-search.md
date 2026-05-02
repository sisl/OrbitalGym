# T4 — Plan with short-horizon search

Wire `POMDPAdapter` to a small random-shooting planner that simulates
candidate action sequences forward through the env and picks the best
by cumulative reward, ending with one planned step from a sampled
initial state. Builds on
[T3 — Write an active-pursuer policy](t3-active-pursuer.md); reference
implementation at
[`examples/planners/receding_horizon.py`](https://github.com/duncaneddy/orbital-game/blob/main/examples/planners/receding_horizon.py).

## What this is and isn't

This is **random shooting** — sample K action sequences, simulate
each, take the first action of the best. It is *not* MCTS: there is
no tree, no UCB, no expansion-by-uncertainty. The teaching value is
the API patterns: how to call `transition` / `reward`, how to thread
PRNG keys through a horizon loop, how the action vector is laid out,
and how a pure adapter composes with `jax.vmap` and `jax.lax.scan`.

For real planning, swap the search loop for [`mctx`](https://github.com/google-deepmind/mctx)
or [`pomdp-py`](https://github.com/h2r/pomdp-py). The wiring stays
the same.

## The POMDP adapter surface

```python
--8<-- "tests/docs/test_tut_t4_short_horizon_search.py:imports"
```

```python
--8<-- "tests/docs/test_tut_t4_short_horizon_search.py:adapter-setup"
```

`POMDPAdapter` exposes a [POMDPs.jl](https://juliapomdp.github.io/POMDPs.jl/latest/)-shaped
interface over `OrbitalGameEnv`:

- `states_dim` — flat-state dimensionality. The vector packs
  `StateLayout.flatten` (per-side guard/bandit truth) followed by
  two scalar tail entries: `t` and `step`. The reference orbit is
  captured at `__init__` time as a per-scenario constant, so it
  doesn't need to live in the flat vector.
- `action_dim_per_side` — components of `Δv` per vehicle (3 for
  RTN dynamics, 2 for in-plane).
- `discount() → float` — episode-bounded games return `1.0`. Override
  by subclassing if your problem needs `γ < 1`.
- `initialstate(key) → s_flat` — sample an initial state and return
  its flat vector.
- `transition(s_flat, a_flat, key) → s_flat'` — apply one env step.
- `observation(s, a, s', side) → obs` — per-side observation at `s'`.
- `reward(s, a, s', side) → float` — per-side scalar reward.

The action vector layout is **guards-first then bandits**. For
`n_g` guards and `n_b` bandits, each with `d = action_dim_per_side`
components, the flat action is

```
a_flat = [guard_0_dv(d), ..., guard_{n_g-1}_dv(d),
          bandit_0_dv(d), ..., bandit_{n_b-1}_dv(d)]
```

with total length `(n_g + n_b) * d`.

## Sample an initial state

```python
--8<-- "tests/docs/test_tut_t4_short_horizon_search.py:initial-state"
```

`s0` is a flat `(states_dim,)` array — the per-side truth plus the
two-scalar `(t, step)` tail. There is no hidden mutable bookkeeping;
every call to `transition` reads its bookkeeping out of the input
vector and writes the new bookkeeping into the output vector.

## Build the planner

```python
--8<-- "tests/docs/test_tut_t4_short_horizon_search.py:planner-class"
```

The planner is a single `jax.vmap(jax.lax.scan)`:

1. Outer `vmap` over `K` candidates — each candidate gets its own
   PRNG sub-key. JAX runs them in parallel on whatever device is
   available.
2. Inner `lax.scan` over `H` horizon steps — `transition` advances
   the flat state, `reward` scores it for the controlled side, and
   the scan accumulates the per-step rewards.
3. `argmax` over the per-candidate cumulative rewards picks the
   best sequence; the planner returns its first action.

Because `POMDPAdapter` is pure, all of this compiles to a single
JIT call. No Python loops, no per-candidate state snapshots — the
horizon-rollout primitive is a JAX function from `(seq_key) →
(score, first_action)` and `vmap` does the parallelism.

## Plan one step

```python
--8<-- "tests/docs/test_tut_t4_short_horizon_search.py:plan-one-step"
```

`a_planner` has shape `(full_action_dim,) = ((n_g + n_b) * d,)` —
the same flat layout the adapter consumes. Feeding it back into
`adapter.transition` advances one real env step from `s0`, and
`adapter.reward(s0, a_planner, s1, Side.BANDIT)` scores it from the
bandit's perspective.

## When to override discount

`POMDPAdapter.discount()` returns `1.0` because the bundled games
(`make_pursuit_evasion`, `make_lady_bandit_guard`, `make_sun_blocking`)
are episode-bounded — terminal conditions and `max_horizon_s` cap
the trajectory, so undiscounted cumulative reward is well-defined.
If you wire the adapter into an unbounded MDP or want to emphasise
near-term reward, subclass `POMDPAdapter` and override `discount()`
to return your `γ`. The planner above ignores discount; a serious
search loop should multiply it into the inner-loop accumulator.

## Going further

- [`mctx`](https://github.com/google-deepmind/mctx) — DeepMind's
  JAX-native MCTS library. Plugs into a *pure* dynamics function;
  `POMDPAdapter.transition` is exactly that, so it slots in as
  `recurrent_fn` directly.
- [`pomdp-py`](https://github.com/h2r/pomdp-py) — Python POMDPs
  framework with a similar interface to `POMDPAdapter`. The flat
  vector here is intentionally compatible with that ecosystem.

## Next

→ **[T5 — Run on GPU / MPS](t5-acceleration.md)** moves the same
rollouts off CPU and onto an accelerator.
