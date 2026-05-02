# T4 — Plan with short-horizon search

**Time:** ~30 minutes.
**You will:** wire `POMDPAdapter` to a small random-shooting planner
that simulates candidate action sequences forward through the env and
picks the best by cumulative reward. End with one planned step from
a sampled initial state.
**Prerequisite:** [T3 — Write an active-pursuer policy](t3-active-pursuer.md).
**Reference implementation:** [`examples/planners/receding_horizon.py`](https://github.com/duncaneddy/orbital-game/blob/main/examples/planners/receding_horizon.py).

## What this is and isn't

This is **random shooting** — sample K action sequences, simulate
each, take the first action of the best. It is *not* MCTS: there is
no tree, no UCB, no expansion-by-uncertainty. The teaching value is
the API patterns: how to call `transition` / `reward`, how to thread
PRNG keys through a horizon loop, how the action vector is laid out,
and — critically — how to drive an adapter that carries internal
mutable state.

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

- `states_dim` — flat-state dimensionality (per-side guard/bandit
  truth, flattened by `StateLayout`). Scalar fields like `t`, `step`,
  and the `reference_orbit` are *not* in the flat vector — the adapter
  carries them in a stashed "context state" between calls.
- `action_dim_per_side` — components of `Δv` per vehicle (3 for
  RTN dynamics, 2 for in-plane).
- `discount() → float` — episode-bounded games return `1.0`. Override
  by subclassing if your problem needs `γ < 1`.
- `initialstate(key) → s_flat` — sample an initial state, return its
  flat vector, and stash the full env state internally.
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

`s0` is a flat `(states_dim,)` array. The full env state — including
`t`, `step`, and `reference_orbit` — is now stashed inside the
adapter as `adapter._last_state`. Hold on to that fact; the planner
below depends on it.

## Build the planner

```python
--8<-- "tests/docs/test_tut_t4_short_horizon_search.py:planner-class"
```

The planner has two nested Python loops: an outer loop over `K`
candidate sequences, an inner loop of length `H` advancing the
state. For each candidate it:

1. Splits `key` into per-step subkeys.
2. Samples a fresh `(H, action_dim)` Gaussian action sequence.
3. Resets `adapter._last_state` to the snapshot taken before the
   search began.
4. Steps the adapter forward through the horizon, summing per-step
   reward to the controlled side.
5. Tracks the best-scoring candidate's first action.

### Why not `vmap`/`scan`?

The load-bearing teaching point: **`POMDPAdapter` is stateful**. Each
call to `transition` mutates `self._last_state` to thread the env's
`t`, `step`, and `reference_orbit` through a flat-vector API that
otherwise only carries the per-side guard/bandit truth. The `act`
method *snapshots* `_last_state` once before the search, *restores*
it before each candidate's rollout so they all start from the same
root, and *restores* it again at exit so callers see no side effect.

This rules out `jax.vmap(planner_step)` or `jax.lax.scan` over
`transition`. Tracers from a transformed call would land in
`_last_state` and the next post-tracing call to `transition` would
raise `UnexpectedTracerError`. The POMDPPlanners-shape interface is
imperative by design — drive it with imperative Python.

If you need vectorised search, build a *pure-functional* clone of
the adapter that returns the next env state explicitly instead of
stashing it, and `vmap` that.

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
  wrap a functional version of the adapter and pass it as the
  `recurrent_fn`.
- [`pomdp-py`](https://github.com/h2r/pomdp-py) — Python POMDPs
  framework with a similar interface to `POMDPAdapter`. The flat
  vector here is intentionally compatible with that ecosystem.

## Next

→ **[T5 — Run on GPU / MPS](t5-acceleration.md)** moves the same
rollouts off CPU and onto an accelerator.
