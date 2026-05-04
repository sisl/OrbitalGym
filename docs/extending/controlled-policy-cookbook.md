# Controlled-side policy cookbook

Patterns for the side under your control. Each pattern is a *recipe*,
not a single fixed implementation — the gallery (linked from each
section) ships canonical implementations of three of these.

## Pattern: scripted-with-fallback

A learned primary policy backed by a known-safe fallback that takes
over when a confidence signal drops below threshold. The confidence
signal is threaded through `policy_state` (typically: belief covariance
trace, critic-head value, or any other uncertainty proxy).

```python
--8<-- "tests/docs/test_extending_controlled_policy_cookbook.py:imports"
```

```python
--8<-- "tests/docs/test_extending_controlled_policy_cookbook.py:scripted-with-fallback-pattern"
```

Both branches are evaluated every step (this is JAX-native — `jnp.where`
is the right primitive). The trade-off is intentional: tracing-friendly
routing always pays both costs.

Gallery: [`HeuristicWithFallbackPolicy`](gallery.md#heuristicwithfallbackpolicy).

## Pattern: belief-conditioned

A policy that augments its observation with the belief mean (or any
belief statistic). Usually paired with a `BeliefRollout` that supplies
the belief; the wrapper takes care of concatenation so the underlying
policy sees a single flat input.

```python
--8<-- "tests/docs/test_extending_controlled_policy_cookbook.py:belief-conditioned-pattern"
```

`policy_state` shape: `(belief_mean, base_state)`. The wrapper
prepends `belief_mean` to `obs`, calls the base policy, and threads
`(belief_mean, next_base_state)` back out.

Gallery: [`BeliefConditionedPolicy`](gallery.md#beliefconditionedpolicy).

## Pattern: composite action

A learned offset policy summed onto a scripted base. Lets you add a
small NN correction to a known-reasonable controller — usually faster
to train than learning the entire control law from scratch.

```python
--8<-- "tests/docs/test_extending_controlled_policy_cookbook.py:composite-action-pattern"
```

Gallery: [`CompositeActionPolicy`](gallery.md#compositeactionpolicy).

## Pattern: A/B harness

To compare two candidate policies head-to-head against the same
heuristic opponent, run them in parallel under `jax.vmap` over a seed
batch and compute the metric difference. This is a *recipe*, not a
class — see [How-to → vmap rollouts](../how-to/vmap-rollouts.md) for
the underlying mechanics.

The pattern: `vmap` the rollout over a `jax.random.split(key, n_seeds)`,
swap the controlled-side policy across two batches, and reduce the
per-seed metrics with `jnp.mean` / `jnp.std`.

## See also

- [Gallery](gallery.md) — concrete classes that ship in the package.
- [Heuristic policy cookbook](heuristic-policy-cookbook.md) — patterns
  for the *opposing* side.
- [Switch the controlled side](../how-to/switch-controlled-side.md) —
  train the bandit instead of the guard.
- [T3 — Active-pursuer policy](../tutorials/t3-active-pursuer.md).
