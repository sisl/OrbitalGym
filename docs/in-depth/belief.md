# Belief

A **belief** is what a side knows about every entity it tracks. The
shape is `(N_obs, N_total, d)` — one row per (observer, target) pair,
each carrying a `d`-dimensional estimate.

## The shape, in one diagram

For a side with `N_obs` observers and `N_total = N_obs + N_targets`
entities being tracked, a `KFBelief` carries:

| Field | Shape | Meaning |
|---|---|---|
| `mean` | `(N_obs, N_total, d)` | observer i's estimated state of entity k |
| `cov` | `(N_obs, N_total, d, d)` | observer i's covariance for entity k |

```python
--8<-- "tests/docs/test_indepth_belief.py:kf-belief-shape"
```

## Indexing recipes

```python
--8<-- "tests/docs/test_indepth_belief.py:belief-indexing"
```

The first axis is **who's observing**; the second is **whom they
think about** (own-side first, opposing-side second). For a guard
side with `n_guards=1, n_bandits=2`, the second axis is `[guard_0,
bandit_0, bandit_1]` — guard 0's estimate of bandit 1 is at
`belief.mean[0, 2]` (i.e. `belief.mean[0, n_guards + 1]`).

## Linear Kalman update

`KFBeliefUpdater` runs predict-and-correct each step:

**Predict.** Apply the side's STM `F` to every (observer, target)
state vector. For the self-pair (observer i tracking itself), add the
control offset `B @ action[i]`. Process noise `Q` is added to every
covariance.

**Correct.** For each `Observation` channel returned by the
observation function, run the per-pair correction
`x ← x + K(z − Hx)` gated by the channel's `visible` mask. Pairs with
`visible=False` keep the predicted estimate, no correction.

`KFBeliefUpdater` rejects channels with `obs_fn` set — those are
nonlinear and require `EKFBeliefUpdater`.

## EKF when measurements are nonlinear

`EKFBelief` and `EKFBeliefUpdater` carry the same shape but evaluate
the measurement function `obs_fn(state)` at the current mean and
linearise locally. Reach for EKF when your sensor model is nonlinear
in the tracked state — e.g. range-only or angle-only measurements.

## `BeliefRollout`

The `BeliefRollout` helper time-steps a belief through a recorded
trajectory: at each step it pulls the channel measurements, calls
`updater(belief, observations, action, side, key)`, and stacks the
result. Use it when you've already rolled out the env and want to
post-hoc reconstruct what each side believed at every step.

## Writing a custom updater

The practical "how to write one" walkthrough lives in the Extending
section. See [Extending → Customize observations](../extending/customize-observations.md)
for the observation-side protocol, two worked examples, and per-game
applicability — the belief consumes whatever channels you build.

## Where to next

- [Observations](observations.md) — what produces the `Observation`
  tuples that the belief consumes.
- [API reference → Components → Belief](../api/components.md) —
  `BeliefInitializer`, `BeliefUpdater`, `KFBelief`, `EKFBelief`.
