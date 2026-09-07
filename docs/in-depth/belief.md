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

The `action` an updater receives is the Δv the env actually imparted,
not the Δv the policy commanded — `belief_rollout` reads it from
`StepOutput.info["applied_dv"]`, sliced to the truth frame's width. A
policy is free to command more than its thruster or its tank can
deliver, and a belief that predicted under the command would walk its
own-state estimate off truth every step the command was clipped.

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

## Particle filter when the prior is non-Gaussian

`ParticleFilterBelief` represents the posterior as a weighted cloud
of `K` state samples per `(observer, target)` pair instead of a
mean+cov. This is the right choice when the prior is **multi-modal**
or **lives on a manifold** — e.g. an LBG ring scenario where the
guard knows the bandit is somewhere on the 2:1 RT-plane natural-motion
ring but uniformly distributed in phase. A Kalman filter would have
to collapse that prior onto the centroid of the ring (the lady
herself), which is the worst possible point estimate.

| Field | Shape | Meaning |
|---|---|---|
| `particles` | `(N_obs, N_total, K, d)` | per-pair particle cloud |
| `log_weights` | `(N_obs, N_total, K)` | log-softmax-normalized weights |
| `n_eff` | `(N_obs, N_total)` | pre-resample effective sample size |
| `weight_entropy` | `(N_obs, N_total)` | pre-resample Shannon entropy (nats) |
| `resampled` | `(N_obs, N_total)` bool | did this pair resample this step? |

`mean` is exposed as a `@property` computing the weighted mean over
the K axis, so policies that read `belief.mean` work unchanged across
KF / EKF / PF.

```python
--8<-- "tests/docs/test_indepth_belief.py:pf-belief-shape"
```

`ParticleFilterBeliefUpdater` runs propagate → measurement-conditioned
proposal → per-channel weight update → pre-resample metrics →
systematic resampling per pair when `N_eff / K < n_eff_threshold`
(default `0.5`). Non-visible pairs leave weights unchanged unless a
gated channel supplies negative information. Resampled particles get
uniform log-weights and an optional jitter draw to combat sample
impoverishment.

The proposal is what keeps a directly measured cell on its
measurement. Every linear channel visible for a pair is stacked into
one measurement, and the new particles are drawn from that pair's
prior conditioned on it, with the importance weight
`log N(z; H f(x), H P Hᵀ + R)` to match. The prior `P = h² C + Q` is
the pair's own predicted spread `C`, kernel-smoothed by Silverman's
bandwidth `h`, plus the process noise, so a measurement sharper than a
broad cloud — a ring prior meeting its first detection, a cell that
drifted while unseen — repositions the cloud instead of collapsing it
onto whichever particle happened to sit closest.

Inflating the transition kernel that way makes this a **regularized**
particle filter: it targets the posterior under a kernel-widened
transition, not the exact posterior. On a pair with a visible linear
channel the update is Gaussian-approximate — one shared gain moves
every particle, and multimodality in the subspace the measurement does
not resolve is smoothed away with it. Pairs with nothing visible keep
the plain `N(f(x), Q)` draw and the exact bootstrap kernel, which is
what a ring prior needs until the sensor first fires.

The three pair-level metrics are recorded **pre-resample** — N_eff and
entropy describe the posterior produced by the latest measurement, so
the *collapse signal* survives in `belief_history` rather than being
masked by the post-resample uniform reset. Read `resampled` as what it
says: the pairs whose weights had degenerated far enough to be redrawn.
It is not a detection marker — the conditioned proposal keeps weights
well conditioned through a detection, so a pair can be measured every
step and rarely resample.

### Negative information

A gated channel (`ConicalObservation`, `RangeLimitedObservation`) can
supply a `visibility_score_fn`, and the PF's `negative_info` mode then
reads a non-detection as evidence: `Hard` crushes particles inside the
gate, `Soft` tapers them across the boundary, `Off` leaves them alone.

A channel must score every pair its `visible` mask can never set — self
pairs, same-side pairs — as `OUT_OF_SCOPE_SCORE`. Those pairs are
outside the sensor's reach rather than inside its gate and undetected,
so a non-detection carries no information about them. Scoring them on
their geometry instead lets the update crush the observer's own-state
particles: the cone would truncate the own-state cloud along the
boresight every step and push the estimate away without bound.

### Initializers

- `ParticleFilterFromTruthInitializer` — particles concentrated at
  truth + small isotropic jitter. Use when the prior should be a
  near-delta on the true state (e.g. cheat-prior baselines).
- `ParticleFilterRingInitializer` — cross-pair particles sampled
  uniformly in phase along the 2:1 RT natural-motion ellipse, with
  velocities set from the closed-form natural-motion solution so each
  particle is itself a valid drift trajectory under HCW. Self-pair
  particles are delta-on-truth. `along_track_offset_m` moves the ring's
  centre along-track, matching the target's
  `RelativeEllipse.along_track_offset_m`.
- `ParticleFilterTrackedInitializer` — cross-pair particles are the
  opponent's true state plus Gaussian noise, `sigma_pos_m` on each
  position component and `sigma_vel_mps` on each velocity component.
  Use for an observer holding a stale but roughly correct track.
  Self-pair particles are delta-on-truth.

### When to pick which belief

- **Truth ground-truth available** → skip belief entirely; use
  `rollout` (or feed truth state directly to the policy).
- **Linear dynamics + linear sensor + Gaussian prior** → `KFBelief`.
- **Nonlinear dynamics or sensor + Gaussian prior** → `EKFBelief`.
- **Multi-modal or manifold prior** (e.g. ring, donut, two-cluster
  intruder hypothesis) → `ParticleFilterBelief`.

See `examples/lbg_ring_pf_intercept.ipynb` for the full demo —
uniform-on-ring prior, range-limited sensor, particle-cloud collapse
on first detection, and `RolloutScene` animation with the integrated
particle-cloud renderer.

## `belief_rollout`

`orbitalgym.rollout.belief_rollout` is the belief-aware sibling of
`rollout`. Instead of passing the flattened observation to each side's
policy, it threads a per-side `Belief` (initialised by your
`BeliefInitializer` and updated each tick by your `BeliefUpdater`) and
hands it to the policy as `agent_view`. Policies that consume belief
read `agent_view.mean` directly — there is no special belief-aware
protocol; the same `Policy` works for obs-only and belief-aware
pipelines, and the rollout decides what `agent_view` is.

```python
from orbitalgym.rollout import belief_rollout
traj, belief_history = belief_rollout(
    env,
    policies,                 # BySide[Policy]
    init_policy_state_fns,    # BySide[Callable]
    belief_initializers,      # BySide[BeliefInitializer]
    belief_updaters,          # BySide[BeliefUpdater]
    key=jax.random.PRNGKey(0),
    n_steps=200,
)
```

`belief_history` is a `BySide` whose leaves are belief leaves with a
leading time axis — feeds straight into
`orbitalgym.viz.animation.RolloutScene`'s `belief_history` parameter.
The renderer detects the belief type by duck-typing: KF/EKF beliefs
(with `cov`) draw 2σ ellipses/ellipsoids; particle-filter beliefs (with
`particles` + `log_weights`) draw a weighted scatter cloud with a
weighted-mean overlay. The same `belief_history` keyword serves both.

## Writing a custom updater

The practical "how to write one" walkthrough lives in the Extending
section. See [Extending → Customize observations](../extending/customize-observations.md)
for the observation-side protocol, two worked examples, and per-game
applicability — the belief consumes whatever channels you build.

## Where to next

- [Observations](observations.md) — what produces the `Observation`
  tuples that the belief consumes.
- [API reference → Components → Belief](../api/components.md) —
  `BeliefInitializer`, `BeliefUpdater`, `KFBelief`, `EKFBelief`,
  `ParticleFilterBelief`.
