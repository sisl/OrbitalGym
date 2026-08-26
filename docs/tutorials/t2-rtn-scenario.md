# T2 — Build an RTN scenario with visualisations

Build a 2-guard / 1-bandit Pursuit-Evasion scenario in 3D RTN, run a
multi-agent rollout, and render trajectory and reward plots. Builds on
[T1 — First rollout (RT 2D)](t1-first-rollout.md); a companion notebook
lives at [`examples/workflow_3d_rtn.ipynb`](https://github.com/sisl/OrbitalGym/blob/main/examples/workflow_3d_rtn.ipynb).

## Imports

```python
--8<-- "tests/docs/test_tut_t2_rtn_scenario.py:imports"
```

## Build the scenario

```python
--8<-- "tests/docs/test_tut_t2_rtn_scenario.py:build-config"
```

`make_pursuit_evasion` defaults to HCW-RTN, so we get the 6-state form
`[r, θ, n, ṙ, θ̇, ṅ]` — radial / along-track / cross-track position
and velocity in the rotating reference-orbit frame. Two guards, one
bandit, 200 steps at 10 s.

## Roll the episode out (multi-agent)

```python
--8<-- "tests/docs/test_tut_t2_rtn_scenario.py:run-rollout"
```

This is `rollout`, not `rollout_single_agent` — both sides have an
explicit policy and both sides' actions are recorded. The symmetric
core takes a `BySide[Policy]` for the policies and a
`BySide[Callable]` for the per-side policy-state initialisers; the
`lambda c, s, k: None` callables are no-op initialisers because
`ZeroControl` is stateless.

## Read off the shapes

```python
--8<-- "tests/docs/test_tut_t2_rtn_scenario.py:shapes"
```

- `traj.env_state.guards.rtn` — shape `(T, N_g, 6)`. **Outer-to-inner:
  time, vehicle, feature.**
- `traj.env_state.bandits.rtn` — shape `(T, N_b, 6)`. Same convention.
- `episode_mask(traj)` — `(T,)` bool, `True` for steps before and
  including termination.

To grab the i-th guard's full time series of position:
`traj.env_state.guards.rtn[:, i, :3]`.

To grab the t-th step's full state across guards:
`traj.env_state.guards.rtn[t]`.

## Compute capture distance over time

```python
--8<-- "tests/docs/test_tut_t2_rtn_scenario.py:capture-distance"
```

Pairwise distance between guard 0 and bandit 0. With zero control on
both sides, this oscillates (relative ellipses precess against each
other) without falling below `cfg.game.capture_distance_m`.

## Render the trajectory in 3D

```python
--8<-- "tests/docs/test_tut_t2_rtn_scenario.py:plot-rtn"
```

`plot_rtn_3d` accepts a `(T, N, 6)` array and overlays each vehicle's
ellipse on the same axes. The reference orbit appears as the origin.

## Render the reward curve

```python
--8<-- "tests/docs/test_tut_t2_rtn_scenario.py:plot-reward"
```

For Pursuit-Evasion (zero-sum) the bandit reward is the negation of
the guard reward — plot one and the other is implied.

## Next

→ **[T3 — Write an active-pursuer policy](t3-active-pursuer.md)**
replaces `ZeroControl` on the bandit with a HCW-predict-and-thrust
policy that actually closes on the guard.
