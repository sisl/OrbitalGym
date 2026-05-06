# T1 — First rollout (RT 2D)

Install `orbital-game`, run a planar (2D) HCW rollout with zero control
on both sides, and read three pieces of data off the resulting
`Trajectory`. A companion notebook lives at
[`examples/workflow_2d_rt.ipynb`](https://github.com/duncaneddy/orbital-game/blob/main/examples/workflow_2d_rt.ipynb).

## Install

```bash
git clone https://github.com/duncaneddy/orbital-game.git
cd orbital-game
uv sync --extra dev
```

## Imports

```python
--8<-- "tests/docs/test_tut_t1_first_rollout.py:imports"
```

## Build a 2D Lady-Bandit-Guard scenario

```python
--8<-- "tests/docs/test_tut_t1_first_rollout.py:build-config"
```

`HCW_RT` is the planar 4-state Clohessy–Wiltshire dynamics. Vehicle
state is `[r, θ, ṙ, θ̇]` — radial and along-track position and
velocity in the rotating reference-orbit frame. The full 6-state RTN
form is what T2 will use.

Each side carries an explicit list of state components that must
match the chosen dynamics — `RT` is required by `HCW_RT`, and the
guard additionally tracks `MASS` so propellant is bookkept across
impulses.

## Roll the episode out

```python
--8<-- "tests/docs/test_tut_t1_first_rollout.py:run-rollout"
```

`rollout_single_agent` compiles down to one `jax.lax.scan` call. The
opposing side (bandit) is driven internally by
`cfg.bandit_policy` — defaulting to `ZeroControl`.

## Inspect the trajectory

```python
--8<-- "tests/docs/test_tut_t1_first_rollout.py:inspect"
```

- `traj.sides.guard.reward.sum()` is the cumulative guard reward —
  for `make_lady_bandit_guard()` this is `LbgZeroSumReward` (dense
  shaping based on guard-bandit distance, plus terminal events at
  catch / breach). With both sides at zero thrust neither side scores
  a terminal event, so the cumulative return is dominated by the dense
  shaping term.
- `traj.episode_done.argmax() + 1` is the step at which the episode
  terminated. With no terminal event triggering, this equals
  `cfg.max_steps`.
- `traj.env_state.guards.rt` has shape `(T, N_g, 4)` — see [In depth →
  Symmetric core](../in-depth/symmetric-core.md) for the full axis
  convention.

## Next

→ **[T2 — Build an RTN scenario](t2-rtn-scenario.md)** scales up to
3D dynamics, adds vehicles, and renders trajectories you can spin in
matplotlib.
