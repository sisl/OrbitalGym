# Concepts

The mental model behind `orbital-game`'s symmetric core. This page explains what the types mean and why they exist; for call signatures see the [API reference](api/index.md).

## Sides

Every game is symmetric in two roles: a **guard** and a **bandit**. The naming comes from the Lady-Bandit-Guard taxonomy and applies across all four bundled games — in Pursuit-Evasion the guard is the evader and the bandit is the pursuer; in Sun-Blocking and Observation-Blocking the guard is the observer and the bandit is the blocker.

`Side` is a Python-level `StrEnum`, never a traced JAX array. That distinction matters: side identity is a static structural property of the computation graph, not data flowing through it.

## `BySide[T]`

A universal one-per-side container, registered as a JAX pytree so `vmap`, `lax.scan`, and `tree.map` traverse it transparently. Anywhere a value is naturally per-side — actions, observations, rewards, beliefs, scripted policies — the type is `BySide[T]`.

```python
bs = BySide(guard=42.0, bandit=-1.0)
bs.map(lambda x: x * 2)             # BySide(guard=84.0, bandit=-2.0)
```

## Actions and step output

`env.step` takes a single `Actions` container (per-side stacked arrays) and returns a single `StepOutput` carrying:

- `state` — the next `EnvState`
- `outputs.guard` and `outputs.bandit` — per-side `(obs, reward, done)` bundles
- `episode_done` — a scalar bool indicating episode termination

The per-side `done` field is a broadcast of `episode_done`, kept on each side so per-side trajectories have uniform shape.

## Single-agent vs multi-agent framing

The same env supports both styles. In multi-agent framing, callers pass actions for both sides and read both sides' outputs — this is what the PettingZoo adapter and the symmetric `rollout` use.

In single-agent framing, `SingleAgentView` reads `cfg.controlled_side` (default `Side.GUARD`) and runs the opposite side's scripted policy internally. The view's `step` takes only the controlled side's action and returns only the controlled side's reward and observation. The Gymnasium adapter wraps this view.

The "agent perspective" is therefore a property of the *adapter*, not the core. The core is always symmetric.

## Scope: per-vehicle vs per-side

Observations and rewards declare a `scope` attribute that determines output shape:

- `PER_VEHICLE` is the Dec-POMDP framing: every vehicle on a side gets its own observation and reward. Shapes are `(N_side, obs_dim)` and `(N_side,)`.
- `PER_SIDE` is the team framing: one observation and reward shared across the side. Shapes are `(obs_dim,)` and `()`.

The bundled `FullObservation` and `DistanceToReferenceOrbit` are both `PER_SIDE`. Custom implementations declare their own scope and are responsible for emitting the right shape.

## Axis ordering

Outer to inner: **batch → time → vehicle → feature**.

| Axis | When present |
|---|---|
| `B` (batch) | Only when the caller wraps with `jax.vmap` over seeds or configs |
| `T` (time) | Only on `Trajectory` leaves (added by `lax.scan` in `rollout`) |
| `N_side` (vehicle) | Always present per side, even if `N=1` |
| feature | Innermost (e.g. `action_dim=3`, RTN's `6`) |

For a vmapped rollout over `(B,)` seeds, `traj.sides.guard.action` has shape `(B, T, N_g, 3)`.

## Trajectory shape

A `Trajectory` is what `rollout` returns: an `env_state` pytree with a leading time axis on every leaf, plus `BySide[SideTrajectory]` carrying per-side `(obs, action, reward, done)` time series. `episode_done` is a `(T,)` bool latched after termination, and `controlled_side` is preserved as a Python-level enum so single-agent consumers know which side the trajectory was generated for.

`episode_mask(traj)` returns a `(T,)` bool mask covering steps up to and including termination — useful for restricting analysis to valid steps.

## Scripted policies

Both sides carry a scripted policy on `ScenarioConfig` (`guard_scripted_policy`, `bandit_scripted_policy`, both default to `ZeroControl`). `SingleAgentView` reads the *opposite* side's scripted policy at construction; multi-agent adapters ignore both fields.

A `Policy` is role-agnostic: the same class can serve as a guard scripted opponent, a bandit scripted opponent, or a learned controlled-side fallback. Per-side dimensions (`n_vehicles`, `action_dim`) are populated by the env at construction.
