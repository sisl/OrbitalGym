# Concepts

The mental model behind `orbital-game`'s symmetric core.

## Sides

`Side` is a Python-level enum (a `StrEnum`, never a traced JAX array):

```python
from orbital_game import Side

Side.GUARD       # the protected/observing asset
Side.BANDIT      # the adversarial actor
Side.GUARD.opposite() == Side.BANDIT
```

The naming is from the Lady-Bandit-Guard taxonomy and applies across all four games:

| Game | Guard | Bandit |
|---|---|---|
| Lady-Bandit-Guard | protector | threat |
| Pursuit-Evasion | evader | pursuer |
| Sun-Blocking | observer | blocker |
| Observation-Blocking | observer | blocker |

## `BySide[T]`

The universal one-per-side container — itself a JAX pytree, so `vmap` / `lax.scan` / `tree.map` traverse it transparently:

```python
from orbital_game import BySide, Side

bs = BySide(guard=42.0, bandit=-1.0)
bs.guard            # 42.0
bs.bandit           # -1.0
bs.get(Side.GUARD)  # 42.0  (static-key access)
bs.map(lambda x: x * 2)             # BySide(guard=84.0, bandit=-2.0)
for side, value in bs.items(): ...  # iterate over both sides
```

`BySide` shows up wherever a value is naturally one-per-side: actions, side outputs, observations, scripted policies, beliefs.

## Actions and step output

`env.step` takes a single `Actions` container (per-side stacked arrays) and returns a single `StepOutput` (per-side outputs + episode-level metadata):

```python
from orbital_game import Actions, BySide, OrbitalGameEnv
import jax, jax.numpy as jnp

env = OrbitalGameEnv(cfg)
state, _outputs = env.reset(jax.random.PRNGKey(0))

actions = Actions(sides=BySide(
    guard=jnp.zeros((cfg.n_guards, action_dim)),
    bandit=jnp.zeros((cfg.n_bandits, action_dim)),
))

step_out = env.step(jax.random.PRNGKey(1), state, actions)
step_out.state                        # next EnvState
step_out.outputs.guard.obs            # guard's observation
step_out.outputs.guard.reward         # guard's reward
step_out.outputs.bandit.obs           # bandit's observation
step_out.outputs.bandit.reward        # bandit's reward
step_out.episode_done                 # scalar bool — episode termination
step_out.outputs.guard.done           # broadcast of episode_done (shape uniformity)
```

## Single-agent vs multi-agent framing

The same env supports both styles:

- **Symmetric (multi-agent)** — call `env.step(...)` directly with both sides' actions. The PettingZoo adapter and the per-side `rollout` use this path.
- **Single-agent** — `SingleAgentView` reads `cfg.controlled_side` (default `Side.GUARD`) and the *opposite* side's scripted policy from `cfg.<side>_scripted_policy`. The view's `step` takes only the controlled side's action; the opponent runs internally:

```python
from orbital_game import SingleAgentView

view = SingleAgentView(env)        # reads cfg.controlled_side + scripted policy
state, obs, opp_ps = view.reset(key)
next_state, obs, reward, done, opp_ps, info = view.step(
    key, state, controlled_action, opp_ps
)
```

The Gymnasium adapter wraps `SingleAgentView`.

## Scope: per-vehicle vs per-side

Observations and rewards declare a `scope` attribute that determines output shape:

| Scope | Observation shape | Reward shape |
|---|---|---|
| `PER_VEHICLE` | `(N_side, obs_dim)` | `(N_side,)` |
| `PER_SIDE` | `(obs_dim,)` | `()` |

`PER_VEHICLE` is the Dec-POMDP framing — every vehicle gets its own observation. `PER_SIDE` is a shared/team framing — one observation broadcast to all vehicles on the side.

The default `FullObservation` is `PER_SIDE`. The default `DistanceToReferenceOrbit` reward is `PER_SIDE`. Custom implementations declare their own scope.

## Axis ordering

Outer-to-inner: `BATCH → TIME → VEHICLE → FEATURE`

| Axis | When present |
|---|---|
| `B` (batch) | Only when caller wraps with `jax.vmap` over seeds/configs |
| `T` (time) | Only on `Trajectory` leaves (added by `jax.lax.scan` in `rollout`) |
| `N_side` (vehicle) | Always present per side, even if `N=1` |
| feature dims | Innermost (e.g. `action_dim=3`, `obs_dim`, RTN's 6) |

For a vmapped rollout over `(B,)` seeds: `Trajectory.sides.guard.action.shape == (B, T, N_g, 3)`.

## Trajectory shape

```python
traj.env_state                       # full pytree, leading T axis on every leaf
traj.sides.guard.obs                 # (T, [N_g,] obs_dim) per scope
traj.sides.guard.action              # (T, N_g, action_dim)
traj.sides.guard.reward              # (T, [N_g])
traj.sides.guard.done                # (T,) bool, latched
traj.sides.bandit.obs / .action / .reward / .done  # mirror of guard
traj.episode_done                    # (T,) bool — same scalar broadcast
traj.controlled_side                 # Side enum (Python-level, not traced)
```

`episode_mask(traj)` returns a `(T,)` bool mask True up to and including the terminating step, useful for masking analysis to valid steps.

## Scripted policies

Both sides' scripted policies live on `ScenarioConfig`:

```python
cfg.controlled_side                  # Side enum, default Side.GUARD
cfg.guard_scripted_policy            # Policy instance, default ZeroControl()
cfg.bandit_scripted_policy           # Policy instance, default ZeroControl()
```

`SingleAgentView` reads the *opposite* side's scripted policy at construction; the multi-agent adapters ignore both fields.

A `Policy` is role-agnostic — the same `ZeroControl` class works as guard scripted-opponent, bandit scripted-opponent, or learned controlled-side fallback. Dimensions (`n_vehicles`, `action_dim`) are populated by the env at construction via `dataclasses.replace`.

## Game catalog

A `Game` is a typed knob bundle attached to `cfg.game`:

```python
from orbital_game import LadyBanditGuard, PursuitEvasion, SunBlocking, ObservationBlocking

cfg.game = LadyBanditGuard(breach_distance_m=10.0)
cfg.game = PursuitEvasion(capture_distance_m=10.0)
cfg.game = SunBlocking(angle_sigma_deg=5.0)
cfg.game = ObservationBlocking(target_lat_deg=37.4, target_lon_deg=-122.2,
                               min_elevation_deg=5.0, angle_sigma_deg=5.0)
```

Game-specific reward/termination implementations check `cfg.game` via `isinstance` and read knobs from there. The `make_<game>(...)` builders wire matching reward + termination automatically.

See [Game Guides](games/index.md) for per-game details.
