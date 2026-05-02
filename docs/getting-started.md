# Getting started

This page walks from a fresh checkout to a rendered RTN trajectory plot of a 1v1 Lady-Bandit-Guard rollout. Allow about five minutes.

## Install

`orbital-game` uses `uv` for dependency management. From the repo root:

```bash
uv sync --extra dev
```

The Gymnasium and PettingZoo adapters live behind extras (the POMDP adapter has no external dependency and is always importable):

```bash
uv sync --extra gymnasium --extra pettingzoo
```

## Run the reference scenario

```bash
uv run python -m examples.reference_scenario
```

The script builds a 1-guard / 1-bandit Lady-Bandit-Guard, rolls it out for 200 steps with zero control on both sides, and prints a path to a temp directory holding the trajectory log (`run.h5`), a 3D RTN trajectory plot, the reward time series, and the guard's propellant mass over time. Open `guard_rtn_3d.png`: with both sides at zero thrust, the guard's relative motion traces a closed ellipse around the reference orbit.

## Build it yourself

The same scenario in code:

```python
import jax
from orbital_game import OrbitalGameEnv, SingleAgentView, make_lady_bandit_guard
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout_single_agent

cfg = make_lady_bandit_guard()
env = OrbitalGameEnv(cfg)
view = SingleAgentView(env)

guard_policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
traj = rollout_single_agent(
    view, guard_policy, lambda c, s, k: None,
    jax.random.PRNGKey(0), n_steps=cfg.max_steps,
)

print("guard reward sum:", float(traj.sides.guard.reward.sum()))
print("episode length:", int(traj.episode_done.argmax()) + 1)
```

`rollout_single_agent` runs a `jax.lax.scan` under the hood, so the whole episode compiles down to one JAX call.

## Try a different game

Swap to Pursuit-Evasion by changing one line:

```python
from orbital_game import make_pursuit_evasion

cfg = make_pursuit_evasion(capture_distance_m=20.0)
```

The same `OrbitalGameEnv` / `SingleAgentView` / `rollout_single_agent` pipeline runs it. Pursuit-Evasion is zero-sum, so `traj.sides.guard.reward + traj.sides.bandit.reward` is zero at every step.

## Where to next

- [Concepts](concepts.md) for the data model behind `BySide`, `Actions`, and `StepOutput`.
- [Game guides](games/index.md) for what each scenario rewards and the knobs that shape it.
- [Adapter how-tos](adapters/index.md) to drive the env from Gymnasium, PettingZoo, or a POMDP planner.
- [Extending](extending/index.md) to write your own observations, rewards, policies, or games.
