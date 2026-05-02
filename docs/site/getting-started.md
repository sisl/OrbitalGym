# Getting Started

A 5-minute path from zero to a rendered RTN trajectory plot.

## Install

`orbital-game` uses `uv` for dependency management. From the repo root:

```bash
uv sync --extra dev
```

This installs the core dependencies (JAX, flax, astrojax, h5py, matplotlib) plus the dev tooling (pytest, ruff, pyrefly).

For the Gymnasium and PettingZoo adapters, install the matching extras:

```bash
uv add orbital-game --extra gymnasium
uv add orbital-game --extra pettingzoo
```

The `POMDPAdapter` has no external dependency and is always importable.

## Run the reference scenario

The reference scenario is the bootstrap acceptance test — a 1-guard / 1-bandit Lady-Bandit-Guard rollout with zero-control policies on both sides:

```bash
uv run python -m examples.reference_scenario
```

This prints the path to a temp directory containing:
- `run.h5` — full trajectory in HDF5
- `guard_rtn_3d.png` — 3D RTN trajectory plot
- `reward.png` — guard's reward time series
- `guard_mass.png` — guard propellant mass over time

## A minimal Python example

```python
import jax
import jax.numpy as jnp

from orbital_game import (
    OrbitalGameEnv,
    SingleAgentView,
    make_lady_bandit_guard,
)
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout_single_agent

# Build the scenario via the Lady-Bandit-Guard preset.
cfg = make_lady_bandit_guard(
    n_guards=1,
    n_bandits=1,
    breach_distance_m=10.0,
    max_horizon_s=2000.0,
    dt=10.0,
    seed=0,
)

# Construct env + single-agent view (guard is the controlled side; bandit
# is scripted via the default ZeroControl in cfg.bandit_scripted_policy).
env = OrbitalGameEnv(cfg)
view = SingleAgentView(env)

# Define the guard's policy — here just zero-control as a placeholder.
guard_policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)

def init_none(c, s, k):
    return None

# Roll out for cfg.max_steps timesteps.
traj = rollout_single_agent(
    view, guard_policy, init_none,
    jax.random.PRNGKey(0), n_steps=cfg.max_steps,
)

# Inspect: per-side trajectory has the same shape for guard and bandit.
print("guard reward sum:", float(jnp.sum(traj.sides.guard.reward)))
print("episode steps until done:", int(jnp.argmax(traj.episode_done)) + 1)
print("guard final RTN position:", traj.env_state.guards.rtn[-1, 0, :3])
```

## What's next

- [Concepts](concepts.md) — the data model: `Side`, `BySide`, `Actions`, `StepOutput`, axis ordering
- [Game Guides](games/index.md) — pick a game and learn its reward shape
- [Adapter How-Tos](adapters/index.md) — drive the env from Gymnasium, PettingZoo, or POMDPPlanners-shape consumers
- [Extending](extending/index.md) — write your own policies, observations, rewards, or games
