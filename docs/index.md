# orbital-game

A JAX-native framework for guard/bandit decision-making in orbital scenarios.

`orbital-game` provides a symmetric multi-agent core for spacecraft pursuit, evasion, and observation games. Both sides — the **guard** (protecting an asset or evading capture) and the **bandit** (the adversarial actor) — are first-class agents driven by their own policies. The agent can play either role, with a scripted policy on the opposite side.

## What's included

- **Symmetric JAX core** — `OrbitalGameEnv.step(state, Actions)` runs both sides through identical machinery, with `vmap`/`scan`/`jit`-clean pytrees throughout.
- **Single-agent and multi-agent views** — `SingleAgentView` projects the symmetric core to a Gymnasium-style single-agent interface; the symmetric core directly drives PettingZoo's Parallel API.
- **Four bootstrap games** —
  - **Lady-Bandit-Guard (LBG)**: guard protects the reference orbit from the bandit
  - **Pursuit-Evasion (PE)**: 1v1 pursuit; bandit minimizes relative distance, guard maximizes
  - **Sun-Blocking (SB)**: bandit occludes the guard's view of the Sun
  - **Observation-Blocking (OB)**: bandit blocks the guard's view of an Earth target, gated on visibility
- **Three framework adapters** — `GymnasiumAdapter`, `PettingZooAdapter`, `POMDPAdapter` (POMDPPlanners-shape) as optional pip extras.

## Quick start

```python
from orbital_game import OrbitalGameEnv, SingleAgentView, ZeroControl, make_lady_bandit_guard
from orbital_game.rollout import rollout_single_agent
import jax

cfg = make_lady_bandit_guard()
env = OrbitalGameEnv(cfg)
view = SingleAgentView(env)

guard_policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
traj = rollout_single_agent(
    view, guard_policy, lambda c, s, k: None,
    jax.random.PRNGKey(0), n_steps=cfg.max_steps,
)
```

See [Getting Started](getting-started.md) for the full path from install to a rendered trajectory plot.

## Documentation map

| Section | What's there |
|---|---|
| [Getting Started](getting-started.md) | Install, run the reference scenario, plot output |
| [Concepts](concepts.md) | Symmetric core, sides, scope, axis convention, `BySide` |
| [Game Guides](games/index.md) | Per-game pages with reward formulas, knobs, builders |
| [Adapter How-Tos](adapters/index.md) | Gymnasium, PettingZoo, POMDPPlanners-shape |
| [Architecture](architecture.md) | Condensed design overview (full spec linked) |
| [Extending](extending/index.md) | Custom observations, rewards, policies, games |

## Project status

`orbital-game` is in active development. The four-phase API overhaul (rename → symmetric core → game catalog → framework adapters) shipped in 2026-Q2; documentation is being filled in incrementally.
