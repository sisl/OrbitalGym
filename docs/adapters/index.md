# Adapter how-tos

Three adapters expose `OrbitalGameEnv` to standard RL and POMDP toolchains. Each converts JAX pytrees to numpy arrays at the framework boundary; all physics lives in the unchanged JAX core.

| Adapter | Wraps | Framework | Install |
|---|---|---|---|
| [GymnasiumAdapter](gymnasium.md) | `SingleAgentView` | Gymnasium (single-agent) | `pip install orbital-game[gymnasium]` |
| [PettingZooAdapter](pettingzoo.md) | `OrbitalGameEnv` | PettingZoo Parallel API | `pip install orbital-game[pettingzoo]` |
| [POMDPAdapter](pomdp.md) | `OrbitalGameEnv` | POMDPPlanners-shape protocol | (no extra) |

## Picking an adapter

- One learned agent against a scripted opponent → [Gymnasium](gymnasium.md).
- Two or more learned agents simultaneously → [PettingZoo](pettingzoo.md).
- Belief-space planning, classical POMDP solvers, POMDPs.jl-style protocols → [POMDPPlanners-shape](pomdp.md).

## Top-level convenience exports

The adapters are re-exported from the top-level package:

```python
import orbital_game

# These work if the matching extra is installed; else they're None:
orbital_game.GymnasiumAdapter
orbital_game.PettingZooAdapter

# Always available:
orbital_game.POMDPAdapter
```

For an explicit `ImportError` on a missing extra (rather than `None`), import the submodule directly:

```python
from orbital_game.adapters.gymnasium import GymnasiumAdapter
```
