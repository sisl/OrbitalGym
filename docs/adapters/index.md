# Adapter How-Tos

Three thin framework adapters expose `OrbitalGameEnv` to standard RL/POMDP toolchains. Adapters are opt-in pip extras.

| Adapter | Wraps | Framework | Pip extra |
|---|---|---|---|
| [`GymnasiumAdapter`](gymnasium.md) | `SingleAgentView` | Gymnasium (single-agent) | `pip install orbital-game[gymnasium]` |
| [`PettingZooAdapter`](pettingzoo.md) | `OrbitalGameEnv` | PettingZoo Parallel API (multi-agent) | `pip install orbital-game[pettingzoo]` |
| [`POMDPAdapter`](pomdp.md) | `OrbitalGameEnv` | POMDPPlanners-shape protocol | (no extra needed) |

## Architecture

Adapters sit at the host-array boundary of the JAX core:

```
External Frameworks
   │
   ▼
adapters/{gymnasium,pettingzoo,pomdp}/   ← host-array boundary
   │
   ▼
SingleAgentView   (Gymnasium only)
   │
   ▼
OrbitalGameEnv  ← symmetric JAX core (jit/vmap/scan-clean)
```

Each adapter is a thin Python class that holds env state in mutable attributes, materializes JAX arrays as numpy at the boundary, and delegates the actual physics to the unchanged JAX core.

## JAX consistency invariant

Every adapter is a *pure projection* of the JAX core, not a parallel reimplementation. The cross-adapter consistency tests (`tests/test_adapter_jax_consistency.py`) prove this: given the same key + same actions, an adapter step produces the same numerical result as a direct `env.step` projected to the relevant view.

This means:
- Bug fixes in the core auto-propagate to all adapters
- Performance improvements in the core (e.g. better dynamics) are inherited
- Adapter behavior never drifts from the core's semantics

## Picking an adapter

- **Single learned agent vs. scripted opponent** → [Gymnasium](gymnasium.md)
- **Two (or more) learned agents simultaneously** → [PettingZoo](pettingzoo.md)
- **Belief-space planning, POMDPs.jl-style protocols, classical POMDP solvers** → [POMDPPlanners-shape](pomdp.md)

## Top-level convenience exports

The adapters are also re-exported from the top-level package for ergonomic access:

```python
import orbital_game

# These work if the matching extra is installed; else they're None:
orbital_game.GymnasiumAdapter
orbital_game.PettingZooAdapter

# Always available:
orbital_game.POMDPAdapter
```

For loud failure on missing extras, import the submodule directly:

```python
from orbital_game.adapters.gymnasium import GymnasiumAdapter   # ImportError if not installed
```
