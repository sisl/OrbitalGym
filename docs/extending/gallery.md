# Gallery

Every named, importable policy that ships with `orbital-game`. Use a
class directly, or read its source as a starting point for your own.

## Roster

| Class | Role | Best in | Summary |
|---|---|---|---|
| [`ZeroControl`](#zerocontrol) | Both | All | Always emits zero — the baseline default. |
| [`UniformRandomDiscretePolicy`](#uniformrandomdiscretepolicy) | Both | All | Uniform sample from a discrete Δv grid. Default `MCTSPolicy.opponent_model`. |
| [`MCTSPolicy`](#mctspolicy) | Both | All | JAX-native classic UCT search via `mctx`. |
| [`LeadInterceptPursuer`](#leadinterceptpursuer) | Heuristic | PE, LBG | Closes on opponent's predicted next-step position. |
| [`OrthogonalEvader`](#orthogonalevader) | Heuristic | PE | Thrusts perpendicular to relative velocity. |
| [`SunTrackerBlocker`](#suntrackerblocker) | Heuristic | SB | Maintains Sun-line geometry vs. opponent. |
| [`JitteredPolicy`](#jitteredpolicy) | Heuristic (wrapper) | All | Wraps any policy and adds Gaussian jitter. |
| [`HeuristicWithFallbackPolicy`](#heuristicwithfallbackpolicy) | Controlled (wrapper) | All | Confidence-gated routing between primary and fallback. |
| [`CompositeActionPolicy`](#compositeactionpolicy) | Controlled (wrapper) | All | Sums actions from a base and offset policy. |

## Importing

```python
from orbital_game.policies import UniformRandomDiscretePolicy, ZeroControl
from orbital_game.policies.mcts import MCTSPolicy
from orbital_game.policies.heuristic import (
    LeadInterceptPursuer,
    OrthogonalEvader,
    SunTrackerBlocker,
    JitteredPolicy,
)
from orbital_game.policies.controlled import (
    HeuristicWithFallbackPolicy,
    CompositeActionPolicy,
)
```

## Class reference

### ZeroControl

::: orbital_game.policies.zero.ZeroControl

### UniformRandomDiscretePolicy

::: orbital_game.policies.uniform_random.UniformRandomDiscretePolicy

### MCTSPolicy

See [In depth → MCTS policy](../in-depth/mcts.md) for the full
walkthrough; the API entry below is the bare class reference.

::: orbital_game.policies.mcts.MCTSPolicy

### LeadInterceptPursuer

::: orbital_game.policies.heuristic.lead_intercept.LeadInterceptPursuer

### OrthogonalEvader

::: orbital_game.policies.heuristic.evader.OrthogonalEvader

### SunTrackerBlocker

::: orbital_game.policies.heuristic.sun_tracker.SunTrackerBlocker

### JitteredPolicy

::: orbital_game.policies.heuristic.jittered.JitteredPolicy

### HeuristicWithFallbackPolicy

::: orbital_game.policies.controlled.heuristic_with_fallback.HeuristicWithFallbackPolicy

### CompositeActionPolicy

::: orbital_game.policies.controlled.composite_action.CompositeActionPolicy
