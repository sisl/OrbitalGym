# Gallery

Every named, importable policy that ships with `orbital-game`. Use a
class directly, or read its source as a starting point for your own.

## Roster

| Class | Role | Best in | Summary |
|---|---|---|---|
| [`ZeroControl`](#zerocontrol) | Both | All | Always emits zero — the baseline default. |
| [`LeadInterceptPursuer`](#leadinterceptpursuer) | Heuristic | PE, LBG | Closes on opponent's predicted next-step position. |
| [`OrthogonalEvader`](#orthogonalevader) | Heuristic | PE | Thrusts perpendicular to relative velocity. |
| [`SunTrackerBlocker`](#suntrackerblocker) | Heuristic | SB | Maintains Sun-line geometry vs. opponent. |
| [`JitteredPolicy`](#jitteredpolicy) | Heuristic (wrapper) | All | Wraps any policy and adds Gaussian jitter. |
| [`HeuristicWithFallbackPolicy`](#heuristicwithfallbackpolicy) | Controlled (wrapper) | All | Confidence-gated routing between primary and fallback. |
| [`BeliefConditionedPolicy`](#beliefconditionedpolicy) | Controlled (wrapper) | All | Pre-pends belief vector to obs before delegating. |
| [`CompositeActionPolicy`](#compositeactionpolicy) | Controlled (wrapper) | All | Sums actions from a base and offset policy. |

## Importing

```python
from orbital_game.policies import ZeroControl
from orbital_game.policies.heuristic import (
    LeadInterceptPursuer,
    OrthogonalEvader,
    SunTrackerBlocker,
    JitteredPolicy,
)
from orbital_game.policies.controlled import (
    HeuristicWithFallbackPolicy,
    BeliefConditionedPolicy,
    CompositeActionPolicy,
)
```

## Class reference

### ZeroControl

::: orbital_game.policies.zero.ZeroControl

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

### BeliefConditionedPolicy

::: orbital_game.policies.controlled.belief_conditioned.BeliefConditionedPolicy

### CompositeActionPolicy

::: orbital_game.policies.controlled.composite_action.CompositeActionPolicy
