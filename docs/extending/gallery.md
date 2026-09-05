# Gallery

Every named, importable policy that ships with `orbitalgym`. Use a
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
| [`LQRGoToLadyWithAvoidance`](#lqrgotoladywithavoidance) | Heuristic | LBG | In-plane HCW LQR to the lady plus Gaussian repulsion from opponents. |
| [`LQRIntercept`](#lqrintercept) | Heuristic | LBG, PE | In-plane HCW LQR driving the relative state to the nearest opponent to zero. |
| [`JitteredPolicy`](#jitteredpolicy) | Heuristic (wrapper) | All | Wraps any policy and adds Gaussian jitter. |
| [`HeuristicWithFallbackPolicy`](#heuristicwithfallbackpolicy) | Controlled (wrapper) | All | Confidence-gated routing between primary and fallback. |
| [`CompositeActionPolicy`](#compositeactionpolicy) | Controlled (wrapper) | All | Sums actions from a base and offset policy. |

## Importing

```python
from orbitalgym.policies import UniformRandomDiscretePolicy, ZeroControl
from orbitalgym.policies.mcts import MCTSPolicy
from orbitalgym.policies.heuristic import (
    LeadInterceptPursuer,
    LQRGoToLadyWithAvoidance,
    LQRIntercept,
    OrthogonalEvader,
    SunTrackerBlocker,
    JitteredPolicy,
)
from orbitalgym.policies.controlled import (
    HeuristicWithFallbackPolicy,
    CompositeActionPolicy,
)
```

## Class reference

### ZeroControl

::: orbitalgym.policies.zero.ZeroControl

### UniformRandomDiscretePolicy

::: orbitalgym.policies.uniform_random.UniformRandomDiscretePolicy

### MCTSPolicy

See [In depth → MCTS policy](../in-depth/mcts.md) for the full
walkthrough; the API entry below is the bare class reference.

::: orbitalgym.policies.mcts.MCTSPolicy

### LeadInterceptPursuer

::: orbitalgym.policies.heuristic.lead_intercept.LeadInterceptPursuer

### OrthogonalEvader

::: orbitalgym.policies.heuristic.evader.OrthogonalEvader

### SunTrackerBlocker

::: orbitalgym.policies.heuristic.sun_tracker.SunTrackerBlocker

### LQRGoToLadyWithAvoidance

::: orbitalgym.policies.heuristic.lqr_avoid.LQRGoToLadyWithAvoidance

### LQRIntercept

::: orbitalgym.policies.heuristic.lqr_intercept.LQRIntercept

### JitteredPolicy

::: orbitalgym.policies.heuristic.jittered.JitteredPolicy

### HeuristicWithFallbackPolicy

::: orbitalgym.policies.controlled.heuristic_with_fallback.HeuristicWithFallbackPolicy

### CompositeActionPolicy

::: orbitalgym.policies.controlled.composite_action.CompositeActionPolicy
