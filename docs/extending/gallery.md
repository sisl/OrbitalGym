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
| [`PhasedBandit`](#phasedbandit) | Heuristic | LBG | Coast, transfer onto a standoff ring, hold there for free, then commit to the lady. |
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
    PhasedBandit,
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

### PhasedBandit

A bandit on a schedule: it stays quiet for `t_coast_s`, spends fuel once to
move onto a standoff ring of radial amplitude `standoff_m`, waits there for
`t_hold_s`, and only then closes on the lady with the
`LQRGoToLadyWithAvoidance` command. Timing and information matter to the
guards, and a game runs across several ground passes rather than resolving in
one approach.

The hold is free because the standoff ring is a natural bounded relative
orbit — the same 2:1 ellipse family `RelativeEllipse` samples for the initial
conditions, with no secular drift — so a vehicle that reaches it coasts around
it. Holding station at a fixed RTN point would have to be paid for every step.
The transfer therefore targets the ellipse *state*, position and velocity, at
the vehicle's current phase angle, and ends only once both are inside
`settle_tol_m` and `settle_tol_mps`. The velocity tolerance sets how tight the
hold is: a residual `dv` leaves an along-track drift of about
`2 pi dv / (1.5 n)` metres per orbit. Measured on a 2 km standoff ring
approached from 3 km at a 0.4545 m/s per-step cap, over six start phases,
`settle_tol_mps=0.2` settles in 470 to 840 s and holds the amplitude within
1.6 to 5.5 percent of the standoff over an orbit; the default 0.02 buys margin
on that, keeping the hold within about 5 percent across start phases.

Avoidance applies only in the commit phase. The transfer regulates onto the
ring with no repulsion term, so a nearby guard cannot push the vehicle off the
target it is trying to reach.

Phases are per-vehicle and live in the policy state, so each bandit in a fleet
runs its own schedule; transitions are `jnp.where` selects on traced values,
so the policy runs under `jit` and `vmap`.

::: orbitalgym.policies.heuristic.phased_bandit.PhasedBandit

### JitteredPolicy

::: orbitalgym.policies.heuristic.jittered.JitteredPolicy

### HeuristicWithFallbackPolicy

::: orbitalgym.policies.controlled.heuristic_with_fallback.HeuristicWithFallbackPolicy

### CompositeActionPolicy

::: orbitalgym.policies.controlled.composite_action.CompositeActionPolicy
