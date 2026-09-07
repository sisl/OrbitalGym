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
| [`GlideslopeToLady`](#glideslopetolady) | Heuristic | LBG | Saturation-aware glideslope to the lady plus Gaussian repulsion from opponents. |
| [`GlideslopeIntercept`](#glideslopeintercept) | Heuristic | LBG, PE | Saturation-aware glideslope onto the nearest opponent's state. |
| [`PhasedBandit`](#phasedbandit) | Heuristic | LBG | Coast, transfer onto a standoff ring, hold there for free, then commit to the lady. |
| [`JitteredPolicy`](#jitteredpolicy) | Heuristic (wrapper) | All | Wraps any policy and adds Gaussian jitter. |
| [`HeuristicWithFallbackPolicy`](#heuristicwithfallbackpolicy) | Controlled (wrapper) | All | Confidence-gated routing between primary and fallback. |
| [`CompositeActionPolicy`](#compositeactionpolicy) | Controlled (wrapper) | All | Sums actions from a base and offset policy. |

## Importing

```python
from orbitalgym.policies import UniformRandomDiscretePolicy, ZeroControl
from orbitalgym.policies.mcts import MCTSPolicy
from orbitalgym.policies.heuristic import (
    GlideslopeIntercept,
    GlideslopeToLady,
    LeadInterceptPursuer,
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

### GlideslopeToLady

Both glideslope policies command the closing speed as the smallest of three
terms: a linear glideslope `rho / slope_s + arrival_mps`, the braking curve
`sqrt(2 a_brake rho)` the per-step budget can shed, and the sustainable speed
`hcw_fraction * max_dv_mps / (2 n dt)`. The last term exists because the
Coriolis coupling of HCW dynamics charges about `2 n v dt` of impulse per step
to hold a relative velocity `v` that is not natural motion, so a vehicle that
commands more than its budget can hold walks off the line of sight and the
range grows instead of closing.

`max_dv_mps / (2 n dt)` bounds the Coriolis term alone, not the whole cost of
the approach. The `3 n^2 x` gradient term, the rotation of the line of sight as
the range closes, and the finite step each add to or subtract from it, so the
law spends only `hcw_fraction` of the bound and keeps the rest as margin. The
default 0.5 is measured: over eight start phases on a 2 km ring offset
along-track by 20 km and by 40 km, `hcw_fraction = 1.0` arrives in six of the
eight 20 km cases and none of the 40 km ones, `0.75` in eight and six, and
`0.5` in all sixteen. At a 0.4545 m/s per-step cap, a 10 s step and
`n = 1.078e-3 rad/s`, that leaves a sustainable speed of 10.5 m/s, which binds
beyond about 2.4 km.

::: orbitalgym.policies.heuristic.glideslope.GlideslopeToLady

### GlideslopeIntercept

::: orbitalgym.policies.heuristic.glideslope.GlideslopeIntercept

### PhasedBandit

A bandit on a schedule: it stays quiet for `t_coast_s`, spends fuel once to
move onto a standoff ring of radial amplitude `standoff_m`, waits there for
`t_hold_s`, and only then closes on the lady with the
`GlideslopeToLady` command. Timing and information matter to the
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
`settle_tol_mps=0.2` settles in 510 to 630 s and holds the amplitude within
47 to 75 percent of the standoff over an orbit; the default 0.02 settles in
750 to 840 s and keeps the hold within 3.4 to 7.7 percent.

Avoidance applies only in the commit phase. The transfer glides onto the
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
