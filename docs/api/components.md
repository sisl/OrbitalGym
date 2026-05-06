# Components

Protocols and reference implementations for the env's swappable building blocks.

## Policies

::: orbital_game.policies.base.Policy

::: orbital_game.policies.zero.ZeroControl

### Planners (state-aware, outside `lax.scan`)

`Planner` is the sibling protocol of `Policy` for tree-search /
MCTS-style controllers that need full env state and Python-level
control flow. Driven by `rollout_with_planner` (see [API → Core][1])
rather than the `lax.scan`-based `rollout`.

[1]: core.md

::: orbital_game.policies.planner.Planner

### Heuristic-policy gallery

::: orbital_game.policies.heuristic.lead_intercept.LeadInterceptPursuer

::: orbital_game.policies.heuristic.evader.OrthogonalEvader

::: orbital_game.policies.heuristic.sun_tracker.SunTrackerBlocker

::: orbital_game.policies.heuristic.jittered.JitteredPolicy

### Controlled-side wrappers

::: orbital_game.policies.controlled.heuristic_with_fallback.HeuristicWithFallbackPolicy

::: orbital_game.policies.controlled.belief_conditioned.BeliefConditionedPolicy

::: orbital_game.policies.controlled.composite_action.CompositeActionPolicy

## Observations

::: orbital_game.observations.base.ObservationFn

::: orbital_game.observations.reference.FullObservation

## Rewards

::: orbital_game.rewards.base.RewardFn

::: orbital_game.rewards.base.RewardScope

::: orbital_game.rewards.reference.DistanceToReferenceOrbit

## Termination

::: orbital_game.termination.base.TerminationFn

::: orbital_game.termination.reference.MaxStepsOrBreach

## Belief

::: orbital_game.belief.base.BeliefInitializer

::: orbital_game.belief.base.BeliefUpdater

## State

::: orbital_game.state.layout.StateLayout

::: orbital_game.state.assemble.build_state_class

## IC sampling

::: orbital_game.sampling.spec.ICSpec

::: orbital_game.sampling.side.RelativeKeplerian

::: orbital_game.sampling.side.RelativeEllipse

::: orbital_game.sampling.mass.ConstantMass

::: orbital_game.sampling.mass.UniformMass

::: orbital_game.sampling.validators.MinSeparation

::: orbital_game.sampling.validators.MaxRange
