# Components

Protocols and reference implementations for the env's swappable building blocks.

## Policies

A single `Policy` protocol covers reactive obs-only policies and
state-aware planners alike. State-aware policies (e.g. `MCTSPolicy`)
require an `agent_view` carrying enough state to step a model — wire
through `belief_rollout` with a Belief whose `mean` is the flat state,
or pass the flat state directly when calling the policy manually.

::: orbital_game.policies.base.Policy

::: orbital_game.policies.zero.ZeroControl

::: orbital_game.policies.uniform_random.UniformRandomDiscretePolicy

### Search policies

::: orbital_game.policies.mcts.MCTSPolicy

### Heuristic-policy gallery

::: orbital_game.policies.heuristic.lead_intercept.LeadInterceptPursuer

::: orbital_game.policies.heuristic.evader.OrthogonalEvader

::: orbital_game.policies.heuristic.sun_tracker.SunTrackerBlocker

::: orbital_game.policies.heuristic.jittered.JitteredPolicy

### Controlled-side wrappers

::: orbital_game.policies.controlled.heuristic_with_fallback.HeuristicWithFallbackPolicy

::: orbital_game.policies.controlled.composite_action.CompositeActionPolicy

## Observations

::: orbital_game.observations.base.ObservationFn

::: orbital_game.observations.reference.FullObservation

## Rewards

::: orbital_game.rewards.base.RewardFn

::: orbital_game.rewards.base.RewardScope

::: orbital_game.rewards.reference.DistanceToReferenceOrbit

::: orbital_game.rewards.reference.ZeroReward

::: orbital_game.rewards.lbg_zero_sum.LbgZeroSumReward

## Termination

::: orbital_game.termination.base.TerminationFn

::: orbital_game.termination.reference.MaxStepsOnly

::: orbital_game.termination.lbg_events.LbgEventTermination

## Belief

::: orbital_game.belief.base.Belief

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
