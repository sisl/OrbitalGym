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

::: orbital_game.observations.range_limited.RangeLimitedObservation

::: orbital_game.observations.conical.ConicalObservation

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

### Protocols

::: orbital_game.belief.base.Belief

::: orbital_game.belief.base.BeliefInitializer

::: orbital_game.belief.base.BeliefUpdater

### Kalman filter

::: orbital_game.belief.kf.KFBelief

::: orbital_game.belief.kf.KFBeliefUpdater

::: orbital_game.belief.kf.KFFromTruthInitializer

::: orbital_game.belief.kf.KFUniformDefaultInitializer

### Extended Kalman filter

::: orbital_game.belief.ekf.EKFBelief

::: orbital_game.belief.ekf.EKFBeliefUpdater

::: orbital_game.belief.ekf.EKFFromTruthInitializer

::: orbital_game.belief.ekf.EKFUniformDefaultInitializer

### Particle filter

::: orbital_game.belief.pf.ParticleFilterBelief

::: orbital_game.belief.pf.ParticleFilterBeliefUpdater

::: orbital_game.belief.pf.ParticleFilterFromTruthInitializer

::: orbital_game.belief.pf.ParticleFilterRingInitializer

## State

::: orbital_game.state.layout.StateLayout

::: orbital_game.state.assemble.build_state_class

::: orbital_game.state.components.RTState

::: orbital_game.state.components.RTNState

::: orbital_game.state.components.ECIState

::: orbital_game.state.components.Mass

::: orbital_game.state.components.Power

::: orbital_game.state.components.Attitude

::: orbital_game.state.components.BodyRates

::: orbital_game.state.components.AppliedDV

::: orbital_game.state.components.AppliedTorque

## Action components

::: orbital_game.actions.components.ActionComponent

::: orbital_game.actions.components.ImpulsiveManeuver

::: orbital_game.actions.components.Communicate

::: orbital_game.actions.components.AttitudeControl

## Attitude dynamics

::: orbital_game.dynamics.attitude.AttitudeParams

::: orbital_game.dynamics.attitude.rigid_body_attitude_step

## IC sampling

::: orbital_game.sampling.spec.ICSpec

::: orbital_game.sampling.side.RelativeKeplerian

::: orbital_game.sampling.side.RelativeEllipse

::: orbital_game.sampling.mass.ConstantMass

::: orbital_game.sampling.mass.UniformMass

::: orbital_game.sampling.validators.MinSeparation

::: orbital_game.sampling.validators.MaxRange

::: orbital_game.sampling.attitude.IdentityAttitude

::: orbital_game.sampling.attitude.FixedAttitude

::: orbital_game.sampling.attitude.UniformAttitude

::: orbital_game.sampling.attitude.UniformBodyRates

::: orbital_game.sampling.attitude.UniformAttitudeAndRates
