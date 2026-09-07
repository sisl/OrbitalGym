# Components

Protocols and reference implementations for the env's swappable building blocks.

## Policies

A single `Policy` protocol covers reactive obs-only policies and
state-aware planners alike. State-aware policies (e.g. `MCTSPolicy`)
require an `agent_view` carrying enough state to step a model — wire
through `belief_rollout` with a Belief whose `mean` is the flat state,
or pass the flat state directly when calling the policy manually.

::: orbitalgym.policies.base.Policy

::: orbitalgym.policies.zero.ZeroControl

::: orbitalgym.policies.uniform_random.UniformRandomDiscretePolicy

### Search policies

Both planners take a `coordination` mode for multi-vehicle teams. `"joint"`
plans the whole fleet at once: one MCTS tree over the joint action space, or
one MPPI sampler over the fleet's Delta-v sequence. `"independent"` plans one
vehicle at a time and models the teammates with `teammate_model`, the
same-side analogue of `opponent_model`; cost then grows linearly rather than
exponentially in the fleet size, and the per-vehicle results are stacked into
one command. The modes coincide for a single vehicle.

::: orbitalgym.policies.mcts.MCTSPolicy

::: orbitalgym.policies.mppi.MPPIPolicy

### Heuristic-policy gallery

::: orbitalgym.policies.heuristic.lead_intercept.LeadInterceptPursuer

::: orbitalgym.policies.heuristic.evader.OrthogonalEvader

::: orbitalgym.policies.heuristic.sun_tracker.SunTrackerBlocker

::: orbitalgym.policies.heuristic.jittered.JitteredPolicy

### Controlled-side wrappers

::: orbitalgym.policies.controlled.heuristic_with_fallback.HeuristicWithFallbackPolicy

::: orbitalgym.policies.controlled.composite_action.CompositeActionPolicy

## Observations

::: orbitalgym.observations.base.ObservationFn

::: orbitalgym.observations.reference.FullObservation

::: orbitalgym.observations.range_limited.RangeLimitedObservation

::: orbitalgym.observations.conical.ConicalObservation

## Rewards

::: orbitalgym.rewards.base.RewardFn

::: orbitalgym.rewards.base.RewardScope

::: orbitalgym.rewards.reference.DistanceToReferenceOrbit

::: orbitalgym.rewards.reference.ZeroReward

::: orbitalgym.rewards.lbg_zero_sum.LbgZeroSumReward

## Termination

::: orbitalgym.termination.base.TerminationFn

::: orbitalgym.termination.reference.MaxStepsOnly

::: orbitalgym.termination.lbg_events.LbgEventTermination

## Belief

### Protocols

::: orbitalgym.belief.base.Belief

::: orbitalgym.belief.base.BeliefInitializer

::: orbitalgym.belief.base.BeliefUpdater

### Kalman filter

::: orbitalgym.belief.kf.KFBelief

::: orbitalgym.belief.kf.KFBeliefUpdater

::: orbitalgym.belief.kf.KFFromTruthInitializer

::: orbitalgym.belief.kf.KFUniformDefaultInitializer

### Extended Kalman filter

::: orbitalgym.belief.ekf.EKFBelief

::: orbitalgym.belief.ekf.EKFBeliefUpdater

::: orbitalgym.belief.ekf.EKFFromTruthInitializer

::: orbitalgym.belief.ekf.EKFUniformDefaultInitializer

### Particle filter

::: orbitalgym.belief.pf.ParticleFilterBelief

::: orbitalgym.belief.pf.ParticleFilterBeliefUpdater

::: orbitalgym.belief.pf.ParticleFilterFromTruthInitializer

::: orbitalgym.belief.pf.ParticleFilterRingInitializer

::: orbitalgym.belief.pf.ParticleFilterTrackedInitializer

## State

::: orbitalgym.state.layout.StateLayout

::: orbitalgym.state.assemble.build_state_class

::: orbitalgym.state.components.RTState

::: orbitalgym.state.components.RTNState

::: orbitalgym.state.components.ECIState

::: orbitalgym.state.components.Mass

::: orbitalgym.state.components.Power

::: orbitalgym.state.components.Attitude

::: orbitalgym.state.components.BodyRates

::: orbitalgym.state.components.AppliedDV

::: orbitalgym.state.components.AppliedTorque

## Action components

::: orbitalgym.actions.components.ActionComponent

::: orbitalgym.actions.components.ImpulsiveManeuver

::: orbitalgym.actions.components.Communicate

::: orbitalgym.actions.components.AttitudeControl

## Attitude dynamics

::: orbitalgym.dynamics.attitude.AttitudeParams

::: orbitalgym.dynamics.attitude.rigid_body_attitude_step

## IC sampling

::: orbitalgym.sampling.spec.ICSpec

::: orbitalgym.sampling.side.RelativeKeplerian

::: orbitalgym.sampling.side.RelativeEllipse

::: orbitalgym.sampling.mass.ConstantMass

::: orbitalgym.sampling.mass.UniformMass

::: orbitalgym.sampling.validators.MinSeparation

::: orbitalgym.sampling.validators.MaxRange

::: orbitalgym.sampling.attitude.IdentityAttitude

::: orbitalgym.sampling.attitude.FixedAttitude

::: orbitalgym.sampling.attitude.UniformAttitude

::: orbitalgym.sampling.attitude.UniformBodyRates

::: orbitalgym.sampling.attitude.UniformAttitudeAndRates
