"""Composable action components.

Each ActionComponent contributes a set of named fields to a side's per-scenario
Command class (built by actions.assemble.build_command_class) and a stateful
`apply` method that mutates EnvState in response to its slice of the command.

Components are orthogonal: a side's command pytree is the union of its
configured components' fields, and env.step() folds over them in registration
order. ImpulsiveManeuver is the batteries-included default that encapsulates
impulsive Δv + propellant burn + truth-dynamics propagation.
"""
