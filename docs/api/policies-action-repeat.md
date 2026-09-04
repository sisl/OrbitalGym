# `orbitalgym.policies.action_repeat`

Wrapper that calls an inner `Policy` every `repeat` env steps and
re-emits its command in between. Pair it with
`POMDPAdapter(env, action_repeat=repeat)` so the trajectory the env
flies matches the one the planner searched. See
[How-to → Use the POMDP adapter](../how-to/use-pomdp-adapter.md).

## Policy state

::: orbitalgym.policies.action_repeat.ActionRepeatState

## Wrapper

::: orbitalgym.policies.action_repeat.ActionRepeatPolicy
