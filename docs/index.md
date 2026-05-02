# orbital-game

A JAX-native framework for guard / bandit decision-making in orbital scenarios.

Guard and bandit are symmetric agents: either side can be the learning agent, with the opposite side driven by a scripted policy. The same symmetric core drives single-agent (Gymnasium-shaped), multi-agent (PettingZoo-shaped), and POMDP-shaped views of the same underlying game. Four scenarios ship in the box: Lady-Bandit-Guard, Pursuit-Evasion, Sun-Blocking, and Observation-Blocking.

## Where to start

- New here? [Getting started](getting-started.md) walks from install to a rendered trajectory plot in five minutes.
- Want the mental model? [Concepts](concepts.md) explains sides, per-side outputs, and observation/reward scope.
- Looking for a specific game? Browse the [game guides](games/index.md).
- Plugging into a framework? See the [adapter how-tos](adapters/index.md).
- Writing custom observations, rewards, or games? See [Extending](extending/index.md).
- Looking up a type or function? Browse the [API reference](api/index.md).
- Curious about the design? [Architecture](architecture.md) sketches the layered decomposition.
