# Architecture

A condensed design overview. The full design spec lives at [`superpowers/specs/2026-05-01-orbital-game-api-overhaul-design.md`](https://github.com/duncaneddy/orbital-game/blob/main/superpowers/specs/2026-05-01-orbital-game-api-overhaul-design.md).

## Layered design

```
┌──────────────────────────────────────────────────────────────────┐
│  External Frameworks                                              │
│  Gymnasium  │  PettingZoo (Parallel)  │  POMDPPlanners-shape     │
└──────┬──────────────┬──────────────────┬───────────────────────--┘
       │              │                  │
┌──────▼──────────────▼──────────────────▼─────────────────────────┐
│  adapters/  (optional extras — host-array boundary)               │
│  GymnasiumAdapter | PettingZooAdapter | POMDPAdapter              │
└──────┬───────────────────────────────────────────────────────────-┘
       │
┌──────▼───────────────────────────────────────────────────────────┐
│  env.SingleAgentView  (controlled-side projection)                │
└──────┬───────────────────────────────────────────────────────────-┘
       │
┌──────▼───────────────────────────────────────────────────────────┐
│  env.OrbitalGameEnv  (symmetric JAX core — pure functions)        │
│  step(key, state, Actions{guard, bandit}) → StepOutput            │
└──────┬───────────────────────────────────────────────────────────-┘
       │
┌──────▼───────────────────────────────────────────────────────────┐
│  Pluggable protocols (role-agnostic):                             │
│  Dynamics | Actuator | ObservationFn | RewardFn | TerminationFn   │
│  Policy | BeliefInitializer | BeliefUpdater                       │
└──────┬───────────────────────────────────────────────────────────-┘
       │
┌──────▼───────────────────────────────────────────────────────────┐
│  Pure JAX building blocks                                          │
│  state/ | dynamics/ | actuators/ | sampling/                       │
└──────────────────────────────────────────────────────────────────-┘
```

## Architectural invariants

The design rests on six invariants that the test suite enforces:

**`OrbitalGameEnv` is the single canonical core.** Every adapter is a wrapper, never a parallel reimplementation. The cross-adapter consistency test proves a Gymnasium step and a direct `env.step` produce the same numerical result given the same key and actions.

**JAX purity stops at the adapter boundary.** Inside the core: `jit`/`vmap`/`scan`-clean, pytree-only, no Python state. Inside adapters: host arrays, mutable attributes, Python exceptions are all fine.

**Policies are side-agnostic.** The same `Policy` class can be wired as a guard scripted opponent, a bandit scripted opponent, or a learned-side fallback — only the `(n_vehicles, action_dim)` injection changes.

**Games are typed configs, not classes-with-logic.** All four games run on the same core; what varies is the `Game` dataclass on `ScenarioConfig` plus the matching reward, termination, and observation choices.

**World bodies are computed, not stored.** Sun ECI and Earth-target ECI are recovered from `(t, epoch, cfg.game)` inside reward and observation. `EnvState` shape is invariant across all games.

**Per-side everything.** Observations, rewards, beliefs, scripted policies, and `done` flags are all per-side. The "agent perspective" is a property of the adapter, not the core.

For the conceptual treatment of these abstractions, see [Concepts](concepts.md). For type signatures, see the [API reference](api/index.md).
