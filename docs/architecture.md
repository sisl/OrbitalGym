# Architecture

Condensed overview. The full design spec is at [`docs/superpowers/specs/2026-05-01-orbital-game-api-overhaul-design.md`](https://github.com/duncaneddy/orbital-game/blob/main/docs/superpowers/specs/2026-05-01-orbital-game-api-overhaul-design.md).

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
│  env.SingleAgentView  (controlled_side projection)                │
│  reads cfg.{guard,bandit}_scripted_policy + cfg.controlled_side   │
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

1. **`OrbitalGameEnv` is the single canonical core.** Every adapter is a wrapper, never a parallel reimplementation. `test_adapter_jax_consistency` enforces this.
2. **JAX purity stops at the adapter boundary.** Inside the core: `jit`/`vmap`/`scan`-clean, pytree-only. Inside adapters: host arrays, Python state, Python exceptions are all fine.
3. **Policies are side-agnostic.** Same `Policy` class can be wired as guard scripted-opponent, bandit scripted-opponent, or learned-side fallback — only the `(n_vehicles, action_dim)` injection changes.
4. **Games are typed configs, not classes-with-logic.** All four games run on the same core; what varies is the `Game` dataclass on `ScenarioConfig` plus the matching reward/termination/observation choices.
5. **World bodies are computed, not stored.** Sun ECI and Earth-target ECI are recovered from `(t, epoch, cfg.game)` inside reward/observation. EnvState shape is invariant across all games.
6. **Per-side everything.** Observations, rewards, beliefs, scripted policies, dones are all per-side. The "agent perspective" is a property of the *adapter*, not the core.

## Package layout

```
src/orbital_game/
├── config.py              # ScenarioConfig
├── reference_orbit.py     # ReferenceOrbitState (renamed from HVAState)
├── registry.py            # Enum keys + register/resolve
├── state/                 # state pytrees + StateLayout
├── dynamics/              # HCW step functions
├── actuators/             # impulsive actuator
├── observations/          # ObservationFn protocol + reference impls
├── rewards/               # RewardFn protocol + reference impls
├── termination/           # TerminationFn protocol + reference impls
├── policies/              # Role-agnostic Policy library
├── belief/                # Per-side belief initializers + updaters
├── sampling/              # ICSpec, side samplers, validators
├── env/
│   ├── core.py            # OrbitalGameEnv (symmetric JAX core)
│   ├── single_agent.py    # SingleAgentView projection
│   └── types.py           # EnvState, Actions, StepOutput, BySide, Side
├── games/                 # Game subtypes + builders
├── adapters/              # Optional-extra framework adapters
├── rollout.py             # Per-side Trajectory + symmetric scan
├── logging/               # HDF5 reader/writer
└── viz/                   # Plotting
```

## Phased implementation

The current state is the result of a four-phase API overhaul:

| Phase | Scope | Tests at exit |
|---|---|---|
| **0** | Rename `defender→guard`, `intruder→bandit`, `hva→reference_orbit` | 139 (existing tests under new names) |
| **1** | Symmetric core + `BySide`/`Actions`/`Trajectory` + `SingleAgentView` + role-agnostic Policy | 170 (+31) |
| **2** | Game catalog: `Game` subtypes, `LBG/PE/SB/OB`, builders, serialization | 208 (+38) |
| **3** | Framework adapters: Gymnasium, PettingZoo, POMDPPlanners-shape | 229 (+21) |

The byte-identity regression test (`tests/test_phase_0_baseline.py`) was locked at the start of Phase 0 and continues to pass — every refactor preserved the exact numerical output of the reference scenario.

## Type system at a glance

| Type | What it is | Where it lives |
|---|---|---|
| `Side` | `StrEnum`: `GUARD`, `BANDIT` | `env/types.py` |
| `BySide` | One-per-side flax dataclass | `env/types.py` |
| `Actions` | Wrapper holding `BySide` of per-side action arrays | `env/types.py` |
| `SideOutput` | Per-side `(obs, reward, done)` bundle | `env/types.py` |
| `StepOutput` | `(state, BySide[SideOutput], episode_done, info)` | `env/types.py` |
| `EnvState` | `(t, step, guards, bandits, reference_orbit, ic_valid)` | `env/core.py` |
| `Trajectory` | `(env_state, BySide[SideTrajectory], episode_done, controlled_side)` | `env/types.py` |

See [Concepts](concepts.md) for the conceptual treatment.
