# Extending orbital-game

`orbital-game` is built around composable pluggable types — observation functions, reward functions, termination conditions, policies, IC samplers, validators, dynamics, actuators, and games are all swappable.

The pattern is uniform: each pluggable is a `@register(<Key>.<MEMBER>) @dataclass(frozen=True)` class that is simultaneously the user-facing config (typed knobs) and the JAX-traceable runtime callable (`__call__`).

## Pluggable types

| Type | Role | Protocol | Example reference impl |
|---|---|---|---|
| `Dynamics` | Per-step state propagator | `(state, dv, params, dt) → next_state` | `hcw_rt_step`, `hcw_rtn_step` |
| `Actuator` | Action → applied control + propellant | `apply(action, state, params, dt)` | `ImpulsiveActuator` |
| `ObservationFn` | State → per-side observation | `(state, side, params, key, t) → obs` | `FullObservation` |
| `RewardFn` | (prev, action, next) → per-side reward | `(prev, action, next, side, params, t) → reward` | `DistanceToReferenceOrbit` |
| `TerminationFn` | Episode-level termination | `(state, params, t) → bool` | `MaxStepsOrBreach` |
| `Policy` | Role-agnostic action generator | `(ps, obs, key, t) → (action, ps')` | `ZeroControl` |
| `BeliefInitializer` | Initialize belief from state | `(env_state, side, key) → belief` | `GaussianBeliefInitializer` |
| `BeliefUpdater` | Update belief on new obs | `(belief, obs, action, side, key) → belief'` | `GaussianKalmanUpdater` |
| `SideSampler` | Sample one side's IC | `(config, key, n_vehicles, components, class_name) → state` | `RelativeKeplerian`, `RelativeEllipse` |
| `Validator` | Post-sample IC check | `(config, guards, bandits) → bool` | `MinSeparation`, `MaxRange` |
| `Game` | Typed game-specific knob bundle | (just a dataclass) | `LadyBanditGuard`, `PursuitEvasion` |

## Pluggable convention

```python
from dataclasses import dataclass
from orbital_game.registry import RewardFnKey, register
from orbital_game.rewards.base import RewardScope


@register(RewardFnKey.MY_REWARD)   # adds an enum member to RewardFnKey
@dataclass(frozen=True)
class MyReward:
    """Concise docstring of what this reward measures."""

    # User knobs:
    weight: float = 1.0
    cutoff_m: float = 100.0
    # No env-populated fields needed for rewards (those are policy-only).

    scope: RewardScope = RewardScope.PER_SIDE

    def __call__(self, prev_state, action, next_state, side, params, t):
        # Read game knobs off params.game if needed:
        # if not isinstance(params.game, MyGame):
        #     raise TypeError(...)
        # Compute and return scope-shaped reward.
        ...
```

Any class registered with `@register(<Key>.<MEMBER>)` round-trips through `ScenarioConfig.to_json()` / `from_json()` automatically. The enum value is the type discriminator; `dataclasses.asdict` walks the knob fields.

## Worked examples

The bootstrap reference implementations are good templates:

- **Custom observation function**: see `src/orbital_game/observations/reference.py:FullObservation`. Replace the body to mask/noise/slice the state.
- **Custom reward**: see `src/orbital_game/rewards/reference.py:DistanceToReferenceOrbit` and `src/orbital_game/games/pursuit_evasion.py:PursuitEvasionReward`. The latter shows the `isinstance(params.game, ...)` guard pattern.
- **Custom termination**: see `src/orbital_game/termination/reference.py:MaxStepsOrBreach` and `src/orbital_game/games/pursuit_evasion.py:PursuitEvasionTermination`.
- **Custom policy**: see `src/orbital_game/policies/library.py:ZeroControl`. The `n_vehicles`/`action_dim` fields default to 0 and are populated by the env at construction.
- **Custom game**: see all four `src/orbital_game/games/*.py` modules. Each game is `(Game subtype) + (Reward) + (Termination if needed) + (builder)`.

## Topics

Full per-topic pages are work-in-progress. The patterns above are what to follow until the dedicated pages land:

- Custom observation functions
- Custom reward functions
- Custom termination conditions
- Custom IC samplers and validators
- Custom dynamics
- Custom actuators
- Custom state components
- Custom games
- Custom policies (stateless, stateful, MPC, learned)
- Registry mechanics

## Registry mechanics

`@register(<Key>.<MEMBER>)` populates two dicts:
- forward: `<Key>.<MEMBER> → class` for deserialization
- reverse: `class → (<Key>.<MEMBER>.value, <Key>)` for serialization

For Game subclasses, use `@register_game(<GameKey>.<MEMBER>)` instead — same pattern with separate `_GAME_REGISTRY`/`_GAME_REVERSE` dicts.

Register only when you need JSON round-trip. In-memory composition doesn't require registration.
