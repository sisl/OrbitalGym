# Extending orbital-game

Almost every component of the env is swappable. Observation functions, reward functions, termination conditions, policies, IC samplers, validators, dynamics, actuators, and games are all pluggables — uniform `@register(<Key>.<MEMBER>) @dataclass(frozen=True)` classes that double as user-facing typed config and JAX-traceable runtime callable.

## Pluggable types

| Type | Role | Protocol | Reference impl |
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

## The convention

Every pluggable follows the same shape: a frozen dataclass that registers itself under an enum key. The dataclass fields are the user-facing knobs; `__call__` is the JAX-traceable runtime body.

```python
from dataclasses import dataclass
from orbital_game.registry import RewardFnKey, register
from orbital_game.rewards.base import RewardScope

@register(RewardFnKey.MY_REWARD)   # adds the enum member
@dataclass(frozen=True)
class MyReward:
    """One-line description of what this reward measures."""

    weight: float = 1.0
    cutoff_m: float = 100.0

    scope: RewardScope = RewardScope.PER_SIDE

    def __call__(self, prev_state, action, next_state, side, params, t):
        # Read game-specific knobs off params.game if the reward is game-coupled:
        #   if not isinstance(params.game, MyGame):
        #       raise TypeError(...)
        # Return a scope-shaped reward.
        ...
```

Any class registered with `@register(<Key>.<MEMBER>)` automatically round-trips through `ScenarioConfig.to_json()` / `from_json()`. The enum value is the type discriminator on serialization; `dataclasses.asdict` walks the knob fields. Game subclasses use `@register_game(<GameKey>.<MEMBER>)` instead — same pattern, separate registry.

Registration is only required if you need JSON round-trip. In-memory composition works without it.

## Worked examples

The bundled reference implementations are good templates:

- **Custom observation function** — `src/orbital_game/observations/reference.py:FullObservation`. Replace the body to mask, noise, or slice the state.
- **Custom reward** — `src/orbital_game/rewards/reference.py:DistanceToReferenceOrbit` for a simple game-agnostic case; `src/orbital_game/games/pursuit_evasion.py:PursuitEvasionReward` for the `isinstance(params.game, ...)` guard pattern when a reward is coupled to a specific game.
- **Custom termination** — `src/orbital_game/termination/reference.py:MaxStepsOrBreach`, and again the per-game variant in `pursuit_evasion.py`.
- **Custom policy** — `src/orbital_game/policies/library.py:ZeroControl`. The `n_vehicles` and `action_dim` fields default to 0 and are populated by the env at construction via `dataclasses.replace`.
- **Custom game** — all four `src/orbital_game/games/*.py` modules. Each game is a tuple of `(Game subtype, Reward, Termination if game-specific, builder)`.
