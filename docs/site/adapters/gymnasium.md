# Gymnasium Adapter

`GymnasiumAdapter` wraps `SingleAgentView` as a `gymnasium.Env`. The controlled side (default `Side.GUARD`) is the agent; the opposite side is driven by `cfg.<side>_scripted_policy` internally.

## Install

```bash
pip install orbital-game[gymnasium]
```

`gymnasium` is also pulled in transitively via `gymnax`, but the explicit extra makes the dependency intent clear.

## Minimal example

```python
import numpy as np

from orbital_game import OrbitalGameEnv, make_lady_bandit_guard
from orbital_game.adapters.gymnasium import GymnasiumAdapter

cfg = make_lady_bandit_guard()
env = GymnasiumAdapter(OrbitalGameEnv(cfg), seed=0)

obs, info = env.reset(seed=0)
done = False
total_reward = 0.0
while not done:
    action = np.zeros(env.action_space.shape, dtype=np.float32)  # zero-control demo
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"episode reward: {total_reward}")
```

## Action and observation spaces

Both spaces are `Box([-inf, +inf], dtype=float32)`:

- `action_space.shape == (n_controlled * action_dim,)` — flattened from the per-vehicle (N, action_dim) shape so single-agent learners see a 1-D vector.
- `observation_space.shape` — inferred from the wrapped observation function. For the default `FullObservation` (`PER_SIDE` scope), this is the layout's flat dim.

If you have multiple controlled vehicles and want decentralized control instead of a centralized concatenation, prefer the [PettingZoo](pettingzoo.md) adapter.

## Action shape conversion

The adapter accepts either:
- A 1-D numpy array `(n_controlled * action_dim,)` — flattened
- A 2-D numpy array `(n_controlled, action_dim)` — explicit per-vehicle

Internally the adapter reshapes to `(n_controlled, action_dim)` before passing to the JAX core.

## Reward, terminated, truncated

- `reward` is a Python `float`, materialized from the `()`-shape JAX array.
- `terminated` is a Python `bool` reflecting `episode_done` from the wrapped env (max steps OR game-specific termination condition).
- `truncated` is always `False` — the env's `TerminationFn` already accounts for max-step truncation; we don't distinguish step-limit truncation from natural termination.

## Verified against `gymnasium.utils.env_checker`

The adapter passes `gymnasium.utils.env_checker.check_env(adapter, skip_render_check=True, skip_close_check=True)`. The only warnings emitted are about unbounded `Box` spaces (`-inf, +inf`) — these are intentional and don't affect functionality.

## Switching the controlled side

To make the bandit the agent:

```python
import dataclasses
from orbital_game import Side, make_lady_bandit_guard

cfg = make_lady_bandit_guard()
cfg = dataclasses.replace(cfg, controlled_side=Side.BANDIT)
# Now cfg.guard_scripted_policy drives the guard (defaults to ZeroControl);
# the adapter's action and obs are the bandit's.
env = GymnasiumAdapter(OrbitalGameEnv(cfg))
```

To customize the scripted opponent:

```python
from orbital_game.policies.library import ZeroControl

# Replace ZeroControl with your own Policy implementation:
cfg = dataclasses.replace(cfg, bandit_scripted_policy=MyPursuitPolicy(...))
```

Any class that satisfies the `Policy` protocol (callable `(ps, obs, key, t) → (action, ps')`) works as a scripted policy.

## Determinism

The adapter is deterministic given a seed:

```python
env1 = GymnasiumAdapter(OrbitalGameEnv(cfg), seed=42)
env2 = GymnasiumAdapter(OrbitalGameEnv(cfg), seed=42)
obs1, _ = env1.reset(seed=42)
obs2, _ = env2.reset(seed=42)
assert np.array_equal(obs1, obs2)   # always true
```

## Source

- [`src/orbital_game/adapters/gymnasium/adapter.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/adapters/gymnasium/adapter.py)
- Tests: [`tests/test_adapter_gymnasium.py`](https://github.com/duncaneddy/orbital-game/blob/main/tests/test_adapter_gymnasium.py)
