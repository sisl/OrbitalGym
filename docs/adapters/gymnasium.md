# Gymnasium adapter

`GymnasiumAdapter` wraps `SingleAgentView` as a `gymnasium.Env`. The controlled side (default `Side.GUARD`) is the agent; the opposite side is driven internally by `cfg.<side>_scripted_policy`.

## Install

```bash
pip install orbital-game[gymnasium]
```

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
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
```

## Action and observation spaces

Both spaces are unbounded `Box(float32)`. The action space is flat — `(n_controlled * action_dim,)` — so single-agent learners see a 1-D vector. The observation space's shape depends on the wrapped observation function; for the bundled `FullObservation` (which is `PER_SIDE` scope) it's the layout's flat dim.

For multiple controlled vehicles with decentralized control instead of a centralized concatenation, prefer the [PettingZoo adapter](pettingzoo.md).

The adapter accepts either a 1-D `(n_controlled * action_dim,)` array or a 2-D `(n_controlled, action_dim)` array — it reshapes internally before handing off to the JAX core.

## Termination semantics

`reward` is a Python `float`. `terminated` reflects `episode_done` — both `max_steps` exhaustion and game-specific conditions count as terminations here, so `truncated` is always `False`. The env's `TerminationFn` already accounts for the step limit, and the adapter does not separate truncation from natural termination.

## Switching the controlled side

```python
import dataclasses
from orbital_game import Side, make_lady_bandit_guard

cfg = make_lady_bandit_guard()
cfg = dataclasses.replace(cfg, controlled_side=Side.BANDIT)
env = GymnasiumAdapter(OrbitalGameEnv(cfg))
```

Now `cfg.guard_scripted_policy` drives the guard; the adapter's action and observation are the bandit's.

## Customizing the scripted opponent

Replace either side's scripted policy with anything that satisfies the `Policy` protocol — the callable `(ps, obs, key, t) → (action, ps')`:

```python
cfg = dataclasses.replace(cfg, bandit_scripted_policy=MyPursuitPolicy(...))
```

## Determinism

Given the same seed, two adapter instances produce identical observations and actions. The seed flows into both the env's IC sampling and the scripted opponent's RNG.
