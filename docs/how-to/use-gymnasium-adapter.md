# Use the Gymnasium adapter

Wrap `OrbitalGameEnv` as a `gymnasium.Env` for single-agent RL training.

## Recipe

```python
import numpy as np
from orbital_game import OrbitalGameEnv, make_lady_bandit_guard
from orbital_game.adapters.gymnasium import GymnasiumAdapter

cfg = make_lady_bandit_guard()
env = GymnasiumAdapter(OrbitalGameEnv(cfg), seed=0)

obs, info = env.reset(seed=0)
done = False
total = 0.0
while not done:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated
```

## Spaces

Both `action_space` and `observation_space` are unbounded
`Box(float32)`. The action shape is flat:
`(n_controlled * action_dim,)`. The adapter accepts either flat or
`(n_controlled, action_dim)`-shaped input and reshapes internally.

For the bundled `FullObservation`, the observation shape is the
layout's flat dim.

## Termination semantics

`terminated` reflects `episode_done`. Both `max_steps` exhaustion and
game-specific conditions count as termination, so `truncated` is
always `False`. The env's `TerminationFn` already handles the step
limit.

## Switching the controlled side

See [Switch the controlled side](switch-controlled-side.md).

## Customising the opponent

See [Customize the opponent](customize-opponent.md).

## See also

- [API reference → Adapters](../api/adapters.md).
