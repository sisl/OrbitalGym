# PettingZoo adapter

`PettingZooAdapter` wraps `OrbitalGameEnv` as a `pettingzoo.ParallelEnv`. Every vehicle is a separate agent; all agents step simultaneously each cycle.

## Install

```bash
pip install orbital-game[pettingzoo]
```

## Agent IDs

Each vehicle is named `<side>_<index>`. For a 1v1 game the agents are `['guard_0', 'bandit_0']`; for `n_guards=2, n_bandits=1` they are `['guard_0', 'guard_1', 'bandit_0']`.

## Minimal example

```python
import numpy as np
from orbital_game import OrbitalGameEnv, make_pursuit_evasion
from orbital_game.adapters.pettingzoo import PettingZooAdapter

cfg = make_pursuit_evasion()
env = PettingZooAdapter(OrbitalGameEnv(cfg), seed=0)

obs_dict, info_dict = env.reset(seed=0)
total = {agent: 0.0 for agent in env.agents}

while env.agents:
    actions = {
        agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
        for agent in env.agents
    }
    obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(actions)
    for agent, r in reward_dict.items():
        total[agent] += r
```

In a zero-sum game like Pursuit-Evasion, `total['guard_0'] + total['bandit_0']` is approximately zero (exactly zero if all rewards are summed losslessly).

## How scope maps to per-agent observations

For `PER_VEHICLE` scope, the leading vehicle axis is sliced into per-agent observations: `guard_0.obs = side_obs[0]`, `guard_1.obs = side_obs[1]`. For `PER_SIDE` scope (the bundled `FullObservation` default), the same observation is broadcast to every agent on the side.

## Episode termination

When `episode_done` fires, every agent has `terminated=True` simultaneously. The adapter then clears `self.agents = []`, which is the Parallel API signal for "loop is done." Call `reset(seed=...)` to start another episode.

## Determinism

Same as the Gymnasium adapter — given a seed, two instances produce identical observations.
