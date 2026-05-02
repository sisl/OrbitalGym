# Use the PettingZoo adapter

Wrap `OrbitalGameEnv` as a `pettingzoo.ParallelEnv` for multi-agent
RL training where every vehicle is a separate agent.

## Install

```bash
pip install orbital-game[pettingzoo]
```

## Recipe

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

## Agent IDs

Each vehicle is `<side>_<index>`. For 1v1: `['guard_0', 'bandit_0']`.
For `n_guards=2, n_bandits=1`: `['guard_0', 'guard_1', 'bandit_0']`.

## Per-agent observations

Per-pair shape `(N_self, N_total, m)` from the bundled channels means
each agent sees its row of the per-pair tensor. See
[In depth → Observations](../in-depth/observations.md) for the shape
diagrams.

## Termination

When `episode_done` fires, every agent has `terminated=True`
simultaneously and the adapter sets `self.agents = []` — the Parallel
API signal for "loop is done." Call `reset()` to restart.

## See also

- [API reference → Adapters](../api/adapters.md).
