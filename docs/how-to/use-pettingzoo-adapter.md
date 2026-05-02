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

Each channel returns a per-pair tensor of shape `(N_self, N_total, m)`.
The adapter takes row `i` for agent `<side>_i` and concatenates across
channels: agent `i` sees only the i-th observer's view of every tracked
entity, flattened to a 1-D vector. Different agents on the same side
see different observations.

For per-vehicle channels like `OnboardGPSObservation` this gives each
vehicle its own GPS reading. For broadcast channels like
`FullObservation` every observer's row is identical, so all agents on
a side end up with the same vector — consistent with the channel's
semantics.

See [In depth → Observations](../in-depth/observations.md) for the
shape diagrams.

## Termination

When `episode_done` fires, every agent has `terminated=True`
simultaneously and the adapter sets `self.agents = []` — the Parallel
API signal for "loop is done." Call `reset()` to restart.

## See also

- [API reference → Adapters](../api/adapters.md).
