# PettingZoo Adapter

`PettingZooAdapter` wraps `OrbitalGameEnv` as a `pettingzoo.ParallelEnv`. Every vehicle is a separate agent; all step simultaneously each cycle.

## Install

```bash
pip install orbital-game[pettingzoo]
```

## Agent IDs

Each vehicle is a separate agent, named `<side>_<index>`:

```python
adapter.possible_agents
# ['guard_0', 'guard_1', ..., 'bandit_0', 'bandit_1', ...]
```

For a 1v1 game, the agents are `['guard_0', 'bandit_0']`.

## Minimal example

```python
import numpy as np

from orbital_game import OrbitalGameEnv, make_pursuit_evasion
from orbital_game.adapters.pettingzoo import PettingZooAdapter

cfg = make_pursuit_evasion()
env = PettingZooAdapter(OrbitalGameEnv(cfg), seed=0)

obs_dict, info_dict = env.reset(seed=0)
total_rewards = {agent: 0.0 for agent in env.agents}

while env.agents:
    actions = {
        agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
        for agent in env.agents
    }
    obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(actions)
    for agent, r in reward_dict.items():
        total_rewards[agent] += r

print(total_rewards)
# In a zero-sum game like PE, total_rewards['guard_0'] + total_rewards['bandit_0'] ≈ 0
```

## Per-side scope handling

For `PER_VEHICLE`-scope observations: the leading vehicle axis is sliced into per-agent observations.

For `PER_SIDE`-scope observations (the default `FullObservation`): the same observation is broadcast to every agent on the side.

```python
# PER_VEHICLE scope, n_guards=2:
#   side_obs.shape == (2, obs_dim) → guard_0.obs = side_obs[0]; guard_1.obs = side_obs[1]
#
# PER_SIDE scope:
#   side_obs.shape == (obs_dim,) → guard_0.obs = guard_1.obs = side_obs
```

## Verified against `pettingzoo.test.parallel_api_test`

The adapter passes `parallel_api_test(adapter, num_cycles=3)` cleanly — agents, action_spaces, observation_spaces, and the reset/step/agents lifecycle all match the Parallel API contract.

## Episode termination

When `episode_done` fires, all agents simultaneously have `terminated=True`. The adapter clears `self.agents = []` after termination, signaling the loop should stop. To run another episode, call `reset(seed=...)`.

## Determinism

Same as Gymnasium — given a seed, two adapter instances produce identical observations.

## Multi-vehicle scenarios

For scenarios with multiple guards or bandits:

```python
cfg = make_pursuit_evasion(n_guards=2, n_bandits=1)
adapter = PettingZooAdapter(OrbitalGameEnv(cfg))
adapter.possible_agents
# ['guard_0', 'guard_1', 'bandit_0']
```

## Source

- [`src/orbital_game/adapters/pettingzoo/adapter.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/adapters/pettingzoo/adapter.py)
- Tests: [`tests/test_adapter_pettingzoo.py`](https://github.com/duncaneddy/orbital-game/blob/main/tests/test_adapter_pettingzoo.py)
