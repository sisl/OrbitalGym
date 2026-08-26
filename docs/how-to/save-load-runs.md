# Save and load runs

Persist a `Trajectory` plus its `ScenarioConfig` to HDF5 for later
analysis.

## Recipe

```python
from pathlib import Path

import jax

from orbitalgym import (
    OrbitalGymEnv,
    SingleAgentView,
    load_run,
    make_lady_bandit_guard,
    save_run,
)
from orbitalgym.policies import ZeroControl
from orbitalgym.rollout import rollout_single_agent

cfg = make_lady_bandit_guard()
env = OrbitalGymEnv(cfg)
view = SingleAgentView(env)
policy = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
traj = rollout_single_agent(view, policy, lambda c, s, k: None,
                            jax.random.PRNGKey(0), n_steps=cfg.max_steps)

save_run(Path("run.h5"), cfg, traj)

# Later:
cfg_loaded, traj_loaded = load_run(Path("run.h5"))
```

## What's stored

- The full `ScenarioConfig` JSON (round-trippable via
  `to_json` / `from_json` — see [Round-trip config to JSON](json-roundtrip-config.md)).
- Every leaf of `Trajectory` — env state, per-side outputs, episode mask.
- `controlled_side` as metadata.

## See also

- [API reference → Core → Logging](../api/core.md).
