# vmap rollouts over seeds

Run `B` parallel rollouts in a single JIT-compiled call.

## Recipe

```python
import jax
import jax.numpy as jnp

from orbital_game import OrbitalGameEnv, SingleAgentView, make_pursuit_evasion
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout_single_agent

cfg = make_pursuit_evasion(seed=0)
env = OrbitalGameEnv(cfg)
view = SingleAgentView(env)
policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)

def run_one(seed):
    return rollout_single_agent(view, policy, lambda c, s, k: None,
                                seed, n_steps=cfg.max_steps)

n_seeds = 1024
seeds = jax.vmap(jax.random.PRNGKey)(jnp.arange(n_seeds))
trajs = jax.jit(jax.vmap(run_one))(seeds)
```

## What you get back

Every leaf of `trajs` has a leading `(B,)` axis — `B → T → N →
feature`:

- `trajs.sides.guard.reward.shape == (B, T)`
- `trajs.env_state.guards.rtn.shape == (B, T, N_g, 6)`

## Performance

This is the per-rollout-throughput pattern. On a single GPU, 1024
seeds typically run in roughly the same wall time as one rollout —
JIT amortises the compilation, vmap dispatches the inner loop in
parallel.

## See also

- [T5 — Run on GPU / MPS](../tutorials/t5-acceleration.md) for the
  full acceleration story.
- [In depth → Symmetric core](../in-depth/symmetric-core.md) for the
  axis order under vmap.
