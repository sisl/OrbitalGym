# MCTS policy

`MCTSPolicy` is a JAX-native Gumbel MuZero / PUCT search backed by
[mctx](https://github.com/google-deepmind/mctx). It conforms to the
unified `Policy` protocol — same shape as every other policy in the
gallery — and slots into either side. Both sides may run an
`MCTSPolicy` independently; each carries its own *opponent model* (see
below).

The whole search compiles into a single `jax.lax.fori_loop`-based JIT
region (mctx is fully traced). MPS / GPU backends pay for themselves on
the simulation budget — there are no host↔device round-trips per
simulation, unlike Python-loop UCB1 patterns that came before.

## What it is, in two minutes

- **Gumbel MuZero by default.** Root selection uses Gumbel noise and
  sequential halving; interior nodes use deterministic Gumbel MuZero
  selection. The alternative `variant="muzero"` uses PUCT. Both use
  uniform priors, with leaf value defaulting to zero. Pass a custom
  `leaf_value_fn` for cheap heuristic value backups.
- **Discrete action space.** Build an `(A, dv_dim)` `action_grid` of
  candidate Δv vectors. Search picks a row index per call.
- **Opponent model inside the tree.** During simulation the *other*
  side's actions come from the `opponent_model` field — itself a
  Policy. This is the searcher's *belief about the opponent*, not a
  "scripted opponent" baked into the planner. Default is
  `UniformRandomDiscretePolicy(opponent_action_grid)`; for sharper
  search wire any vmappable Policy (e.g. `GlideslopeToLady`,
  `LeadInterceptPursuer`).
- **agent_view contract.** Either a flat state vector
  (`adapter.states_dim` length) or a `Belief` whose `mean` field is
  flat-state-shaped. The duck-type dispatch is `isinstance(view,
  jax.Array)`; everything else reads `.mean`.

## Minimal recipe

```python
import jax, jax.numpy as jnp, numpy as np
from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.policies.mcts import MCTSPolicy
from orbitalgym.policies.uniform_random import UniformRandomDiscretePolicy

cfg = make_lady_bandit_guard(seed=0)
env = OrbitalGymEnv(cfg)
adapter = POMDPAdapter(env)

# 9-direction RT-plane Δv grid (8 dirs + no-op).
angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
grid_2d = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
action_grid = jnp.asarray(np.concatenate([grid_2d, np.zeros((1, 2))], axis=0))

opponent_model = UniformRandomDiscretePolicy(
    action_grid=action_grid,
    n_vehicles=cfg.n_bandits,
    command_cls=env.bandit_command_cls,
)

mcts = MCTSPolicy(
    env_model=adapter,
    side=Side.GUARD,
    action_grid=action_grid,
    opponent_model=opponent_model,
    opponent_action_grid=action_grid,
    num_simulations=32,
    max_num_considered_actions=16,
    n_vehicles=cfg.n_guards,
    command_cls=env.guard_command_cls,
)

state, _ = env.reset(jax.random.PRNGKey(0))
cmd, _ = mcts(None, adapter.pack(state), jax.random.PRNGKey(1), state.t)
print(cmd.dv)   # (1, 3) — chosen Δv padded into the side's command dim
```

## Belief support

For belief-aware MCTS, drive via `belief_rollout` and use a Belief
representation whose `mean` field is the adapter's flat state vector.
The simplest pattern is a thin wrapper:

```python
import flax.struct
import jax

@flax.struct.dataclass
class FlatStateBelief:
    """Trivial Belief whose `mean` is the adapter's flat state vector."""
    mean: jax.Array
```

Build a matching `BeliefInitializer`/`BeliefUpdater` (e.g. a Kalman
filter that maintains the flat state directly, or an "oracle" pair that
trusts the truth). Real `KFBelief`/`EKFBelief` whose `mean` is per-pair
`(N_obs, N_total, d)` need a small adapter — flatten to the
`adapter.states_dim` layout before handing it to the policy.

## Tuning

| Knob | Default | When to change |
|------|---------|----------------|
| `num_simulations` | 32 | Simulations per search tree. Increasing this allocates more search effort but does not raise the Gumbel root candidate limit. |
| `max_num_considered_actions` | 16 | Positive root candidate limit for Gumbel MuZero, bounded by the available actions and simulation budget. Applies to joint actions or each independent vehicle's actions. Ignored by `"muzero"`. |
| `variant` | `"gumbel_muzero"` | Sequential halving + Gumbel noise at the root; `"muzero"` uses PUCT. Both currently use uniform priors. |
| `max_depth` | None | Cap planning horizon; useful when the discount is 1.0 and trajectories are long. |
| `discount` | `None` | `None` reads `env_model.discount()`, the adapter's per-macro-step discount. Set a float to override. |
| `leaf_value_fn` | zero | Cheap heuristic estimate at leaves often beats pure rollouts. |
| `opponent_model` | uniform-random | Wire e.g. `GlideslopeToLady` for adversary-aware search. |

Tune root coverage and simulation effort separately. With nine actions
per vehicle and three jointly controlled vehicles, the joint action space
contains 729 actions, but the default Gumbel search visits at most 16 root
actions even with 2048 simulations. Raise `max_num_considered_actions`
explicitly to explore more root actions, and provide enough simulations
to evaluate and compare those candidates. In independent coordination,
each vehicle searches its own action space with these same per-tree
budgets. Inspect `search_tree` root visit counts when measuring coverage;
`action_weights` can be nonzero for actions that were never visited.

## CPU vs MPS

The dedicated benchmark lives in
[`examples/mcts_cpu_vs_mps.ipynb`](https://github.com/sisl/OrbitalGym/blob/main/examples/mcts_cpu_vs_mps.ipynb).
The expectation: MPS pulls ahead at higher `num_simulations` because
more on-device work amortizes the JIT-compile cost. If you observe MPS
lagging at every budget, that's a regression — investigate before
declaring victory. Common culprits: stale JIT cache, an inadvertent
host-roundtrip in `recurrent_fn`, or a float64 leaf forcing MPS to fall
back to CPU.

## Limitations and future work

- **POMCP-style particle MCTS** is not implemented. The current `Belief`
  protocol exposes only `mean`. When particle-based beliefs ship the
  protocol grows a `.sample(key, n)` method; `MCTSPolicy` then runs
  per-particle searches and aggregates. Until then, MCTS-from-mean is
  the supported path.
- **Two-MCTS-as-mutual-opponent-models** is fine in principle but
  combinatorially expensive — each `MCTSPolicy.opponent_model` invocation
  inside the tree spawns another full search. Use cheap surrogates (glideslope,
  uniform-random) as opponent models even when the *real* opponent in
  the game is itself an MCTS policy.
- **Warm-starting / tree reuse** across calls is future work. Today
  every call rebuilds the search tree from scratch.
