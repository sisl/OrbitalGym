# orbital-game

A JAX-native framework for two-side decision-making in orbital
scenarios. Same symmetric core drives single-agent (Gymnasium-shaped),
multi-agent (PettingZoo-shaped), and POMDP-shaped views of the same
underlying game. Four scenarios ship in the box.

## Choose your path

<div class="grid cards" markdown>

- :material-school: **New here**
  Start with [Tutorials](tutorials/index.md). Five guided walkthroughs
  from `pip install` to a working short-horizon planner running on GPU.

- :material-book-search: **Looking up an API**
  Jump to the [API reference](api/index.md). Auto-generated from
  in-code docstrings.

- :material-target: **Building a scenario**
  See [Game guides](games/index.md) for what each game rewards, then
  the [How-to recipes](how-to/index.md) for the task-by-task steps.

- :material-cog: **Understanding the design**
  Read [In depth](in-depth/index.md). Eleven deep-dives on the
  symmetric core, state layout, dynamics, observations, belief, and
  more.

</div>

## What's in the box

- **Four games:** [Lady-Bandit-Guard](games/lady-bandit-guard.md),
  [Pursuit-Evasion](games/pursuit-evasion.md),
  [Sun-Blocking](games/sun-blocking.md),
  [Observation-Blocking](games/observation-blocking.md).
- **Three adapters:** Gymnasium (single-agent RL), PettingZoo
  (multi-agent RL), and a POMDPPlanners-shape duck-typed protocol
  (belief-space planners).

## How fast

A whole episode compiles to one `jax.lax.scan`, so a single rollout is
fast. The bigger win is `vmap` over rollouts: 1000 rollouts can be run in parallel on GPU or MPS.