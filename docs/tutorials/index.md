# Tutorials

Five tutorials walk you from a fresh checkout to a working planner on
GPU. Each ends with a pointer to the next.

| # | Tutorial | What you do |
|---|---|---|
| T1 | [First rollout (RT 2D)](t1-first-rollout.md) | Run a planar HCW rollout with zero control on both sides and inspect the resulting `Trajectory`. |
| T2 | [Build an RTN scenario](t2-rtn-scenario.md) | Build a 2-guard / 1-bandit Pursuit-Evasion scenario in 3D RTN and render trajectory and reward plots. |
| T3 | [Write an active-pursuer policy](t3-active-pursuer.md) | Replace `ZeroControl` with a lead-intercept policy and A/B-compare it on closest-approach distance. |
| T4 | [Plan with short-horizon search](t4-short-horizon-search.md) | Wire `POMDPAdapter` to a random-shooting planner that picks the best action sequence by cumulative reward. |
| T5 | [Run on GPU / MPS](t5-acceleration.md) | Install GPU- or MPS-backed JAX and vmap a rollout over a batch of seeds. |

Every code block on these pages is a snippet pulled from a pytest test
in `tests/docs/` that runs in CI — what you read is what runs.

For a per-game entry point that links across tutorials, how-tos, and
the API reference, see [Game guides](../games/index.md).
