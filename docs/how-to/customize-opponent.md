# Customize the opponent

The detailed walkthrough now lives in
[Extending → Heuristic policy cookbook](../extending/heuristic-policy-cookbook.md),
which covers four named patterns (lead-intercept chaser, evader,
sun-tracker, randomized) and includes runnable examples that use the
gallery classes shipping with the package.

For a one-liner replacement: `cfg = dataclasses.replace(cfg, bandit_policy=YourPolicy())`.
