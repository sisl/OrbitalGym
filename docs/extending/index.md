# Extending

The four games shipping in `orbitalgym` (Lady-Bandit-Guard,
Pursuit-Evasion, Sun-Blocking, Observation-Blocking) are designed to be
*composed*. Each scenario is a `ScenarioConfig`, a frozen dataclass whose
fields you can swap with `dataclasses.replace`. This section is a
Gymnasium-style "custom envs" track: by-axis customization on the left,
patterns and a gallery of named, ready-to-use policies on the right.

## Axes of variation

| Axis | What you swap | Page |
|---|---|---|
| Dynamics | `cfg.truth_dynamics`, `cfg.policy_dynamics`, `cfg.belief_dynamics`, `cfg.reference_orbit_dynamics` | [Customize dynamics](customize-dynamics.md) |
| Reward | `cfg.reward_fn` | [Customize rewards](customize-rewards.md) |
| Termination | `cfg.termination_fn` | [Customize termination](customize-termination.md) |
| Initial conditions | `cfg.ic_sampler` | [Customize IC sampling](customize-ic-sampling.md) |
| Observations | `cfg.guard_observation_fn`, `cfg.bandit_observation_fn` | [Customize observations](customize-observations.md) |
| Controlled-side policy | the policy you pass to `rollout_single_agent` | [Controlled-side policy cookbook](controlled-policy-cookbook.md) |
| Heuristic policy | `cfg.{guard,bandit}_policy` | [Heuristic policy cookbook](heuristic-policy-cookbook.md) |
| Game | restricted to LBG / PE / SB / OB; this section does not author new games | (see [Game guides](../games/index.md)) |

## Two cookbooks

- **[Controlled-side policy cookbook](controlled-policy-cookbook.md)** —
  patterns for the side under your control (heuristic-with-fallback,
  belief-conditioned, composite action, A/B harness).
- **[Heuristic policy cookbook](heuristic-policy-cookbook.md)** —
  patterns for the opposing heuristic policy (lead-intercept chaser,
  evasive maneuver, sun-tracker, randomized).

## Gallery

[Gallery](gallery.md) catalogs every named, importable policy that
ships with the package. Use a gallery class directly or read its source
as a starting point for your own.
