# Heuristic policy cookbook

Patterns for the *opposing* side — the side that runs as a heuristic
policy rather than a controlled (learned or planned) policy. Each
pattern is a recipe; the gallery ships canonical implementations of all
four.

In single-agent framing, the opposite side runs `cfg.guard_policy` or
`cfg.bandit_policy` (depending on which side `cfg.controlled_side`
points to). Multi-agent and POMDP framings ignore the opposite side's
field — both sides are driven by the caller.

## Pattern: lead-intercept chaser

Predict the opponent's next-step position under a free-drift assumption,
then thrust along the line to that predicted point. T3 walks through
this in tutorial form; the gallery class is `LeadInterceptPursuer`.

```python
--8<-- "tests/docs/test_extending_heuristic_policy_cookbook.py:imports"
```

```python
--8<-- "tests/docs/test_extending_heuristic_policy_cookbook.py:lead-intercept-wire"
```

Gallery: [`LeadInterceptPursuer`](gallery.md#leadinterceptpursuer).

## Pattern: evasive maneuver

Move perpendicular to the opponent's relative velocity. With near-zero
relative velocity the orthogonal subspace is degenerate; the canonical
implementation falls back to a fixed reference axis to keep the action
finite.

```python
--8<-- "tests/docs/test_extending_heuristic_policy_cookbook.py:evader-wire"
```

Gallery: [`OrthogonalEvader`](gallery.md#orthogonalevader).

## Pattern: sun-tracker

Maintain Sun-line geometry between the controlled vehicle and the
opponent. Useful for Sun-Blocking. The simple version takes the Sun
direction in the local RTN frame as a static knob; a dynamic Sun
version (re-derived per step from epoch + time) is out of scope for the
gallery — see `games/sun_blocking.py:_sun_eci_at_mjd` for the dynamic
approach.

Gallery: [`SunTrackerBlocker`](gallery.md#suntrackerblocker).

## Pattern: randomized heuristic

Wrap any deterministic policy with PRNG-keyed Gaussian jitter. Great
for population-of-opponents training where you want a smooth
distribution of adversaries.

```python
--8<-- "tests/docs/test_extending_heuristic_policy_cookbook.py:randomized-jitter-wire"
```

Gallery: [`JitteredPolicy`](gallery.md#jitteredpolicy).

## Pitfalls

- **Why the `n_vehicles=0` and `command_cls=None` defaults?** The env
  populates these via `dataclasses.replace` at construction;
  hard-coding them defeats portability across scenarios.
- **Why must side identity be static?** `Side` is a Python `StrEnum`,
  never a traced JAX array. JAX specialises the trace per-side via the
  static identity check. See [Symmetric core](../in-depth/symmetric-core.md).
- **The `__call__` signature is rigid.**
  `(self, policy_state, obs, key, t) → (action, next_policy_state)`.
  Stateless? Pass `None` and forward it unchanged.
- **`JitteredPolicy` requires its `base` keyword.** No default — pass
  `JitteredPolicy(base=..., sigma=...)`.

## See also

- [Switch the controlled side](../how-to/switch-controlled-side.md).
- [Controlled-side policy cookbook](controlled-policy-cookbook.md) —
  patterns for the side under your control.
- [Gallery](gallery.md).
- [T3 — Active-pursuer policy](../tutorials/t3-active-pursuer.md).
