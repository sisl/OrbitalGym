# T3 — Write an active-pursuer policy

Replace `ZeroControl` on the bandit with a policy that predicts the
guard's next-step position and thrusts toward it, then A/B-compare it
against `ZeroControl` on closest-approach distance averaged over 10
seeds. Builds on [T2 — Build an RTN scenario](t2-rtn-scenario.md);
reference implementation at
[`examples/policies/lead_intercept.py`](https://github.com/sisl/OrbitalGym/blob/main/examples/policies/lead_intercept.py).

## Why ZeroControl is the wrong baseline

`ZeroControl` ignores the observation and emits a zero command. Against
a Pursuit-Evasion guard that's also on `ZeroControl`, this gives two
closed ellipses that never close on each other meaningfully. Replacing
the bandit with a policy that *uses* the observation is the smallest
meaningful upgrade.

## The Policy protocol

Every `Policy` is a callable with this signature:

```python
def __call__(
    self,
    policy_state: Any,
    obs: jax.Array,
    key: jax.Array,
    t: jax.Array,
) -> tuple[jax.Array, Any]:
    ...
```

- `policy_state` — opaque per-policy carry (None for stateless).
- `obs` — flat observation vector for the side. For the bundled
  `FullObservation` and a 1v1 RTN game, the layout is
  `[own_truth(6), opp_truth(6)]` — **the policy's own side first**, then
  the opposing side. From the bandit's perspective that means
  `[bandit_rtn(6), guard_rtn(6)]`; from the guard's it would be
  `[guard_rtn(6), bandit_rtn(6)]`. This is convenient: a policy can
  always read its own state from the leading slice without knowing
  which side it's wired into.
- `key` — JAX PRNG subkey.
- `t` — step counter as a scalar JAX array.

Returns `(action, next_policy_state)`. `action` is the per-side
`Command` pytree (built by `build_command_class` from the registered
action components). For a 1-bandit RTN game with the default
`(IMPULSIVE_MANEUVER,)` component tuple, `action.dv` has shape `(1, 3)` —
three components of `Δv` in the rotating frame.

## n_vehicles and command_cls injection

Implementations declare `n_vehicles: int = 0` and
`command_cls: Any = None` as fields. The env populates them at
construction via `dataclasses.replace`. Keep the defaults — never
hard-code.

## Build the policy

T3 used to define the lead-intercept pursuer inline. The gallery now
ships [`LeadInterceptPursuer`](../extending/gallery.md#leadinterceptpursuer);
T3 imports it directly:

```python
--8<-- "tests/docs/test_tut_t3_active_pursuer.py:imports"
```

The math: with both vehicles in HCW the guard's next-step position
under zero control is approximately `r + v·dt` (linear short-step
extrapolation; one *real* HCW propagation would be more accurate but
the linear form is the right teaching example). Thrust at maximum
magnitude along the line from the bandit's current position to that
predicted point.

## Wire it in

```python
--8<-- "tests/docs/test_tut_t3_active_pursuer.py:wire-it-up"
```

`dataclasses.replace` because `ScenarioConfig` is frozen.

## Did it close?

```python
--8<-- "tests/docs/test_tut_t3_active_pursuer.py:closing-distance"
```

The closest-approach distance (`distance.min()`) is the right metric
here, not `distance[-1]`. With a 2000-second horizon the relative
orbit naturally swings past closest approach and opens up again, so
the *final* distance can be larger than the initial — but the
trajectory clearly *did* close. Plotting the full `distance[t]` curve
makes the closing dynamic visible.

## A/B vs ZeroControl over 10 seeds

`vmap` over seeds (recipe: [How-to → vmap rollouts](../how-to/vmap-rollouts.md))
to compare the two policies' closest-approach distances. The
`LeadInterceptPursuer` mean is meaningfully smaller than the
`ZeroControl` mean. See the test file's `test_t3_ab_comparison_vmap`
for the exact comparison code.

## Going further: Proportional Navigation

The classical guidance law for a pursuer with a measured line-of-sight
rate is **Proportional Navigation (PN)**. The acceleration command is

> `a = N · V_c · ω_LOS`

where `V_c` is closing velocity, `ω_LOS` is the LOS rate, and `N` is
the navigation gain (typically 3–5). Implementing PN in this codebase
needs (a) a finite-difference or filtered LOS-rate estimate from
consecutive observations and (b) the impulsive actuator wrapping the
continuous acceleration command into per-step `Δv`. Left as an
exercise; the reference texts are Zarchan, *Tactical and Strategic
Missile Guidance*, and Kabamba & Girard, *Fundamentals of Aerospace
Navigation and Guidance*.

## Next

→ **[T4 — Plan with short-horizon search](t4-short-horizon-search.md)**
moves from a hand-coded reactive policy to a planner that simulates
candidate action sequences forward through the env.
