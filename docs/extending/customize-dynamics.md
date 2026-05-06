# Customize dynamics

`ScenarioConfig` carries four dynamics roles —
`truth_dynamics`, `policy_dynamics`, `belief_dynamics`, and
`reference_orbit_dynamics` — and each accepts either a
`DynamicsKey` (registered step function) or a typed-instance dynamics
that exposes the protocol below. Use this page to author a custom
dynamics, register or wire it without registering, and compose
mixed-frame scenarios where truth runs in ECI but the policy and belief
stay in RTN.

For the conceptual story (truth/policy/belief roles, what the env
calls each step, what ships) read [In depth → Dynamics](../in-depth/dynamics.md).

## The dynamics protocol

A dynamics is a callable

```
dynamics: (state, dv, params, dt) -> next_state
```

with two class-level metadata attributes:

- `frame: Frame` — the spatial frame of the input/output state array
  (`Frame.RT`, `Frame.RTN`, or `Frame.ECI`).
- `kind: DynamicsKind` — `RELATIVE` (rotating-frame propagation; HCW
  and friends) or `ABSOLUTE` (inertial; Keplerian, J2, full astrojax).

Inputs:

- `state` — raw per-side dynamics array. Shape `(N_side, 4)` for `RT`,
  `(N_side, 6)` for `RTN` and `ECI`. One row per vehicle.
- `dv` — applied impulsive Δv, shape `(N_side, action_dim)`. Already
  produced by the actuator and rotated into the dynamics frame by the
  env.
- `params` — per-side `VehicleParams` (mass, max-thrust, dry-mass,
  mean motion). Some dynamics ignore it (Keplerian, AstrojaxOrbit);
  HCW reads `params.mean_motion`.
- `dt` — scalar timestep in seconds.

`ScenarioConfig.__post_init__` runs a validator
(`_validate_components_match_dynamics`) that reads `frame` from your
dynamics and asserts the matching state component is present on each
side (`Frame.RTN` ⇒ `StateComponentKey.RTN`, `Frame.ECI` ⇒
`StateComponentKey.ECI`, etc.). The same `frame`/`kind` metadata also
drives the env's per-step frame conversions and the reference-orbit
propagator's `ABSOLUTE`-kind check.

`RELATIVE` dynamics propagate state expressed in the *rotating*
reference-orbit frame (RT or RTN). `ABSOLUTE` dynamics propagate state
expressed in *inertial* coordinates (ECI). Mixed-frame scenarios
(ECI truth + RTN belief/policy) are explicitly supported — the env
converts between frames at each step using astrojax helpers.

## Authoring a custom relative-frame dynamics

Two patterns. Both produce a callable with the right `frame`/`kind`
metadata; the difference is whether you put it in the registry.

**Pattern A — register against an existing `DynamicsKey`.** Useful if
you want to override the bundled HCW with a drop-in alternative *and*
serialize the config to JSON later (registry-keyed dynamics survive
JSON round-trips; typed instances raise `NotImplementedError` on
`to_json` today).

```python
import jax
import jax.numpy as jnp

from orbital_game.registry import (
    DynamicsKey,
    DynamicsKind,
    Frame,
    register,
    _clear_registry_for_tests,  # only if you need to swap an existing key
)


@register(DynamicsKey.HCW_RTN, frame=Frame.RTN, kind=DynamicsKind.RELATIVE)
def my_hcw_rtn(state, dv, params, dt):
    # state: (N, 6), dv: (N, 3) — radial / along-track / cross-track.
    n = params.mean_motion
    s, c = jnp.sin(n * dt), jnp.cos(n * dt)
    # ... your STM / integrator goes here ...
    s0 = state.at[:, 3:].add(dv)
    return s0  # placeholder: real implementation propagates by dt
```

`DynamicsKey` is a closed `StrEnum`, so to register against a *new* key
you'd need to extend the enum — which is a project-level change, not
something most users want to do. Pattern B is the idiomatic answer.

**Pattern B — skip the registry entirely.** Pass the function directly
to the role field. Attach `frame` and `kind` as plain attributes so
the validator finds them.

```python
import jax.numpy as jnp

from orbital_game.registry import DynamicsKind, Frame


def my_hcw_rtn(state, dv, params, dt):
    n = params.mean_motion
    s, c = jnp.sin(n * dt), jnp.cos(n * dt)
    # ... your STM / integrator ...
    return state.at[:, 3:].add(dv)  # placeholder


my_hcw_rtn.frame = Frame.RTN
my_hcw_rtn.kind = DynamicsKind.RELATIVE
```

Then wire it onto the cfg:

```python
import dataclasses

from orbital_game import make_lady_bandit_guard

cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1)
cfg = dataclasses.replace(cfg, truth_dynamics=my_hcw_rtn, policy_dynamics=my_hcw_rtn)
```

Pattern B is the right choice for one-off experiments and ablations.
The trade-off is that `cfg.to_json()` will raise `NotImplementedError`
for typed-instance dynamics (the registry is the persistence boundary).

## Authoring a custom absolute-frame ECI dynamics

For an inertial-frame dynamics (truth = full physics, belief = HCW),
two paths.

**Path A — from scratch.** Write a callable on a 6D ECI state and
attach `frame=Frame.ECI`, `kind=DynamicsKind.ABSOLUTE`:

```python
import jax
import jax.numpy as jnp

from orbital_game.registry import DynamicsKind, Frame


def my_eci_step(state, dv, params, dt):
    # state: (N, 6) — (rx, ry, rz, vx, vy, vz) in ECI [m, m/s].
    # dv:    (N, 3) — impulsive Δv applied at start of interval.
    s0 = state.at[:, 3:].add(dv)
    # ... your integrator on s0 by dt ...
    return s0  # placeholder


my_eci_step.frame = Frame.ECI
my_eci_step.kind = DynamicsKind.ABSOLUTE
```

**Path B — wrap astrojax with a `ForceModelConfig`.** This is the
headline pattern. `AstrojaxOrbitDynamics` is a configurable typed
instance that composes any combination of spherical-harmonics gravity,
drag, SRP, and third-body perturbations through one
`ForceModelConfig`. Instantiate it directly and pass the instance as
`truth_dynamics`.

The 5×5 spherical-harmonics worked example:

```python
from astrojax import ForceModelConfig, GravityModel

from orbital_game.dynamics.astrojax_orbit import AstrojaxOrbitDynamics

gm = GravityModel.from_type("JGM3")
gm.set_max_degree_order(5, 5)

config = ForceModelConfig(
    gravity_type="spherical_harmonics",
    gravity_model=gm,
    gravity_degree=5,
    gravity_order=5,
)
truth = AstrojaxOrbitDynamics(force_model=config)
```

`AstrojaxOrbitDynamics` carries `frame = Frame.ECI` and
`kind = DynamicsKind.ABSOLUTE` as class attributes — the validator and
reference-orbit kind-check find them the same way they find them on
registered step functions. Default `integrator="rk4"` and
`sub_steps=1` work for most LEO scenarios; bump `sub_steps` if you
need finer integration on high-eccentricity or long-step regimes.

Default `ForceModelConfig()` (no kwargs) is point-mass two-body and is
numerically equivalent to `DynamicsKey.KEPLERIAN_ECI`. Use that
equivalence as a sanity check when you start adding perturbations.

## Wiring custom truth, HCW belief and policy

The headline mixed-frame scenario: ground-truth runs full ECI physics,
the policy and belief use the cheaper HCW model. This is the recipe
you want for robustness studies — the policy is built against an
idealized model, but the env propagates against perturbed truth.

```python
import dataclasses

from astrojax import ForceModelConfig, GravityModel

from orbital_game import make_lady_bandit_guard
from orbital_game.dynamics.astrojax_orbit import AstrojaxOrbitDynamics
from orbital_game.registry import DynamicsKey, StateComponentKey

gm = GravityModel.from_type("JGM3")
gm.set_max_degree_order(5, 5)
config = ForceModelConfig(
    gravity_type="spherical_harmonics",
    gravity_model=gm,
    gravity_degree=5,
    gravity_order=5,
)
truth = AstrojaxOrbitDynamics(force_model=config)

cfg = make_lady_bandit_guard(
    n_guards=1,
    n_bandits=1,
    truth_dynamics=truth,                      # full ECI physics
    policy_dynamics=DynamicsKey.HCW_RTN,       # cheap RTN model for the planner
    # belief_dynamics=None  ⇒ defaults to policy_dynamics (HCW_RTN)
    guard_components=(StateComponentKey.ECI,),
    bandit_components=(StateComponentKey.ECI,),
)
```

A few subtleties to keep in mind:

- **`belief_dynamics` defaults to `policy_dynamics`** when unset. Read
  the resolved value as `cfg.belief_dynamics_resolved` /
  `env.belief_dynamics`, never the input field.
- **`reference_orbit_dynamics` defaults to `truth_dynamics` if it's
  `ABSOLUTE`-kind, else a `KeplerianEciDynamics` instance carrying the
  scenario's epoch (`cfg.epoch_mjd_utc`).** Read
  `cfg.reference_orbit_dynamics_resolved`. If you supply your own,
  the validator enforces `kind=ABSOLUTE` (the reference orbit lives in
  ECI by definition).
- **`KeplerianEciDynamics` is a typed-instance class** (just like
  `AstrojaxOrbitDynamics`) — pass `DynamicsKey.KEPLERIAN_ECI` and the
  env instantiates with J2000 defaults; pass an instance directly to
  bind the scenario's epoch:
  `truth_dynamics=KeplerianEciDynamics(epoch_mjd_utc=58849.0)`.
- **You only need `StateComponentKey.ECI` in your component tuples.**
  `__post_init__` auto-extends each side's components with derived-frame
  views for every frame consumed by the resolved dynamics roles, plus
  `Frame.RTN` for visualization. Read the resolved tuples as
  `cfg.guard_components_extended` / `cfg.bandit_components_extended`.
  Observations and viz keep working on the RTN view; you do not need
  to add it manually.
- **Mixed-frame scenarios incur per-step ECI ↔ RTN conversion** via
  astrojax helpers. The cost is small relative to the dynamics step
  itself and `vmap`s cleanly across vehicles and seeds.
- **JSON round-trips of typed-instance dynamics are deferred.**
  `cfg.to_json()` raises `NotImplementedError` if any of the four
  dynamics roles holds a typed instance (e.g. `AstrojaxOrbitDynamics`).
  Use enum-keyed dynamics for the persistence boundary; use typed
  instances for in-memory experiments.
- **Visualization stays RTN-only.** ECI-truth scenarios are projected
  to RTN for plotting via the auto-extended `RTN` derived view, so
  the existing notebooks and viz helpers work unchanged.

## See also

- [In depth → Dynamics](../in-depth/dynamics.md) — the conceptual story
  for truth/policy/belief roles and what ships in the registry.
- [API → Components](../api/components.md) — `Dynamics` protocol and
  the bundled step functions.
- [State layout & adding a Power component](../in-depth/state-layout.md)
  — the parallel "extend the per-side state" recipe.
