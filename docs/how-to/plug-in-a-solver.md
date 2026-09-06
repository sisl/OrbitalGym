# Plug in a solver

Mahdi's solvers integrate into OrbitalGym's batched evaluation pipeline
when they satisfy the policy contract: a callable with a predictable
signature, return types, and behavior under JAX transformations. This
page covers the contract, validation, and integration patterns.

## 1. The contract

A solver is any callable that maps the policy state, the current
observation, and time to a command and the next policy state:

```python
def my_solver(policy_state, agent_view, key, t):
    """
    Args:
        policy_state: Any pytree threaded across rollout steps. Use None if
            stateless (e.g., reactive controllers).
        agent_view: ContactAwareBelief with .inner (particle belief) and
            .contact (link mask). The inner belief has .particles (shape
            N_observers × N_total × K × d) and .log_weights.
        key: jax.random.PRNGKey for sampling or stochastic decisions.
        t: Current simulation time (scalar or array).

    Returns:
        (command, new_policy_state) where command is an instance of the
        side's command class (e.g., env.guard_command_cls.zeros(n) with
        fields like .dv and .target_dir replaced).
    """
    # Example: zero control (stateless)
    n_vehicles = agent_view.inner.particles.shape[0]
    command = make_some_command_class().zeros(n_vehicles)
    return command, policy_state  # Unchanged state
```

The `.inner` belief is a `ParticleFilterBelief` with particles indexed
as `[observer_idx, target_idx, particle_idx, state_component]` and
weights per target. `.contact` is a boolean array of length `n_vehicles`
indicating which side members are in communication this tick.

## 2. Checking a solver

Before integration, validate your solver against the contract using
`check_policy_conforms`. It exercises the solver in isolation and under
JAX transformations (jit + vmap), and reports which checks passed:

```python
import jax
from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.eval import check_policy_conforms

cfg = make_lady_bandit_guard(n_guards=2)
env = OrbitalGymEnv(cfg)
init = ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=8)

report = check_policy_conforms(
    my_solver,
    env,
    Side.GUARD,
    init,
    n_steps=5,
)

print(f"Command shape OK: {report.command_ok}")
print(f"Traceable under jit+vmap: {report.traceable}")
print(f"Runs in belief_rollout: {report.rollout_ok}")
if not all([report.command_ok, report.traceable, report.rollout_ok]):
    print(f"Message: {report.message}")
```

The report has fields `command_ok` (shape and type match the expected
command), `traceable` (no data-dependent control flow or host callbacks
under jit), and `rollout_ok` (integrates into a full episode rollout).
All three must pass for integration into batched evaluation.

## 3. Running from a bank

Evaluation banks hold scenario initial conditions (guards, bandits,
reference orbits) sampled from a distribution. Solvers are integrated
via the `policies` dict and a belief updater for each side:

```python
import dataclasses
import math
import jax
import jax.numpy as jnp
from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.belief.pf import (
    ParticleFilterFromTruthInitializer,
    ParticleFilterBeliefUpdater,
)
from orbitalgym.belief.sync import PFTeamFusion
from orbitalgym.env.types import BySide
from orbitalgym.eval import sample_bank, evaluate_bank
from orbitalgym.links import PointingConeLink
from orbitalgym.dynamics.hcw import hcw_rtn_step
from orbitalgym.policies.zero import ZeroControl

cfg = make_lady_bandit_guard(n_guards=2, n_bandits=1)
env = OrbitalGymEnv(cfg)

# Sample 100 random scenarios from the distribution.
bank_states = sample_bank(env, n_episodes=100, seed=0)

# Build a belief initializer and updater for your environment.
initializer = ParticleFilterFromTruthInitializer(
    layout=env.layout, n_particles=64
)
d = env.layout.dynamics_state_dim

def dyn_fn(x, u, dt):
    return hcw_rtn_step(x[None, :], u[None, :], env, dt)[0]

updater = ParticleFilterBeliefUpdater(
    dynamics_fn=dyn_fn,
    process_noise=jnp.eye(d) * 1e-4,
    dt=cfg.dt,
)

# Set up zero control policy for the opposing side.
bandit_policy = dataclasses.replace(
    ZeroControl(),
    command_cls=env.bandit_command_cls,
    n_vehicles=cfg.n_bandits,
)

# Run evaluation: guard uses your solver, bandit uses zero control.
metrics_per_scenario = evaluate_bank(
    env,
    cfg,
    bank_states,
    jax.random.PRNGKey(1),
    n_steps=200,
    policies=BySide(guard=my_solver, bandit=bandit_policy),
    init_policy_state_fns=BySide(
        guard=lambda c, s, k: None,
        bandit=lambda c, s, k: None,
    ),
    belief_initializers=BySide(guard=initializer, bandit=initializer),
    belief_updaters=BySide(guard=updater, bandit=updater),
    guard_link=PointingConeLink(half_angle_rad=math.radians(10.0)),
    team_sync_fns=BySide(guard=PFTeamFusion(), bandit=None),
)
```

`evaluate_bank` batches scenarios via `jax.vmap` (one vmap per scenario).
Each scenario runs a full episode; belief state is updated per timestep
using the particle filter updater. The `team_sync_fns` fuse particle
beliefs across vehicles on the same side (e.g., guards share detection
windows via the provided link).

### What comes back

`evaluate_bank` returns an `EpisodeMetrics` with one entry per scenario.
Alongside the outcome and the resource accounting (`outcome`, `steps`,
`min_d_gb`, `min_d_bl`, `dv_guard`, `dv_bandit`, `link_events_guard`,
`ic_valid`, `in_cone_fraction_guard`) it scores what the guard team knew
about the bandit nearest the lady. The lady sits at the origin of the
truth frame, and the belief error of one guard is the distance between
its belief mean for that bandit and the bandit's true position.

| Metric | Definition |
| --- | --- |
| `belief_err_guard` | Metres. The smallest belief error among the guards, averaged over the episode's live steps. |
| `belief_err_guard_at_commit` | Metres. The same quantity at the first live step where a bandit is within `commit_radius_m` of the lady. NaN when no bandit commits. |
| `time_to_detect_guard` | Seconds from the episode start to the first live step where the smallest belief error drops below `detect_error_m`. NaN when it never does. |
| `belief_age_guard` | Seconds since any guard last held a bandit in its sensor cone, averaged over the live steps. Counts from the episode start while no guard has seen a bandit yet. |

`commit_radius_m` (2500 m) and `detect_error_m` (100 m) are
`evaluate_bank` keywords. These four metrics separate scenarios that a
win rate cannot: when the outcome is settled before the first
communication link opens, only the information metrics move.

`metrics_to_records(metrics, **constants)` turns the batch into one
plain-Python dict per episode, carrying every field above plus the
constants, ready for a data frame or a parquet file.

## 4. Planning on macro steps

A solver that searches at the env's tick rate covers `dt` seconds per
decision, which is far short of an intercept. Build the adapter with
`action_repeat = k` so one planner action spans `k * dt` seconds, and
wrap the solver in `ActionRepeatPolicy` with the same `repeat` so the
executed trajectory matches the searched one:

```python
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.policies.action_repeat import ActionRepeatPolicy

adapter = POMDPAdapter(env, action_repeat=8, discount=0.99)
solver = my_solver_using(adapter)          # sees macro_dt = cfg.dt * 8

policy = ActionRepeatPolicy(
    inner=solver,
    repeat=adapter.action_repeat,
    n_vehicles=cfg.n_guards,
    command_cls=env.guard_command_cls,
)
```

`ActionRepeatPolicy.init_state(config, env_state, key)` follows the
`init_policy_state_fns` convention, so it can be passed straight into
`evaluate_bank`, and every argument has a default so
`check_policy_conforms` resolves it without an override.

## 5. Non-traceable solvers

If your solver uses host callbacks, data-dependent loops, or other
non-traceable constructs, wrap it in `jax.pure_callback` to run on the
slow path:

```python
import jax
import jax.numpy as jnp

def my_traced_solver(policy_state, agent_view, key, t):
    # Pytree output shapes must be known at trace time.
    def callback_impl(view_arrays):
        # Host-side solver: returns command arrays.
        return my_untraceable_solver(view_arrays)

    # Specify output shapes and dtypes.
    view_flat, tree_def = jax.tree_util.tree_flatten(agent_view)
    command_shape = (env.guard_command_cls.zeros(2).dv.shape,)
    command_dtype = jnp.float64

    result = jax.pure_callback(
        callback_impl,
        jax.ShapeDtypeStruct(command_shape, command_dtype),
        agent_view,
        vectorized=False,
    )
    command = env.guard_command_cls(dv=result, ...)
    return command, policy_state
```

When wrapped, the solver is called once per scenario (not vmapped into
a batch). Batched evaluation still works but runs one episode at a time.
Each scenario gets its own callback invocation.

## See also

- [Belief representations](../in-depth/belief.md) for `ParticleFilterBelief` and its updater.
- [Ground stations and links](../in-depth/ground-stations.md) for contact-gated fusion.
- [API reference](../api/index.md).
