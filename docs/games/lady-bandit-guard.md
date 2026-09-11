# Lady-Bandit-Guard

## Quickstart

```python
import jax
from orbitalgym import OrbitalGymEnv, SingleAgentView, make_lady_bandit_guard
from orbitalgym.policies import ZeroControl
from orbitalgym.rollout import rollout_single_agent

cfg = make_lady_bandit_guard()
env = OrbitalGymEnv(cfg)
view = SingleAgentView(env)
guard = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
traj = rollout_single_agent(view, guard, lambda c, s, k: None,
                            jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

For a full walkthrough see [T1 — First rollout](../tutorials/t1-first-rollout.md).

The guard defends the lady — a virtual point at the reference-orbit origin (the RTN frame) — from one or more bandits. The bandit wins by reaching within `breach_radius_m` of the lady; the guard wins by catching a bandit within `catch_radius_m`, or by repelling every bandit so none can threaten the lady again. Episode termination is event-driven (`LbgEventTermination`); the reward (`LbgZeroSumReward`) pays zero-sum terminal payoffs on those events plus a per-side shaping term that is not zero-sum.

## What the reward shapes

`LbgZeroSumReward` pays each side a shaping term plus the mirrored terminal events:

- **Guard**: `shaping_gain·(g·Φ_g(s′) − Φ_g(s)) + R_catch·1[catch or repelled] − R_breach·1[breach]`.
- **Bandit**: `shaping_gain·(g·Φ_b(s′) − Φ_b(s)) + R_breach·1[breach] − R_catch·1[catch or repelled]`.

Each side has its own potential, over the distance that side is trying to close:

```
Φ_g(s) = (−d_guard_bandit_min − home_weight·d_guard_lady_min) / shaping_scale_m
Φ_b(s) = −d_bandit_lady_min / shaping_scale_m
```

The guard's rises as it closes on the nearest bandit and, with `home_weight` above zero, as it stays near the asset it defends; the bandit's rises as it closes on the lady. `shaping_scale_m` sets the distance that makes a potential unity, keeping the shaping reward order one per step.

A single zero-sum potential `(d_bl − d_gb)/L` reads more elegantly but is flat exactly where the game is decided. With the guard parked near the lady, a bandit run at the lady shortens `d_bl` and `d_gb` by nearly the same amount, so the difference barely moves and the bandit sees no gradient at all. Per-side potentials have no such cancellation.

Only the *difference* of a side's potential is paid, which makes this potential-based shaping in the sense of [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/ml99-shaping.pdf), applied per agent. [Devlin and Kudenko (AAMAS 2011)](https://dl.acm.org/doi/10.5555/2031678.2031716) show that giving each agent its own potential in a multi-agent setting leaves the Nash equilibria of the underlying game unchanged, whatever the potentials and gains, and that a side's shaped value is its unshaped value less its own potential when `shaping_discount` matches the planner's per-step discount. `shaping_gain=0` recovers a purely terminal game.

Each potential is **zero at an absorbing state**, the remaining condition those results ask of an episodic game. On the step that terminates — a catch, a breach, or the bandits repelled, read from the same events the termination uses — the shaping pays `−shaping_gain·Φ_side(s)` and nothing more, so a whole episode's shaping sums to `−shaping_gain·Φ_side(s₀)`, a constant of the initial state, with no residual `shaping_gain·g^T·Φ_side(s_T)` left to bias which terminal state a side steers toward.

Because the two potentials are unrelated functions, **the shaping is not zero-sum between the sides** — their differences do not cancel, and both sides can gain on the same step. The terminal payoffs stay exactly zero-sum, which is what makes this a zero-sum game; the shaping only redistributes each side's own return along the path to the same equilibria.

Search leaves take the other side of the same equivalence. Shaping a reward with a potential and initializing values with that potential produce the same greedy behavior (Wiewiora 2003), so the leaf values in `orbitalgym.policies.leaf_values` *add* `shaping_gain·Φ_side` to their terminal estimate. Subtracting it would cancel the shaping that telescoped along the search path and leave the ranking with the terminal estimate alone.

Repelling the bandits pays the guard exactly what a catch pays: either way the bandit team is out of the fight, so the guard is indifferent between intercepting a bandit and driving it off.

Two further terms are genuine per-side costs rather than shaping, and they *do* move the equilibria:

- **Fuel.** With `dv_cost > 0` each side pays `dv_cost` reward units per m/s of delta-v **its own** vehicles spent over the step, recovered from the propellant drawn down through the rocket equation rather than from the commanded impulse — thrust limits and an empty tank both clip a command, so the commanded magnitude would overstate the spend. A side whose state carries no `MASS` component has no propellant trace and pays nothing.
- **Separation.** The guard side alone pays `separation_cost·(1 − d/r)²` for every guard pair closer than `r = guard_separation_m` and for every guard closer than `r = lady_keepout_m` to the lady. The charge is zero at the radius, grows quadratically inward, and reaches `separation_cost` at zero distance. It buys a spread-out formation that does not fly through the asset it protects. A one-guard side has no pairs and pays only the lady term.

Neither cost is mirrored, so each side bears only what it incurs. Of the four terms, only the terminal payoffs are zero-sum.

`shaping_scale_m=300.0`, `shaping_gain=50.0`, `shaping_discount=1.0`, `home_weight=0.0`, `guard_separation_m=lady_keepout_m=20.0`, `separation_cost=10.0`, `R_catch=R_breach=1000.0`, `dv_cost=0.0` by default; `shaping_scale_m` must be positive. The radii and the speed gates come from `cfg.game`. The potentials and the separation cost are state functions read off the step endpoints; the catch and breach events are resolved over the whole step, and the reward and the termination read the same ones, so a terminal bonus is paid in exactly the step the episode ends and the same booleans are what make the arriving state absorbing for the shaping.

## Termination

`LbgEventTermination` ends the episode as soon as any of the following holds:

1. `state.step >= cfg.max_steps` (horizon exhausted).
2. Some bandit passes within `breach_radius_m` of the lady at a relative speed below `breach_speed_mps` (bandit win).
3. Some guard passes within `catch_radius_m` of any bandit at a relative speed below `catch_speed_mps` (guard win).
4. Every bandit is repelled (guard win without an intercept).

Conditions 2 and 3 may instead require a dwell; see [Dwell](#dwell) below.

Both spatial events are resolved over the whole step, not just at its endpoints: the relative motion between two samples is treated as a straight line and the closest approach along it is what the radius test sees. A 10 s decision step at 7000 km altitude departs from that chord by well under 0.1 m, so the approximation is far finer than any usable event radius.

The speed gates default to infinity, which tests the radius alone. Setting them finite distinguishes a capture from a high-speed flyby that merely passes close.

### Dwell

`catch_dwell_steps` and `breach_dwell_steps` turn a win condition from a single step into a hold: the event fires once some bandit has been inside the corresponding radius for that many *consecutive* steps. Both default to `0`, which is the single-step condition above.

Each bandit carries two counters on the environment state, `state.dwell_catch` and `state.dwell_breach`, one entry per bandit. The game advances them every step: a step whose within-step closest approach is inside the radius, and slower than the speed gate, increments that bandit's counter, and any other step resets it to zero. So the speed gate applies on every step of a dwell, and with the gate at infinity the condition is the radius alone. The catch counter asks only that *some* guard is inside the radius on each step, not that it is the same guard throughout.

The step a bandit departs on still counts toward its dwell: its closest approach is the position it started from, which is inside the radius. The first step spent wholly outside is the one that breaks the hold.

The termination, the reward's terminal bonuses, and `Outcome` in the episode metrics all read the same counters, so a dwell-gated win ends the episode, pays its bonus, and classifies on the same step. `EpisodeMetrics` also reports `dwell_catch_steps` and `dwell_breach_steps`, the longest dwell any bandit reached over the live episode history, including the final transition, whether or not the win conditions require one. A later reset does not erase an earlier peak, and padded steps after termination do not count.

The leaf values in `orbitalgym.policies.leaf_values` price the dwell only as delay: `catch_dwell_s` and `breach_dwell_s` push the estimated event that much further away. Nothing there models the counter resetting, which is a planning approximation, not a rule of the game.

### Repelled

A bandit is repelled when either of two independent gates fires, and the episode ends only once *every* bandit is repelled:

- **Distance**: the bandit is farther than `escape_radius_m` from the lady. `escape_radius_m=0`, the default, disables this gate.
- **Empty tank**: with `repel_on_empty_tank=True`, the bandit has no propellant left and its ballistic coast cannot reach the breach sphere in the steps the horizon still allows. The coast is propagated with the powers of the one-step HCW state transition matrix for the reference orbit, so a bandit stranded on a bounded relative ellipse is repelled while one still drifting toward the lady is not. The gate therefore requires `truth_dynamics=HCW_RTN`.

The empty-tank gate reads `state.bandits.propellant_mass` and coasts the bandit's RTN state, so it requires both a `MASS` and an `RTN` component on the bandit side. Setting it on a layout that has neither raises when the `ScenarioConfig` is built.

An episode ended this way classifies as `Outcome.REPELLED`.

`max_steps` and `dt` are read from the cfg at call time — the termination class itself does not store them.

## Builder

```python
from orbitalgym import make_lady_bandit_guard

cfg = make_lady_bandit_guard(
    breach_radius_m=5.0,
    catch_radius_m=50.0,
    breach_speed_mps=float("inf"),
    catch_speed_mps=float("inf"),
    catch_dwell_steps=0,
    breach_dwell_steps=0,
    escape_radius_m=0.0,
    repel_on_empty_tank=False,
    dv_cost=0.0,
    shaping_scale_m=300.0,
    shaping_gain=50.0,
    shaping_discount=1.0,
    home_weight=0.0,
    guard_separation_m=20.0,
    lady_keepout_m=20.0,
    separation_cost=10.0,
    max_horizon_s=2000.0,
    seed=0,
)
```

`LadyBanditGuard` owns these knobs and provides the matching reward + termination via `default_reward_fn` / `default_termination_fn`. `ScenarioConfig.__post_init__` calls those automatically — the builder never sets `reward_fn` or `termination_fn` explicitly. To override either, pass it as a constructor kwarg on `ScenarioConfig` directly:

```python
from orbitalgym.config import ScenarioConfig
from orbitalgym.games import LadyBanditGuard
from my_project.rewards import MyCustomReward

cfg = ScenarioConfig(
    ...,
    game=LadyBanditGuard(breach_radius_m=5.0, catch_radius_m=50.0),
    reward_fn=MyCustomReward(),  # game default skipped; termination_fn still flows
)
```

## Knobs at a glance

| Field | Type | Default | What it does |
|---|---|---|---|
| `n_guards` | `int` | `1` | Number of guard vehicles. |
| `n_bandits` | `int` | `1` | Number of bandit vehicles. |
| `breach_radius_m` | `float` | `5.0` | Bandit wins when any bandit-to-lady distance falls below this radius. |
| `catch_radius_m` | `float` | `50.0` | Guard wins when any guard-to-bandit distance falls below this radius. |
| `catch_dwell_steps` | `int` | `0` | Consecutive steps a bandit must hold the catch radius of some guard before the catch fires. `0` fires on the first step inside it. |
| `breach_dwell_steps` | `int` | `0` | Consecutive steps a bandit must hold the breach radius of the lady before the breach fires. `0` fires on the first step inside it. |
| `escape_radius_m` | `float` | `0.0` | Bandit counts as repelled beyond this distance from the lady. `0` disables the gate. |
| `repel_on_empty_tank` | `bool` | `False` | Treat an out-of-propellant bandit that can no longer coast to the lady as repelled. Needs a mass-tracked bandit in an RTN frame. |
| `dv_cost` | `float` | `0.0` | Reward units charged per m/s of delta-v. Each side pays for its own fuel only; a side without a `MASS` component pays nothing. |
| `max_horizon_s` | `float` | `2000.0` | Total episode duration in seconds. |
| `dt` | `float` | `10.0` | Step size in seconds. |
| `seed` | `int` | `0` | PRNG seed for IC sampling. |

For the full field list see [API → `LadyBanditGuard`](../api/games.md).

## Communication variant

`make_lady_bandit_guard(with_communication=True)` gives the guard side the
`COMMUNICATE` action component and installs `LbgWithCommsReward`, which
scores distance to the reference orbit plus a charge of `comm_cost` per
active broadcast. That reward carries none of the LBG event geometry: no
catch or breach radius, no speed gate, no repel gate, no `dv_cost`. The
termination still enforces all of it, so a scenario that tuned those knobs
would be terminated on one set of rules and rewarded on another.

The builder therefore rejects `with_communication=True` together with any
of `breach_radius_m`, `catch_radius_m`, `breach_speed_mps`,
`catch_speed_mps`, `catch_dwell_steps`, `breach_dwell_steps`,
`escape_radius_m`, `repel_on_empty_tank`, or `dv_cost`
set away from its default. To combine communication with tuned event
geometry, build the scenario with `with_communication=False` and pass an
explicit `reward_fn` that scores both.

## Suggested experiments

- **Sanity baseline.** Run with zero control on both sides — neither side wins, the episode ends at `max_steps` and cumulative return is dominated by the dense distance terms.
- **Bandit attack.** Wire `examples/policies/glideslope_bandit.py`'s `GlideslopeBanditPolicy` as the bandit's policy — the bandit closes on the lady; the zero-control guard should eventually lose by breach.
- **Guard interception.** Train a guard policy (e.g. MCTS, see `examples/lbg_ring_intercept.ipynb`) to maximize guard return — the guard should learn to close on the bandit before it reaches the lady.

## Variants

The four customization axes (reward, termination, IC, observation) all
swap by passing a kwarg to `ScenarioConfig` (or via `dataclasses.replace`).
Two LBG-flavored recipes:

### Variant 1: jittered chasing bandit

Wrap a `LeadInterceptPursuer` in `JitteredPolicy` to give the guard a
randomized chaser threat — the bandit closes on the lady but
with stochastic per-step deviation, preventing a guard from exploiting a
deterministic threat trajectory.

```python
--8<-- "tests/docs/test_games_lady_bandit_guard_variants.py:variant-jittered-bandit"
```

### Variant 2: tighter IC for evaluation

Shrink the IC sampler's `sigma_radial_ellipse_m` so evaluation rollouts
draw from a narrow band around the canonical phase-π geometry. Useful
when comparing learned guard policies head-to-head: lower IC variance
means lower episode-return variance, which means fewer rollouts needed
to separate two policies.

```python
--8<-- "tests/docs/test_games_lady_bandit_guard_variants.py:variant-tighter-ic"
```

For more axes, see [Extending → Customize rewards](../extending/customize-rewards.md),
[…termination](../extending/customize-termination.md),
[…IC sampling](../extending/customize-ic-sampling.md), and
[…observations](../extending/customize-observations.md).

## Built-in policies and adversaries

Sensible gallery picks for Lady-Bandit-Guard:

- **Heuristic policies (opponent):** [`LeadInterceptPursuer`](../extending/gallery.md#leadinterceptpursuer),
  [`JitteredPolicy`](../extending/gallery.md#jitteredpolicy) (wrap any of the above).
- **Controlled side:** any class from [Controlled-side cookbook](../extending/controlled-policy-cookbook.md)
  ([`HeuristicWithFallbackPolicy`](../extending/gallery.md#heuristicwithfallbackpolicy),
  [`CompositeActionPolicy`](../extending/gallery.md#compositeactionpolicy),
  [`MCTSPolicy`](../extending/gallery.md#mctspolicy)).

## Sanity-check notebook

[`examples/games/lady_bandit_guard.ipynb`](https://github.com/sisl/OrbitalGym/blob/main/examples/games/lady_bandit_guard.ipynb) is a full walkthrough that builds an LBG scenario, runs a rollout, and renders the rollout diagnostic plus the 2D guard-position reward surface.

## Where to next

- **Tutorial:** [T1 — First rollout](../tutorials/t1-first-rollout.md).
- **API:** [`LadyBanditGuard`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [State layout](../in-depth/state-layout.md).

## Ground-station extension

LBG can be extended with ground-station-gated communications. In this
mode, one (or both) sides only update their plans during contact
windows; the rest of the time they execute the most recently uplinked
plan open-loop. This models real space-domain operational constraints
where vehicles can't update from belief continuously.

The example notebook
[`examples/lbg_groundstations_delayed_planning.ipynb`](https://github.com/sisl/OrbitalGym/blob/main/examples/lbg_groundstations_delayed_planning.ipynb)
demonstrates a 2-guard / 1-bandit scenario where the bandit's planning
is gated by a 2-station network (Alaska + Australia) with a 1-contact
upload lag. A lag sweep over `replan_contacts_lag ∈ {0, 1, 2, 3}` shows
how the guards can exploit the bandit's planning gaps as lag grows.

See:

- [Ground stations (in-depth)](../in-depth/ground-stations.md)
- [Modeling comms windows (how-to)](../how-to/model-comms-windows.md)
