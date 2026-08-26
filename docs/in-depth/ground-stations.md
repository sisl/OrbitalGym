# Ground stations

Real space-domain operations don't get to plan continuously from a fresh
belief. Vehicles update their on-board plans only when they pass through
a ground-station contact window — moments later, that plan executes
**open-loop** until the next contact, with no replanning in between.
The `orbitalgym.groundstations` and `orbitalgym.policies.plan_cache`
modules model that operational dynamic explicitly: contact-windowed
comms, an intentional uplink lag between belief and plan, and per-side
asymmetry in who has visibility of whom.

## What this models

Two operationally distinct effects:

1. **Contact-windowed comms.** A vehicle can only receive a fresh plan
   while it's in line-of-sight of a ground station. Outside the window,
   the cached plan executes open-loop.
2. **Uplink lag.** Even inside a contact, the freshest belief used to
   build the plan you're uploading might lag by one or more contacts —
   ops loops are not instantaneous.

Together these recover the operational picture: contact `c` produces a
belief; contact `c+1` uplinks the plan that belief drove; the belief
gathered during `c+1` lands in the ring buffer for contact `c+2`.

## Geometry layer

The geometry layer is plain JAX pytrees:

- **`GroundStation`** — a fixed Earth point: name, lat/lon/alt, and an
  elevation mask. Validates lat/lon/alt at construction time.
- **`ContactSchedule`** — a side's precomputed contact windows over the
  episode horizon. Internally a fixed-shape `(N_max, 2)` padded array
  with sentinel rows guarded by an `idx < n_valid` mask, so the runtime
  predicate runs cleanly under `jit` and `lax.scan`.
- **`GroundStationNetwork`** — pairs the static-tuple stations with the
  precomputed schedule. The schedule field is filled by
  `precompute_contact_schedule` at scenario construction.

The brahe-backed builder
[`precompute_contact_schedule`](../api/groundstations.md) runs **once**,
at construction time, outside the JAX trace. brahe handles
EOP-aware ECI↔ECEF rotation and elevation-constraint search via
`KeplerianPropagator` + `location_accesses`. The runtime predicate
[`in_contact_now`](../api/groundstations.md), in contrast, is pure JAX
— a single `jnp.any` over the padded windows array — so it composes
freely under `jit`/`vmap`/`scan`.

This split is deliberate: brahe does precise once-per-scenario geometry;
JAX does cheap per-tick predicate evaluation.

## Decision layer: `PlanCachePolicy`

The contact-gated policy wrapper lives in
[`orbitalgym.policies.plan_cache`](../api/policies-plan-cache.md).
It wraps any inner `Policy` and converts continuous replanning into
contact-gated replanning with an explicit lag.

Two knobs control the lag:

| `replan_contacts_lag` | Operational meaning |
|---|---|
| `0` | This contact's belief drives this contact's plan (no operational lag). |
| `1` (default) | Last contact's belief drives this contact's plan; this contact's belief feeds next contact. |
| `N >= 2` | Belief from N contacts ago drives this contact's plan. |

| `upload_delay_s` | Operational meaning |
|---|---|
| `0.0` (default) | Plan uploads at the start of contact. |
| `M > 0` | Wait `M` seconds inside a contact before plan upload (handshake / negotiation lag). |
| `> contact length` | Upload silently skipped this contact (handles short passes correctly). |

The default `replan_contacts_lag=1` matches the operational picture
described above.

## Open-loop execution between contacts

The inner planner is invoked **only at upload events** (typically once
per contact, after `upload_delay_s`, gated by the lag predicate).
Out-of-contact ticks do exactly one thing: pop the next slot from the
cached plan and emit it. There is no MCTS rerun, no LQR resolve, no
heuristic re-evaluation between contacts — the wrapper executes a
stored open-loop plan.

This matters for two reasons. First, it's the right operational model:
on-orbit vehicles don't run a fresh planner at every tick. Second, it
makes the cost of `replan_contacts_lag` legible — at lag `N`, the agent
goes longer between upload events, which is exactly the regime where
the *opposing* side gets to exploit a stale cached plan.

## First-contact behavior at `replan_contacts_lag >= 1`

The upload trigger requires `contact_count > replan_contacts_lag`, so
at lag `N` the **first N contacts cannot upload** — there's no lagged
belief to act on yet. The cached command stays at its initial value
(computed from the initial belief by `init_state`) and the agent
executes that command every tick until the (N+1)th contact, when the
first upload fires. This is the correct operational analogue of "you
can't act on data you haven't received yet".

## Sequence diagram for one contact

```text
t = t_entered_contact:    enter contact, push belief into ring buffer
t = t_entered + delay:    upload trigger (if lag satisfied) —
                          call inner(belief[lag], ...) → store cached_dv
t = t_entered + dt:       execute cached_dv (every tick)
...
t = t_exit_contact:       leave contact; continue executing the
                          cached_dv until next upload event
```

## Belief fusion

When teammates are simultaneously in contact, their beliefs about the
opposing side can be **fused** into a sharper joint posterior. The
[`BeliefSyncFn`](../api/belief-sync.md) protocol runs once per side per
tick, after the per-side belief update and before policy invocation.
Three concrete implementations ship:

- **`KFTeamFusion`** — information-form pooling for `KFBelief`. For
  each opposing target, observers in contact share information state
  `(P^-1 mu, P^-1)`; the pooled posterior overwrites every in-contact
  observer's `(mu_i, P_i)` for that target. Out-of-contact observers
  untouched. Optimal under independent observers.
- **`EKFTeamFusion`** — same algorithm, EKFBelief shapes; distinct
  registry key for serialization.
- **`PFTeamFusion`** — joint-resample fusion for
  `ParticleFilterBelief`. Per opposing target, gather particles from
  every in-contact observer (excluding self-pairs), build a combined
  weighted cloud, and systematic-resample to K particles per
  in-contact observer.

Default is information-form pooling, which is optimal under independent
observers. Covariance intersection (for known-correlated cases — e.g.
all teammates pulling from the same operator estimate) is a follow-up
under a separate registry key.

The `ContactAwareBelief` wrapper exposes `.mean` (delegated to the
inner belief) plus `.contact` (per-agent bool mask). Existing policies
that read only `.mean` are unaffected; `PlanCachePolicy` is the only
consumer that reads `.contact`.

## Side asymmetry

Contact networks are wired per side, so four configurations come for
free:

- **Both sides know all stations.** Symmetric model — both sides treat
  the same network as ground truth and both sides plan with the same
  lag.
- **Only one side has stations.** Pass `None` for the other side's
  `*_ground_station_network` kwarg; that side plans continuously, the
  other side plays the contact-gated game. The classic operational
  asymmetry: defenders own the ground network, attackers don't.
- **Asymmetric stations.** Each side has its own network with
  different station locations or different schedules.
- **Jittered opponent-model knowledge.** The opposing side's MCTS
  `opponent_model` is built against a *perturbed* copy of the real
  network — models "I think I know where their stations are, but my
  estimate is imperfect."

See the [how-to](../how-to/model-comms-windows.md) for runnable code
snippets for each.

## What this does NOT model

- **Intermittent station outages** — the precomputed schedule is
  static for the episode.
- **Bandwidth budgets / partial uploads** — uploads are atomic; the
  full H-step plan transfers in one event.
- **Sub-tick contact entry/exit** — `in_contact_now` evaluates at the
  tick boundary; sub-`dt` window edges round to the nearest tick.
- **Mobile platforms** (ship-borne, airborne) — stations are
  Earth-fixed lat/lon/alt.

## Where to next

- [Modeling comms windows](../how-to/model-comms-windows.md) — runnable
  recipes: single-station network, configuring lag, asymmetric
  knowledge.
- [`examples/lbg_groundstations_delayed_planning.ipynb`](https://github.com/sisl/OrbitalGym/blob/main/examples/lbg_groundstations_delayed_planning.ipynb) — full
  2-guard / 1-bandit lag-sweep with rendered animations.
- [API → groundstations](../api/groundstations.md),
  [API → plan-cache](../api/policies-plan-cache.md),
  [API → belief-sync](../api/belief-sync.md).
- [Spec](https://github.com/sisl/OrbitalGym/blob/main/superpowers/specs/2026-05-07-ground-station-comms-design.md) — design rationale and acceptance criteria.
