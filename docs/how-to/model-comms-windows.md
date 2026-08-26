# Model comms windows

Three recipes for adding ground-station-gated comms to a scenario:
build a basic single-station network, configure the upload-lag knobs,
and set up asymmetric per-side knowledge.

For the operational rationale and design discussion, see
[Ground stations (in-depth)](../in-depth/ground-stations.md).

## 1. Build a single-station network

The minimum viable comms network: one ground station, a precomputed
contact schedule against the reference orbit, attached to one side of
the LBG scenario.

```python
import jax.numpy as jnp
from orbitalgym.groundstations import (
    GroundStation, GroundStationNetwork
)
from orbitalgym.groundstations.contacts import precompute_contact_schedule
from orbitalgym.reference_orbit import ReferenceOrbitState

reference = ReferenceOrbitState.from_keplerian(
    semi_major_axis_m=6878e3, eccentricity=0.0, inclination=51.6,
    raan=0.0, argument_of_perigee=0.0, mean_anomaly=0.0,
)

stations = (
    GroundStation(
        name="alaska",
        lat_deg=jnp.asarray(64.8),
        lon_deg=jnp.asarray(-147.7),
        altitude_m=jnp.asarray(150.0),
        elevation_mask_deg=jnp.asarray(5.0),
    ),
)

schedule = precompute_contact_schedule(
    stations=stations,
    reference_orbit=reference,
    epoch_mjd_utc=60067.0,
    horizon_s=3 * 3600.0,
)
network = GroundStationNetwork(stations=stations, schedule=schedule)

# Attach to the scenario:
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
cfg = make_lady_bandit_guard(
    n_guards=1, n_bandits=1,
    guard_ground_station_network=network,
)
```

`precompute_contact_schedule` runs brahe **once** at construction time
to find every (station, satellite) access window over `horizon_s`. The
runtime predicate `in_contact_now` is then pure JAX — no brahe at tick
time.

## 2. Configure `replan_contacts_lag` and `upload_delay_s`

`PlanCachePolicy` wraps any inner planner and gates uploads on the
contact schedule. The two knobs that shape its behaviour:

| `replan_contacts_lag` | Meaning |
|---|---|
| 0 | This contact's belief drives this contact's plan (no operational lag). |
| 1 (default) | Last contact's belief drives this contact's plan; this contact's belief feeds next contact. |
| N >= 2 | Belief from N contacts ago drives this contact's plan. |

| `upload_delay_s` | Meaning |
|---|---|
| 0.0 (default) | Plan uploads at the start of contact. |
| 300.0 | Plan uploads 5 minutes after contact entry (handshake delay). |
| > contact length | Upload silently skipped this contact. |

```python
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.policies.plan_cache import PlanCachePolicy
from orbitalgym.policies.mcts import MCTSPolicy  # or any inner planner

env = OrbitalGymEnv(cfg)

inner = MCTSPolicy(...)
cached = PlanCachePolicy(
    inner=inner,
    schedule=network.schedule,
    dt=cfg.dt,
    replan_contacts_lag=1,    # operational default
    upload_delay_s=300.0,     # 5-minute uplink handshake
    n_vehicles=cfg.n_bandits,
    command_cls=env.bandit_command_cls,
)
```

At `replan_contacts_lag=N` with `N >= 1`, the first N contacts cannot
upload (no lagged belief exists yet) and the agent free-drifts until
the (N+1)th contact — see
[Ground stations → first-contact behavior](../in-depth/ground-stations.md).

## 3. Asymmetric knowledge

Networks are wired per side, so common operational asymmetries fall out
of the builder kwargs.

### 3a. Only one side has stations

The most common operational picture: defenders run the ground network,
attackers do not. The unwired side plans continuously; the wired side
is contact-gated.

```python
cfg = make_lady_bandit_guard(
    guard_ground_station_network=None,         # guard plans continuously
    bandit_ground_station_network=bandit_net,  # bandit gated
)
```

### 3b. Different stations per side

Each side has its own network with its own stations and schedule.
Useful for studying how station-geometry differences shape strategic
windows.

```python
cfg = make_lady_bandit_guard(
    guard_ground_station_network=guard_net,
    bandit_ground_station_network=bandit_net,
)
```

### 3c. Jittered opponent-model knowledge

Build the bandit's MCTS `opponent_model` against a **perturbed** copy
of the guard's real network — models "I think I know where their
stations are, but my model is imperfect."

```python
import jax.random as jr

def jittered(network, key, sigma_deg=2.0):
    keys = jr.split(key, len(network.stations))
    new_stations = tuple(
        GroundStation(
            name=st.name + "_jittered",
            lat_deg=st.lat_deg + sigma_deg * jr.normal(k_i, ()),
            lon_deg=st.lon_deg + sigma_deg * jr.normal(jr.fold_in(k_i, 1), ()),
            altitude_m=st.altitude_m,
            elevation_mask_deg=st.elevation_mask_deg,
        )
        for st, k_i in zip(network.stations, keys, strict=True)
    )
    schedule = precompute_contact_schedule(
        stations=new_stations,
        reference_orbit=reference,
        epoch_mjd_utc=cfg.epoch_mjd_utc,
        horizon_s=3 * 3600.0,
    )
    return GroundStationNetwork(stations=new_stations, schedule=schedule)

bandits_belief_about_guard = jittered(guard_net, jr.PRNGKey(0))
# Use this in the bandit's MCTS opponent_model construction — the
# bandit's tree search treats the perturbed network as ground truth.
```

The bandit's tree search now reasons about a guard whose comms windows
fire at the *jittered* schedule, while the actual rollout uses the
real `guard_net`.

## See also

- [Ground stations (in-depth)](../in-depth/ground-stations.md) — design
  rationale, lag semantics, side-asymmetry discussion.
- [`examples/lbg_groundstations_delayed_planning.ipynb`](https://github.com/sisl/OrbitalGym/blob/main/examples/lbg_groundstations_delayed_planning.ipynb) — full
  worked example with a lag sweep.
- [API → groundstations](../api/groundstations.md),
  [API → plan-cache](../api/policies-plan-cache.md).
