# `orbitalgym.groundstations`

Ground-station network and contact-schedule primitives.

## Network types

::: orbitalgym.groundstations.network.GroundStation

::: orbitalgym.groundstations.network.ContactSchedule

::: orbitalgym.groundstations.network.GroundStationNetwork

## Contact computation

The runtime predicate `in_contact_now` is pure JAX and JIT-stable; the
brahe-backed builder `precompute_contact_schedule` runs once at
scenario construction.

::: orbitalgym.groundstations.contacts.precompute_contact_schedule

::: orbitalgym.groundstations.contacts.in_contact_now
