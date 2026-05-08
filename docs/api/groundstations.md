# `orbital_game.groundstations`

Ground-station network and contact-schedule primitives.

## Network types

::: orbital_game.groundstations.network.GroundStation

::: orbital_game.groundstations.network.ContactSchedule

::: orbital_game.groundstations.network.GroundStationNetwork

## Contact computation

The runtime predicate `in_contact_now` is pure JAX and JIT-stable; the
brahe-backed builder `precompute_contact_schedule` runs once at
scenario construction.

::: orbital_game.groundstations.contacts.precompute_contact_schedule

::: orbital_game.groundstations.contacts.in_contact_now
