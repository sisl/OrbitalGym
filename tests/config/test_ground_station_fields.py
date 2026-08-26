import jax.numpy as jnp
import pytest

from orbitalgym.groundstations import (
    ContactSchedule,
    GroundStation,
    GroundStationNetwork,
)


def _network_with_schedule(n_valid=1):
    station = GroundStation(
        name="A",
        lat_deg=jnp.asarray(0.0),
        lon_deg=jnp.asarray(0.0),
        altitude_m=jnp.asarray(0.0),
        elevation_mask_deg=jnp.asarray(5.0),
    )
    sch = ContactSchedule(
        windows=jnp.asarray([[100.0, 200.0]] + [[-1.0, -1.0]] * 7, dtype=jnp.float32),
        n_valid=jnp.asarray(n_valid),
        station_ix=jnp.asarray([0] + [-1] * 7),
    )
    return GroundStationNetwork(stations=(station,), schedule=sch)


def test_scenarioconfig_accepts_ground_station_networks(make_minimal_lbg_config):
    cfg = make_minimal_lbg_config(
        guard_ground_station_network=_network_with_schedule(),
        bandit_ground_station_network=_network_with_schedule(),
    )
    assert cfg.guard_ground_station_network is not None
    assert cfg.bandit_ground_station_network is not None


def test_scenarioconfig_rejects_empty_schedule(make_minimal_lbg_config):
    with pytest.raises(ValueError, match="schedule has no valid windows"):
        make_minimal_lbg_config(
            guard_ground_station_network=_network_with_schedule(n_valid=0),
        )


def test_scenarioconfig_defaults_to_none(make_minimal_lbg_config):
    cfg = make_minimal_lbg_config()
    assert cfg.guard_ground_station_network is None
    assert cfg.bandit_ground_station_network is None
