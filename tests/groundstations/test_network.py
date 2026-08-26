import jax
import jax.numpy as jnp
import pytest

from orbitalgym.groundstations import ContactSchedule, GroundStation, GroundStationNetwork


def test_ground_station_construction():
    s = GroundStation(
        name="alaska",
        lat_deg=jnp.asarray(64.8),
        lon_deg=jnp.asarray(-147.7),
        altitude_m=jnp.asarray(150.0),
        elevation_mask_deg=jnp.asarray(5.0),
    )
    assert s.name == "alaska"
    assert float(s.lat_deg) == pytest.approx(64.8)


def test_ground_station_rejects_bad_latitude():
    with pytest.raises(ValueError, match="lat_deg"):
        GroundStation(
            name="bad",
            lat_deg=jnp.asarray(95.0),
            lon_deg=jnp.asarray(0.0),
            altitude_m=jnp.asarray(0.0),
            elevation_mask_deg=jnp.asarray(5.0),
        )


def test_ground_station_rejects_bad_elevation_mask():
    with pytest.raises(ValueError, match="elevation_mask_deg"):
        GroundStation(
            name="bad",
            lat_deg=jnp.asarray(0.0),
            lon_deg=jnp.asarray(0.0),
            altitude_m=jnp.asarray(0.0),
            elevation_mask_deg=jnp.asarray(95.0),
        )


def test_contact_schedule_construction():
    sch = ContactSchedule(
        windows=jnp.asarray([[0.0, 600.0], [3600.0, 4200.0], [-1.0, -1.0]]),
        n_valid=jnp.asarray(2),
        station_ix=jnp.asarray([0, 0, -1]),
    )
    assert int(sch.n_valid) == 2
    assert sch.windows.shape == (3, 2)


def test_ground_station_network_rejects_empty_stations():
    with pytest.raises(ValueError, match="at least one station"):
        GroundStationNetwork(stations=())


def test_ground_station_pytree_roundtrip():
    """Verify `name` (pytree_node=False) is preserved through tree flatten/unflatten."""
    s = GroundStation(
        name="alaska",
        lat_deg=jnp.asarray(64.8),
        lon_deg=jnp.asarray(-147.7),
        altitude_m=jnp.asarray(150.0),
        elevation_mask_deg=jnp.asarray(5.0),
    )
    leaves, treedef = jax.tree.flatten(s)
    s2 = jax.tree.unflatten(treedef, leaves)
    assert s2.name == "alaska"
    assert float(s2.lat_deg) == pytest.approx(64.8)


def test_ground_station_jit_compatible():
    """Verify the dataclass survives a JIT trace — name stays static, arrays trace."""
    s = GroundStation(
        name="alaska",
        lat_deg=jnp.asarray(64.8),
        lon_deg=jnp.asarray(-147.7),
        altitude_m=jnp.asarray(150.0),
        elevation_mask_deg=jnp.asarray(5.0),
    )
    f = jax.jit(lambda x: x.lat_deg + 1.0)
    out = f(s)
    assert float(out) == pytest.approx(65.8)
