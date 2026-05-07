"""Tests for ReferenceOrbitState."""

import jax
import jax.numpy as jnp
import pytest

from orbital_game.reference_orbit import MU_EARTH, ReferenceOrbitState


def test_reference_orbit_state_construction_has_expected_fields():
    state = ReferenceOrbitState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
    )
    assert state.position_eci.shape == (3,)
    assert state.velocity_eci.shape == (3,)


def test_reference_orbit_state_is_pytree():
    """Must flatten/unflatten as a pytree so jax.tree_util treats it as a leaf container."""
    state = ReferenceOrbitState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
    )
    flat, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, flat)
    assert jnp.allclose(restored.position_eci, state.position_eci)
    assert jnp.allclose(restored.velocity_eci, state.velocity_eci)


# ---------------------------------------------------------------------------
# from_keplerian — alternate constructor accepting Keplerian elements.
# Convention: [a (m), e, i, RAAN, omega, M]; M is mean anomaly.
# Angles default to degrees (as_degrees=True); pass as_degrees=False for radians.
# ---------------------------------------------------------------------------


def test_from_keplerian_circular_equatorial_matches_analytic():
    """Circular equatorial orbit at periapsis: r=(a,0,0), v=(0, sqrt(mu/a), 0)."""
    a = 7000e3
    state = ReferenceOrbitState.from_keplerian(
        semi_major_axis_m=a,
        eccentricity=0.0,
        inclination=0.0,
        raan=0.0,
        argument_of_perigee=0.0,
        mean_anomaly=0.0,
    )
    expected_speed = jnp.sqrt(MU_EARTH / a)
    assert jnp.allclose(state.position_eci, jnp.array([a, 0.0, 0.0]), atol=1e-3)
    assert jnp.allclose(state.velocity_eci, jnp.array([0.0, expected_speed, 0.0]), atol=1e-6)


def test_from_keplerian_returns_reference_orbit_state():
    state = ReferenceOrbitState.from_keplerian(
        semi_major_axis_m=7000e3,
        eccentricity=0.001,
        inclination=0.1,
        raan=0.2,
        argument_of_perigee=0.3,
        mean_anomaly=0.4,
        as_degrees=False,
    )
    assert isinstance(state, ReferenceOrbitState)
    assert state.position_eci.shape == (3,)
    assert state.velocity_eci.shape == (3,)


def test_from_keplerian_round_trip_via_astrojax_eci_to_koe():
    """Building from KOE (radians path) then converting back via astrojax recovers inputs."""
    from astrojax import state_eci_to_koe

    inputs = dict(
        semi_major_axis_m=7100e3,
        eccentricity=0.01,
        inclination=0.5,
        raan=1.0,
        argument_of_perigee=0.7,
        mean_anomaly=0.3,
    )
    state = ReferenceOrbitState.from_keplerian(**inputs, as_degrees=False)
    eci6 = jnp.concatenate([state.position_eci, state.velocity_eci])
    koe = state_eci_to_koe(eci6)
    assert jnp.allclose(koe[0], inputs["semi_major_axis_m"], rtol=1e-6)
    assert jnp.allclose(koe[1], inputs["eccentricity"], atol=1e-9)
    assert jnp.allclose(koe[2], inputs["inclination"], atol=1e-9)
    assert jnp.allclose(koe[3], inputs["raan"], atol=1e-9)
    assert jnp.allclose(koe[4], inputs["argument_of_perigee"], atol=1e-9)
    assert jnp.allclose(koe[5], inputs["mean_anomaly"], atol=1e-9)


def test_from_keplerian_defaults_to_degrees():
    """Default as_degrees=True: 90° inclination should equal π/2 rad inclination."""
    deg_state = ReferenceOrbitState.from_keplerian(
        semi_major_axis_m=7000e3,
        eccentricity=0.0,
        inclination=90.0,  # degrees
        raan=0.0,
        argument_of_perigee=0.0,
        mean_anomaly=0.0,
    )
    rad_state = ReferenceOrbitState.from_keplerian(
        semi_major_axis_m=7000e3,
        eccentricity=0.0,
        inclination=jnp.pi / 2,  # radians
        raan=0.0,
        argument_of_perigee=0.0,
        mean_anomaly=0.0,
        as_degrees=False,
    )
    assert jnp.allclose(deg_state.position_eci, rad_state.position_eci, atol=1e-3)
    assert jnp.allclose(deg_state.velocity_eci, rad_state.velocity_eci, atol=1e-6)


def test_from_keplerian_degrees_and_radians_paths_agree():
    """Same orbit specified in deg vs rad should produce identical ECI states."""
    deg_state = ReferenceOrbitState.from_keplerian(
        semi_major_axis_m=7100e3,
        eccentricity=0.01,
        inclination=45.0,
        raan=30.0,
        argument_of_perigee=60.0,
        mean_anomaly=15.0,
    )
    rad_state = ReferenceOrbitState.from_keplerian(
        semi_major_axis_m=7100e3,
        eccentricity=0.01,
        inclination=jnp.deg2rad(45.0),
        raan=jnp.deg2rad(30.0),
        argument_of_perigee=jnp.deg2rad(60.0),
        mean_anomaly=jnp.deg2rad(15.0),
        as_degrees=False,
    )
    assert jnp.allclose(deg_state.position_eci, rad_state.position_eci, atol=1e-3)
    assert jnp.allclose(deg_state.velocity_eci, rad_state.velocity_eci, atol=1e-6)


def test_from_keplerian_rejects_negative_eccentricity():
    with pytest.raises(ValueError, match="eccentricity"):
        ReferenceOrbitState.from_keplerian(
            semi_major_axis_m=7000e3,
            eccentricity=-0.1,
            inclination=0.0,
            raan=0.0,
            argument_of_perigee=0.0,
            mean_anomaly=0.0,
        )


def test_from_keplerian_rejects_unbounded_eccentricity():
    with pytest.raises(ValueError, match="eccentricity"):
        ReferenceOrbitState.from_keplerian(
            semi_major_axis_m=7000e3,
            eccentricity=1.0,
            inclination=0.0,
            raan=0.0,
            argument_of_perigee=0.0,
            mean_anomaly=0.0,
        )


def test_from_keplerian_rejects_non_positive_semi_major_axis():
    with pytest.raises(ValueError, match="semi_major_axis"):
        ReferenceOrbitState.from_keplerian(
            semi_major_axis_m=0.0,
            eccentricity=0.0,
            inclination=0.0,
            raan=0.0,
            argument_of_perigee=0.0,
            mean_anomaly=0.0,
        )
