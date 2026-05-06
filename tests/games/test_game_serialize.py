"""Round-trip serialization tests for all four games + NoGame."""

from __future__ import annotations

import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.games import (
    LadyBanditGuard,
    NoGame,
    ObservationBlocking,
    PursuitEvasion,
    SunBlocking,
    make_game,
    make_lady_bandit_guard,
    make_observation_blocking,
    make_pursuit_evasion,
    make_sun_blocking,
)
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import GameKey, StateComponentKey
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec


def test_lbg_roundtrip():
    cfg = make_lady_bandit_guard(breach_radius_m=12.0)
    cfg2 = ScenarioConfig.from_json(cfg.to_json())
    assert isinstance(cfg2.game, LadyBanditGuard)
    assert cfg2.game.breach_radius_m == 12.0


def test_pe_roundtrip():
    cfg = make_pursuit_evasion(capture_distance_m=20.0)
    cfg2 = ScenarioConfig.from_json(cfg.to_json())
    assert isinstance(cfg2.game, PursuitEvasion)
    assert cfg2.game.capture_distance_m == 20.0


def test_sb_roundtrip():
    cfg = make_sun_blocking(target_viewing_distance_m=250.0, range_decay_coef=2.0e-6)
    cfg2 = ScenarioConfig.from_json(cfg.to_json())
    assert isinstance(cfg2.game, SunBlocking)
    assert cfg2.game.target_viewing_distance_m == 250.0
    assert cfg2.game.range_decay_coef == 2.0e-6


def test_ob_roundtrip():
    cfg = make_observation_blocking(
        target_lat_deg=51.5,
        target_lon_deg=-0.1,
        min_elevation_deg=15.0,
    )
    cfg2 = ScenarioConfig.from_json(cfg.to_json())
    assert isinstance(cfg2.game, ObservationBlocking)
    assert cfg2.game.target_lat_deg == 51.5
    assert cfg2.game.min_elevation_deg == 15.0


def test_no_game_roundtrip():
    """A hand-built ScenarioConfig with no explicit game keeps NoGame on roundtrip."""
    cfg = ScenarioConfig(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        guard_components=(StateComponentKey.RTN,),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        bandit_params=VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=jnp.pi,
                sigma_radial_ellipse_m=10.0,
            ),
            validators=(),
            max_attempts=100,
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )
    assert isinstance(cfg.game, NoGame)
    cfg2 = ScenarioConfig.from_json(cfg.to_json())
    assert isinstance(cfg2.game, NoGame)


def test_make_game_dispatches_correctly():
    cfg = make_game(GameKey.PURSUIT_EVASION, capture_distance_m=42.0)
    assert isinstance(cfg.game, PursuitEvasion)
    assert cfg.game.capture_distance_m == 42.0


def test_make_game_lbg():
    cfg = make_game(GameKey.LADY_BANDIT_GUARD, breach_radius_m=8.0)
    assert isinstance(cfg.game, LadyBanditGuard)
    assert cfg.game.breach_radius_m == 8.0


def test_make_game_sb():
    cfg = make_game(GameKey.SUN_BLOCKING, target_viewing_distance_m=750.0)
    assert isinstance(cfg.game, SunBlocking)
    assert cfg.game.target_viewing_distance_m == 750.0


def test_make_game_ob():
    cfg = make_game(
        GameKey.OBSERVATION_BLOCKING,
        target_lat_deg=10.0,
        target_lon_deg=20.0,
    )
    assert isinstance(cfg.game, ObservationBlocking)
    assert cfg.game.target_lat_deg == 10.0


def test_make_game_unknown_key_raises():
    """make_game with an unsupported key raises KeyError."""
    try:
        make_game(GameKey.NONE)
    except KeyError as e:
        assert "NONE" in str(e) or "no_game" in str(e).lower()
    else:
        raise AssertionError("expected KeyError for GameKey.NONE")
