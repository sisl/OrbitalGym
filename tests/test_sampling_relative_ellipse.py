"""Tests for RelativeEllipse per-side sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.dynamics.hcw import hcw_rtn_step
from orbital_game.hva import HVAState
from orbital_game.registry import StateComponentKey
from orbital_game.sampling.side import RelativeEllipse

MU_EARTH = 3.986004418e14


def _config(n_def=1):
    from orbital_game.sampling.spec import ICSpec
    return ScenarioConfig(
        n_defenders=n_def,
        n_intruders=1,
        epoch_mjd_utc=60067.0,
        hva=HVAState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            # Circular equatorial HVA — fine for RelativeEllipse since it
            # doesn't go through Keplerian elements (no singularity).
            velocity_eci=jnp.array([0.0, 7546.05, 0.0]),
        ),
        defender_components=(StateComponentKey.RTN,),
        intruder_components=(StateComponentKey.RTN,),
        defender_params=VehicleParamsSpec(100.0, 220.0, 5.0),
        intruder_params=VehicleParamsSpec(50.0, 200.0, 2.0),
        ic_sampler=ICSpec(
            defender_sampler=RelativeEllipse(radial_ellipse_m=0.0, phase_rad=0.0),
            intruder_sampler=RelativeEllipse(radial_ellipse_m=0.0, phase_rad=0.0),
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )


def _hva_mean_motion(cfg):
    r = jnp.linalg.norm(cfg.hva.position_eci)
    return jnp.sqrt(MU_EARTH / r**3)


def test_phase_zero_radial_ellipse_only_puts_vehicle_at_radial_max():
    sampler = RelativeEllipse(radial_ellipse_m=100.0, phase_rad=0.0)
    cfg = _config()
    out = sampler(
        cfg,
        jax.random.PRNGKey(0),
        n_vehicles=1,
        components=(StateComponentKey.RTN,),
        class_name="DefenderState",
    )
    rtn = out.rtn[0]
    # phase=0 closed form: R = -radial_e * cos(0) = -100, T = 0, N = 0
    assert jnp.isclose(rtn[0], -100.0, atol=1.0)
    assert jnp.isclose(rtn[1], 0.0, atol=1.0)
    assert jnp.isclose(rtn[2], 0.0, atol=1.0)


def test_along_track_offset_only_yields_co_orbital_companion():
    sampler = RelativeEllipse(along_track_offset_m=500.0, phase_rad=0.0)
    cfg = _config()
    out = sampler(
        cfg,
        jax.random.PRNGKey(0),
        n_vehicles=1,
        components=(StateComponentKey.RTN,),
        class_name="DefenderState",
    )
    rtn = out.rtn[0]
    # Pure along-track offset, no in-plane motion -> R=0, T=500, N=0
    assert jnp.isclose(rtn[0], 0.0, atol=1.0)
    assert jnp.isclose(rtn[1], 500.0, atol=1.0)
    assert jnp.isclose(rtn[2], 0.0, atol=1.0)
    # Velocity satisfies bounded-orbit condition: Tdot ≈ 0 for pure offset
    assert jnp.isclose(rtn[4], 0.0, atol=1e-3)


def test_boundedness_invariant_radial_ellipse():
    """Propagate a sampled IC under HCW for 10 orbits; max radial excursion stays
    within configured radial_ellipse_m + small tolerance."""
    sampler = RelativeEllipse(radial_ellipse_m=200.0, phase_rad=0.0)
    cfg = _config()
    out = sampler(
        cfg,
        jax.random.PRNGKey(0),
        n_vehicles=1,
        components=(StateComponentKey.RTN,),
        class_name="DefenderState",
    )
    n = _hva_mean_motion(cfg)
    period = 2 * jnp.pi / n

    class _Params:
        mean_motion = float(n)

    state = out.rtn
    dt = float(period / 200.0)
    radial_max = jnp.abs(state[0, 0])
    for _ in range(2000):  # 10 periods
        state = hcw_rtn_step(state, jnp.zeros((1, 3)), _Params(), dt)
        radial_max = jnp.maximum(radial_max, jnp.abs(state[0, 0]))
    assert float(radial_max) <= 200.0 + 1.0


def test_phase_uniform_when_phase_rad_is_none():
    sampler = RelativeEllipse(radial_ellipse_m=100.0, phase_rad=None)
    cfg = _config(n_def=64)
    out = sampler(
        cfg,
        jax.random.PRNGKey(0),
        n_vehicles=64,
        components=(StateComponentKey.RTN,),
        class_name="DefenderState",
    )
    radial = out.rtn[:, 0]
    assert jnp.min(radial) < -50.0
    assert jnp.max(radial) > 50.0
