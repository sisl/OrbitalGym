"""Tests for RelativeEllipse per-side sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.dynamics.hcw import hcw_rtn_step
from orbital_game.hva import HVAState, mean_motion
from orbital_game.registry import StateComponentKey
from orbital_game.sampling.side import RelativeEllipse


def _config(n_def=1, velocity_y_mps=7546.05):
    """Build a minimal scenario config. Default velocity is circular at r=7000km;
    pass a different value to construct an elliptical HVA orbit."""
    from orbital_game.sampling.spec import ICSpec
    return ScenarioConfig(
        n_defenders=n_def,
        n_intruders=1,
        epoch_mjd_utc=60067.0,
        hva=HVAState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, velocity_y_mps, 0.0]),
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


def _propagate_max_excursion(cfg, sampler, n_periods=10):
    """Sample an IC and propagate via HCW, returning the (max|R|, max|T|) pair."""
    out = sampler(
        cfg,
        jax.random.PRNGKey(0),
        n_vehicles=1,
        components=(StateComponentKey.RTN,),
        class_name="DefenderState",
    )
    n = mean_motion(cfg.hva)
    period = 2 * jnp.pi / n

    class _Params:
        pass
    _Params.mean_motion = float(n)

    state = out.rtn
    dt = float(period / 200.0)
    max_r = float(jnp.abs(state[0, 0]))
    max_t = float(jnp.abs(state[0, 1]))
    for _ in range(200 * n_periods):
        state = hcw_rtn_step(state, jnp.zeros((1, 3)), _Params(), dt)
        max_r = max(max_r, float(jnp.abs(state[0, 0])))
        max_t = max(max_t, float(jnp.abs(state[0, 1])))
    return max_r, max_t


def test_boundedness_invariant_radial_ellipse():
    """Circular HVA: sampled IC produces a closed 2:1 relative ellipse over 10 periods."""
    sampler = RelativeEllipse(radial_ellipse_m=200.0, phase_rad=0.0)
    cfg = _config()  # circular HVA (default v=7546.05 m/s at r=7000 km)
    max_r, max_t = _propagate_max_excursion(cfg, sampler, n_periods=10)
    assert max_r <= 200.0 + 1.0
    assert max_t <= 400.0 + 1.0  # 2x radial — canonical HCW relative ellipse


def test_boundedness_invariant_with_elliptical_hva():
    """Elliptical HVA: sampler must agree with env's mean motion (vis-viva), not
    assume circular. With v=7500 m/s vs circular 7546.05 at r=7000 km, the orbit
    has e≈0.012. The previous `a = |r|` shortcut produced ~700m drift per period;
    the vis-viva fix keeps closure tight. This test would have caught that bug.
    """
    sampler = RelativeEllipse(radial_ellipse_m=200.0, phase_rad=0.0)
    cfg = _config(velocity_y_mps=7500.0)  # below circular -> elliptical HVA
    max_r, max_t = _propagate_max_excursion(cfg, sampler, n_periods=10)
    # With the broken sampler this would be O(thousands of meters); with the
    # vis-viva fix it's exact under HCW (sampler n == dynamics n).
    assert max_r <= 200.0 + 1.0
    assert max_t <= 400.0 + 1.0


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
