"""Tests for sampling/reference.py — GaussianAroundNominal."""

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.hva import HVAState
from orbital_game.registry import StateComponentKey
from orbital_game.sampling.reference import GaussianAroundNominal


def _make_cfg(seed: int) -> ScenarioConfig:
    return ScenarioConfig(
        n_defenders=1,
        n_intruders=1,
        epoch_mjd_utc=60067.0,
        hva=HVAState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        defender_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        intruder_components=(StateComponentKey.RTN,),
        defender_params=VehicleParamsSpec(100.0, 10.0, 220.0, 5.0),
        intruder_params=VehicleParamsSpec(50.0, 5.0, 200.0, 2.0),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=seed,
    )


def test_gaussian_around_nominal_is_seed_deterministic():
    cfg = _make_cfg(seed=0)
    sampler = GaussianAroundNominal(
        nominal_defender_rtn=jnp.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        nominal_intruder_rtn=jnp.array([[-100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        sigma_pos=1.0,
        sigma_vel=0.1,
    )
    a = sampler(cfg, jax.random.PRNGKey(42))
    b = sampler(cfg, jax.random.PRNGKey(42))
    assert jnp.allclose(a[0].rtn, b[0].rtn)
    assert jnp.allclose(a[1].rtn, b[1].rtn)
