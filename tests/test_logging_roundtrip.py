"""HDF5 save/load round-trip for rollouts + scenario config."""

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.logging.reader import load_run
from orbital_game.logging.writer import save_run
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import StateComponentKey
from orbital_game.rollout import rollout
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec


def _make_cfg() -> ScenarioConfig:
    return ScenarioConfig(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(100.0, 220.0, 5.0),
        bandit_params=VehicleParamsSpec(50.0, 200.0, 2.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=0.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=jnp.pi,
            ),
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )


def test_hdf5_save_load_roundtrip(tmp_path):
    """Save a rollout + config to HDF5, load it back, and verify key fields match."""
    cfg = _make_cfg()
    env = OrbitalGameEnv(cfg)
    traj = rollout(
        env,
        lambda ps, obs, k, t: (jnp.zeros((1, 3)), ps),
        lambda c, es, k: None,
        jax.random.PRNGKey(0),
        n_steps=10,
    )
    path = tmp_path / "run.h5"
    save_run(path, cfg, traj)
    loaded_cfg, loaded_traj = load_run(path)

    assert loaded_cfg.n_guards == cfg.n_guards
    assert loaded_cfg.dt == cfg.dt
    assert loaded_cfg.max_horizon_s == cfg.max_horizon_s
    # Reward and action survive byte-equal through the round-trip.
    assert jnp.allclose(loaded_traj["reward"], traj.reward)
    assert jnp.allclose(loaded_traj["action"], traj.action)
    # Done bool array round-trips too.
    assert jnp.array_equal(loaded_traj["done"], traj.done)


def test_hdf5_rejects_unknown_schema_version(tmp_path):
    """Reader must reject files with unexpected schema_version."""
    import h5py

    path = tmp_path / "bad_schema.h5"
    with h5py.File(path, "w") as f:
        f.attrs["config"] = "{}"
        f.attrs["schema_version"] = "999"
        f.create_group("trajectory")

    import pytest

    with pytest.raises(ValueError, match="schema_version"):
        load_run(path)


def test_hdf5_rejects_file_missing_config_attribute(tmp_path):
    """Reader must raise a clear error when 'config' root attribute is absent."""
    import h5py

    path = tmp_path / "no_config.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2"
        f.create_group("trajectory")

    import pytest

    with pytest.raises(ValueError, match="config"):
        load_run(path)


def test_hdf5_rejects_file_missing_schema_version_attribute(tmp_path):
    """Reader must raise a clear error when 'schema_version' root attribute is absent."""
    import h5py

    path = tmp_path / "no_schema.h5"
    with h5py.File(path, "w") as f:
        f.attrs["config"] = "{}"
        f.create_group("trajectory")

    import pytest

    with pytest.raises(ValueError, match="schema_version"):
        load_run(path)
