"""Multi-vehicle ROE example: 3 defenders + 2 intruders with min-separation rejection.

Demonstrates:
  - Heterogeneous per-vehicle ROE specs (per-vehicle radial_ellipse_m array).
  - Cross-side MinSeparation validator with rejection re-sampling.
  - ic_valid propagation through trajectories.

Run: uv run python -m examples.multi_vehicle_roe
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.hva import HVAState
from orbital_game.registry import StateComponentKey
from orbital_game.rollout import rollout
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec
from orbital_game.sampling.validators import MinSeparation, SeparationScope


def build_config() -> ScenarioConfig:
    return ScenarioConfig(
        n_defenders=3,
        n_intruders=2,
        epoch_mjd_utc=60067.0,
        hva=HVAState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        defender_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        intruder_components=(StateComponentKey.RTN,),
        defender_params=VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        intruder_params=VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0),
        ic_sampler=ICSpec(
            defender_sampler=RelativeEllipse(
                # Heterogeneous: 3 defenders with different ellipse sizes
                radial_ellipse_m=jnp.array([500.0, 750.0, 1000.0]),
                cross_track_m=jnp.array([100.0, 100.0, 100.0]),
                phase_rad=jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3]),
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            intruder_sampler=RelativeEllipse(
                radial_ellipse_m=jnp.array([1500.0, 1500.0]),
                cross_track_m=jnp.array([200.0, 200.0]),
                phase_rad=None,  # uniform random
                sigma_radial_ellipse_m=50.0,
            ),
            # Reject if any pair (defender-defender, intruder-intruder, or cross)
            # is closer than 50m at t=0.
            validators=(MinSeparation(distance_m=50.0, scope=SeparationScope.ALL),),
            max_attempts=200,
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )


def run() -> None:
    cfg = build_config()
    env = OrbitalGameEnv(cfg)

    def policy(ps, obs, key, t):
        del obs, key, t
        return jnp.zeros((cfg.n_defenders, 3)), ps

    def init_ps(config, env_state, key):
        del config, env_state, key
        return None

    # Seed-parallel: 64 lanes
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), 64)
    batched_rollout = jax.vmap(
        lambda k: rollout(env, policy, init_ps, k, n_steps=cfg.max_steps)
    )
    traj = batched_rollout(keys)

    # ic_valid is shape (B, T) under freeze-on-done; first step = sample-time validity
    valid_rate = float(jnp.mean(traj.env_state.ic_valid[:, 0]))
    print(f"IC validity rate across 64 seeds: {valid_rate:.3f}")
    print(
        "Mean defender propellant at t=0: "
        f"{float(jnp.mean(traj.env_state.defenders.propellant_mass[:, 0])):.2f} kg"
    )


if __name__ == "__main__":
    run()
