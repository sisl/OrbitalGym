"""OPPONENT_SCAN finds a bandit that OPPONENT_BELIEF never points at.

The guard rides a 150 m natural-motion ring and the bandit coasts on a
3 km ring at the same phase, so the two sit on a common ray from the
lady. A ring-prior particle belief has its mean on the lady, exactly
opposite the bandit, so a guard pointing at the belief mean holds its
30-degree cone 180 degrees away from the bandit for the whole orbit.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.belief.pf import (
    ParticleFilterBelief,
    ParticleFilterBeliefUpdater,
    ParticleFilterRingInitializer,
)
from orbitalgym.config import VehicleParamsSpec
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.dynamics.hcw import hcw_rt_step
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.policies.pointing import PointingPolicy, PointingTarget
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.reference_orbit import mean_motion as ref_mean_motion
from orbitalgym.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from orbitalgym.rollout import belief_rollout
from orbitalgym.sampling.attitude import IdentityAttitude
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec

_GUARD_RING_M = 150.0
_BANDIT_RING_M = 3000.0
_RING_PHASE_RAD = 0.0
_HALF_ANGLE_RAD = float(jnp.deg2rad(30.0))
_DT = 20.0
_N_PARTICLES = 64


@dataclasses.dataclass(frozen=True)
class _EvenRingInitializer:
    """Guard-side ring prior with evenly spaced phases, so its mean is the lady.

    `ParticleFilterRingInitializer` draws phases at random, which leaves a
    sampling-noise offset in the prior mean that is large next to the
    150 m guard ring. Even spacing pins the mean on the origin so the
    OPPONENT_BELIEF baseline is exactly anti-parallel to the bandit.
    """

    ring_radius_m: float
    mean_motion_rad_s: float
    n_particles: int

    def __call__(self, env_state, side, key) -> ParticleFilterBelief:
        del side, key
        own_truth = env_state.guards.rt
        n_self = own_truth.shape[0]
        n_opp = env_state.bandits.rt.shape[0]
        n_total = n_self + n_opp
        k = self.n_particles

        own_block = jnp.broadcast_to(
            own_truth[None, :, None, :], (n_self, n_self, k, own_truth.shape[-1])
        )
        phases = jnp.linspace(0.0, 2.0 * jnp.pi, k, endpoint=False)
        a = self.ring_radius_m
        n = self.mean_motion_rad_s
        ring = jnp.stack(
            [
                -a * jnp.cos(phases),
                2.0 * a * jnp.sin(phases),
                n * a * jnp.sin(phases),
                2.0 * n * a * jnp.cos(phases),
            ],
            axis=-1,
        )
        opp_block = jnp.broadcast_to(ring[None, None], (n_self, n_opp, k, ring.shape[-1]))
        particles = jnp.concatenate([own_block, opp_block], axis=1)
        return ParticleFilterBelief(
            particles=particles,
            log_weights=jnp.full((n_self, n_total, k), -jnp.log(float(k))),
            n_eff=jnp.full((n_self, n_total), float(k)),
            weight_entropy=jnp.full((n_self, n_total), jnp.log(float(k))),
            resampled=jnp.zeros((n_self, n_total), dtype=bool),
        )


def _scenario() -> Any:
    return make_lady_bandit_guard(
        n_guards=1,
        n_bandits=1,
        dt=_DT,
        max_horizon_s=1.0e6,
        truth_dynamics=DynamicsKey.HCW_RT,
        policy_dynamics=DynamicsKey.HCW_RT,
        action_frame=Frame.RT,
        guard_components=(
            StateComponentKey.RT,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        bandit_components=(StateComponentKey.RT,),
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.POINT_AT,
        ),
        bandit_action_components=(ActionComponentKey.IMPULSIVE_MANEUVER,),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=AttitudeParams(inertia_diag=jnp.ones(3), omega_max=jnp.ones(3)),
        pointing_boresight_body=(1.0, 0.0, 0.0),
        guard_params=VehicleParamsSpec(
            dry_mass_kg=10.0, isp_s=200.0, max_thrust_n=1.0, slew_rate_rad_s=0.2
        ),
        bandit_params=VehicleParamsSpec(dry_mass_kg=10.0, isp_s=200.0, max_thrust_n=1.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=_GUARD_RING_M,
                phase_rad=_RING_PHASE_RAD,
                attitude_sampler=IdentityAttitude(),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=_BANDIT_RING_M, phase_rad=_RING_PHASE_RAD
            ),
        ),
    )


def _visibility_duty(target: PointingTarget) -> float:
    cfg = _scenario()
    n_motion = float(ref_mean_motion(cfg.reference_orbit))
    cfg = dataclasses.replace(
        cfg,
        guard_observation_fn=ConicalObservation(
            layout=cfg.layout,
            sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]]),
            half_angle_rad=_HALF_ANGLE_RAD,
            sigma_floor=1.0,
        ),
    )
    env = OrbitalGymEnv(cfg)

    def dynamics_fn(x, u, dt):
        return hcw_rt_step(x[None, :], u[None, :], SimpleNamespace(mean_motion=n_motion), dt)[0]

    updater = ParticleFilterBeliefUpdater(
        dynamics_fn=dynamics_fn,
        process_noise=jnp.diag(jnp.array([1.0, 1.0, 1e-4, 1e-4])),
        dt=_DT,
    )
    guard_init = _EvenRingInitializer(
        ring_radius_m=_BANDIT_RING_M,
        mean_motion_rad_s=n_motion,
        n_particles=_N_PARTICLES,
    )
    bandit_init = ParticleFilterRingInitializer(
        layout=env.layout,
        ring_radius_m=_GUARD_RING_M,
        mean_motion_rad_s=n_motion,
        n_particles=_N_PARTICLES,
    )

    guard_zero = dataclasses.replace(
        ZeroControl(), command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards
    )
    bandit_zero = dataclasses.replace(
        ZeroControl(), command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits
    )
    policies = BySide(guard=PointingPolicy(inner=guard_zero, target=target), bandit=bandit_zero)

    n_steps = int(round((2.0 * jnp.pi / n_motion) / _DT))
    traj, _ = belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None),
        belief_initializers=BySide(guard=guard_init, bandit=bandit_init),
        belief_updaters=BySide(guard=updater, bandit=updater),
        key=jax.random.PRNGKey(0),
        n_steps=n_steps,
    )
    assert traj.visible.guard.shape == (n_steps, cfg.n_guards)
    return float(jnp.mean(traj.visible.guard))


def test_scan_detects_a_ring_bandit_that_belief_pointing_never_sees():
    assert _visibility_duty(PointingTarget.OPPONENT_BELIEF) == 0.0
    assert _visibility_duty(PointingTarget.OPPONENT_SCAN) > 0.2
