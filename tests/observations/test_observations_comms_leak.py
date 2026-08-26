"""CommsLeakObservation emits a high-precision guard-position channel
for the bandit side iff any guard's communicate.active is True this step."""

import jax
import jax.numpy as jnp

from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.observations.comms_leak import CommsLeakObservation
from orbitalgym.reference_orbit import ReferenceOrbitState
from orbitalgym.registry import ActionComponentKey, StateComponentKey
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec


def _build_cfg_with_comms():
    # Intentional decoupling: this fixture predates `make_lady_bandit_guard`'s
    # `with_communication=True` builder path. We construct ScenarioConfig
    # directly here so the unit test stays independent of the LBG builder —
    # if the LBG builder grows side effects, these tests still exercise just
    # the CommsLeakObservation contract. Don't refactor onto the builder.
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
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=jnp.pi,
                sigma_radial_ellipse_m=10.0,
            ),
            validators=(),
            max_attempts=100,
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.COMMUNICATE,
        ),
    )


def _build_actions_with_comms(env, comms_active: bool):
    guard_cmd = env.guard_command_cls.zeros(env.config.n_guards).replace(
        active=jnp.array([comms_active]),
    )
    bandit_cmd = env.bandit_command_cls.zeros(env.config.n_bandits)
    return Actions(sides=BySide(guard=guard_cmd, bandit=bandit_cmd))


def test_comms_leak_silent_when_no_guard_communicates():
    cfg = _build_cfg_with_comms()
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    actions = _build_actions_with_comms(env, comms_active=False)
    obs_fn = CommsLeakObservation(layout=cfg.layout)
    (channel,) = obs_fn(state, actions, Side.BANDIT, cfg, jax.random.PRNGKey(1), state.t)
    assert not channel.visible.any()


def test_comms_leak_visible_when_guard_communicates():
    cfg = _build_cfg_with_comms()
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    actions = _build_actions_with_comms(env, comms_active=True)
    obs_fn = CommsLeakObservation(layout=cfg.layout)
    (channel,) = obs_fn(state, actions, Side.BANDIT, cfg, jax.random.PRNGKey(1), state.t)
    # Opposing-side columns must be visible; own-side must not.
    n_self = cfg.n_bandits
    n_total = cfg.n_guards + cfg.n_bandits
    assert channel.visible.shape == (n_self, n_total)
    assert not channel.visible[:, :n_self].any()
    assert channel.visible[:, n_self:].all()


def test_comms_leak_shape_matches_full_observation_layout():
    """CommsLeak emits (n_self, n_total, d) — same layout as FullObservation
    and the other reference observers. Own-side columns must be visible=False
    even when comms is active (a bandit doesn't observe itself via leak)."""
    cfg = _build_cfg_with_comms()
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    actions = _build_actions_with_comms(env, comms_active=True)
    obs_fn = CommsLeakObservation(layout=cfg.layout)
    (channel,) = obs_fn(state, actions, Side.BANDIT, cfg, jax.random.PRNGKey(1), state.t)

    n_self = cfg.n_bandits  # bandit is the observer
    n_total = cfg.n_guards + cfg.n_bandits
    d = cfg.layout.dynamics_state_dim
    assert channel.obs.shape == (n_self, n_total, d)
    assert channel.visible.shape == (n_self, n_total)
    # Own-side columns mask is False even with comms active.
    assert not channel.visible[:, :n_self].any()
