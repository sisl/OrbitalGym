"""Link predicates: always, ground network, pointing cone."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.groundstations import ContactSchedule
from orbitalgym.links import AlwaysLinked, GroundNetworkLink, PointingConeLink
from orbitalgym.registry import AttitudeDynamicsKey, StateComponentKey


def _two_guard_state(guard_quats):
    cfg = make_lady_bandit_guard(
        n_guards=2,
        guard_components=(
            StateComponentKey.RTN,
            StateComponentKey.MASS,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=AttitudeParams(inertia_diag=jnp.ones(3), omega_max=jnp.ones(3)),
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    # Guard 0 at +100 m along-track of guard 1, both on the R axis otherwise.
    rtn = jnp.array([[0.0, 100.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    guards = state.guards.replace(rtn=rtn, quat=jnp.asarray(guard_quats))
    return state.replace(guards=guards)


IDENTITY = (1.0, 0.0, 0.0, 0.0)
YAW_180 = (0.0, 0.0, 0.0, 1.0)  # 180 deg about N: +T becomes -T


def test_always_linked_is_all_true():
    state = _two_guard_state([IDENTITY, IDENTITY])
    mask = AlwaysLinked()(state, Side.GUARD, jnp.asarray(0.0))
    assert mask.shape == (2,)
    assert bool(jnp.all(mask))


def test_ground_network_link_follows_schedule():
    sch = ContactSchedule(
        windows=jnp.asarray([[100.0, 200.0]] + [[-1.0, -1.0]] * 3, dtype=jnp.float32),
        n_valid=jnp.asarray(1),
        station_ix=jnp.asarray([0, -1, -1, -1]),
    )
    state = _two_guard_state([IDENTITY, IDENTITY])
    link = GroundNetworkLink(schedule=sch)
    assert bool(jnp.all(link(state, Side.GUARD, jnp.asarray(150.0))))
    assert not bool(jnp.any(link(state, Side.GUARD, jnp.asarray(50.0))))


def test_pointing_cone_link_requires_mutual_pointing_by_default():
    # Boresight is body +T. Guard 1 at the origin points +T toward guard 0
    # (at +100 m T); guard 0 with identity also points +T, away from guard 1.
    link = PointingConeLink(half_angle_rad=jnp.deg2rad(10.0))
    one_way = _two_guard_state([IDENTITY, IDENTITY])
    assert not bool(jnp.any(link(one_way, Side.GUARD, jnp.asarray(0.0))))
    facing = _two_guard_state([YAW_180, IDENTITY])
    assert bool(jnp.all(link(facing, Side.GUARD, jnp.asarray(0.0))))


def test_pointing_cone_link_one_way_when_not_mutual():
    link = PointingConeLink(half_angle_rad=jnp.deg2rad(10.0), mutual=False)
    state = _two_guard_state([IDENTITY, IDENTITY])
    mask = link(state, Side.GUARD, jnp.asarray(0.0))
    # Guard 1 sees guard 0 in its cone; guard 0 does not see guard 1.
    assert bool(mask[1]) and not bool(mask[0])


def test_pointing_cone_link_single_agent_is_never_linked():
    cfg = make_lady_bandit_guard()
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    link = PointingConeLink(half_angle_rad=0.5)
    mask = link(state, Side.BANDIT, jnp.asarray(0.0))
    assert mask.shape == (1,) and not bool(mask[0])
