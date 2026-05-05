"""ImpulsiveManeuver applied to RTN state with HCW dynamics: rocket-equation
propellant burn + truth_dynamics propagation, with the track_mass flag
gating propellant tracking."""

import flax
import jax
import jax.numpy as jnp

from orbital_game.actions.components import ImpulsiveManeuver
from orbital_game.dynamics.hcw import hcw_rtn_step
from orbital_game.registry import Frame
from orbital_game.state.assemble import build_state_class
from orbital_game.state.components import Mass, RTNState


@flax.struct.dataclass
class _Params:
    dry_mass_kg: jax.Array
    isp_s: jax.Array
    max_thrust_n: jax.Array
    mean_motion: jax.Array


def _make_params() -> _Params:
    return _Params(
        dry_mass_kg=jnp.asarray([100.0]),
        isp_s=jnp.asarray([220.0]),
        max_thrust_n=jnp.asarray([5.0]),
        mean_motion=jnp.asarray(0.001),
    )


@flax.struct.dataclass
class _Command:
    dv: jax.Array


def _make_side_state(n_vehicles: int):
    Cls = build_state_class([RTNState, Mass], n_vehicles, "TestState")  # noqa: N806
    return Cls.zeros(n_vehicles).replace(
        rtn=jnp.array([[100.0, 0.0, 0.0, 0.0, 0.1, 0.0]]),
        propellant_mass=jnp.array([10.0]),
    )


def test_impulsive_maneuver_zero_dv_no_state_change():
    state = _make_side_state(1)
    m = ImpulsiveManeuver(
        truth_dynamics=hcw_rtn_step,
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=True,
    )
    cmd = _Command(dv=jnp.zeros((1, 3)))
    ref_eci6 = jnp.array([7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    key = jax.random.PRNGKey(0)
    new_state = m.apply(cmd, state, _make_params(), dt=10.0, ref_eci6=ref_eci6, key=key)
    # State propagates by HCW even with zero dv; check propellant unchanged.
    assert jnp.allclose(new_state.propellant_mass, state.propellant_mass)


def test_impulsive_maneuver_nonzero_dv_burns_propellant():
    state = _make_side_state(1)
    m = ImpulsiveManeuver(
        truth_dynamics=hcw_rtn_step,
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=True,
    )
    cmd = _Command(dv=jnp.array([[0.5, 0.0, 0.0]]))
    ref_eci6 = jnp.array([7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    key = jax.random.PRNGKey(0)
    new_state = m.apply(cmd, state, _make_params(), dt=10.0, ref_eci6=ref_eci6, key=key)
    assert new_state.propellant_mass[0] < state.propellant_mass[0]


def test_impulsive_maneuver_track_mass_false_preserves_propellant():
    state = _make_side_state(1)
    m = ImpulsiveManeuver(
        truth_dynamics=hcw_rtn_step,
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=False,
    )
    cmd = _Command(dv=jnp.array([[0.5, 0.0, 0.0]]))
    ref_eci6 = jnp.array([7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    key = jax.random.PRNGKey(0)
    new_state = m.apply(cmd, state, _make_params(), dt=10.0, ref_eci6=ref_eci6, key=key)
    assert jnp.allclose(new_state.propellant_mass, state.propellant_mass)
