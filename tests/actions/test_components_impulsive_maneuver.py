"""ImpulsiveManeuver: Δv producer that writes applied_dv onto state.

After the Task 2.2+2.3 refactor, ImpulsiveManeuver no longer calls
truth_dynamics. It writes dv (converted to truth frame, padded to width 3)
onto side_state.applied_dv and optionally deducts propellant. The actual
truth-dynamics propagation is done by env.step.
"""

import flax
import jax
import jax.numpy as jnp

from orbitalgym.actions.components import ImpulsiveManeuver
from orbitalgym.registry import Frame
from orbitalgym.state.assemble import build_state_class
from orbitalgym.state.components import AppliedDV, Mass, RTNState


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
    Cls = build_state_class([RTNState, Mass, AppliedDV], n_vehicles, "TestState")  # noqa: N806
    return Cls.zeros(n_vehicles).replace(
        rtn=jnp.array([[100.0, 0.0, 0.0, 0.0, 0.1, 0.0]]),
        propellant_mass=jnp.array([10.0]),
    )


def test_impulsive_maneuver_zero_dv_no_propellant_burn():
    """Zero Δv leaves propellant unchanged."""
    state = _make_side_state(1)
    m = ImpulsiveManeuver(
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=True,
    )
    cmd = _Command(dv=jnp.zeros((1, 3)))
    ref_eci6 = jnp.array([7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    key = jax.random.PRNGKey(0)
    new_state = m.apply(cmd, state, _make_params(), dt=10.0, ref_eci6=ref_eci6, key=key)
    assert jnp.allclose(new_state.propellant_mass, state.propellant_mass)


def test_impulsive_maneuver_nonzero_dv_burns_propellant():
    """Non-zero Δv deducts propellant via the rocket equation."""
    state = _make_side_state(1)
    m = ImpulsiveManeuver(
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
    """track_mass=False leaves propellant unchanged even with non-zero dv."""
    state = _make_side_state(1)
    m = ImpulsiveManeuver(
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=False,
    )
    cmd = _Command(dv=jnp.array([[0.5, 0.0, 0.0]]))
    ref_eci6 = jnp.array([7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    key = jax.random.PRNGKey(0)
    new_state = m.apply(cmd, state, _make_params(), dt=10.0, ref_eci6=ref_eci6, key=key)
    assert jnp.allclose(new_state.propellant_mass, state.propellant_mass)


def test_apply_writes_applied_dv_rtn():
    """RTN scenario: 3-D dv command is written to applied_dv unchanged."""
    state = _make_side_state(1)
    m = ImpulsiveManeuver(
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=False,
    )
    dv = jnp.array([[0.1, 0.2, 0.3]])
    cmd = _Command(dv=dv)
    ref_eci6 = jnp.array([7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    key = jax.random.PRNGKey(0)
    new_state = m.apply(cmd, state, _make_params(), dt=10.0, ref_eci6=ref_eci6, key=key)
    assert jnp.allclose(new_state.applied_dv, dv)
    # truth state must NOT be mutated by apply
    assert jnp.allclose(new_state.rtn, state.rtn)


def test_apply_does_not_invoke_dynamics():
    """After refactor, ImpulsiveManeuver has no truth_dynamics attribute."""
    comp = ImpulsiveManeuver(
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=False,
    )
    assert not hasattr(comp, "truth_dynamics")
