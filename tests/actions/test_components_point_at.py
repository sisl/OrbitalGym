"""PointAt: slew-limited kinematic pointing action component."""

import flax
import jax
import jax.numpy as jnp

from orbitalgym.actions.components import PointAt
from orbitalgym.dynamics.quaternion import quat_to_rotation_matrix
from orbitalgym.state.assemble import build_state_class
from orbitalgym.state.components import Attitude, BodyRates, RTNState


@flax.struct.dataclass
class _Params:
    slew_rate_rad_s: jax.Array


@flax.struct.dataclass
class _Command:
    target_dir: jax.Array


def _state(n):
    cls = build_state_class([RTNState, Attitude, BodyRates], n, "PointState")
    return cls.zeros(n)


def _boresight(quat, body=(1.0, 0.0, 0.0)):
    return quat_to_rotation_matrix(quat) @ jnp.asarray(body)


def test_point_at_slews_by_rate_times_dt():
    comp = PointAt(boresight_body=(1.0, 0.0, 0.0))
    state = _state(1)
    cmd = _Command(target_dir=jnp.array([[0.0, 1.0, 0.0]]))
    params = _Params(slew_rate_rad_s=jnp.asarray(jnp.deg2rad(1.0)))
    new = comp.apply(cmd, state, params, dt=10.0, ref_eci6=jnp.zeros(6), key=jax.random.PRNGKey(0))
    b = _boresight(new.quat[0])
    angle = jnp.arccos(jnp.clip(b[0], -1, 1))
    assert jnp.allclose(angle, jnp.deg2rad(10.0), atol=1e-6)


def test_point_at_zero_target_holds_attitude():
    comp = PointAt(boresight_body=(1.0, 0.0, 0.0))
    state = _state(2)
    cmd = _Command(target_dir=jnp.zeros((2, 3)))
    params = _Params(slew_rate_rad_s=jnp.asarray(1.0))
    new = comp.apply(cmd, state, params, dt=10.0, ref_eci6=jnp.zeros(6), key=jax.random.PRNGKey(0))
    assert jnp.allclose(new.quat, state.quat)


def test_point_at_fields_and_zeros():
    comp = PointAt(boresight_body=(0.0, 1.0, 0.0))
    assert comp.fields() == {"target_dir": (3,)}
    assert comp.zeros(3)["target_dir"].shape == (3, 3)
