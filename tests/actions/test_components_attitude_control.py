import jax
import jax.numpy as jnp

from orbitalgym.actions.components import AttitudeControl


def test_fields_rotation_dim_1():
    comp = AttitudeControl(rotation_dim=1)
    assert comp.fields() == {"torque": (1,)}


def test_fields_rotation_dim_3():
    comp = AttitudeControl(rotation_dim=3)
    assert comp.fields() == {"torque": (3,)}


def test_apply_lifts_rt_command_to_3vec():
    """rotation_dim=1: (n, 1) command → applied_torque (n, 3) = [0, 0, τz]."""
    comp = AttitudeControl(rotation_dim=1)

    class _State:
        def __init__(self):
            self.applied_torque = jnp.zeros((1, 3), dtype=jnp.float32)

        def replace(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            return self

    class _Cmd:
        def __init__(self):
            self.torque = jnp.array([[0.5]], dtype=jnp.float32)

    new = comp.apply(
        _Cmd(),
        _State(),
        None,
        dt=0.1,
        ref_eci6=jnp.zeros(6, dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(new.applied_torque, jnp.array([[0.0, 0.0, 0.5]], dtype=jnp.float32))


def test_apply_writes_3vec_directly_for_rotation_dim_3():
    comp = AttitudeControl(rotation_dim=3)

    class _State:
        def __init__(self):
            self.applied_torque = jnp.zeros((2, 3), dtype=jnp.float32)

        def replace(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            return self

    class _Cmd:
        def __init__(self):
            self.torque = jnp.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=jnp.float32)

    new = comp.apply(
        _Cmd(),
        _State(),
        None,
        dt=0.1,
        ref_eci6=jnp.zeros(6, dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(new.applied_torque, _Cmd().torque)


def test_torque_max_clip():
    comp = AttitudeControl(rotation_dim=3, torque_max=(0.1, 0.1, 0.1))

    class _State:
        def __init__(self):
            self.applied_torque = jnp.zeros((1, 3), dtype=jnp.float32)

        def replace(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            return self

    class _Cmd:
        def __init__(self):
            self.torque = jnp.array([[5.0, -5.0, 0.05]], dtype=jnp.float32)

    new = comp.apply(
        _Cmd(),
        _State(),
        None,
        dt=0.1,
        ref_eci6=jnp.zeros(6, dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(new.applied_torque, jnp.array([[0.1, -0.1, 0.05]], dtype=jnp.float32))


def test_zeros_shape():
    """zeros(n) returns the right shape per rotation_dim."""
    assert AttitudeControl(rotation_dim=1).zeros(4)["torque"].shape == (4, 1)
    assert AttitudeControl(rotation_dim=3).zeros(2)["torque"].shape == (2, 3)
