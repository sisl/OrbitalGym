import jax.numpy as jnp
import numpy as np


def _circular_leo_ref():
    # 500 km circular equatorial LEO.
    r = 6378137.0 + 500e3
    v = (3.986004418e14 / r) ** 0.5
    return jnp.array([r, 0.0, 0.0, 0.0, v, 0.0])


def test_eci_to_rtn_roundtrip_zero_relative():
    from orbital_game.frames.conversions import eci_to_rtn, rtn_to_eci

    ref = _circular_leo_ref()
    deputy_eci = ref[None, :]  # (1, 6)
    rtn = eci_to_rtn(deputy_eci, ref)
    assert rtn.shape == (1, 6)
    np.testing.assert_allclose(rtn, 0.0, atol=1e-6)
    back = rtn_to_eci(rtn, ref)
    np.testing.assert_allclose(back, deputy_eci, atol=1e-6)


def test_eci_to_rtn_roundtrip_nonzero():
    from orbital_game.frames.conversions import eci_to_rtn, rtn_to_eci

    ref = _circular_leo_ref()
    rtn = jnp.array([[100.0, -50.0, 25.0, 0.5, 0.1, -0.2]])  # (1, 6)
    eci = rtn_to_eci(rtn, ref)
    rtn_back = eci_to_rtn(eci, ref)
    np.testing.assert_allclose(rtn_back, rtn, atol=1e-6)


def test_eci_to_rtn_vmap_over_vehicles():
    from orbital_game.frames.conversions import eci_to_rtn

    ref = _circular_leo_ref()
    deputies_eci = jnp.tile(ref, (4, 1)) + jnp.array(
        [[0, 0, 0, 0, 0, 0], [10.0, 0, 0, 0, 0, 0], [0, 20.0, 0, 0, 0, 0], [0, 0, 30.0, 0, 0, 0]]
    )
    rtn = eci_to_rtn(deputies_eci, ref)
    assert rtn.shape == (4, 6)
    np.testing.assert_allclose(rtn[0], 0.0, atol=1e-6)


def test_convert_action_rtn_to_eci_to_rtn_identity():
    from orbital_game.frames.conversions import convert_action
    from orbital_game.registry import Frame

    ref = _circular_leo_ref()
    dv_rtn = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
    dv_eci = convert_action(dv_rtn, Frame.RTN, Frame.ECI, ref)
    dv_back = convert_action(dv_eci, Frame.ECI, Frame.RTN, ref)
    np.testing.assert_allclose(dv_back, dv_rtn, atol=1e-9)


def test_convert_action_rt_to_rtn_pads_zero():
    from orbital_game.frames.conversions import convert_action
    from orbital_game.registry import Frame

    ref = _circular_leo_ref()
    dv_rt = jnp.array([[0.1, 0.2], [-0.3, 0.4]])
    dv_rtn = convert_action(dv_rt, Frame.RT, Frame.RTN, ref)
    np.testing.assert_allclose(dv_rtn[:, 2], 0.0)
    np.testing.assert_allclose(dv_rtn[:, :2], dv_rt)
