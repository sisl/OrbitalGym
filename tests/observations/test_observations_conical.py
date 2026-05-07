import jax
import jax.numpy as jnp

from orbital_game.env.types import Side
from orbital_game.observations.conical import ConicalObservation


def _layout_rtn(n_self=1, n_tgt=1):
    class _L:
        n_guards = n_self
        n_bandits = n_tgt
        dynamics_state_dim = 6

    return _L()


def _layout_rt(n_self=1, n_tgt=1):
    class _L:
        n_guards = n_self
        n_bandits = n_tgt
        dynamics_state_dim = 4

    return _L()


def _state_rtn(self_pos, target_pos, self_quat=None):
    """Build a stub env_state with .guards.rtn/.quat and .bandits.rtn/.quat."""
    n_self = self_pos.shape[0]
    n_tgt = target_pos.shape[0]
    if self_quat is None:
        self_quat = jnp.zeros((n_self, 4), dtype=jnp.float32).at[:, 0].set(jnp.float32(1.0))
    own = jnp.concatenate([self_pos, jnp.zeros_like(self_pos)], axis=1).astype(jnp.float32)
    opp = jnp.concatenate([target_pos, jnp.zeros_like(target_pos)], axis=1).astype(jnp.float32)

    class _Side:
        def __init__(self, rtn, quat):
            self.rtn = rtn
            self.quat = quat

    class _State:
        guards = _Side(own, self_quat.astype(jnp.float32))
        bandits = _Side(
            opp, jnp.zeros((n_tgt, 4), dtype=jnp.float32).at[:, 0].set(jnp.float32(1.0))
        )

    return _State()


def _state_rt(own_rt, opp_rt, self_quat=None):
    """RT layout: rt is shape (n, 4)."""
    n_self = own_rt.shape[0]
    n_tgt = opp_rt.shape[0]
    if self_quat is None:
        self_quat = jnp.zeros((n_self, 4), dtype=jnp.float32).at[:, 0].set(jnp.float32(1.0))

    class _Side:
        def __init__(self, rt, quat):
            self.rt = rt
            self.quat = quat

    class _State:
        guards = _Side(own_rt.astype(jnp.float32), self_quat.astype(jnp.float32))
        bandits = _Side(
            opp_rt.astype(jnp.float32),
            jnp.zeros((n_tgt, 4), dtype=jnp.float32).at[:, 0].set(jnp.float32(1.0)),
        )

    return _State()


def test_target_on_boresight_visible():
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma=0.0,
    )
    env_state = _state_rtn(
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.array([[10.0, 0.0, 0.0]], dtype=jnp.float32),
    )
    out = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    obs = out[0]
    assert bool(obs.visible[0, 1])  # target visible
    assert not bool(obs.visible[0, 0])  # self pair masked


def test_target_outside_cone_invisible():
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma=0.0,
    )
    # 45° off boresight: outside 30° cone.
    env_state = _state_rtn(
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.array([[10.0, 10.0, 0.0]], dtype=jnp.float32),
    )
    out = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert not bool(out[0].visible[0, 1])


def test_two_opposing_sensors_cover_both_sides():
    """Boresights ±x: target on +x visible, on -x visible, on +y not visible."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma=0.0,
    )
    cases = [
        (jnp.array([[10.0, 0.0, 0.0]], dtype=jnp.float32), True),
        (jnp.array([[-10.0, 0.0, 0.0]], dtype=jnp.float32), True),
        (jnp.array([[0.0, 10.0, 0.0]], dtype=jnp.float32), False),
    ]
    for tgt, want in cases:
        env_state = _state_rtn(jnp.zeros((1, 3), dtype=jnp.float32), tgt)
        out = obs_fn(
            env_state,
            None,
            Side.GUARD,
            None,
            jax.random.PRNGKey(0),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        assert bool(out[0].visible[0, 1]) is want


def test_boundary_inclusive():
    """Target exactly at the half-angle is visible (>= comparison)."""
    half = jnp.deg2rad(jnp.float32(30.0))
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=half,
        sigma=0.0,
    )
    target = jnp.array([[float(jnp.cos(half)), float(jnp.sin(half)), 0.0]], dtype=jnp.float32)
    env_state = _state_rtn(jnp.zeros((1, 3), dtype=jnp.float32), target)
    out = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert bool(out[0].visible[0, 1])


def test_sigma_zero_returns_ground_truth_state():
    """sigma=0 + visible target ⇒ obs[i, j] equals target's full 6-D state."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma=0.0,
    )
    own_full = jnp.zeros((1, 6), dtype=jnp.float32)
    target_full = jnp.array([[10.0, 0.0, 0.0, 0.5, -0.5, 0.1]], dtype=jnp.float32)

    class _Side:
        def __init__(self, rtn, quat):
            self.rtn = rtn
            self.quat = quat

    class _State:
        guards = _Side(own_full, jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32))
        bandits = _Side(target_full, jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32))

    out = obs_fn(
        _State(), None, Side.GUARD, None, jax.random.PRNGKey(0), jnp.asarray(0.0, dtype=jnp.float32)
    )
    obs = out[0]
    assert jnp.allclose(obs.obs[0, 1], target_full[0])


def test_obs_matrix_is_identity():
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma=1.0,
    )
    env_state = _state_rtn(
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1, 3), dtype=jnp.float32),
    )
    out = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    obs = out[0]
    assert obs.obs_matrix.shape == (6, 6)
    assert jnp.allclose(obs.obs_matrix, jnp.eye(6))
    assert obs.obs_noise.shape == (6, 6)
    assert jnp.allclose(obs.obs_noise, jnp.eye(6))


def test_rt_2d_planar_wedge():
    """RT layout (d=4): positions are 2-D; cone math zero-pads internally."""
    layout = _layout_rt()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma=0.0,
    )
    own = jnp.array([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    opp = jnp.array([[10.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    env_state = _state_rt(own, opp)
    out = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert bool(out[0].visible[0, 1])


def test_sigma_zero_r_mat_is_epsilon_identity():
    """sigma=0 must return R_mat = 1e-12 * I, not I (Fix 3)."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma=0.0,
    )
    env_state = _state_rtn(
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.array([[10.0, 0.0, 0.0]], dtype=jnp.float32),
    )
    out = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    obs = out[0]
    expected_r = jnp.eye(6, dtype=jnp.float32) * jnp.float32(1e-12)
    assert jnp.allclose(obs.obs_noise, expected_r, atol=0.0, rtol=0.0), (
        f"sigma=0 R_mat should be 1e-12*I, got diagonal={jnp.diag(obs.obs_noise)}"
    )


def _state_rtn_f64(self_pos, target_pos, self_quat=None):
    """Like _state_rtn but with float64 arrays."""
    n_self = self_pos.shape[0]
    n_tgt = target_pos.shape[0]
    if self_quat is None:
        self_quat = jnp.zeros((n_self, 4), dtype=jnp.float64).at[:, 0].set(jnp.float64(1.0))
    own = jnp.concatenate([self_pos, jnp.zeros_like(self_pos)], axis=1).astype(jnp.float64)
    opp = jnp.concatenate([target_pos, jnp.zeros_like(target_pos)], axis=1).astype(jnp.float64)

    class _Side:
        def __init__(self, rtn, quat):
            self.rtn = rtn
            self.quat = quat

    class _State:
        guards = _Side(own, self_quat.astype(jnp.float64))
        bandits = _Side(
            opp, jnp.zeros((n_tgt, 4), dtype=jnp.float64).at[:, 0].set(jnp.float64(1.0))
        )

    return _State()


def test_dtype_preserved_float64_sigma_nonzero():
    """obs, obs_matrix, obs_noise must all be float64 when state is float64 (Fix 2)."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64),
        half_angle_rad=jnp.deg2rad(jnp.float64(30.0)),
        sigma=1.0,
    )
    env_state = _state_rtn_f64(
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.array([[10.0, 0.0, 0.0]], dtype=jnp.float64),
    )
    out = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    obs = out[0]
    assert obs.obs.dtype == jnp.float64, f"obs dtype={obs.obs.dtype}"
    assert obs.obs_matrix.dtype == jnp.float64, f"obs_matrix dtype={obs.obs_matrix.dtype}"
    assert obs.obs_noise.dtype == jnp.float64, f"obs_noise dtype={obs.obs_noise.dtype}"


def test_dtype_preserved_float64_sigma_zero():
    """sigma=0 path must also return float64 when state is float64 (Fix 2+3)."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64),
        half_angle_rad=jnp.deg2rad(jnp.float64(30.0)),
        sigma=0.0,
    )
    env_state = _state_rtn_f64(
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.array([[10.0, 0.0, 0.0]], dtype=jnp.float64),
    )
    out = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    obs = out[0]
    assert obs.obs.dtype == jnp.float64, f"obs dtype={obs.obs.dtype}"
    assert obs.obs_matrix.dtype == jnp.float64, f"obs_matrix dtype={obs.obs_matrix.dtype}"
    assert obs.obs_noise.dtype == jnp.float64, f"obs_noise dtype={obs.obs_noise.dtype}"
    # R should be 1e-12 * I in float64
    expected_r = jnp.eye(6, dtype=jnp.float64) * jnp.float64(1e-12)
    assert jnp.allclose(obs.obs_noise, expected_r, atol=0.0, rtol=0.0)


def test_rotated_attitude_changes_visibility():
    """Same target, rotate observer 180° about z: in-cone → out-of-cone."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma=0.0,
    )
    target = jnp.array([[10.0, 0.0, 0.0]], dtype=jnp.float32)

    state_id = _state_rtn(
        jnp.zeros((1, 3), dtype=jnp.float32),
        target,
        self_quat=jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
    )
    out_id = obs_fn(
        state_id, None, Side.GUARD, None, jax.random.PRNGKey(0), jnp.asarray(0.0, dtype=jnp.float32)
    )
    assert bool(out_id[0].visible[0, 1])

    # 180° rotation about z: (w, x, y, z) = (0, 0, 0, 1)
    state_rot = _state_rtn(
        jnp.zeros((1, 3), dtype=jnp.float32),
        target,
        self_quat=jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32),
    )
    out_rot = obs_fn(
        state_rot,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert not bool(out_rot[0].visible[0, 1])
