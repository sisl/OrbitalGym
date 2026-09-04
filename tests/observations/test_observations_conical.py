import jax
import jax.numpy as jnp

from orbitalgym.belief.kf import KFBelief, KFBeliefUpdater
from orbitalgym.belief.pf import ParticleFilterBelief, ParticleFilterBeliefUpdater
from orbitalgym.env.types import Side
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.observations.onboard_gps import OnboardGPSObservation


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
        sigma_floor=0.0,
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
        sigma_floor=0.0,
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
        sigma_floor=0.0,
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
        sigma_floor=0.0,
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


def test_zero_sigma_floor_returns_ground_truth_state():
    """Zero sigma + visible target ⇒ obs[i, j] equals target's full 6-D state."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma_floor=0.0,
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
        sigma_floor=1.0,
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
    assert obs.obs_noise.shape == (1, 2, 6, 6)
    assert jnp.allclose(obs.obs_noise, jnp.eye(6))


def test_rt_2d_planar_wedge():
    """RT layout (d=4): positions are 2-D; cone math zero-pads internally."""
    layout = _layout_rt()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma_floor=0.0,
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


def test_zero_sigma_floor_r_mat_is_epsilon_identity():
    """Zero sigma must return R = 1e-12 * I, not I."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma_floor=0.0,
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
    expected_r = jnp.broadcast_to(jnp.eye(6, dtype=jnp.float32) * jnp.float32(1e-12), (1, 2, 6, 6))
    assert jnp.allclose(obs.obs_noise, expected_r, atol=0.0, rtol=0.0), (
        f"zero-sigma R should be 1e-12*I, got diagonal={jnp.diag(obs.obs_noise[0, 1])}"
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


def test_dtype_preserved_float64_sigma_floor_nonzero():
    """obs, obs_matrix and obs_noise must all be float64 when state is float64."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64),
        half_angle_rad=jnp.deg2rad(jnp.float64(30.0)),
        sigma_floor=1.0,
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


def test_dtype_preserved_float64_sigma_floor_zero():
    """The zero-sigma path must also return float64 when state is float64."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64),
        half_angle_rad=jnp.deg2rad(jnp.float64(30.0)),
        sigma_floor=0.0,
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
    expected_r = jnp.broadcast_to(jnp.eye(6, dtype=jnp.float64) * jnp.float64(1e-12), (1, 2, 6, 6))
    assert jnp.allclose(obs.obs_noise, expected_r, atol=0.0, rtol=0.0)


def test_conical_attaches_visibility_score_fn():
    """ConicalObservation must attach a closure returning
    max_s (half_angle_s - angle_s_to_particle) per (observer, target, particle)."""
    layout = _layout_rtn()
    half = jnp.float32(0.5)  # ~28.6 degrees
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=half,
        sigma_floor=1.0,
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
    channel = out[0]
    assert channel.visibility_score_fn is not None

    # Particle directly along +x at any distance → angle = 0 → score = +half.
    particles_on_boresight = jnp.array(
        [
            [
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # self-pair (irrelevant)
                [[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # cross-pair on boresight
            ]
        ],
        dtype=jnp.float32,
    )  # (N_obs=1, N_total=2, K=1, d=6)
    scores_on = channel.visibility_score_fn(particles_on_boresight)
    assert scores_on.shape == (1, 2, 1)
    assert float(scores_on[0, 1, 0]) > 0.0  # cross-pair particle inside cone
    assert jnp.isclose(scores_on[0, 1, 0], half, atol=1e-5)

    # Particle perpendicular (along +y) → angle = pi/2 → score = half - pi/2.
    particles_perp = jnp.array(
        [
            [
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]],  # cross-pair perpendicular
            ]
        ],
        dtype=jnp.float32,
    )
    scores_perp = channel.visibility_score_fn(particles_perp)
    assert float(scores_perp[0, 1, 0]) < 0.0  # cross-pair particle outside cone
    assert jnp.isclose(scores_perp[0, 1, 0], half - jnp.float32(jnp.pi / 2), atol=1e-5)


def test_conical_visibility_score_fn_rt_layout():
    """Closure must work with RT (d=4) layout — pos_dim=2 zero-pad path."""
    layout = _layout_rt()
    half = jnp.float32(0.5)
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=half,
        sigma_floor=1.0,
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
    channel = out[0]
    assert channel.visibility_score_fn is not None

    # Particle on boresight (+x) — d=4 (RT layout).
    particles_on = jnp.array(
        [
            [
                [[0.0, 0.0, 0.0, 0.0]],
                [[50.0, 0.0, 0.0, 0.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    scores_on = channel.visibility_score_fn(particles_on)
    assert scores_on.shape == (1, 2, 1)
    assert float(scores_on[0, 1, 0]) > 0.0
    assert jnp.isclose(scores_on[0, 1, 0], half, atol=1e-5)

    particles_perp = jnp.array(
        [
            [
                [[0.0, 0.0, 0.0, 0.0]],
                [[0.0, 50.0, 0.0, 0.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    scores_perp = channel.visibility_score_fn(particles_perp)
    assert float(scores_perp[0, 1, 0]) < 0.0


def test_conical_visibility_score_fn_max_over_sensors():
    """With opposing boresights ±x, score is max — particle on +x is positive,
    particle on +y is negative for both sensors."""
    layout = _layout_rtn()
    half = jnp.float32(0.5)
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=half,
        sigma_floor=1.0,
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
    channel = out[0]
    assert channel.visibility_score_fn is not None

    # Particle on -x boresight: covered by sensor 1 (best margin = +half).
    particles_neg_x = jnp.array(
        [
            [
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[-100.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    scores = channel.visibility_score_fn(particles_neg_x)
    assert float(scores[0, 1, 0]) > 0.0
    assert jnp.isclose(scores[0, 1, 0], half, atol=1e-5)


def test_rotated_attitude_changes_visibility():
    """Same target, rotate observer 180° about z: in-cone → out-of-cone."""
    layout = _layout_rtn()
    obs_fn = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma_floor=0.0,
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


# ---- range-conditioned noise -------------------------------------------


def _two_target_state_on_boresight():
    """One observer at the origin, targets at 100 m and 3000 m on +R."""
    return _state_rtn(
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.array([[100.0, 0.0, 0.0], [3000.0, 0.0, 0.0]], dtype=jnp.float32),
    )


def _conical_two_targets(sigma_floor: float, sigma_range_frac: float):
    return ConicalObservation(
        layout=_layout_rtn(n_self=1, n_tgt=2),
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma_floor=sigma_floor,
        sigma_range_frac=sigma_range_frac,
    )


def test_noise_std_scales_with_range():
    """Monte-Carlo noise std must follow sigma_floor + sigma_range_frac * range."""
    sigma_floor = 1.0
    sigma_range_frac = 0.01
    obs_fn = _conical_two_targets(sigma_floor, sigma_range_frac)
    env_state = _two_target_state_on_boresight()
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)

    def one(k):
        return obs_fn(env_state, None, Side.GUARD, None, k, jnp.asarray(0.0, dtype=jnp.float32))[
            0
        ].obs

    samples = jax.vmap(one)(keys)  # (2000, 1, 3, 6)
    truth = jnp.array([[100.0, 0.0, 0.0], [3000.0, 0.0, 0.0]], dtype=jnp.float32)
    residual_close = samples[:, 0, 1, :3] - truth[0]
    residual_far = samples[:, 0, 2, :3] - truth[1]

    expected_close = sigma_floor + sigma_range_frac * 100.0
    expected_far = sigma_floor + sigma_range_frac * 3000.0
    std_close = float(jnp.std(residual_close))
    std_far = float(jnp.std(residual_far))

    assert abs(std_close - expected_close) / expected_close < 0.10, std_close
    assert abs(std_far - expected_far) / expected_far < 0.10, std_far
    expected_ratio = expected_far / expected_close
    assert abs(std_far / std_close - expected_ratio) / expected_ratio < 0.10


def test_velocity_rows_use_the_same_per_pair_sigma():
    """Velocity noise scales with range identically to position noise."""
    obs_fn = _conical_two_targets(1.0, 0.01)
    env_state = _two_target_state_on_boresight()
    keys = jax.random.split(jax.random.PRNGKey(1), 2000)

    def one(k):
        return obs_fn(env_state, None, Side.GUARD, None, k, jnp.asarray(0.0, dtype=jnp.float32))[
            0
        ].obs

    samples = jax.vmap(one)(keys)
    vel_std_far = float(jnp.std(samples[:, 0, 2, 3:]))
    assert abs(vel_std_far - 31.0) / 31.0 < 0.10, vel_std_far


def test_obs_noise_is_per_pair():
    """obs_noise has shape (N_obs, N_total, m, m) with per-pair variances."""
    obs_fn = _conical_two_targets(1.0, 0.01)
    env_state = _two_target_state_on_boresight()
    obs = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )[0]
    assert obs.obs_noise.shape == (1, 3, 6, 6)
    assert jnp.allclose(jnp.diag(obs.obs_noise[0, 1]), 2.0**2, rtol=1e-4)
    assert jnp.allclose(jnp.diag(obs.obs_noise[0, 2]), 31.0**2, rtol=1e-4)
    assert jnp.allclose(obs.noise_for(0, 2), obs.obs_noise[0, 2])


def test_zero_range_fraction_gives_constant_per_pair_noise():
    """sigma_range_frac=0 reproduces a constant sigma_floor**2 * I for every pair."""
    obs_fn = _conical_two_targets(3.0, 0.0)
    env_state = _two_target_state_on_boresight()
    obs = obs_fn(
        env_state,
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype=jnp.float32),
    )[0]
    expected = jnp.broadcast_to(jnp.eye(6, dtype=jnp.float32) * 9.0, (1, 3, 6, 6))
    assert obs.obs_noise.shape == (1, 3, 6, 6)
    assert jnp.allclose(obs.obs_noise, expected, rtol=1e-5)


# ---- mixed noise shapes through a composite channel ---------------------


def _mixed_shape_channels():
    """(m, m) GPS noise plus per-pair conical noise, one observer, two targets."""
    layout = _layout_rtn(n_self=1, n_tgt=2)
    cone = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        half_angle_rad=jnp.deg2rad(jnp.float32(30.0)),
        sigma_floor=1.0,
        sigma_range_frac=0.01,
    )
    gps = OnboardGPSObservation(layout=layout, sigma_gps=0.1)
    comp = CompositeObservation(constituents=(gps, cone))
    channels = comp(
        _two_target_state_on_boresight(),
        None,
        Side.GUARD,
        None,
        jax.random.PRNGKey(3),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert channels[0].obs_noise.shape == (6, 6)
    assert channels[1].obs_noise.shape == (1, 3, 6, 6)
    return channels


def test_mixed_noise_shapes_through_jitted_kf_update():
    """A composite of shared-R and per-pair-R channels updates under jit."""
    channels = _mixed_shape_channels()
    d = 6
    upd = KFBeliefUpdater(
        stm=jnp.eye(d, dtype=jnp.float32),
        control_matrix=jnp.zeros((d, 3), dtype=jnp.float32),
        process_noise=jnp.eye(d, dtype=jnp.float32) * 1e-6,
    )
    belief = KFBelief(
        mean=jnp.zeros((1, 3, d), dtype=jnp.float32),
        cov=jnp.broadcast_to(jnp.eye(d, dtype=jnp.float32) * 1e4, (1, 3, d, d)),
    )

    @jax.jit
    def step(b, chans):
        return upd(b, chans, jnp.zeros((1, 3), dtype=jnp.float32), Side.GUARD, None)

    out = step(belief, channels)
    trace_close = float(jnp.trace(out.cov[0, 1]))
    trace_far = float(jnp.trace(out.cov[0, 2]))
    assert trace_close < trace_far  # tighter R at 100 m than at 3000 m
    # The shared-R GPS channel still corrects the observer's own slot.
    assert float(jnp.trace(out.cov[0, 0])) < trace_close


def test_mixed_noise_shapes_through_pf_update():
    """The same composite drives a PF update with the same per-pair asymmetry."""
    channels = _mixed_shape_channels()
    d = 6
    k = 512
    cloud = jax.random.normal(jax.random.PRNGKey(5), (k, d), dtype=jnp.float32) * 50.0
    truth = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    particles = truth[None, :, None, :] + cloud[None, None, :, :]
    log_w = jnp.full((1, 3, k), -jnp.log(float(k)), dtype=jnp.float32)
    belief = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=jnp.full((1, 3), float(k)),
        weight_entropy=jnp.full((1, 3), float(jnp.log(k))),
        resampled=jnp.zeros((1, 3), dtype=bool),
    )
    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=lambda x, u, dt: x,
        process_noise=jnp.zeros((d, d), dtype=jnp.float32),
        dt=1.0,
        n_eff_threshold=0.0,
    )
    out = upd(
        belief,
        observations=channels,
        action=jnp.zeros((1, 3), dtype=jnp.float32),
        side=Side.GUARD,
        key=jax.random.PRNGKey(7),
    )
    err_close = float(jnp.linalg.norm(out.mean[0, 1, :3] - truth[1, :3]))
    err_far = float(jnp.linalg.norm(out.mean[0, 2, :3] - truth[2, :3]))
    assert err_close < err_far
