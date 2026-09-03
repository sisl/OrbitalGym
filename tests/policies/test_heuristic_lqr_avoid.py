"""LQR-to-lady with guard avoidance, driven by a belief mean or a flat observation."""

import flax
import jax
import jax.numpy as jnp
import pytest

from orbitalgym.policies.heuristic.lqr_avoid import LQRGoToLadyWithAvoidance
from tests.policies._helpers import make_impulsive_maneuver_command_cls

N_MOTION = 1.078e-3


@flax.struct.dataclass
class _Belief:
    mean: jax.Array


def _view(bandit_rtn, guard_rtn):
    rows = jnp.stack([bandit_rtn, guard_rtn])  # own first, then opposing
    return _Belief(mean=rows[None])  # (1, 2, 6)


def _policy(avoidance_gain, max_dv_mps: float = 1.0):
    return LQRGoToLadyWithAvoidance.build(
        mean_motion=N_MOTION,
        dt=10.0,
        n_vehicles=1,
        n_opponents=1,
        state_dim=6,
        command_cls=make_impulsive_maneuver_command_cls(1),
        max_dv_mps=max_dv_mps,
        avoidance_gain=avoidance_gain,
        avoidance_sigma_m=300.0,
    )


BANDIT = jnp.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0])
GUARD = jnp.array([600.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_pure_lqr_moves_toward_lady():
    cmd, _ = _policy(0.0)(None, _view(BANDIT, GUARD), None, 0.0)
    assert cmd.dv.shape == (1, 3)
    assert cmd.dv[0, 0] < 0.0  # radial component points toward the origin


def test_avoidance_pushes_away_from_guard():
    pure, _ = _policy(0.0, max_dv_mps=100.0)(None, _view(BANDIT, GUARD), None, 0.0)
    avoid, _ = _policy(1.0, max_dv_mps=100.0)(None, _view(BANDIT, GUARD), None, 0.0)
    assert avoid.dv[0, 0] < pure.dv[0, 0]  # guard sits at +R; repulsion adds -R


def test_cross_track_is_zero_and_clip_holds():
    cmd, _ = _policy(5.0)(None, _view(BANDIT, GUARD), None, 0.0)
    assert cmd.dv[0, 2] == 0.0
    assert bool(jnp.all(jnp.abs(cmd.dv) <= 1.0 + 1e-9))


def test_flat_observation_matches_belief_path():
    view = _view(BANDIT, GUARD)
    flat = view.mean.reshape(-1)
    a, _ = _policy(1.0)(None, view, None, 0.0)
    b, _ = _policy(1.0)(None, flat, None, 0.0)
    assert jnp.allclose(a.dv, b.dv)


def test_rejects_wrong_sized_flat_observation():
    policy = _policy(0.0)
    wrong_size_flat = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="full-state"):
        policy(None, wrong_size_flat, None, 0.0)


def test_uses_last_channel_of_a_composite_flat_observation():
    """Two equal-sized full-state channels concatenated (as CompositeObservation
    would produce): the trailing channel is used, matching the single-channel
    flat-observation result."""
    view = _view(BANDIT, GUARD)
    single_flat = view.mean.reshape(-1)
    noise_channel = single_flat + 1000.0  # a differing leading channel
    composite_flat = jnp.concatenate([noise_channel, single_flat])
    a, _ = _policy(1.0)(None, single_flat, None, 0.0)
    b, _ = _policy(1.0)(None, composite_flat, None, 0.0)
    assert jnp.allclose(a.dv, b.dv)
