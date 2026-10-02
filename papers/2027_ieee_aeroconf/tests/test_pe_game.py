"""The pursuit-evasion game: capture test, budgets, sensing, and the planners."""

import jax.numpy as jnp
import numpy as np
import pytest
from pe_game import Game, Planner, approach_penalty, evaluate_games

SMALL = Planner(samples=32, segments=10)


def test_in_interval_capture_detects_a_crossing_between_samples():
    # Both coast; the pursuer crosses the evader about 5 s into the first interval.
    initial = np.array([[0.0, -45.0, 3.0, 0.0, 9.0, 0.0]])
    endpoint = Game(pursuer="coast", evader="coast", horizon_s=10.0, substeps=1)
    inside = Game(pursuer="coast", evader="coast", horizon_s=10.0, substeps=10)
    assert evaluate_games([endpoint], initial, [0])["outcome"][0] == 0
    result = evaluate_games([inside], initial, [0])
    assert result["outcome"][0] == 1
    assert result["closest_m"][0] < 10.0
    assert result["capture_time_s"][0] == 10.0


def test_sides_draw_on_separate_budgets():
    initial = np.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    game = Game(
        pursuer="direct",
        evader="flee",
        horizon_s=600.0,
        pursuer_budget_mps=np.inf,
        evader_budget_mps=0.3,
    )
    result = evaluate_games([game], initial, [0])
    assert result["evader_dv"][0] == pytest.approx(0.3)
    assert result["pursuer_dv"][0] > 0.3


def test_mppi_pursuer_captures_a_coasting_evader():
    initial = np.array([[0.0, 300.0, 0.0, 0.0, 0.0, 0.0]])
    game = Game(pursuer="mppi_coast", evader="coast", horizon_s=900.0, pursuer_budget_mps=5.0)
    result = evaluate_games([game], initial, [0], planner=SMALL)
    assert result["outcome"][0] == 1
    assert result["pursuer_dv"][0] <= 5.0 + 1e-9


def test_play_stops_at_capture():
    initial = np.array([[0.0, 300.0, 0.0, 0.0, 0.0, 0.0]])
    game = Game(pursuer="lqr", evader="coast", horizon_s=1800.0)
    result = evaluate_games([game], initial, [0])
    assert result["outcome"][0] == 1
    assert result["duration_s"][0] == pytest.approx(result["capture_time_s"][0])


def test_accurate_omnidirectional_sensing_matches_full_information():
    initial = np.array([[0.0, 300.0, 0.0, 0.0, 0.0, 0.0]] * 4)
    full = Game(pursuer="lqr", evader="flee", evader_cap_mps=0.05, horizon_s=900.0)
    sensed = Game(
        pursuer="lqr",
        evader="flee",
        evader_cap_mps=0.05,
        horizon_s=900.0,
        sensor="radial",
        noise_fraction=1e-6,
    )
    a = evaluate_games([full] * 4, initial, np.arange(4))
    b = evaluate_games([sensed] * 4, initial, np.arange(4))
    np.testing.assert_array_equal(a["outcome"], b["outcome"])
    np.testing.assert_allclose(a["capture_time_s"], b["capture_time_s"])
    assert np.all(b["pursuer_detection_fraction"] == 1.0)
    assert np.all(b["pursuer_mean_error_m"] < 0.2)


def test_out_of_range_target_is_never_detected():
    initial = np.array([[0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]])
    game = Game(
        pursuer="coast",
        evader="coast",
        horizon_s=100.0,
        sensor="radial",
        sensor_range_m=500.0,
        prior_sigma_m=50.0,
        prior_sigma_mps=0.01,
    )
    result = evaluate_games([game], initial, [0])
    assert result["pursuer_detection_fraction"][0] == 0.0
    assert result["pursuer_mean_error_m"][0] > 1.0


def test_cone_loses_a_target_outside_the_estimated_line_of_sight():
    # A 90 degree prior error with a 10 degree cone: the target is not in view.
    initial = np.array([[0.0, 300.0, 0.0, 0.0, 0.0, 0.0]])
    wide = Game(
        pursuer="coast",
        evader="coast",
        horizon_s=10.0,
        sensor="cone",
        cone_half_angle_deg=170.0,
        prior_sigma_m=300.0,
    )
    narrow = Game(
        pursuer="coast",
        evader="coast",
        horizon_s=10.0,
        sensor="cone",
        cone_half_angle_deg=1.0,
        prior_sigma_m=300.0,
    )
    assert evaluate_games([wide], initial, [0])["pursuer_detection_fraction"][0] == 1.0
    assert evaluate_games([narrow], initial, [0])["pursuer_detection_fraction"][0] == 0.0


def test_velocity_matching_rejects_a_fast_crossing():
    # The same crossing as above at 9 m/s: captured without a velocity-matching condition,
    # not with one of 1 m/s.
    initial = np.array([[0.0, -45.0, 3.0, 0.0, 9.0, 0.0]])
    free = Game(pursuer="coast", evader="coast", horizon_s=10.0)
    gated = Game(pursuer="coast", evader="coast", horizon_s=10.0, capture_speed_mps=1.0)
    assert evaluate_games([free], initial, [0])["outcome"][0] == 1
    assert evaluate_games([gated], initial, [0])["outcome"][0] == 0
    assert gated.key() != free.key()
    assert "capture_speed_mps" not in free.record()


def test_mcts_pursuer_captures_a_coasting_evader():
    initial = np.array([[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]])
    game = Game(pursuer="mcts", evader="coast", horizon_s=300.0, pursuer_budget_mps=5.0)
    result = evaluate_games([game], initial, [0])
    assert result["outcome"][0] == 1


def test_approach_penalty_vanishes_without_a_speed_condition():
    x = jnp.array([100.0, 0.0, 0.0, -5.0, 0.0, 0.0])
    assert float(approach_penalty(x, jnp.inf)) == 0.0
    assert float(approach_penalty(x, 0.5)) > 0.0
