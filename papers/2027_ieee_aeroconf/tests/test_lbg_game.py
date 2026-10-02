"""The lady-bandit-guard game: breach, capture, their order, and the timeout."""

import numpy as np
from lbg_game import LBGGame, evaluate_lbg


def state(guard, bandit):
    return np.array([[*guard, 0.0, 0.0, 0.0, *bandit, 0.0, 0.0, 0.0]])


def test_bandit_flying_through_the_lady_breaches():
    # The bandit crosses the lady at 9 m/s; the guard is far away and coasts.
    initial = np.array([[0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 0.0, -45.0, 3.0, 0.0, 9.0, 0.0]])
    game = LBGGame(guard="coast", bandit="coast", horizon_s=10.0, breach_m=10.0)
    result = evaluate_lbg([game], initial, [0])
    assert result["breach"][0] == 1 and result["outcome"][0] == 0


def test_speed_condition_blocks_a_fast_breach():
    initial = np.array([[0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 0.0, -45.0, 3.0, 0.0, 9.0, 0.0]])
    game = LBGGame(
        guard="coast", bandit="coast", horizon_s=10.0, breach_m=10.0, breach_speed_mps=1.0
    )
    assert evaluate_lbg([game], initial, [0])["breach"][0] == 0


def test_capture_before_breach_wins_for_the_guard():
    # The capture sphere about the guard contains the breach sphere about the lady, so any
    # breach is preceded by, or ties with, a capture.
    initial = state((0.0, 30.0, 0.0), (0.0, 1000.0, 0.0))
    game = LBGGame(
        guard="coast",
        bandit="lqr",
        horizon_s=1800.0,
        capture_m=60.0,
        breach_m=20.0,
        bandit_budget_mps=np.inf,
    )
    result = evaluate_lbg([game], initial, [0])
    assert result["capture"][0] == 1 and result["breach"][0] == 0


def test_timeout_is_a_guard_win():
    initial = state((0.0, 0.0, 300.0), (0.0, 1000.0, 0.0))
    game = LBGGame(guard="coast", bandit="coast", horizon_s=100.0)
    result = evaluate_lbg([game], initial, [0])
    assert result["outcome"][0] == 1 and result["capture"][0] == 0 and result["breach"][0] == 0
