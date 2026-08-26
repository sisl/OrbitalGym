"""Get-Off-My-Lawn variants — exercises the snippets shown in
docs/games/get-off-my-lawn.md so they keep compiling.
"""

from __future__ import annotations


def test_strict_zero_sum_variant():
    from orbitalgym import make_get_off_my_lawn

    cfg = make_get_off_my_lawn(alpha_chase=0.0, r_pushout=0.0)
    assert cfg.reward_fn.alpha_chase == 0.0
    assert cfg.reward_fn.r_pushout == 0.0


def test_catch_dominated_variant():
    from orbitalgym import make_get_off_my_lawn

    cfg = make_get_off_my_lawn(r_catch=10_000.0, r_loiter=1.0)
    assert cfg.reward_fn.r_catch == 10_000.0
    assert cfg.reward_fn.r_loiter == 1.0


def test_tight_keep_band_variant():
    from orbitalgym import make_get_off_my_lawn

    cfg = make_get_off_my_lawn(
        r_min_keep_m=180.0,
        r_max_keep_m=220.0,
        catch_radius_m=40.0,
        keep_out_radius_m=1500.0,
    )
    assert cfg.game.r_min_keep_m == 180.0
    assert cfg.game.r_max_keep_m == 220.0
    assert cfg.game.catch_radius_m == 40.0
