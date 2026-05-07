"""Visualization utilities — static plots, glyph primitives, and animations.

Three layers:

- ``trajectories`` / ``summaries`` — original static plots that take raw arrays
- ``glyphs`` — 3D primitives (cubes, sensor-range spheres, thrust arrows,
  belief ellipsoids) used by the animation system
- ``per_rollout`` — single-rollout multi-actor static views (all guards +
  bandits color-coded together)
- ``animation`` — interactive ipywidgets viewer + ffmpeg MP4 export
- ``reward_diagnostics`` — game-aware plots that overlay the signal driving
  the reward against the reward itself
"""

from orbital_game.viz.animation import (
    RolloutScene,
    interactive_viewer,
    render_frame,
    save_animation,
)
from orbital_game.viz.glyphs import (
    draw_belief_ellipse_2d,
    draw_belief_ellipsoid,
    draw_body_axes,
    draw_circle,
    draw_cube,
    draw_sphere,
    draw_thrust_arrow,
    quat_to_rotation_matrix,
)
from orbital_game.viz.per_rollout import (
    plot_rollout_3d,
    plot_rollout_mass,
    plot_rollout_panels,
    plot_rollout_rewards,
)
from orbital_game.viz.reward_diagnostics import (
    plot_get_off_my_lawn_diagnostic,
    plot_get_off_my_lawn_position_sweep,
    plot_lady_bandit_guard_diagnostic,
    plot_lady_bandit_guard_position_sweep,
    plot_observation_blocking_animated_diagnostic,
    plot_observation_blocking_diagnostic,
    plot_observation_blocking_position_sweep,
    plot_pursuit_evasion_diagnostic,
    plot_pursuit_evasion_position_sweep,
    plot_reward_diagnostic,
    plot_reward_position_sweep,
    plot_sun_blocking_animated_diagnostic,
    plot_sun_blocking_diagnostic,
    plot_sun_blocking_position_sweep,
)
from orbital_game.viz.summaries import plot_mass_curve, plot_reward_curve
from orbital_game.viz.trajectories import plot_rn, plot_rt, plot_rtn_3d, plot_tn

__all__ = [
    "RolloutScene",
    "draw_belief_ellipse_2d",
    "draw_belief_ellipsoid",
    "draw_body_axes",
    "draw_circle",
    "draw_cube",
    "draw_sphere",
    "draw_thrust_arrow",
    "interactive_viewer",
    "plot_get_off_my_lawn_diagnostic",
    "plot_get_off_my_lawn_position_sweep",
    "plot_lady_bandit_guard_diagnostic",
    "plot_lady_bandit_guard_position_sweep",
    "plot_mass_curve",
    "plot_observation_blocking_animated_diagnostic",
    "plot_observation_blocking_diagnostic",
    "plot_observation_blocking_position_sweep",
    "plot_pursuit_evasion_diagnostic",
    "plot_pursuit_evasion_position_sweep",
    "plot_reward_curve",
    "plot_reward_diagnostic",
    "plot_reward_position_sweep",
    "plot_rn",
    "plot_rollout_3d",
    "plot_rollout_mass",
    "plot_rollout_panels",
    "plot_rollout_rewards",
    "plot_rt",
    "plot_rtn_3d",
    "plot_sun_blocking_animated_diagnostic",
    "plot_sun_blocking_diagnostic",
    "plot_sun_blocking_position_sweep",
    "plot_tn",
    "quat_to_rotation_matrix",
    "render_frame",
    "save_animation",
]
