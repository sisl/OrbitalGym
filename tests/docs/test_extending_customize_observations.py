"""Customize observations — source-of-truth for snippets in
docs/extending/customize-observations.md.
"""

from __future__ import annotations

import jax


def test_customize_observations_walkthrough():
    # --8<-- [start:imports]
    import dataclasses
    from dataclasses import dataclass
    from typing import Any

    import jax.numpy as jnp

    from orbital_game import OrbitalGameEnv, SingleAgentView, make_pursuit_evasion
    from orbital_game.observations.types import Observation
    # --8<-- [end:imports]

    # --8<-- [start:position-only-channel]
    @dataclass(frozen=True)
    class PositionOnlyObservation:
        """One channel: own + opponent position only (drops velocity).

        Returns shape `(N_self, N_total, 3)` for RTN dynamics. The
        obs_matrix projects the full `d`-dim state to its first 3
        position components.
        """

        layout: Any  # carries n_guards, n_bandits, dynamics_state_dim

        def __call__(self, env_state, actions, side, params, key, t):
            del actions, params, key, t
            from orbital_game.belief._common import _truth_arrays_for_side

            own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
            n_self = own_truth.shape[0]
            n_tgt = opp_truth.shape[0]
            n_total = n_self + n_tgt
            d = self.layout.dynamics_state_dim
            m = 3  # position-only

            stacked = jnp.concatenate([own_truth, opp_truth], axis=0)
            positions = stacked[:, :m]  # (N_total, 3)
            obs = jnp.broadcast_to(positions[None, :, :], (n_self, n_total, m))
            visible = jnp.ones((n_self, n_total), dtype=bool)
            H = jnp.zeros((m, d)).at[:m, :m].set(jnp.eye(m))  # noqa: N806
            R = jnp.eye(m) * 1e-6  # noqa: N806
            return (Observation(obs=obs, visible=visible, obs_matrix=H, obs_noise=R),)

    # --8<-- [end:position-only-channel]

    # --8<-- [start:wire-it-up]
    cfg = make_pursuit_evasion(seed=0, max_horizon_s=200.0)
    cfg = dataclasses.replace(
        cfg,
        guard_observation_fn=PositionOnlyObservation(layout=cfg.layout),
        bandit_observation_fn=PositionOnlyObservation(layout=cfg.layout),
    )
    env = OrbitalGameEnv(cfg)
    # --8<-- [end:wire-it-up]

    view = SingleAgentView(env)
    state, obs, opp_ps = view.reset(jax.random.PRNGKey(0))
    del obs
    controlled_cmd = env.guard_command_cls.zeros(cfg.n_guards)
    next_state, next_obs, reward, done, next_opp_ps, info = view.step(
        jax.random.PRNGKey(1), state, controlled_cmd, opp_ps
    )
    del next_state, reward, done, next_opp_ps, info
    # Position-only channel: per-side flat obs is (N_self * N_total * 3) entries.
    expected = cfg.n_guards * (cfg.n_guards + cfg.n_bandits) * 3
    assert next_obs.shape == (expected,)
