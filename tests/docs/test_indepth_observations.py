"""In-depth: Observations. Source-of-truth for snippets in
docs/in-depth/observations.md.
"""

from __future__ import annotations


def test_indepth_observations_full():
    # --8<-- [start:full-observation]
    import jax

    from orbital_game import OrbitalGameEnv, make_lady_bandit_guard
    from orbital_game.env.types import Side

    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1, seed=0)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Both sides expose their observation function as `env.<side>_observation_fn`.
    # The protocol is `(env_state, side, params, key, t) -> tuple[Observation, ...]`.
    # The bundled `FullObservation` returns a single channel.
    channels = env.guard_observation_fn(state, Side.GUARD, cfg, jax.random.PRNGKey(1), state.t)
    assert isinstance(channels, tuple)
    channel = channels[0]
    # Each channel carries `obs`, `visible`, `obs_matrix`, `obs_noise`.
    assert hasattr(channel, "obs")
    # FullObservation is per-pair: shape (N_self, N_total, dynamics_state_dim).
    assert channel.obs.shape == (cfg.n_guards, cfg.n_guards + cfg.n_bandits, 6)
    # --8<-- [end:full-observation]


def test_indepth_observations_range_limited():
    # --8<-- [start:range-limited]
    import jax

    from orbital_game import OrbitalGameEnv, make_lady_bandit_guard
    from orbital_game.env.types import Side
    from orbital_game.observations import RangeLimitedObservation

    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1, seed=0)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Swap in a RangeLimitedObservation channel — distance-gated per-pair
    # 3-D position measurement with Gaussian noise.
    obs_fn = RangeLimitedObservation(layout=env.layout, sensor_range_m=2000.0, sigma_range=1.0)
    channels = obs_fn(state, Side.GUARD, cfg, jax.random.PRNGKey(1), state.t)
    channel = channels[0]
    # Per-pair shapes: obs is (N_self, N_total, 3), visible is (N_self, N_total).
    assert channel.obs.ndim == 3
    assert channel.visible.ndim == 2
    # --8<-- [end:range-limited]
