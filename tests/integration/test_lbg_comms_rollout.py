"""End-to-end: LBG with comms enabled. Compare silent vs talking step from
the same starting state — only difference is the communicate.active flag,
so reward delta must equal exactly comm_cost."""

import jax
import jax.numpy as jnp

from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide
from orbital_game.games.lady_bandit_guard import make_lady_bandit_guard
from orbital_game.observations.comms_leak import CommsLeakObservation
from orbital_game.observations.composite import CompositeObservation
from orbital_game.observations.reference import FullObservation


def test_lbg_comms_three_step_rollout():
    cfg = make_lady_bandit_guard(with_communication=True, comm_cost=5.0)
    env = OrbitalGameEnv(cfg)
    key = jax.random.PRNGKey(42)
    state, _ = env.reset(key)

    rewards_silent_then_comms = []

    # Run each comms setting from the SAME starting state so the only difference
    # is the comm cost — distance-to-reference baseline cancels out.
    for comms in [False, True]:
        guard_cmd = env.guard_command_cls.zeros(cfg.n_guards).replace(
            active=jnp.array([comms]),
        )
        bandit_cmd = env.bandit_command_cls.zeros(cfg.n_bandits)
        actions = Actions(sides=BySide(guard=guard_cmd, bandit=bandit_cmd))
        out = env.step(jax.random.PRNGKey(1), state, actions)
        rewards_silent_then_comms.append(float(out.outputs.guard.reward))

    # Silent step has higher reward than comms step by exactly comm_cost (5.0)
    # because (a) same starting state, (b) zero Δv on both, (c) only the
    # `active` flag differs.
    diff = rewards_silent_then_comms[0] - rewards_silent_then_comms[1]
    assert jnp.isclose(diff, 5.0, atol=1e-5)


def test_env_step_with_composite_full_plus_commsleak_emits_two_channels():
    """Wire bandit observation as CompositeObservation(Full + CommsLeak) and
    step the env with comms active — bandit's obs tuple must have two
    channels and the leak channel's opposing-side mask must be True."""
    cfg = make_lady_bandit_guard(with_communication=True, comm_cost=5.0)
    # Replace bandit obs with composite of Full + CommsLeak. The default
    # bandit obs is FullObservation alone — we extend it with the leak channel.
    object.__setattr__(
        cfg,
        "bandit_observation_fn",
        CompositeObservation(
            constituents=(
                FullObservation(layout=cfg.layout),
                CommsLeakObservation(layout=cfg.layout),
            )
        ),
    )

    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    guard_cmd = env.guard_command_cls.zeros(cfg.n_guards).replace(
        active=jnp.array([True]),
    )
    bandit_cmd = env.bandit_command_cls.zeros(cfg.n_bandits)
    actions = Actions(sides=BySide(guard=guard_cmd, bandit=bandit_cmd))
    out = env.step(jax.random.PRNGKey(1), state, actions)

    bandit_obs = out.outputs.bandit.obs
    # Two channels: Full + CommsLeak.
    assert len(bandit_obs) == 2

    full_channel, leak_channel = bandit_obs
    n_self = cfg.n_bandits
    n_total = cfg.n_guards + cfg.n_bandits
    d = cfg.layout.dynamics_state_dim
    # Both channels share the (n_self, n_total, d) layout.
    assert full_channel.obs.shape == (n_self, n_total, d)
    assert leak_channel.obs.shape == (n_self, n_total, d)
    # Leak channel: opposing-side columns (guards) visible because comms active.
    assert leak_channel.visible[:, n_self:].all()
    # Own-side columns are not visible via leak (a bandit doesn't observe
    # itself through guard's broadcasts).
    assert not leak_channel.visible[:, :n_self].any()
