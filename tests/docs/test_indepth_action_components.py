"""In-depth: Action components. Source-of-truth for snippets in
docs/in-depth/action-components.md.
"""

from __future__ import annotations


def test_indepth_action_components_default_impulsive_maneuver():
    # --8<-- [start:default-impulsive-maneuver]
    from orbital_game import OrbitalGameEnv, make_lady_bandit_guard
    from orbital_game.registry import ActionComponentKey

    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1, seed=0)
    # Every bundled scenario defaults both sides to a single
    # IMPULSIVE_MANEUVER component.
    assert cfg.guard_action_components == (ActionComponentKey.IMPULSIVE_MANEUVER,)
    assert cfg.bandit_action_components == (ActionComponentKey.IMPULSIVE_MANEUVER,)

    env = OrbitalGameEnv(cfg)
    # The per-side Command class exposes one field per component field —
    # IMPULSIVE_MANEUVER contributes `dv` of shape (N_side, 3).
    cmd = env.guard_command_cls.zeros(cfg.n_guards)
    assert cmd.dv.shape == (cfg.n_guards, 3)
    # --8<-- [end:default-impulsive-maneuver]


def test_indepth_action_components_compose_communicate():
    # --8<-- [start:compose-communicate]
    from orbital_game import OrbitalGameEnv, make_lady_bandit_guard

    cfg = make_lady_bandit_guard(
        n_guards=1, n_bandits=1, with_communication=True, comm_cost=5.0, seed=0
    )
    env = OrbitalGameEnv(cfg)

    # The guard side now composes (IMPULSIVE_MANEUVER, COMMUNICATE) — its
    # Command class carries `dv` (from IMPULSIVE_MANEUVER) plus `active` and
    # `payload` (from COMMUNICATE).
    cmd = env.guard_command_cls.zeros(cfg.n_guards)
    assert cmd.dv.shape == (cfg.n_guards, 3)
    assert cmd.active.shape == (cfg.n_guards,)
    assert cmd.payload.shape == (cfg.n_guards, 6)

    # The bandit side keeps the (IMPULSIVE_MANEUVER,) default — only `dv`.
    bandit_cmd = env.bandit_command_cls.zeros(cfg.n_bandits)
    assert hasattr(bandit_cmd, "dv")
    assert not hasattr(bandit_cmd, "active")
    # --8<-- [end:compose-communicate]


def test_indepth_action_components_reward_delta():
    # --8<-- [start:reward-delta]
    import jax
    import jax.numpy as jnp

    from orbital_game import Actions, BySide, OrbitalGameEnv, make_lady_bandit_guard

    cfg = make_lady_bandit_guard(with_communication=True, comm_cost=5.0, seed=0)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(42))

    # Step from the SAME state with comms off, then comms on. Δv is zero in
    # both, so the only difference is the `active` flag — and the reward
    # delta should equal exactly comm_cost.
    rewards = []
    for active in (False, True):
        guard_cmd = env.guard_command_cls.zeros(cfg.n_guards).replace(active=jnp.array([active]))
        actions = Actions(
            sides=BySide(
                guard=guard_cmd,
                bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
            )
        )
        out = env.step(jax.random.PRNGKey(1), state, actions)
        rewards.append(float(out.outputs.guard.reward))

    silent_reward, comms_reward = rewards
    assert jnp.isclose(silent_reward - comms_reward, 5.0, atol=1e-5)
    # --8<-- [end:reward-delta]


def test_indepth_action_components_build_command_class():
    # --8<-- [start:build-command-class]
    from orbital_game.actions.assemble import build_command_class
    from orbital_game.actions.components import Communicate, ImpulsiveManeuver
    from orbital_game.registry import Frame

    # Per-side Command classes are built from a tuple of ActionComponent
    # *instances*. Order in this tuple determines apply order in env.step.
    maneuver = ImpulsiveManeuver(
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
    )
    cmd_cls = build_command_class(
        (maneuver, Communicate()),
        n_agents=2,
        class_name="DocCmd",
    )
    cmd = cmd_cls.zeros(2)
    assert cmd.dv.shape == (2, 3)
    assert cmd.active.shape == (2,)
    assert cmd.payload.shape == (2, 6)
    # --8<-- [end:build-command-class]
