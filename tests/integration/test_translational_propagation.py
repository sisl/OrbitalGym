"""Validates the unified-dynamics invariant: translational dynamics ALWAYS
runs in env.step, regardless of whether any action component produced a Δv."""

import jax
import jax.numpy as jnp
import pytest

from orbital_game.config import ScenarioConfig
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide
from orbital_game.registry import (
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from tests.helpers.minimal_scenario import minimal_scenario_kwargs


@pytest.mark.parametrize(
    "dyn_key, frame, truth_field, comp_key",
    [
        (DynamicsKey.HCW_RT, Frame.RT, "rt", StateComponentKey.RT),
        (DynamicsKey.HCW_RTN, Frame.RTN, "rtn", StateComponentKey.RTN),
    ],
)
def test_propagation_runs_without_impulsive_maneuver(dyn_key, frame, truth_field, comp_key):
    cfg = ScenarioConfig(
        **minimal_scenario_kwargs(
            truth_dynamics=dyn_key,
            action_frame=frame,
            guard_components=(comp_key,),
            bandit_components=(comp_key,),
            guard_action_components=(),
            bandit_action_components=(),
        )
    )
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    pre_truth = getattr(state.guards, truth_field)

    identity_actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )
    out = env.step(jax.random.PRNGKey(1), state, identity_actions)
    post_truth = getattr(out.state.guards, truth_field)

    expected = env.truth_dynamics(
        pre_truth,
        jnp.zeros((cfg.n_guards, frame.dim)),
        env.guard_params,
        cfg.dt,
    )
    assert jnp.allclose(post_truth, expected, atol=1e-6)
