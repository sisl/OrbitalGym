"""Search dynamics must retain every available full-state sensor channel."""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Side
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.observations.onboard_gps import OnboardGPSObservation
from orbitalgym.observations.reference import FullObservation
from orbitalgym.observations.teammate_ephemeris import TeammateEphemerisObservation
from orbitalgym.observations.types import merge_full_state_observations
from orbitalgym.policies.heuristic.glideslope import GlideslopeIntercept
from orbitalgym.policies.mcts import MCTSPolicy
from orbitalgym.policies.mppi import MPPIPolicy
from tests.eval.test_information_metrics import _cone_pointed_away_cfg


def _models(composite):
    cfg = _cone_pointed_away_cfg()
    cfg = dataclasses.replace(
        cfg,
        n_guards=2,
        bandit_components=cfg.guard_components,
        bandit_action_components=cfg.guard_action_components,
        bandit_attitude_params=cfg.guard_attitude_params,
    )
    if composite:
        sensor = CompositeObservation(
            constituents=(
                OnboardGPSObservation(cfg.layout, sigma_gps=0.0),
                TeammateEphemerisObservation(cfg.layout, sigma=0.0),
                ConicalObservation(cfg.layout, jnp.array([[1.0, 0.0, 0.0]]), jnp.pi, 0.0, 0.0),
            )
        )
    else:
        sensor = FullObservation(cfg.layout)
    cfg = dataclasses.replace(cfg, guard_observation_fn=sensor, bandit_observation_fn=sensor)
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        guards=state.guards.replace(
            rt=jnp.array([[100.0, 50.0, 0.0, 0.0], [300.0, 200.0, 0.0, 0.0]])
        ),
        bandits=state.bandits.replace(rt=jnp.array([[700.0, -100.0, 0.0, 0.0]])),
    )
    adapter = POMDPAdapter(env)

    def model(n, opponents, cls):
        return GlideslopeIntercept.build(
            mean_motion=0.001,
            dt=cfg.dt,
            max_dv_mps=0.2,
            n_vehicles=n,
            n_opponents=opponents,
            state_dim=4,
            command_cls=cls,
        )

    return (
        env,
        adapter,
        state,
        model(1, 2, env.bandit_command_cls),
        model(2, 1, env.guard_command_cls),
    )


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("planner", ["mppi", "mcts"])
def test_search_models_agree_for_equivalent_full_and_composite_sensors(planner, independent):
    results = []
    for composite in (False, True):
        env, adapter, state, opponent, teammate = _models(composite)
        common = dict(
            env_model=adapter,
            side=Side.GUARD,
            opponent_model=opponent,
            teammate_model=teammate,
            n_vehicles=2,
            command_cls=env.guard_command_cls,
            coordination="independent" if independent else "joint",
            model_view_fn=partial(
                merge_full_state_observations, state_dim=env.layout.dynamics_state_dim
            ),
        )
        packed, key = adapter.pack(state), jax.random.PRNGKey(3)
        if planner == "mppi":
            policy = MPPIPolicy(
                **common, n_samples=2, horizon=2, temperature=0.1, noise_sigma=0.1, dv_max=0.2
            )
            u = jnp.zeros((2, 2)) if independent else jnp.zeros((2, 2, 2))
            result = policy._rollout_cost(packed, u, key, self_index=0 if independent else None)
        else:
            grid = jnp.array([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]])
            policy = MCTSPolicy(
                **common,
                action_grid=grid,
                opponent_action_grid=grid,
                num_simulations=2,
                max_depth=1,
            )
            output = (
                policy._search_one_vehicle_out(packed, key, 0)
                if independent
                else policy._search_joint_out(packed, key)
            )
            result = output.search_tree.embeddings
        results.append(result)
    # These sensors reveal exactly the same noiseless state. The real
    # glideslope modeled commands and simulated transitions must agree.
    for full, composite in zip(
        jax.tree_util.tree_leaves(results[0]), jax.tree_util.tree_leaves(results[1]), strict=True
    ):
        np.testing.assert_allclose(full, composite, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("planner", ["mppi", "mcts"])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_default_model_view_accepts_partial_sensors(planner, independent, composite):
    from orbitalgym import make_lady_bandit_guard
    from orbitalgym.observations.range_limited import RangeLimitedObservation
    from orbitalgym.policies.uniform_random import UniformRandomDiscretePolicy
    from orbitalgym.policies.zero import ZeroControl

    cfg = make_lady_bandit_guard(n_guards=2)
    position = RangeLimitedObservation(cfg.layout, sensor_range_m=1e6, sigma_range=0.0)
    sensor = (
        CompositeObservation((OnboardGPSObservation(cfg.layout), position))
        if composite
        else position
    )
    cfg = dataclasses.replace(cfg, guard_observation_fn=sensor, bandit_observation_fn=sensor)
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env)
    state, _ = env.reset(jax.random.PRNGKey(0))
    grid = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    inner = (
        UniformRandomDiscretePolicy(grid, cfg.n_bandits, env.bandit_command_cls)
        if planner == "mcts"
        else ZeroControl(n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls)
    )
    measurement_width = 9 if composite else 3

    def opponent(ps, view, key, t):
        # The generic model contract retains the complete native flattened
        # measurement layout, even when no full state can be reconstructed.
        assert view.shape == (cfg.n_bandits * 3 * measurement_width,)
        return inner(ps, view, key, t)

    teammate_zero = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)

    def teammate(ps, view, key, t):
        assert view.shape == (cfg.n_guards * 3 * measurement_width,)
        return teammate_zero(ps, view, key, t)

    common = dict(
        env_model=adapter,
        side=Side.GUARD,
        opponent_model=opponent,
        teammate_model=teammate,
        n_vehicles=cfg.n_guards,
        command_cls=env.guard_command_cls,
        coordination="independent" if independent else "joint",
    )
    if planner == "mcts":
        policy = MCTSPolicy(
            **common, action_grid=grid, opponent_action_grid=grid, num_simulations=2, max_depth=1
        )
    else:
        policy = MPPIPolicy(**common, n_samples=2, horizon=1)
    command, _ = jax.jit(lambda s, k: policy(None, s, k, state.t))(
        adapter.pack(state), jax.random.PRNGKey(2)
    )
    assert command.dv.shape == (2, 3)
    assert np.all(np.isfinite(command.dv))
