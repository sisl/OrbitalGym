"""Native pursuit/evasion endpoints, outcome roles, and bank evaluation."""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym import OrbitalGymEnv, make_pursuit_evasion
from orbitalgym.belief.kf import KFBeliefUpdater, KFFromTruthInitializer
from orbitalgym.env.types import BySide
from orbitalgym.eval import metrics as metrics_module
from orbitalgym.eval.bank import sample_bank
from orbitalgym.eval.evaluate import evaluate_bank, metrics_to_records
from orbitalgym.eval.metrics import Outcome
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.registry import StateComponentKey
from orbitalgym.rollout import rollout
from orbitalgym.sampling.mass import ConstantMass


def _policies(env):
    return BySide(
        guard=ZeroControl(n_vehicles=1, command_cls=env.guard_command_cls),
        bandit=ZeroControl(n_vehicles=1, command_cls=env.bandit_command_cls),
    )


def _init():
    return BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)


@pytest.fixture(scope="module")
def pe_trace():
    cfg = make_pursuit_evasion(max_horizon_s=30.0, capture_distance_m=10.0)
    env = OrbitalGymEnv(cfg)
    traj = rollout(env, _policies(env), _init(), jax.random.PRNGKey(0), n_steps=3)
    return traj, cfg


def _geometry(pe_trace, distances, done=(False, False, True)):
    """Set signed x separations at four sampled endpoints in a real trajectory."""
    traj, cfg = pe_trace
    guards = traj.env_state.guards.replace(rtn=jnp.zeros_like(traj.env_state.guards.rtn))
    bandits = traj.env_state.bandits.replace(
        rtn=jnp.zeros_like(traj.env_state.bandits.rtn).at[:, 0, 0].set(jnp.array(distances[:-1]))
    )
    final = traj.final_state.replace(
        guards=traj.final_state.guards.replace(rtn=jnp.zeros_like(traj.final_state.guards.rtn)),
        bandits=traj.final_state.bandits.replace(
            rtn=jnp.zeros_like(traj.final_state.bandits.rtn).at[0, 0].set(distances[-1])
        ),
    )
    return traj.replace(
        env_state=traj.env_state.replace(guards=guards, bandits=bandits),
        final_state=final,
        episode_done=jnp.array(done),
    ), cfg


def test_crossing_between_endpoints_is_not_native_pe_capture(pe_trace):
    traj, cfg = _geometry(pe_trace, [20.0, -20.0, -20.0, -20.0])
    metrics = metrics_module.pe_episode_metrics(traj, cfg)
    assert int(metrics.outcome) == Outcome.TIMEOUT
    assert float(metrics.min_d_gb) == 20.0
    assert np.isnan(metrics.min_d_bl)
    assert np.isnan(metrics.belief_err_guard_at_commit)
    assert int(metrics.dwell_catch_steps) == int(metrics.dwell_breach_steps) == 0


@pytest.mark.parametrize("distance,captured", [(10.0, False), (9.999, True)])
def test_final_endpoint_uses_strict_threshold_and_bandit_capture_code(pe_trace, distance, captured):
    traj, cfg = _geometry(pe_trace, [30.0, 25.0, 20.0, distance])
    metrics = metrics_module.pe_episode_metrics(traj, cfg)
    expected = Outcome.PE_CAPTURE if captured else Outcome.TIMEOUT
    assert int(metrics.outcome) == expected
    assert int(metrics.outcome) != Outcome.CATCH
    assert int(metrics.steps) == 3
    assert float(metrics.min_d_gb) == pytest.approx(distance)


def test_padding_does_not_change_completed_capture_or_resource_totals(pe_trace):
    traj, cfg = _geometry(pe_trace, [20.0, 5.0, float("nan"), float("nan")], (True, True, True))
    dv = jnp.array([[[0.2, 0.0, 0.0]], [[100.0, 0.0, 0.0]], [[float("nan"), 0.0, 0.0]]])
    traj = traj.replace(applied_dv=BySide(guard=dv, bandit=dv))
    metrics = metrics_module.pe_episode_metrics(traj, cfg)
    assert int(metrics.outcome) == Outcome.PE_CAPTURE
    assert int(metrics.steps) == 1
    assert float(metrics.min_d_gb) == 5.0
    assert float(metrics.dv_guard) == pytest.approx(0.2)
    assert float(metrics.dv_bandit) == pytest.approx(0.2)


def test_partial_rollout_is_unresolved_until_actual_game_deadline(pe_trace):
    traj, cfg = _geometry(pe_trace, [40.0, 35.0, 30.0, 25.0], (False, False, False))
    cfg = dataclasses.replace(cfg, max_horizon_s=100.0)
    assert int(metrics_module.pe_episode_metrics(traj, cfg).outcome) == Outcome.UNRESOLVED
    # Classification uses the absolute post-step clock, including resumed runs.
    traj = traj.replace(
        env_state=traj.env_state.replace(step=traj.env_state.step + 7),
        final_state=traj.final_state.replace(step=traj.final_state.step + 7),
    )
    assert int(metrics_module.pe_episode_metrics(traj, cfg).outcome) == Outcome.TIMEOUT


def test_invalid_initial_conditions_override_capture(pe_trace):
    traj, cfg = _geometry(pe_trace, [20.0, 15.0, 12.0, 5.0])
    traj = traj.replace(env_state=traj.env_state.replace(ic_valid=jnp.zeros(3, dtype=bool)))
    metrics = metrics_module.pe_episode_metrics(traj, cfg)
    assert int(metrics.outcome) == Outcome.INVALID_IC
    assert not bool(metrics.ic_valid)


@pytest.mark.parametrize("where", ["position", "velocity", "applied_dv"])
def test_nonfinite_live_trace_cannot_be_counted_as_evader_survival(pe_trace, where):
    traj, cfg = _geometry(pe_trace, [40.0, 35.0, 30.0, 25.0])
    if where == "applied_dv":
        traj = traj.replace(
            applied_dv=traj.applied_dv.replace(guard=traj.applied_dv.guard.at[1, 0, 0].set(jnp.nan))
        )
    else:
        component = 0 if where == "position" else 3
        traj = traj.replace(
            final_state=traj.final_state.replace(
                bandits=traj.final_state.bandits.replace(
                    rtn=traj.final_state.bandits.rtn.at[0, component].set(jnp.nan)
                )
            )
        )
    assert int(metrics_module.pe_episode_metrics(traj, cfg).outcome) == Outcome.INVALID_TRAJECTORY


def test_pe_metrics_require_final_state_and_reject_multiagent_config(pe_trace):
    traj, cfg = pe_trace
    with pytest.raises(ValueError, match="final_state"):
        metrics_module.pe_episode_metrics(traj.replace(final_state=None), cfg)
    with pytest.raises(ValueError, match="1v1"):
        metrics_module.pe_episode_metrics(traj, dataclasses.replace(cfg, n_guards=2))


@dataclass(frozen=True)
class Burn:
    command_cls: Any

    def __call__(self, policy_state, agent_view, key, t):
        command = self.command_cls.zeros(1)
        return command.replace(dv=command.dv.at[0, 0].set(100.0)), policy_state


@pytest.mark.parametrize("with_mass", [False, True])
def test_pe_resource_totals_count_actual_clipped_burns_including_final_step(with_mass):
    components = (
        (StateComponentKey.RTN, StateComponentKey.MASS) if with_mass else (StateComponentKey.RTN,)
    )
    cfg = make_pursuit_evasion(max_horizon_s=30.0, guard_components=components)
    if with_mass:
        cfg = dataclasses.replace(
            cfg,
            ic_sampler=dataclasses.replace(
                cfg.ic_sampler,
                guard_sampler=dataclasses.replace(
                    cfg.ic_sampler.guard_sampler, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
                ),
            ),
        )
    env = OrbitalGymEnv(cfg)
    policies = _policies(env).replace(guard=Burn(env.guard_command_cls))
    traj = rollout(env, policies, _init(), jax.random.PRNGKey(0), n_steps=3)
    metrics = metrics_module.pe_episode_metrics(traj, cfg)
    applied = float(jnp.sum(jnp.linalg.norm(traj.applied_dv.guard, axis=-1)))
    assert 0.0 < applied < 2.0
    assert int(metrics.steps) == 3
    assert float(metrics.dv_guard) == pytest.approx(applied, rel=1e-5)
    assert float(metrics.dv_bandit) == pytest.approx(0.0)


def test_batched_evaluator_dispatches_pe_and_keeps_role_and_invalid_outcomes():
    cfg = make_pursuit_evasion(max_horizon_s=30.0)
    env = OrbitalGymEnv(cfg)
    states = sample_bank(env, n_episodes=3, seed=9)
    states = states.replace(
        bandits=states.bandits.replace(rtn=states.bandits.rtn.at[0].set(states.guards.rtn[0])),
        ic_valid=states.ic_valid.at[2].set(False),
    )
    d = env.layout.dynamics_state_dim
    initializer = KFFromTruthInitializer(layout=env.layout, variance_diag=jnp.ones(d))
    updater = KFBeliefUpdater(
        stm=jnp.eye(d), control_matrix=jnp.zeros((d, 3)), process_noise=jnp.eye(d) * 0.01
    )
    metrics = evaluate_bank(
        env,
        cfg,
        states,
        jax.random.PRNGKey(0),
        n_steps=3,
        policies=_policies(env),
        init_policy_state_fns=_init(),
        belief_initializers=BySide(guard=initializer, bandit=initializer),
        belief_updaters=BySide(guard=updater, bandit=updater),
    )
    np.testing.assert_array_equal(
        metrics.outcome, [Outcome.PE_CAPTURE, Outcome.TIMEOUT, Outcome.INVALID_IC]
    )
    assert np.isnan(metrics.min_d_bl).all()
    assert np.isnan(metrics.belief_err_guard_at_commit).all()
    assert np.isfinite(metrics.belief_err_guard[:2]).all()
    records = metrics_to_records(metrics, game="pe")
    assert records[0]["outcome"] == int(Outcome.PE_CAPTURE)
    assert records[0]["game"] == "pe"


def test_pe_metrics_is_available_from_public_eval_namespace():
    from orbitalgym import eval as public_eval

    assert public_eval.pe_episode_metrics is metrics_module.pe_episode_metrics
