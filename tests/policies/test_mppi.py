"""MPPIPolicy: sampling-based receding-horizon planner over continuous delta-v."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.env.types import Actions, BySide
from orbitalgym.eval.conformance import check_policy_conforms
from orbitalgym.policies.leaf_values import bandit_leaf_value
from orbitalgym.policies.mppi import MPPIPolicy
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec


def _build(n_samples=256, horizon=8, temperature=0.05, noise_sigma=0.3, dv_max=0.5):
    ic = ICSpec(
        guard_sampler=RelativeEllipse(
            radial_ellipse_m=300.0,
            phase_rad=0.0,
            sigma_radial_ellipse_m=0.0,
            mass_sampler=ConstantMass(propellant_mass_kg=10.0),
        ),
        bandit_sampler=RelativeEllipse(
            radial_ellipse_m=2000.0, phase_rad=jnp.pi, sigma_radial_ellipse_m=0.0
        ),
    )
    cfg = make_lady_bandit_guard(ic_sampler=ic, max_horizon_s=600.0)
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env)
    guard_zero = ZeroControl(n_vehicles=1, command_cls=env.guard_command_cls)
    policy = MPPIPolicy(
        env_model=adapter,
        side=Side.BANDIT,
        opponent_model=guard_zero,
        n_samples=n_samples,
        horizon=horizon,
        temperature=temperature,
        noise_sigma=noise_sigma,
        dv_max=dv_max,
        terminal_value_fn=bandit_leaf_value(
            adapter,
            v_close_guard_mps=dv_max / cfg.dt,
            v_close_bandit_mps=dv_max / cfg.dt,
        ),
        n_vehicles=1,
        command_cls=env.bandit_command_cls,
    )
    return env, adapter, policy, guard_zero


def _lady_distance(state):
    return float(jnp.linalg.norm(state.bandits.rtn[0, :3]))


def test_mppi_bandit_closes_on_lady_versus_free_drift():
    env, adapter, policy, guard_zero = _build()
    state, _ = env.reset(jax.random.PRNGKey(0))
    drift = state
    d0 = _lady_distance(state)
    ps = policy.init_state()
    key = jax.random.PRNGKey(1)
    for _i in range(20):
        key, k_act, k_env, k_drift = jax.random.split(key, 4)
        cmd, ps = policy(ps, adapter.pack(state), k_act, state.t)
        actions = Actions(sides=BySide(guard=env.guard_command_cls.zeros(1), bandit=cmd))
        state = env.step(k_env, state, actions).state
        zero = Actions(
            sides=BySide(
                guard=env.guard_command_cls.zeros(1), bandit=env.bandit_command_cls.zeros(1)
            )
        )
        drift = env.step(k_drift, drift, zero).state
    assert _lady_distance(state) < _lady_distance(drift) - 50.0
    assert _lady_distance(state) < d0


def test_mppi_respects_dv_cap_and_is_deterministic():
    env, adapter, policy, _ = _build(dv_max=0.2, noise_sigma=1.0)
    state, _ = env.reset(jax.random.PRNGKey(0))
    s = adapter.pack(state)
    cmd_a, ps_a = policy(None, s, jax.random.PRNGKey(3), state.t)
    cmd_b, ps_b = policy(None, s, jax.random.PRNGKey(3), state.t)
    assert jnp.array_equal(cmd_a.dv, cmd_b.dv)
    assert float(jnp.linalg.norm(cmd_a.dv[0])) <= 0.2 + 1e-6
    assert bool(jnp.all(jnp.linalg.norm(ps_a, axis=-1) <= 0.2 + 1e-6))


def test_mppi_warm_start_shifts_mean_sequence():
    env, adapter, policy, _ = _build(horizon=5)
    state, _ = env.reset(jax.random.PRNGKey(0))
    s = adapter.pack(state)
    cmd, ps = policy(policy.init_state(), s, jax.random.PRNGKey(4), state.t)
    assert ps.shape == (5, 1, 3)
    assert jnp.allclose(ps[-1], 0.0)


def test_mppi_traces_under_jit_and_vmap():
    env, adapter, policy, _ = _build(n_samples=32, horizon=3)
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    s = jax.vmap(adapter.pack)(states)
    keys = jax.random.split(jax.random.PRNGKey(5), 2)
    f = jax.jit(jax.vmap(lambda si, k: policy(None, si, k, jnp.asarray(0.0))[0].dv))
    assert f(s, keys).shape == (2, 1, 3)


def test_mppi_conforms_with_belief_views():
    env, adapter, policy, _ = _build(n_samples=16, horizon=3)
    template, _ = env.reset(jax.random.PRNGKey(0))
    belief_policy = policy.__class__(**{**policy.__dict__, "template_env_state": template})
    init = ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=8)
    report = check_policy_conforms(belief_policy, env, Side.BANDIT, init)
    assert report.command_ok and report.traceable and report.rollout_ok, report.message
