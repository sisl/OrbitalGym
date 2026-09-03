"""K-root determinized MCTS over a particle belief."""

import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.belief.contact_aware import ContactAwareBelief
from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.policies.mcts import MCTSPolicy, ParticleRootMCTSPolicy
from orbitalgym.policies.uniform_random import UniformRandomDiscretePolicy


def _grid(dv=0.3):
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    vecs = dv * np.stack([np.cos(angles), np.sin(angles), np.zeros(8)], axis=-1)
    return jnp.asarray(np.concatenate([vecs, np.zeros((1, 3))]), dtype=jnp.float32)


def _build(coordination="joint", n_guards=1):
    cfg = make_lady_bandit_guard(n_guards=n_guards)
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env)
    grid = _grid()
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    teammate = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls
    )
    inner = MCTSPolicy(
        env_model=adapter,
        side=Side.GUARD,
        action_grid=grid,
        opponent_model=opp,
        opponent_action_grid=grid,
        num_simulations=8,
        n_vehicles=cfg.n_guards,
        command_cls=env.guard_command_cls,
        coordination=coordination,
        teammate_model=teammate if coordination == "independent" else None,
    )
    state, _ = env.reset(jax.random.PRNGKey(0))
    belief = ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=16)(
        state, Side.GUARD, jax.random.PRNGKey(1)
    )
    view = ContactAwareBelief(inner=belief, contact=jnp.zeros((cfg.n_guards,), dtype=bool))
    policy = ParticleRootMCTSPolicy(inner_mcts=inner, template_env_state=state, n_roots=4)
    return env, adapter, policy, view, state, cfg


def test_search_joint_out_exposes_action_weights():
    env, adapter, policy, view, state, cfg = _build()
    out = policy.inner_mcts._search_joint_out(adapter.pack(state), jax.random.PRNGKey(0))
    assert out.action_weights.shape == (1, 9)
    assert jnp.allclose(jnp.sum(out.action_weights), 1.0, atol=1e-5)


def test_particle_root_policy_emits_command_of_right_shape():
    env, adapter, policy, view, state, cfg = _build()
    cmd, ps = policy(None, view, jax.random.PRNGKey(2), state.t)
    assert cmd.dv.shape == env.guard_command_cls.zeros(cfg.n_guards).dv.shape
    assert ps is None


def test_particle_root_policy_is_deterministic_in_key_and_jittable():
    env, adapter, policy, view, state, cfg = _build()
    f = jax.jit(lambda v, k: policy(None, v, k, jnp.asarray(0.0))[0].dv)
    a = f(view, jax.random.PRNGKey(5))
    b = f(view, jax.random.PRNGKey(5))
    assert jnp.array_equal(a, b)


def test_particle_root_policy_supports_independent_two_guards():
    env, adapter, policy, view, state, cfg = _build(coordination="independent", n_guards=2)
    cmd, _ = policy(None, view, jax.random.PRNGKey(2), state.t)
    assert cmd.dv.shape == (2, 3)


def test_flat_state_falls_back_to_inner_search():
    env, adapter, policy, view, state, cfg = _build()
    cmd, _ = policy(None, adapter.pack(state), jax.random.PRNGKey(2), state.t)
    assert cmd.dv.shape == (1, 3)
