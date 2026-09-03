"""belief_rollout accepts link predicates and records contact masks."""

import jax
import jax.numpy as jnp

from orbitalgym.belief.sync import KFTeamFusion
from orbitalgym.env.types import BySide
from orbitalgym.links import AlwaysLinked
from orbitalgym.rollout import belief_rollout, rollout


def test_belief_rollout_records_contact_from_link(make_minimal_lbg_env_with_kf):
    env, belief_init, belief_upd, init_ps_fns, policies = make_minimal_lbg_env_with_kf(n_guards=2)
    traj, beliefs = belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
        key=jax.random.key(0),
        n_steps=8,
        guard_link=AlwaysLinked(),
        team_sync_fns=BySide(guard=KFTeamFusion(), bandit=None),
    )
    assert traj.contact.guard.shape == (8, 2)
    assert bool(jnp.all(traj.contact.guard))
    assert traj.contact.bandit.shape == (8, 1)
    assert not bool(jnp.any(traj.contact.bandit))
    cov = beliefs.guard.cov[5]
    assert jnp.allclose(cov[0, 2], cov[1, 2], atol=1e-6)


def test_plain_rollout_has_no_contact(make_minimal_lbg_env_with_kf):
    env, _, _, init_ps_fns, policies = make_minimal_lbg_env_with_kf()
    traj = rollout(env, policies, init_ps_fns, jax.random.PRNGKey(0), n_steps=3)
    assert traj.contact is None
