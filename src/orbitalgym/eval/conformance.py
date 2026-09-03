"""Conformance check for solvers supplied outside the library.

A solver plugs into evaluation when it satisfies the policy contract:
``policy(policy_state, agent_view, key, t) -> (command, policy_state)`` with
``agent_view`` a :class:`~orbitalgym.belief.contact_aware.ContactAwareBelief`
and ``command`` an instance of the side's assembled command class. Traceable
solvers run under ``jax.jit`` and ``jax.vmap``; the report says which checks
passed so a host-callback solver can still be run on the slow path.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.belief.contact_aware import ContactAwareBelief
from orbitalgym.belief.pf import ParticleFilterBeliefUpdater
from orbitalgym.dynamics.hcw import hcw_rt_step, hcw_rtn_step
from orbitalgym.env.types import BySide, Side
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.rollout import belief_rollout


@dataclasses.dataclass(frozen=True)
class ConformanceReport:
    command_ok: bool
    traceable: bool
    rollout_ok: bool
    message: str


def _expected_command(env: Any, side: Side) -> Any:
    n = env.config.n_guards if side is Side.GUARD else env.config.n_bandits
    cls = env.guard_command_cls if side is Side.GUARD else env.bandit_command_cls
    return cls.zeros(n)


def _shapes_match(command: Any, template: Any) -> tuple[bool, str]:
    if type(command) is not type(template):
        return False, f"command type {type(command).__name__} != {type(template).__name__}"
    for name in template.__dataclass_fields__:
        got = getattr(command, name).shape
        want = getattr(template, name).shape
        if got != want:
            return (
                False,
                f"field {name!r}: shape {got} != {want} (leading axis is the vehicle count)",
            )
    return True, "ok"


def check_policy_conforms(
    policy: Any,
    env: Any,
    side: Side,
    belief_initializer: Any,
    *,
    n_steps: int = 3,
) -> ConformanceReport:
    """Exercise ``policy`` on ``side`` with a particle belief and report what passed."""
    state, _ = env.reset(jax.random.PRNGKey(0))
    belief = belief_initializer(state, side, jax.random.PRNGKey(1))
    n_side = env.config.n_guards if side is Side.GUARD else env.config.n_bandits
    view = ContactAwareBelief(inner=belief, contact=jnp.zeros((n_side,), dtype=bool))
    template = _expected_command(env, side)

    try:
        command, _ = policy(None, view, jax.random.PRNGKey(2), state.t)
    except Exception as e:  # noqa: BLE001 - the report carries the failure
        return ConformanceReport(False, False, False, f"call raised: {e!r}")
    command_ok, message = _shapes_match(command, template)
    if not command_ok:
        return ConformanceReport(False, False, False, message)

    try:
        f = jax.jit(jax.vmap(lambda v, k: policy(None, v, k, jnp.asarray(0.0))[0]))
        batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), view)
        f(batched, jax.random.split(jax.random.PRNGKey(3), 2))
        traceable = True
    except Exception as e:  # noqa: BLE001
        traceable = False
        message = f"not traceable under jit+vmap: {e!r}"

    d = env.layout.dynamics_state_dim
    step = hcw_rtn_step if d == 6 else hcw_rt_step
    mean_motion = env.mean_motion

    def dyn(x, u, dt):
        return step(x[None, :], u[None, :], _MeanMotion(mean_motion), dt)[0]

    updater = ParticleFilterBeliefUpdater(
        dynamics_fn=dyn, process_noise=jnp.eye(d) * 1e-4, dt=env.config.dt
    )
    other = dataclasses.replace(
        ZeroControl(),
        command_cls=env.bandit_command_cls if side is Side.GUARD else env.guard_command_cls,
        n_vehicles=(env.config.n_bandits if side is Side.GUARD else env.config.n_guards),
    )
    policies = (
        BySide(guard=policy, bandit=other)
        if side is Side.GUARD
        else BySide(guard=other, bandit=policy)
    )
    none_init = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
    try:
        belief_rollout(
            env,
            policies,
            none_init,
            BySide(guard=belief_initializer, bandit=belief_initializer),
            BySide(guard=updater, bandit=updater),
            jax.random.PRNGKey(4),
            n_steps,
        )
        rollout_ok = True
    except Exception as e:  # noqa: BLE001
        rollout_ok = False
        message = f"belief_rollout failed: {e!r}"
    return ConformanceReport(command_ok, traceable, rollout_ok, message)


@dataclasses.dataclass(frozen=True)
class _MeanMotion:
    mean_motion: float
