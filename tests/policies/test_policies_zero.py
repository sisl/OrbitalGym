"""ZeroControl emits a Command pytree (not a raw array) populated with each
component's identity values."""

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.policies.zero import ZeroControl


def test_zero_control_emits_command_pytree():
    cfg = make_lady_bandit_guard()
    env = OrbitalGymEnv(cfg)
    policy = ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards)
    cmd, ps = policy(None, jnp.zeros((1, 12)), jax.random.PRNGKey(0), jnp.asarray(0.0))
    assert hasattr(cmd, "dv")
    assert jnp.allclose(cmd.dv, jnp.zeros((cfg.n_guards, 3)))


def test_zero_control_without_env_injection_raises_clear_error():
    """Constructing a policy with default `command_cls=None` is allowed (the env
    later populates it via `dataclasses.replace`), but invoking `__call__` on an
    un-injected policy must raise immediately with a clear message — not the
    opaque `AttributeError: NoneType has no attribute 'zeros'` users would
    otherwise see. The same fail-fast contract applies to all policies in the
    gallery (SunTrackerBlocker, OrthogonalEvader, JitteredPolicy,
    LeadInterceptPursuer, CompositeActionPolicy, HeuristicWithFallbackPolicy,
    UniformRandomDiscretePolicy, MCTSPolicy); ZeroControl exemplifies it.
    """
    policy = ZeroControl()
    with pytest.raises(ValueError, match="ZeroControl"):
        policy(None, jnp.zeros((1, 2)), jax.random.PRNGKey(0), jnp.asarray(0.0))
