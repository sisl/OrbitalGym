"""Regression test for asymmetric attitude configuration.

Only the guard has ATTITUDE+BODY_RATES state components; the bandit has only RTN.
env.step must not raise AttributeError when attitude_dynamics is configured but
the bandit side lacks the attitude fields.
"""

import jax
import jax.numpy as jnp

from orbitalgym.config import ScenarioConfig
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide
from orbitalgym.registry import (
    AttitudeDynamicsKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from tests.helpers.minimal_scenario import minimal_scenario_kwargs


def _one_sided_attitude_kwargs():
    """Guard has RTN+ATTITUDE+BODY_RATES; bandit has only RTN."""
    return minimal_scenario_kwargs(
        truth_dynamics=DynamicsKey.HCW_RTN,
        action_frame=Frame.RTN,
        guard_components=(
            StateComponentKey.RTN,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        bandit_components=(StateComponentKey.RTN,),
        guard_action_components=(),
        bandit_action_components=(),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=AttitudeParams(
            inertia_diag=jnp.ones(3),
            omega_max=jnp.ones(3) * 10.0,
        ),
        # bandit_attitude_params not needed — bandit has no attitude
    )


def _identity_actions(env, cfg):
    return Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )


def test_one_sided_attitude_no_crash():
    """env.step must not raise when only the guard has attitude state."""
    cfg = ScenarioConfig(**_one_sided_attitude_kwargs())
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Inject a nonzero omega so the guard's quaternion will actually advance.
    state = state.replace(
        guards=state.guards.replace(omega=jnp.array([[0.0, 0.0, 1.0]] * cfg.n_guards))
    )

    # Must not raise AttributeError on missing .quat/.omega/.applied_torque for bandit.
    out = env.step(jax.random.PRNGKey(1), state, _identity_actions(env, cfg))

    # Guard's quaternion advanced under nonzero omega.
    assert out.state.guards.quat[0, 0] < 1.0, (
        "guard quaternion should have rotated away from identity"
    )

    # Bandit's RTN propagated normally (free drift — zero dv, so position is unchanged
    # to first order when initial velocity is zero from the sampler default).
    assert out.state.bandits.rtn.shape == state.bandits.rtn.shape, (
        "bandit RTN shape must be preserved"
    )

    # Bandit has no attitude fields (regression: no AttributeError was raised).
    assert not hasattr(out.state.bandits, "quat"), (
        "bandit should not have a quat field in this asymmetric configuration"
    )
