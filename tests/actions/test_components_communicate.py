"""Communicate ActionComponent: schema sanity + state-identity apply()."""

import jax
import jax.numpy as jnp

from orbital_game.actions.assemble import build_command_class
from orbital_game.actions.components import Communicate


def test_communicate_zeros_has_active_field():
    cmd = build_command_class((Communicate,), n_agents=2, class_name="C").zeros(2)
    assert cmd.active.shape == (2,)
    assert cmd.active.dtype == jnp.bool_
    assert not cmd.active.any()


def test_communicate_apply_is_state_identity():
    """Communicate is observation-channel only; it does not mutate state."""
    Cls = build_command_class((Communicate,), n_agents=1, class_name="C")  # noqa: N806
    cmd = Cls.zeros(1).replace(active=jnp.array([True]))
    comp = Communicate()
    side_state = jnp.array([1.0, 2.0, 3.0])  # placeholder — apply must not touch
    out = comp.apply(
        cmd, side_state, None, dt=10.0, ref_eci6=jnp.zeros(6), key=jax.random.PRNGKey(0)
    )
    assert out is side_state
