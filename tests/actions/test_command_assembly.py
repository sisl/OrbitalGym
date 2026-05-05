import jax
import jax.numpy as jnp

from orbital_game.actions.assemble import build_command_class
from orbital_game.actions.components import ImpulsiveManeuver
from orbital_game.dynamics.hcw import hcw_rtn_step
from orbital_game.registry import Frame


def _maneuver():
    return ImpulsiveManeuver(
        truth_dynamics=hcw_rtn_step,
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=True,
    )


def test_single_component_command_has_dv_field():
    Cmd = build_command_class((ImpulsiveManeuver,), n_agents=2, class_name="GuardCommand")  # noqa: N806
    c = Cmd.zeros(2)
    assert c.dv.shape == (2, 3)


def test_command_pytree_roundtrip():
    Cmd = build_command_class((ImpulsiveManeuver,), n_agents=3, class_name="C")  # noqa: N806
    c = Cmd.zeros(3).replace(dv=jnp.ones((3, 3)))
    flat, treedef = jax.tree_util.tree_flatten(c)
    restored = jax.tree_util.tree_unflatten(treedef, flat)
    assert jnp.allclose(restored.dv, c.dv)


def test_command_class_caching():
    """Repeated calls with same args return the same class (jit-friendly)."""
    A = build_command_class((ImpulsiveManeuver,), n_agents=2, class_name="C")  # noqa: N806
    B = build_command_class((ImpulsiveManeuver,), n_agents=2, class_name="C")  # noqa: N806
    assert A is B
