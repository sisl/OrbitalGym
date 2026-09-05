"""LBG with comms: guard pays a per-broadcast cost; bandit sees the leak."""

import inspect

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide
from orbitalgym.games.lady_bandit_guard import _COMMS_UNSUPPORTED_DEFAULTS, make_lady_bandit_guard
from orbitalgym.registry import ActionComponentKey


def test_lbg_comms_builder_wires_communicate():
    cfg = make_lady_bandit_guard(with_communication=True, comm_cost=5.0)
    assert ActionComponentKey.COMMUNICATE in cfg.guard_action_components
    assert ActionComponentKey.COMMUNICATE not in cfg.bandit_action_components


def test_lbg_comms_charges_cost_when_active():
    cfg = make_lady_bandit_guard(with_communication=True, comm_cost=5.0)
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    silent = env.guard_command_cls.zeros(cfg.n_guards)
    bandit_cmd = env.bandit_command_cls.zeros(cfg.n_bandits)
    out_silent = env.step(
        jax.random.PRNGKey(1), state, Actions(sides=BySide(guard=silent, bandit=bandit_cmd))
    )

    talking = silent.replace(active=jnp.array([True]))
    out_talking = env.step(
        jax.random.PRNGKey(1), state, Actions(sides=BySide(guard=talking, bandit=bandit_cmd))
    )

    diff = out_silent.outputs.guard.reward - out_talking.outputs.guard.reward
    assert jnp.isclose(diff, 5.0)


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("breach_radius_m", 10.0),
        ("catch_radius_m", 25.0),
        ("breach_speed_mps", 1.0),
        ("catch_speed_mps", 1.0),
        ("escape_radius_m", 5000.0),
        ("repel_on_empty_tank", True),
        ("dv_cost", 0.1),
    ],
)
def test_lbg_comms_rejects_event_geometry_knobs(knob, value):
    with pytest.raises(ValueError, match=knob):
        make_lady_bandit_guard(with_communication=True, **{knob: value})


def test_lbg_comms_rejected_knobs_match_the_builder_defaults():
    signature = inspect.signature(make_lady_bandit_guard)
    for knob, default in _COMMS_UNSUPPORTED_DEFAULTS.items():
        assert signature.parameters[knob].default == default
