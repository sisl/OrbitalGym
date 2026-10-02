# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""Listing 1 of the paper: pursuit-evasion with the library's MPPI policy, 1,024 encounters.

uv run papers/2027_ieee_aeroconf/listing_1_pursuit_evasion.py
"""

from types import SimpleNamespace as View

import jax
import jax.numpy as jnp

import orbitalgym as og
from orbitalgym import policies as pol
from orbitalgym.rollout import rollout

keys = jax.random.split(jax.random.key(0), 1024)
cfg = og.make_pursuit_evasion(capture_distance_m=10.0)
env = og.OrbitalGymEnv(cfg)

# Guard: random 0.25 m/s impulses along RTN axes.
axes = jnp.concatenate([jnp.eye(3), -jnp.eye(3)])
guard = pol.UniformRandomDiscretePolicy(0.25 * axes, 1, env.guard_command_cls)

# Bandit: MPPI planning against a coasting guard.
coast = pol.ZeroControl(env.guard_command_cls, 1)
mppi = pol.MPPIPolicy(
    og.POMDPAdapter(env),
    og.Side.BANDIT,
    coast,
    n_samples=64,
    horizon=10,
    dv_max=0.5,
    n_vehicles=1,
    command_cls=env.bandit_command_cls,
    template_env_state=env.reset(keys[0])[0],
)


# rollout gives each policy its flat observation;
# MPPI reads it as an estimate of both vehicles.
def bandit(plan, obs, key, t):
    view = View(mean=obs.reshape(1, 2, 6))
    return mppi(plan, view, key, t)


policies = og.BySide(guard=guard, bandit=bandit)
# Each side's initial policy state: MPPI's nominal
# impulse sequence; None for the stateless guard.
init_state = og.BySide(guard=lambda *_: None, bandit=lambda *_: mppi.init_state())


@jax.jit
@jax.vmap
def episode(key):
    s = rollout(env, policies, init_state, key, cfg.max_steps).final_state
    d = s.guards.rtn[0, :3] - s.bandits.rtn[0, :3]
    return jnp.linalg.norm(d), s.step * cfg.dt


dist, t_end = episode(keys)
caught = dist < cfg.game.capture_distance_m
t_mean = t_end[caught].mean()
print(f"capture rate {caught.mean():.2f}, mean capture time {t_mean:.0f} s")
