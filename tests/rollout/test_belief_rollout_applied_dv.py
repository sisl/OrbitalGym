"""Belief prediction must use the Δv the env applied, not the commanded Δv.

The guard commands 5 m/s every step while its thruster can deliver only
``max_thrust_n * dt / wet_mass`` ≈ 0.45 m/s. A belief that predicts under
the command accumulates the ~4.5 m/s per-step difference as velocity error
and walks its own-state estimate tens of kilometres off truth over a few
minutes. Predicting under the applied Δv keeps the estimate on truth.

Catch and breach radii are zeroed so the episode runs the full 50 steps and
the drift is measured over the whole horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.belief.pf import ParticleFilterBeliefUpdater, ParticleFilterRingInitializer
from orbitalgym.dynamics.hcw import hcw_rtn_step
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.reference_orbit import mean_motion
from orbitalgym.rollout import belief_rollout
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec

N_STEPS = 50
DT = 10.0
COMMANDED_DV_MPS = 5.0
THRUST_CAP_MPS = 0.46  # 5 N * 10 s / 110 kg


@dataclass(frozen=True)
class _ConstantRadialBurn:
    """Commands a fixed radial Δv every step, far above the thrust cap."""

    command_cls: Any = None
    n_vehicles: int = 0
    dv_mps: float = COMMANDED_DV_MPS

    def __call__(self, policy_state, agent_view, key, t):
        del agent_view, key, t
        zero = self.command_cls.zeros(self.n_vehicles)
        dv = jnp.zeros_like(zero.dv).at[:, 0].set(self.dv_mps)
        return zero.replace(dv=dv), policy_state


class _Fixture:
    def __init__(self):
        ic = ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=0.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=jnp.pi,
                sigma_radial_ellipse_m=0.0,
            ),
        )
        self.cfg = make_lady_bandit_guard(
            n_guards=1,
            n_bandits=1,
            dt=DT,
            max_horizon_s=N_STEPS * DT,
            ic_sampler=ic,
            catch_radius_m=0.0,
            breach_radius_m=0.0,
        )
        self.env = OrbitalGymEnv(self.cfg)
        n = float(mean_motion(self.cfg.reference_orbit))

        def per_vehicle_dyn(x, u, dt):
            return hcw_rtn_step(x[None, :], u[None, :], _HcwParams(n), dt)[0]

        self.updater = ParticleFilterBeliefUpdater(
            dynamics_fn=per_vehicle_dyn,
            process_noise=jnp.eye(6) * 1e-8,
            dt=DT,
            n_eff_threshold=0.5,
        )
        self.initializer = ParticleFilterRingInitializer(
            layout=self.env.layout,
            ring_radius_m=1000.0,
            mean_motion_rad_s=n,
            n_particles=32,
        )

    def run(self):
        policies = BySide(
            guard=_ConstantRadialBurn(command_cls=self.env.guard_command_cls, n_vehicles=1),
            bandit=ZeroControl(command_cls=self.env.bandit_command_cls, n_vehicles=1),
        )
        init_ps = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
        return belief_rollout(
            self.env,
            policies,
            init_ps,
            BySide(guard=self.initializer, bandit=self.initializer),
            BySide(guard=self.updater, bandit=self.updater),
            key=jax.random.PRNGKey(0),
            n_steps=N_STEPS,
        )


@dataclass(frozen=True)
class _HcwParams:
    mean_motion: float


def _own_state_position_error(traj, history) -> jax.Array:
    """Per-step distance between the guard's own-state belief mean and truth.

    The logged belief at index k is the one entering step k, the same tick
    the logged ``env_state`` is entering, so the two align index for index.
    """
    belief_pos = history.guard.mean[:, 0, 0, :3]  # (T, 3)
    truth_pos = traj.env_state.guards.rtn[:, 0, :3]  # (T, 3)
    return jnp.linalg.norm(belief_pos - truth_pos, axis=-1)


def test_commanded_dv_is_clipped_by_thrust():
    """The command is well above what the thruster can deliver."""
    fixture = _Fixture()
    traj, _ = fixture.run()
    applied = jnp.linalg.norm(traj.applied_dv.guard, axis=-1)  # (T, n)
    assert float(jnp.max(applied)) < THRUST_CAP_MPS
    assert float(jnp.min(applied)) > 0.4
    assert 10.0 * float(jnp.max(applied)) < COMMANDED_DV_MPS


def test_own_state_belief_tracks_truth_under_a_clipped_command():
    """Belief predicted under the applied Δv stays on the guard's truth."""
    fixture = _Fixture()
    traj, history = fixture.run()
    error = float(jnp.max(_own_state_position_error(traj, history)))
    assert error < 5.0, f"own-state belief drifted {error:.1f} m from truth"


def test_applied_dv_is_reported_per_side():
    fixture = _Fixture()
    traj, _ = fixture.run()
    assert traj.applied_dv.guard.shape == (N_STEPS, 1, 3)
    assert traj.applied_dv.bandit.shape == (N_STEPS, 1, 3)
    assert jnp.allclose(traj.applied_dv.bandit, 0.0)
