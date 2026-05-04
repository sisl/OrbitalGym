"""T3 — Write an active-pursuer policy. Source-of-truth for snippets in
docs/tutorials/t3-active-pursuer.md.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


def test_t3_active_pursuer_walkthrough():
    # --8<-- [start:imports]
    import dataclasses
    from dataclasses import dataclass
    from typing import Any

    import jax
    import jax.numpy as jnp

    from orbital_game import (
        OrbitalGameEnv,
        SingleAgentView,
        make_pursuit_evasion,
    )
    from orbital_game.policies import ZeroControl
    from orbital_game.rollout import rollout_single_agent
    # --8<-- [end:imports]

    # --8<-- [start:lead-intercept-class]
    @dataclass(frozen=True)
    class LeadInterceptPursuer:
        """Bandit policy: thrust toward the guard's predicted next-step position.

        FullObservation flattens the per-side state as [own_truth, opp_truth].
        For the bandit, that's [bandit_rtn(6), guard_rtn(6)] in a 1v1 RTN game.
        """

        max_dv_mps: float = 0.05
        dt: float = 10.0

        # Env-populated dims:
        n_vehicles: int = 0
        action_dim: int = 0

        def __call__(
            self,
            policy_state: Any,
            obs: jax.Array,
            key: jax.Array,
            t: jax.Array,
        ) -> tuple[jax.Array, Any]:
            del key, t
            # [own_truth, opp_truth]: bandit reads its own state first, guard second.
            bandit_rtn = obs[0:6]
            guard_rtn = obs[6:12]

            # One-step zero-control HCW prediction of the guard.
            guard_pos = guard_rtn[0:3]
            guard_vel = guard_rtn[3:6]
            guard_pred = guard_pos + guard_vel * self.dt

            # Max-magnitude impulse along the line to the predicted point.
            direction = guard_pred - bandit_rtn[0:3]
            unit = direction / (jnp.linalg.norm(direction) + 1e-9)
            dv = unit * self.max_dv_mps

            action = jnp.broadcast_to(dv, (self.n_vehicles, self.action_dim))
            return action, policy_state

    # --8<-- [end:lead-intercept-class]

    # --8<-- [start:wire-it-up]
    cfg = make_pursuit_evasion(seed=0, max_horizon_s=2000.0)
    cfg = dataclasses.replace(
        cfg,
        bandit_policy=LeadInterceptPursuer(max_dv_mps=0.05, dt=cfg.dt),
    )
    env = OrbitalGameEnv(cfg)
    view = SingleAgentView(env)
    guard_policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)

    traj = rollout_single_agent(
        view,
        guard_policy,
        lambda c, s, k: None,
        jax.random.PRNGKey(0),
        n_steps=cfg.max_steps,
    )
    # --8<-- [end:wire-it-up]

    # --8<-- [start:closing-distance]
    pos_g = traj.env_state.guards.rtn[:, 0, :3]
    pos_b = traj.env_state.bandits.rtn[:, 0, :3]
    distance = jnp.linalg.norm(pos_g - pos_b, axis=-1)
    initial = float(distance[0])
    closest = float(distance.min())  # min distance over the trajectory
    # --8<-- [end:closing-distance]

    # The active pursuer reaches a smaller closest-approach than the IC distance.
    assert closest < initial, f"closest {closest:.1f} >= initial {initial:.1f}"


def test_t3_ab_comparison_vmap():
    """The vmap-over-seeds A/B section of T3 demonstrates a real difference.

    Lives outside snippet regions because the comparison relies on imports
    the tutorial doesn't repeat.
    """
    from examples.policies.lead_intercept import LeadInterceptPursuer
    from orbital_game import (
        OrbitalGameEnv,
        SingleAgentView,
        make_pursuit_evasion,
    )
    from orbital_game.policies import ZeroControl
    from orbital_game.rollout import rollout_single_agent

    def closest_approach(seed, bandit_policy):
        cfg = make_pursuit_evasion(seed=int(seed), max_horizon_s=2000.0)
        cfg = dataclasses.replace(cfg, bandit_policy=bandit_policy)
        env = OrbitalGameEnv(cfg)
        view = SingleAgentView(env)
        guard = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
        traj = rollout_single_agent(
            view,
            guard,
            lambda c, s, k: None,
            jax.random.PRNGKey(int(seed)),
            n_steps=cfg.max_steps,
        )
        pos_g = traj.env_state.guards.rtn[:, 0, :3]
        pos_b = traj.env_state.bandits.rtn[:, 0, :3]
        return float(jnp.linalg.norm(pos_g - pos_b, axis=-1).min())

    seeds = list(range(10))
    zero_min = jnp.array([closest_approach(s, ZeroControl()) for s in seeds])
    lead_min = jnp.array(
        [closest_approach(s, LeadInterceptPursuer(max_dv_mps=0.05, dt=10.0)) for s in seeds]
    )

    assert lead_min.mean() < zero_min.mean(), (
        f"LeadIntercept did not beat ZeroControl on average across {len(seeds)} seeds: "
        f"lead {lead_min.mean():.1f} vs zero {zero_min.mean():.1f}"
    )
