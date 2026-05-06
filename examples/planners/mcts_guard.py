"""MCTS-flavored root planner for the LBG guard.

Root-only MCTS / multi-armed-bandit-with-UCB planner:

1. Discretize the guard's per-step Δv into A candidate actions (8 directions
   on the RT plane at max-thrust magnitude + 1 no-op). The MPS comparison
   benefits from this because each candidate's evaluation is a vmap'd
   JAX rollout.
2. Allocate `n_iterations` simulation budget across the A candidates using
   UCB1 selection: argmax_a [ Q(a) + c * sqrt(log N / N(a)) ].
3. For each selected candidate, simulate forward `horizon` steps using
   simple analytical opponents: the bandit thrusts toward the lady (proxy
   for the real MPC), and the guard plays lead-intercept toward the nearest
   bandit (proxy for "the planner keeps doing something reasonable").
4. Score each rollout by cumulative guard reward (LbgZeroSumReward).
5. Return the action with highest mean Q.

This is *not* a full tree search (no branching past the root) but exercises
the package's pure-JAX primitives that benefit from MPS:
- POMDPAdapter.transition / .reward (jit-compiled)
- vmap over candidates → parallel-rollout speedup

The planner is *not* a Policy — it's called from a manual outer loop
because the UCB iteration over Python state can't go inside lax.scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from orbital_game.adapters._command_flatten import flatten_command
from orbital_game.adapters.pomdp import POMDPAdapter
from orbital_game.env.types import Side


def _build_action_grid(dv_max: float, n_directions: int = 8) -> np.ndarray:
    """(A, 2) grid: n_directions evenly-spaced unit vectors at dv_max + no-op."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)
    vecs = dv_max * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    no_op = np.zeros((1, 2))
    return np.concatenate([vecs, no_op], axis=0)


def _lead_intercept_dv_one(own_pos: jax.Array, target_pos: jax.Array, dv_max: float) -> jax.Array:
    direction = target_pos - own_pos
    norm = jnp.linalg.norm(direction)
    safe_dir = direction / jnp.maximum(norm, 1e-9)
    return safe_dir * dv_max


def _bandit_action_lead_to_origin(bandit_states: jax.Array, dv_max: float) -> jax.Array:
    bandit_positions = bandit_states[:, :2]
    origin = jnp.zeros(2)
    return jax.vmap(lambda p: _lead_intercept_dv_one(p, origin, dv_max))(bandit_positions)


def _guard_action_lead_nearest_bandit(
    guard_states: jax.Array, bandit_states: jax.Array, dv_max: float
) -> jax.Array:
    g_pos = guard_states[:, :2]
    b_pos = bandit_states[:, :2]
    diffs = g_pos[:, None, :] - b_pos[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    nearest = jnp.argmin(dists, axis=-1)
    target_pos = b_pos[nearest]
    return jax.vmap(lambda gp, tp: _lead_intercept_dv_one(gp, tp, dv_max))(g_pos, target_pos)


def _shape_to_action_dim(dv_2d: jax.Array, target_dim: int) -> jax.Array:
    """Adapt a 2D RT-plane Δv to the configured action dim (2 or 3)."""
    if target_dim == 2:
        return dv_2d
    n = dv_2d.shape[0]
    return jnp.concatenate([dv_2d, jnp.zeros((n, target_dim - 2), dv_2d.dtype)], axis=-1)


@dataclass(eq=False)
class MctsGuardPlanner:
    """Root-MCTS planner: UCB over a discrete Δv grid, vmap'd JAX rollouts."""

    adapter: POMDPAdapter
    guard_dv_max: float
    bandit_dv_max: float
    n_directions: int = 8
    horizon: int = 10
    n_iterations: int = 64
    ucb_c: float = 5.0

    # Populated in __post_init__.
    _action_grid: Any = None
    _n_actions: int = 0
    _rollout_one_jit: Any = None
    _rollout_batched_jit: Any = None

    def __post_init__(self):
        grid_np = _build_action_grid(self.guard_dv_max, self.n_directions)
        self._action_grid = jnp.asarray(grid_np)
        self._n_actions = int(grid_np.shape[0])

        adapter = self.adapter
        cfg = adapter.env.config
        n_g = cfg.n_guards
        n_b = cfg.n_bandits
        bandit_dv_max = self.bandit_dv_max
        horizon = int(self.horizon)
        layout = adapter.layout

        guard_command_cls = adapter.env.guard_command_cls
        bandit_command_cls = adapter.env.bandit_command_cls
        guard_dv_dim = int(guard_command_cls.zeros(n_g).dv.shape[-1])
        bandit_dv_dim = int(bandit_command_cls.zeros(n_b).dv.shape[-1])

        def _flatten_pair(guard_dv, bandit_dv):
            g_cmd = guard_command_cls.zeros(n_g).replace(dv=guard_dv)
            b_cmd = bandit_command_cls.zeros(n_b).replace(dv=bandit_dv)
            return jnp.concatenate([flatten_command(g_cmd), flatten_command(b_cmd)])

        def _unpack_states(s_flat):
            guards, bandits = layout.unflatten(s_flat[:-2])
            return guards.rt, bandits.rt

        def _rollout_one(s_flat, candidate_dv_2d, key):
            """One rollout: candidate at root, lead-intercept thereafter.
            Returns cumulative guard reward (scalar)."""
            # Root step: guard uses candidate, bandit uses lead-to-origin.
            guards0_arr, bandits0_arr = _unpack_states(s_flat)
            guard_root_dv_2d = jnp.broadcast_to(candidate_dv_2d[None, :], (n_g, 2))
            bandit0_dv_2d = _bandit_action_lead_to_origin(bandits0_arr, bandit_dv_max)
            a0_flat = _flatten_pair(
                _shape_to_action_dim(guard_root_dv_2d, guard_dv_dim),
                _shape_to_action_dim(bandit0_dv_2d, bandit_dv_dim),
            )

            k0, k_rest = jax.random.split(key, 2)
            s_after = adapter.transition(s_flat, a0_flat, k0)
            r0 = adapter.reward(s_flat, a0_flat, s_after, Side.GUARD)

            def _step_lead(carry, step_key):
                s_flat_cur, total = carry
                g_arr, b_arr = _unpack_states(s_flat_cur)
                g_dv_2d = _guard_action_lead_nearest_bandit(g_arr, b_arr, bandit_dv_max)
                b_dv_2d = _bandit_action_lead_to_origin(b_arr, bandit_dv_max)
                a_flat = _flatten_pair(
                    _shape_to_action_dim(g_dv_2d, guard_dv_dim),
                    _shape_to_action_dim(b_dv_2d, bandit_dv_dim),
                )
                s_next = adapter.transition(s_flat_cur, a_flat, step_key)
                r = adapter.reward(s_flat_cur, a_flat, s_next, Side.GUARD)
                return (s_next, total + r), None

            inner_keys = jax.random.split(k_rest, max(0, horizon - 1))
            (s_final, total), _ = jax.lax.scan(_step_lead, (s_after, r0), inner_keys)
            return total

        self._rollout_one_jit = jax.jit(_rollout_one)
        self._rollout_batched_jit = jax.jit(jax.vmap(_rollout_one, in_axes=(None, 0, 0)))

    def plan(self, env_state, key: jax.Array) -> tuple[Any, None]:
        """Plan: return (Command, None) for the guard side. Conforms to Planner protocol.

        Internally packs the EnvState into the adapter's flat-state vector,
        runs `n_iterations` rollouts (initial sweep + UCB1-driven extras),
        and returns the highest-mean-Q action wrapped in the guard Command pytree.
        """
        s_flat = self.adapter.pack(env_state)

        # MCTS-style state: action count A; per-action Q-value sums + visit counts N.
        A = self._n_actions  # noqa: N806 — action-count, uppercase by convention
        q = np.zeros(A, dtype=np.float64)
        n = np.zeros(A, dtype=np.int64)

        # Initial sweep: each action gets one rollout.
        keys_init = jax.random.split(key, A)
        rewards_init = np.asarray(self._rollout_batched_jit(s_flat, self._action_grid, keys_init))
        q[:] = rewards_init
        n[:] = 1

        remaining = max(0, int(self.n_iterations) - A)
        c = float(self.ucb_c)
        for it in range(remaining):
            total_n = int(n.sum())
            ucb = q / np.maximum(n, 1) + c * np.sqrt(np.log(total_n + 1) / np.maximum(n, 1))
            a_pick = int(np.argmax(ucb))
            sub_key = jax.random.fold_in(key, it + A + 1)
            r = float(self._rollout_one_jit(s_flat, self._action_grid[a_pick], sub_key))
            q[a_pick] += r
            n[a_pick] += 1

        means = q / np.maximum(n, 1)
        a_best = int(np.argmax(means))
        chosen_dv_2d = np.asarray(self._action_grid[a_best])
        n_g = self.adapter.env.config.n_guards
        env = self.adapter.env
        dv_dim = int(env.guard_command_cls.zeros(n_g).dv.shape[-1])
        out = np.zeros((n_g, dv_dim), dtype=np.float32)
        out[:, :2] = chosen_dv_2d
        guard_cmd = env.guard_command_cls.zeros(n_g).replace(dv=jnp.asarray(out))
        return guard_cmd, None
