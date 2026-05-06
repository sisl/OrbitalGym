"""MCTSPolicy — JAX-native classic UCT search via mctx.

Conforms to the unified :class:`orbital_game.policies.base.Policy`
protocol. Slot it on either side; both sides can run an MCTSPolicy
independently. The whole search compiles into a single
``jax.lax.fori_loop``-based JIT region (mctx is fully traced), so MPS / GPU
backends pay for themselves on the search budget — no host↔device
round-trips per simulation.

Design notes:

- **Classic UCT.** Uniform priors, leaf value defaults to zero. Pass a
  custom ``leaf_value_fn`` for cheap heuristic value backups.
- **Opponent model.** Inside the tree, the opposing side's actions come
  from ``opponent_model`` — a Policy used as the searcher's *belief about
  the opponent*. This is **distinct** from the opponent's actual policy in
  the real game; the real game's other side is whatever Policy is wired
  into the env. The default
  :class:`orbital_game.policies.uniform_random.UniformRandomDiscretePolicy`
  is deliberately cheap and uncoupled from any scenario; for sharper
  search, wire e.g. an LQR or LeadIntercept policy as ``opponent_model``.
- **agent_view contract (v1).** Either a 1-D flat state vector matching
  ``env_model.states_dim``, or any object exposing ``mean: jax.Array`` of
  the same shape (a :class:`orbital_game.belief.base.Belief`). KF/EKF
  beliefs whose ``mean`` is per-pair (not flat-state) require a small
  Belief adapter — see ``docs/in-depth/mcts.md``.
- **Variants.** ``gumbel_muzero_policy`` (default) uses sequential halving
  with Gumbel noise — better small-budget behaviour than vanilla PUCT
  with a uniform prior. ``muzero_policy`` is also exposed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import mctx

from orbital_game.adapters._command_flatten import flatten_command
from orbital_game.adapters.pomdp.adapter import POMDPAdapter
from orbital_game.env.types import Actions, BySide, Side
from orbital_game.observations.types import flatten_observations


def _zero_value(s_flat: jax.Array) -> jax.Array:
    return jnp.zeros((), dtype=s_flat.dtype)


@dataclass(frozen=True)
class MCTSPolicy:
    """JAX-native classic UCT search backed by mctx."""

    env_model: POMDPAdapter
    side: Side

    action_grid: jax.Array  # (A, dv_dim)
    opponent_model: Any  # a Policy
    opponent_action_grid: jax.Array  # (A_opp, dv_dim) — used by default opponent_model

    num_simulations: int = 32
    max_depth: int | None = None
    leaf_value_fn: Callable[[jax.Array], jax.Array] | None = None
    discount: float = 1.0
    variant: Literal["muzero", "gumbel_muzero"] = "gumbel_muzero"

    n_vehicles: int = 0
    command_cls: Any = None

    def __post_init__(self) -> None:
        if self.command_cls is None:
            raise ValueError(
                "MCTSPolicy was constructed without `command_cls`. Build via "
                "OrbitalGameEnv (which injects `command_cls`/`n_vehicles` via "
                "dataclasses.replace), or pass them explicitly."
            )
        if self.n_vehicles == 0:
            raise ValueError(
                "MCTSPolicy requires `n_vehicles > 0`. The env normally injects "
                "this; pass it explicitly when constructing manually."
            )

    def __call__(
        self,
        policy_state: Any,
        agent_view: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        del t  # search uses the env-state's own `t` from inside `s_flat`
        # agent_view dispatch:
        # - jax.Array → flat state vector directly (obs-only `rollout`, or
        #   manual call with `adapter.pack(state)`).
        # - anything else → treat as a Belief and read its `mean` attribute
        #   (`belief_rollout` path). NB: jax arrays also expose a `.mean`
        #   method so a hasattr check would alias the two; isinstance is the
        #   robust dispatch.
        s_flat = agent_view if isinstance(agent_view, jax.Array) else agent_view.mean
        action_idx = self._search(s_flat, key)
        cmd = self._action_idx_to_command(action_idx)
        return cmd, policy_state

    # --- internals ---

    def _search(self, s_flat: jax.Array, key: jax.Array) -> jax.Array:
        adapter = self.env_model
        env = adapter.env
        side = self.side
        opp_side = side.opposite()
        opp_obs_fn = (
            env.guard_observation_fn if opp_side is Side.GUARD else env.bandit_observation_fn
        )
        opp_command_cls = (
            env.guard_command_cls if opp_side is Side.GUARD else env.bandit_command_cls
        )
        n_opp = env.config.n_guards if opp_side is Side.GUARD else env.config.n_bandits
        n_self = self.n_vehicles
        action_grid = self.action_grid
        opponent_model = self.opponent_model
        leaf_value = self.leaf_value_fn or _zero_value

        a_count = int(action_grid.shape[0])
        dv_dim_grid = int(action_grid.shape[1])

        # Identity Actions for obs-fn calls — obs_fn requires the Actions
        # pytree but post-step actions don't carry information at root-time
        # observation. Static at trace time.
        identity_self = self.command_cls.zeros(n_self)
        identity_opp = opp_command_cls.zeros(n_opp)
        if side is Side.GUARD:
            identity_actions = Actions(sides=BySide(guard=identity_self, bandit=identity_opp))
        else:
            identity_actions = Actions(sides=BySide(guard=identity_opp, bandit=identity_self))

        discount = jnp.asarray(self.discount, dtype=s_flat.dtype)

        # Pad/trim a 1-D Δv from the action grid to a side's command dv_dim,
        # then broadcast to (n, dv_dim_cmd).
        def _grid_dv_to_full(dv_grid: jax.Array, n: int, target_dim: int) -> jax.Array:
            if dv_dim_grid < target_dim:
                pad = jnp.zeros((target_dim - dv_dim_grid,), dv_grid.dtype)
                dv_grid = jnp.concatenate([dv_grid, pad])
            elif dv_dim_grid > target_dim:
                dv_grid = dv_grid[:target_dim]
            return jnp.broadcast_to(dv_grid, (n, target_dim))

        self_dv_dim = int(self.command_cls.zeros(n_self).dv.shape[-1])

        def _opponent_dv(s: jax.Array, k: jax.Array) -> jax.Array:
            """Predict opponent's per-vehicle dv at the simulated state."""
            state = adapter.unpack(s)
            opp_obs = opp_obs_fn(state, identity_actions, opp_side, env.config, k, state.t)
            opp_view = flatten_observations(opp_obs)
            cmd, _ = opponent_model(None, opp_view, k, state.t)
            return cmd.dv

        def _step_one(s: jax.Array, action_idx: jax.Array, k: jax.Array):
            k_opp, k_step = jax.random.split(k, 2)
            self_dv = _grid_dv_to_full(action_grid[action_idx], n_self, self_dv_dim)
            opp_dv = _opponent_dv(s, k_opp)

            self_cmd = self.command_cls.zeros(n_self).replace(
                dv=self_dv.astype(self.command_cls.zeros(n_self).dv.dtype)
            )
            opp_cmd = opp_command_cls.zeros(n_opp).replace(
                dv=opp_dv.astype(opp_command_cls.zeros(n_opp).dv.dtype)
            )
            self_flat = flatten_command(self_cmd)
            opp_flat = flatten_command(opp_cmd)
            if side is Side.GUARD:
                a_flat = jnp.concatenate([self_flat, opp_flat])
            else:
                a_flat = jnp.concatenate([opp_flat, self_flat])

            s_next = adapter.transition(s, a_flat, k_step)
            r = adapter.reward(s, a_flat, s_next, side)
            return s_next, r

        def recurrent_fn(_params, rng_key, action, embedding):
            keys = jax.random.split(rng_key, action.shape[0])
            s_next, r = jax.vmap(_step_one)(embedding, action, keys)
            value_next = jax.vmap(leaf_value)(s_next)
            batch = action.shape[0]
            recurrent_out = mctx.RecurrentFnOutput(
                reward=r.astype(s_flat.dtype),
                discount=jnp.broadcast_to(discount, (batch,)),
                prior_logits=jnp.zeros((batch, a_count), dtype=s_flat.dtype),
                value=value_next.astype(s_flat.dtype),
            )
            return recurrent_out, s_next

        s_flat_b = s_flat[None, :]
        root = mctx.RootFnOutput(
            prior_logits=jnp.zeros((1, a_count), dtype=s_flat.dtype),
            value=jnp.atleast_1d(leaf_value(s_flat).astype(s_flat.dtype)),
            embedding=s_flat_b,
        )

        # mctx threads `params` through to recurrent_fn; classic UCT has no
        # learned net, so we pass an empty pytree rather than None (which
        # mctx's type stub rejects even though it works at runtime).
        empty_params: dict = {}
        if self.variant == "gumbel_muzero":
            policy_out = mctx.gumbel_muzero_policy(
                params=empty_params,
                rng_key=key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.num_simulations,
                max_depth=self.max_depth,
            )
        else:
            policy_out = mctx.muzero_policy(
                params=empty_params,
                rng_key=key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.num_simulations,
                max_depth=self.max_depth,
            )
        action_array = jnp.asarray(policy_out.action)
        return action_array[0]

    def _action_idx_to_command(self, action_idx: jax.Array) -> Any:
        n = self.n_vehicles
        cmd_template = self.command_cls.zeros(n)
        target_dim = int(cmd_template.dv.shape[-1])
        dv_grid = self.action_grid[action_idx]  # (dv_dim_grid,)
        grid_dim = int(self.action_grid.shape[1])
        if grid_dim < target_dim:
            pad = jnp.zeros((target_dim - grid_dim,), dv_grid.dtype)
            dv_grid = jnp.concatenate([dv_grid, pad])
        elif grid_dim > target_dim:
            dv_grid = dv_grid[:target_dim]
        dv_full = jnp.broadcast_to(dv_grid, (n, target_dim))
        return cmd_template.replace(dv=dv_full.astype(cmd_template.dv.dtype))
