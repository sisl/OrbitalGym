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

Multi-vehicle teams (``n_vehicles > 1``):

- ``coordination="joint"`` (default) — single MCTS over the joint action
  space ``A^n_vehicles``. Cooperative team planning: a single tree picks
  one action per vehicle, optimised jointly. Sound but combinatorial: with
  ``A=9`` and ``n=3`` the action space is 729; ``n=4`` is 6561. Match
  ``num_simulations`` to the joint cardinality so the tree actually
  visits a meaningful fraction of children.
- ``coordination="independent"`` — one single-vehicle MCTS per teammate.
  Non-cooperative: each vehicle plans assuming teammates behave according
  to ``teammate_model`` (the analogue of ``opponent_model``, but for the
  same side). Search cost grows linearly in ``n_vehicles`` instead of
  exponentially, at the price of zero coordination — two guards may both
  chase the same bandit.

For ``n_vehicles == 1`` the two modes are equivalent.
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
    """JAX-native classic UCT search backed by mctx.

    See module docstring for the joint vs independent coordination modes.
    """

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

    # Multi-vehicle coordination. "joint" runs one MCTS over A^n_vehicles
    # cooperatively; "independent" runs n_vehicles searches with teammates
    # modelled by `teammate_model`. Equivalent for n_vehicles == 1.
    coordination: Literal["joint", "independent"] = "joint"
    # Required when coordination=="independent" and n_vehicles>1. A Policy
    # whose `__call__` outputs the whole self-side fleet's per-vehicle Δv;
    # the searched vehicle's slot is overwritten by the searched action.
    teammate_model: Any = None

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
        if self.coordination not in ("joint", "independent"):
            raise ValueError(
                f"MCTSPolicy.coordination must be 'joint' or 'independent', "
                f"got {self.coordination!r}."
            )
        if (
            self.coordination == "independent"
            and self.n_vehicles > 1
            and self.teammate_model is None
        ):
            raise ValueError(
                "coordination='independent' with n_vehicles>1 requires a "
                "`teammate_model` (a Policy returning all teammates' Δv during "
                "simulation). UniformRandomDiscretePolicy(action_grid=..., "
                "n_vehicles=n_self, command_cls=env.<side>_command_cls) is a "
                "sensible cheap default."
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
        per_v_idx = self._search(s_flat, key)
        cmd = self._action_idx_to_command(per_v_idx)
        return cmd, policy_state

    # --- internals ---

    def _search(self, s_flat: jax.Array, key: jax.Array) -> jax.Array:
        """Returns ``(n_vehicles,)`` per-vehicle action indices."""
        if self.coordination == "joint" or self.n_vehicles == 1:
            return self._search_joint(s_flat, key)
        return self._search_independent(s_flat, key)

    def _search_joint(self, s_flat: jax.Array, key: jax.Array) -> jax.Array:
        """Single MCTS over the joint action space ``A^n_vehicles``.

        Returns ``(n_vehicles,)`` per-vehicle action indices.
        """
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
        # Joint cardinality. Python int — static under jit.
        a_joint = a_count**n_self

        identity_self = self.command_cls.zeros(n_self)
        identity_opp = opp_command_cls.zeros(n_opp)
        if side is Side.GUARD:
            identity_actions = Actions(sides=BySide(guard=identity_self, bandit=identity_opp))
        else:
            identity_actions = Actions(sides=BySide(guard=identity_opp, bandit=identity_self))

        discount = jnp.asarray(self.discount, dtype=s_flat.dtype)
        self_dv_dim = int(self.command_cls.zeros(n_self).dv.shape[-1])

        def _decode_joint(joint_idx: jax.Array) -> jax.Array:
            """``joint_idx`` (scalar) → ``(n_self,)`` per-vehicle indices."""
            # Base-A decoding; n_self is a Python int so the comprehension
            # unrolls statically inside jit.
            return jnp.stack([(joint_idx // (a_count**v)) % a_count for v in range(n_self)])

        def _per_v_dv(per_v_idx: jax.Array) -> jax.Array:
            """``(n_self,)`` action indices → ``(n_self, self_dv_dim)`` Δv."""
            dv = action_grid[per_v_idx]  # (n_self, dv_dim_grid)
            if dv_dim_grid < self_dv_dim:
                pad = jnp.zeros((n_self, self_dv_dim - dv_dim_grid), dv.dtype)
                dv = jnp.concatenate([dv, pad], axis=-1)
            elif dv_dim_grid > self_dv_dim:
                dv = dv[:, :self_dv_dim]
            return dv

        def _opponent_dv(s: jax.Array, k: jax.Array) -> jax.Array:
            state = adapter.unpack(s)
            opp_obs = opp_obs_fn(state, identity_actions, opp_side, env.config, k, state.t)
            opp_view = flatten_observations(opp_obs)
            cmd, _ = opponent_model(None, opp_view, k, state.t)
            return cmd.dv

        def _step_one(s: jax.Array, joint_idx: jax.Array, k: jax.Array):
            k_opp, k_step = jax.random.split(k, 2)
            self_dv = _per_v_dv(_decode_joint(joint_idx))
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
                prior_logits=jnp.zeros((batch, a_joint), dtype=s_flat.dtype),
                value=value_next.astype(s_flat.dtype),
            )
            return recurrent_out, s_next

        s_flat_b = s_flat[None, :]
        root = mctx.RootFnOutput(
            prior_logits=jnp.zeros((1, a_joint), dtype=s_flat.dtype),
            value=jnp.atleast_1d(leaf_value(s_flat).astype(s_flat.dtype)),
            embedding=s_flat_b,
        )

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
        joint_idx = jnp.asarray(policy_out.action)[0]
        return _decode_joint(joint_idx)

    def _search_independent(self, s_flat: jax.Array, key: jax.Array) -> jax.Array:
        """One single-vehicle MCTS per teammate.

        Returns ``(n_vehicles,)`` per-vehicle action indices. Each vehicle's
        search treats the others as part of "the world" and rolls them out
        with ``teammate_model``; the opponent uses ``opponent_model`` as in
        joint mode. Search cost is linear in ``n_vehicles``.
        """
        keys = jax.random.split(key, self.n_vehicles)
        per_v = [
            self._search_one_vehicle(s_flat, keys[i], self_index=i) for i in range(self.n_vehicles)
        ]
        return jnp.stack(per_v)

    def _search_one_vehicle(
        self,
        s_flat: jax.Array,
        key: jax.Array,
        self_index: int,
    ) -> jax.Array:
        """Single-vehicle MCTS for the vehicle at ``self_index``.

        Other teammates' Δv during simulation come from ``teammate_model``.
        Returns the chosen action grid index (scalar) for this vehicle.
        """
        adapter = self.env_model
        env = adapter.env
        side = self.side
        opp_side = side.opposite()
        self_obs_fn = env.guard_observation_fn if side is Side.GUARD else env.bandit_observation_fn
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
        teammate_model = self.teammate_model
        leaf_value = self.leaf_value_fn or _zero_value

        a_count = int(action_grid.shape[0])
        dv_dim_grid = int(action_grid.shape[1])

        identity_self = self.command_cls.zeros(n_self)
        identity_opp = opp_command_cls.zeros(n_opp)
        if side is Side.GUARD:
            identity_actions = Actions(sides=BySide(guard=identity_self, bandit=identity_opp))
        else:
            identity_actions = Actions(sides=BySide(guard=identity_opp, bandit=identity_self))

        discount = jnp.asarray(self.discount, dtype=s_flat.dtype)
        self_dv_dim = int(self.command_cls.zeros(n_self).dv.shape[-1])

        def _self_dv_for_action(action_idx: jax.Array) -> jax.Array:
            """Scalar idx → ``(self_dv_dim,)`` Δv for the searched vehicle."""
            dv = action_grid[action_idx]
            if dv_dim_grid < self_dv_dim:
                pad = jnp.zeros((self_dv_dim - dv_dim_grid,), dv.dtype)
                dv = jnp.concatenate([dv, pad])
            elif dv_dim_grid > self_dv_dim:
                dv = dv[:self_dv_dim]
            return dv

        def _teammate_dv(s: jax.Array, k: jax.Array) -> jax.Array:
            """Whole self-side Δv from teammate_model. Shape ``(n_self, self_dv_dim)``."""
            state = adapter.unpack(s)
            self_obs = self_obs_fn(state, identity_actions, side, env.config, k, state.t)
            self_view = flatten_observations(self_obs)
            cmd, _ = teammate_model(None, self_view, k, state.t)
            return cmd.dv

        def _opponent_dv(s: jax.Array, k: jax.Array) -> jax.Array:
            state = adapter.unpack(s)
            opp_obs = opp_obs_fn(state, identity_actions, opp_side, env.config, k, state.t)
            opp_view = flatten_observations(opp_obs)
            cmd, _ = opponent_model(None, opp_view, k, state.t)
            return cmd.dv

        def _step_one(s: jax.Array, action_idx: jax.Array, k: jax.Array):
            k_opp, k_team, k_step = jax.random.split(k, 3)
            # Teammate baseline for the whole self-side fleet, then override
            # the searched vehicle's slot with the searched Δv.
            self_dv = _teammate_dv(s, k_team)  # (n_self, self_dv_dim)
            my_dv = _self_dv_for_action(action_idx).astype(self_dv.dtype)
            self_dv = self_dv.at[self_index].set(my_dv)
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
        return jnp.asarray(policy_out.action)[0]

    def _action_idx_to_command(self, per_v_idx: jax.Array) -> Any:
        """``(n_vehicles,)`` per-vehicle action indices → side Command."""
        n = self.n_vehicles
        cmd_template = self.command_cls.zeros(n)
        target_dim = int(cmd_template.dv.shape[-1])
        grid_dim = int(self.action_grid.shape[1])
        per_v_dv = self.action_grid[per_v_idx]  # (n, grid_dim)
        if grid_dim < target_dim:
            pad = jnp.zeros((n, target_dim - grid_dim), per_v_dv.dtype)
            per_v_dv = jnp.concatenate([per_v_dv, pad], axis=-1)
        elif grid_dim > target_dim:
            per_v_dv = per_v_dv[:, :target_dim]
        return cmd_template.replace(dv=per_v_dv.astype(cmd_template.dv.dtype))
