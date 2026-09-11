"""MCTSPolicy — JAX-native classic UCT search via mctx.

Conforms to the unified :class:`orbitalgym.policies.base.Policy`
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
  :class:`orbitalgym.policies.uniform_random.UniformRandomDiscretePolicy`
  is deliberately cheap and uncoupled from any scenario; for sharper
  search, wire e.g. a glideslope or LeadIntercept policy as ``opponent_model``.
- **agent_view contract (v1).** Either a 1-D flat state vector matching
  ``env_model.states_dim``, or any object exposing ``mean: jax.Array`` of
  the same shape (a :class:`orbitalgym.belief.base.Belief`). KF/EKF
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

from orbitalgym.adapters._command_flatten import flatten_command
from orbitalgym.adapters.pomdp.adapter import POMDPAdapter
from orbitalgym.belief.flatten import belief_mean_to_flat_state
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.observations.types import merge_full_state_observations


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
    # Guidance model receives one merged full-state identity-channel block;
    # partial/transformed sensors need a belief updater before this boundary.
    opponent_model: Any  # a Policy
    opponent_action_grid: jax.Array  # (A_opp, dv_dim) — used by default opponent_model

    num_simulations: int = 32
    max_depth: int | None = None
    leaf_value_fn: Callable[[jax.Array], jax.Array] | None = None
    # Per-tree-edge discount. ``None`` reads ``env_model.discount()``, which
    # is the adapter's per-macro-step discount, so a search over macro steps
    # backs up terminal reward on the same scale the env accumulates it.
    discount: float | None = None
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

    # Optional ContactSchedule. When set, the opponent_model is treated as a
    # *delayed-feedback* controller: its command is recomputed only when the
    # simulated time falls inside a contact window; outside contacts the
    # last-cached command is replayed. The cached Δv is carried as part of
    # the MCTS embedding so it persists across tree-edge advances. Default
    # `None` keeps the original "call opponent_model fresh every step"
    # behaviour.
    opponent_schedule: Any = None

    def __post_init__(self) -> None:
        if self.command_cls is None:
            raise ValueError(
                "MCTSPolicy was constructed without `command_cls`. Build via "
                "OrbitalGymEnv (which injects `command_cls`/`n_vehicles` via "
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

    def _decode_joint_idx(self, joint_idx: jax.Array) -> jax.Array:
        """Base-A decode of a joint action index into ``(n_vehicles,)`` per-vehicle indices."""
        a_count = int(self.action_grid.shape[0])
        return jnp.stack([(joint_idx // (a_count**v)) % a_count for v in range(self.n_vehicles)])

    def _search_joint(self, s_flat: jax.Array, key: jax.Array) -> jax.Array:
        policy_out = self._search_joint_out(s_flat, key)
        return self._decode_joint_idx(jnp.asarray(policy_out.action)[0])

    def _search_joint_out(self, s_flat: jax.Array, key: jax.Array) -> Any:
        """Single MCTS over the joint action space ``A^n_vehicles``.

        Returns the mctx ``PolicyOutput`` with ``action_weights`` shape
        ``(1, A^n_vehicles)``.
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

        discount = jnp.asarray(
            adapter.discount() if self.discount is None else self.discount,
            dtype=s_flat.dtype,
        )
        self_dv_dim = int(self.command_cls.zeros(n_self).dv.shape[-1])

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
            opp_view = merge_full_state_observations(opp_obs, env.layout.dynamics_state_dim)
            cmd, _ = opponent_model(None, opp_view, k, state.t)
            return cmd.dv

        # When `opponent_schedule` is set, the opponent is modelled as a
        # delayed-feedback controller: command is recomputed during contact
        # windows, otherwise the cached Δv is replayed. The cache lives in
        # the MCTS embedding so it persists across tree-edge advances.
        opp_schedule = self.opponent_schedule
        use_schedule = opp_schedule is not None
        if use_schedule:
            from orbitalgym.groundstations.contacts import in_contact_now

        def _step_one_no_schedule(s: jax.Array, joint_idx: jax.Array, k: jax.Array):
            k_opp, k_step = jax.random.split(k, 2)
            self_dv = _per_v_dv(self._decode_joint_idx(joint_idx))
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

            s_next, r = adapter.step(s, a_flat, k_step, side)
            return s_next, r

        def _step_one_with_schedule(
            s: jax.Array,
            opp_cached_dv: jax.Array,
            joint_idx: jax.Array,
            k: jax.Array,
        ):
            """Step with delayed-feedback opponent semantics.

            The opponent's Δv is *recomputed* only when the simulated tick's
            ``state.t`` is inside a contact window of ``opponent_schedule``.
            Otherwise the cached Δv from the most recent contact tick is
            replayed. The (possibly-updated) cached Δv is returned alongside
            ``s_next`` so it can persist as part of the MCTS embedding.
            """
            k_opp, k_step = jax.random.split(k, 2)
            self_dv = _per_v_dv(self._decode_joint_idx(joint_idx))

            state = adapter.unpack(s)
            in_contact = in_contact_now(opp_schedule, state.t)
            new_opp_cached_dv = jax.lax.cond(
                in_contact,
                lambda: _opponent_dv(s, k_opp),
                lambda: opp_cached_dv,
            )
            opp_dv = new_opp_cached_dv

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

            s_next, r = adapter.step(s, a_flat, k_step, side)
            return s_next, new_opp_cached_dv, r

        if use_schedule:

            def recurrent_fn(_params, rng_key, action, embedding):
                s_emb, opp_cached_emb = embedding
                keys = jax.random.split(rng_key, action.shape[0])
                s_next, new_opp_cached, r = jax.vmap(_step_one_with_schedule)(
                    s_emb, opp_cached_emb, action, keys
                )
                value_next = jax.vmap(leaf_value)(s_next)
                batch = action.shape[0]
                recurrent_out = mctx.RecurrentFnOutput(
                    reward=r.astype(s_flat.dtype),
                    discount=jnp.broadcast_to(discount, (batch,)),
                    prior_logits=jnp.zeros((batch, a_joint), dtype=s_flat.dtype),
                    value=value_next.astype(s_flat.dtype),
                )
                return recurrent_out, (s_next, new_opp_cached)
        else:

            def recurrent_fn(_params, rng_key, action, embedding):
                keys = jax.random.split(rng_key, action.shape[0])
                s_next, r = jax.vmap(_step_one_no_schedule)(embedding, action, keys)
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
        if use_schedule:
            # Compute initial cached opponent Δv at the root by calling the
            # opponent_model on the current state. This means: at t=0 (or
            # whenever the search root is evaluated) the bandit assumes the
            # opponent's last commanded Δv is the freshly-solved Δv —
            # the most defensible prior given no prior contact history.
            key_root, key = jax.random.split(key)
            opp_cached_dv_root = _opponent_dv(s_flat, key_root)
            opp_cached_b = opp_cached_dv_root[None, ...]
            embedding_root: Any = (s_flat_b, opp_cached_b)
        else:
            embedding_root = s_flat_b

        root = mctx.RootFnOutput(
            prior_logits=jnp.zeros((1, a_joint), dtype=s_flat.dtype),
            value=jnp.atleast_1d(leaf_value(s_flat).astype(s_flat.dtype)),
            embedding=embedding_root,
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
        return policy_out

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
        policy_out = self._search_one_vehicle_out(s_flat, key, self_index)
        return jnp.asarray(policy_out.action)[0]

    def _search_one_vehicle_out(
        self,
        s_flat: jax.Array,
        key: jax.Array,
        self_index: int,
    ) -> Any:
        """Single-vehicle MCTS for the vehicle at ``self_index``.

        Other teammates' Δv during simulation come from ``teammate_model``.
        Returns the mctx ``PolicyOutput`` with ``action_weights`` shape ``(1, A)``.
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

        discount = jnp.asarray(
            adapter.discount() if self.discount is None else self.discount,
            dtype=s_flat.dtype,
        )
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
            self_view = merge_full_state_observations(self_obs, env.layout.dynamics_state_dim)
            cmd, _ = teammate_model(None, self_view, k, state.t)
            return cmd.dv

        def _opponent_dv(s: jax.Array, k: jax.Array) -> jax.Array:
            state = adapter.unpack(s)
            opp_obs = opp_obs_fn(state, identity_actions, opp_side, env.config, k, state.t)
            opp_view = merge_full_state_observations(opp_obs, env.layout.dynamics_state_dim)
            cmd, _ = opponent_model(None, opp_view, k, state.t)
            return cmd.dv

        opp_schedule = self.opponent_schedule
        use_schedule = opp_schedule is not None
        if use_schedule:
            from orbitalgym.groundstations.contacts import in_contact_now

        def _step_one_no_schedule(s: jax.Array, action_idx: jax.Array, k: jax.Array):
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

            s_next, r = adapter.step(s, a_flat, k_step, side)
            return s_next, r

        def _step_one_with_schedule(
            s: jax.Array,
            opp_cached_dv: jax.Array,
            action_idx: jax.Array,
            k: jax.Array,
        ):
            k_opp, k_team, k_step = jax.random.split(k, 3)
            self_dv = _teammate_dv(s, k_team)
            my_dv = _self_dv_for_action(action_idx).astype(self_dv.dtype)
            self_dv = self_dv.at[self_index].set(my_dv)

            state = adapter.unpack(s)
            in_contact = in_contact_now(opp_schedule, state.t)
            new_opp_cached_dv = jax.lax.cond(
                in_contact,
                lambda: _opponent_dv(s, k_opp),
                lambda: opp_cached_dv,
            )
            opp_dv = new_opp_cached_dv

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

            s_next, r = adapter.step(s, a_flat, k_step, side)
            return s_next, new_opp_cached_dv, r

        if use_schedule:

            def recurrent_fn(_params, rng_key, action, embedding):
                s_emb, opp_cached_emb = embedding
                keys = jax.random.split(rng_key, action.shape[0])
                s_next, new_opp_cached, r = jax.vmap(_step_one_with_schedule)(
                    s_emb, opp_cached_emb, action, keys
                )
                value_next = jax.vmap(leaf_value)(s_next)
                batch = action.shape[0]
                recurrent_out = mctx.RecurrentFnOutput(
                    reward=r.astype(s_flat.dtype),
                    discount=jnp.broadcast_to(discount, (batch,)),
                    prior_logits=jnp.zeros((batch, a_count), dtype=s_flat.dtype),
                    value=value_next.astype(s_flat.dtype),
                )
                return recurrent_out, (s_next, new_opp_cached)
        else:

            def recurrent_fn(_params, rng_key, action, embedding):
                keys = jax.random.split(rng_key, action.shape[0])
                s_next, r = jax.vmap(_step_one_no_schedule)(embedding, action, keys)
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
        if use_schedule:
            key_root, key = jax.random.split(key)
            opp_cached_dv_root = _opponent_dv(s_flat, key_root)
            opp_cached_b = opp_cached_dv_root[None, ...]
            embedding_root: Any = (s_flat_b, opp_cached_b)
        else:
            embedding_root = s_flat_b

        root = mctx.RootFnOutput(
            prior_logits=jnp.zeros((1, a_count), dtype=s_flat.dtype),
            value=jnp.atleast_1d(leaf_value(s_flat).astype(s_flat.dtype)),
            embedding=embedding_root,
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
        return policy_out

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


@dataclass(frozen=True)
class BeliefAdaptedMCTSPolicy:
    """Wrap an :class:`MCTSPolicy` so it accepts a Belief-shaped ``agent_view``.

    `MCTSPolicy.__call__` already dispatches between flat-array and
    ``.mean``-bearing inputs, but a flat ``.mean`` is only valid when the
    belief mean shape matches the env's flat-state width. KF/EKF beliefs
    expose ``mean`` of shape ``(N_obs, N_total, d)`` — three orders of
    magnitude smaller than the env's flat state, and missing the
    mass/attitude tail. This wrapper bridges the two by writing observer
    0's per-target view into a captured ``template_env_state`` and packing
    via the env adapter, then forwarding the resulting flat state to the
    underlying MCTS.

    Construct once at scenario setup::

        env = OrbitalGymEnv(cfg)
        adapter = POMDPAdapter(env)
        mcts = MCTSPolicy(env_model=adapter, ..., command_cls=...)
        template_state, _ = env.reset(jax.random.key(0))
        belief_mcts = BeliefAdaptedMCTSPolicy(
            inner_mcts=mcts, template_env_state=template_state
        )

    The wrapper transparently passes ``ContactAwareBelief`` through to its
    ``.inner.mean`` — the contact gating is handled by `PlanCachePolicy`,
    not by the searcher.

    Everything the belief does not track comes from the template, the
    game's dwell counters included, so a search rooted on a belief starts
    the dwell wherever the template left it. A reset state leaves it at
    zero; supply a template carrying the current counters for the search
    to see a hold already under way.
    """

    inner_mcts: MCTSPolicy
    template_env_state: Any

    @property
    def n_vehicles(self) -> int:
        return self.inner_mcts.n_vehicles

    @property
    def command_cls(self) -> Any:
        return self.inner_mcts.command_cls

    @property
    def side(self) -> Side:
        return self.inner_mcts.side

    def __call__(
        self,
        policy_state: Any,
        agent_view: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        # Flat-array path: degenerate to plain MCTS.
        if isinstance(agent_view, jax.Array):
            return self.inner_mcts(policy_state, agent_view, key, t)

        # ContactAwareBelief wraps an inner Belief — peel it off.
        belief_mean = agent_view.inner.mean if hasattr(agent_view, "inner") else agent_view.mean
        # If the wrapped mean is already 1-D (e.g. an _OracleBelief whose
        # mean *is* the flat state), forward it unchanged.
        if belief_mean.ndim == 1:
            return self.inner_mcts(policy_state, belief_mean, key, t)

        from orbitalgym.belief.flatten import belief_mean_to_flat_state

        s_flat = belief_mean_to_flat_state(
            belief_mean,
            self.inner_mcts.side,
            self.inner_mcts.env_model,
            self.template_env_state,
        )
        return self.inner_mcts(policy_state, s_flat, key, t)


@dataclass(frozen=True)
class ParticleRootMCTSPolicy:
    """Determinized search over a particle belief.

    Samples ``n_roots`` joint states by drawing one particle per tracked
    entity from observer 0's clouds, runs the inner search from every root
    under ``jax.vmap``, averages the root action weights, and acts on the
    argmax. Own-side entities are anchored to truth by the filter, so the
    roots differ only in the opposing side's states.

    Every root inherits the template's non-belief components, the game's
    dwell counters included; see
    :class:`BeliefAdaptedMCTSPolicy` for what that means for a hold already
    under way.
    """

    inner_mcts: MCTSPolicy
    template_env_state: Any
    n_roots: int = 8

    @property
    def n_vehicles(self) -> int:
        return self.inner_mcts.n_vehicles

    @property
    def command_cls(self) -> Any:
        return self.inner_mcts.command_cls

    @property
    def side(self) -> Side:
        return self.inner_mcts.side

    def _sample_roots(self, belief: Any, key: jax.Array) -> jax.Array:
        particles = belief.particles[0]  # (N_total, K, d)
        log_w = belief.log_weights[0]  # (N_total, K)
        n_total = particles.shape[0]
        keys = jax.random.split(key, n_total)
        idx = jax.vmap(lambda k, lw: jax.random.categorical(k, lw, shape=(self.n_roots,)))(
            keys, log_w
        )  # (N_total, n_roots)
        sampled = particles[jnp.arange(n_total)[:, None], idx]  # (N_total, n_roots, d)
        roots = jnp.swapaxes(sampled, 0, 1)  # (n_roots, N_total, d)
        n_obs = belief.particles.shape[0]

        def to_flat(row):
            mean_like = jnp.broadcast_to(row[None], (n_obs,) + row.shape)
            return belief_mean_to_flat_state(
                mean_like, self.inner_mcts.side, self.inner_mcts.env_model, self.template_env_state
            )

        return jax.vmap(to_flat)(roots)

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        if isinstance(agent_view, jax.Array):
            return self.inner_mcts(policy_state, agent_view, key, t)
        belief = agent_view.inner if hasattr(agent_view, "inner") else agent_view
        k_sample, k_search = jax.random.split(key)
        s_roots = self._sample_roots(belief, k_sample)  # (n_roots, states_dim)
        search_keys = jax.random.split(k_search, self.n_roots)
        inner = self.inner_mcts

        if inner.coordination == "joint" or inner.n_vehicles == 1:
            outs = jax.vmap(inner._search_joint_out)(s_roots, search_keys)
            weights = jnp.mean(outs.action_weights[:, 0, :], axis=0)
            per_v_idx = inner._decode_joint_idx(jnp.argmax(weights))
        else:
            per_v = []
            for i in range(inner.n_vehicles):
                # Decorrelate the stochastic opponent samples across
                # vehicles: each vehicle's search draws from its own
                # fold-in of k_search rather than sharing search_keys.
                search_keys_i = jax.random.split(jax.random.fold_in(k_search, i), self.n_roots)
                outs = jax.vmap(lambda s, k, i=i: inner._search_one_vehicle_out(s, k, i))(
                    s_roots, search_keys_i
                )
                weights = jnp.mean(outs.action_weights[:, 0, :], axis=0)
                per_v.append(jnp.argmax(weights))
            per_v_idx = jnp.stack(per_v)

        return inner._action_idx_to_command(per_v_idx), policy_state
