"""ActionRepeatPolicy — hold an inner policy's command for several env steps.

The env ticks at ``config.dt`` while a planner that searches macro steps
decides every ``repeat`` ticks. This wrapper closes that gap: it calls the
inner policy on ticks where the counter is a multiple of ``repeat`` and
re-emits the cached command in between, so the executed trajectory matches
what a macro-step planner simulated (``POMDPAdapter(action_repeat=repeat)``
applies the same command for the same number of substeps, each one capped by
the env's own per-step delta-v limit).

``policy_state`` carries the cached command, the tick counter, and the inner
policy's own state; ``init_state(config, env_state, key)`` follows the
rollout's ``init_policy_state_fns`` convention, and every argument has a
default so the conformance check can resolve it with a zero-argument call.
Only the ``dv`` field of the inner command is cached: every other field of
the side's Command is re-emitted at its zero value, so a side whose action
components carry pointing or communication fields needs a wrapper of its own.

The replan gate is a ``jax.lax.cond``, which preserves semantics everywhere
but saves compute only outside ``jax.vmap``. Under ``vmap`` a ``cond``
becomes a select: both branches execute for every batch element, so the
inner planner runs on every tick of every episode and the wrapper buys
nothing. A batched evaluator should instead let the planner replan every
tick against a macro-step adapter
(``POMDPAdapter(env, action_repeat=k)``), which searches the same macro
horizon at full cadence.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class ActionRepeatState:
    """Cached command, tick counter, and the inner policy's state.

    ``step`` counts ticks since the wrapper was initialised. The inner
    policy runs when ``step % repeat == 0`` — so on the first tick — and
    ``cached_dv`` holds the command replayed on every other tick.
    """

    cached_dv: jax.Array  # (n_vehicles, dv_dim)
    step: jax.Array  # () int32
    inner_state: Any = flax.struct.field(default=None, pytree_node=True)


def _zero_arg_init_state(policy: Any) -> Any:
    """``policy.init_state()`` when it takes no required arguments, else ``None``."""
    init_state = getattr(policy, "init_state", None)
    if init_state is None:
        return None
    params = inspect.signature(init_state).parameters.values()
    requires_args = any(
        p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params
    )
    if requires_args:
        return None
    return init_state()


@dataclass(frozen=True)
class ActionRepeatPolicy:
    """Call ``inner`` every ``repeat`` env steps, replaying its command in between."""

    inner: Any  # a Policy
    repeat: int
    n_vehicles: int = 0
    command_cls: Any = None

    def __post_init__(self) -> None:
        if self.command_cls is None:
            raise ValueError("ActionRepeatPolicy requires command_cls")
        if self.n_vehicles == 0:
            raise ValueError("ActionRepeatPolicy requires n_vehicles > 0")
        if self.repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {self.repeat}")

    def init_state(
        self,
        config: Any = None,
        env_state: Any = None,
        key: jax.Array | None = None,
        inner_init_state: Any = None,
    ) -> ActionRepeatState:
        """Build the initial state with a zero cached command and the counter at 0.

        The counter starts at 0, so the inner policy is called on the very
        first tick and the zero cache is never emitted. ``inner_init_state``
        seeds the wrapped policy; when it is ``None`` the inner policy's own
        zero-argument ``init_state`` is used if it has one.
        """
        del config, env_state, key
        if inner_init_state is None:
            inner_init_state = _zero_arg_init_state(self.inner)
        template = self.command_cls.zeros(self.n_vehicles)
        return ActionRepeatState(
            cached_dv=jnp.zeros_like(template.dv),
            step=jnp.asarray(0, dtype=jnp.int32),
            inner_state=inner_init_state,
        )

    def __call__(
        self,
        policy_state: ActionRepeatState,
        agent_view: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, ActionRepeatState]:
        template = self.command_cls.zeros(self.n_vehicles)
        due = jnp.equal(jnp.mod(policy_state.step, self.repeat), 0)

        def _replan():
            cmd, next_inner = self.inner(policy_state.inner_state, agent_view, key, t)
            return cmd.dv.astype(policy_state.cached_dv.dtype), next_inner

        def _hold():
            return policy_state.cached_dv, policy_state.inner_state

        dv, inner_state = jax.lax.cond(due, _replan, _hold)
        cmd = template.replace(dv=dv.astype(template.dv.dtype))
        new_state = ActionRepeatState(
            cached_dv=dv,
            step=policy_state.step + 1,
            inner_state=inner_state,
        )
        return cmd, new_state
