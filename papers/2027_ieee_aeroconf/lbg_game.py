"""One-on-one lady-bandit-guard: the game and the MPPI policy of either side.

The lady is a fixed point at the origin of the radial, along-track, cross-track
frame. The guard wins by capturing the bandit or by keeping it out until the
horizon; the bandit wins by breaching a sphere about the lady. Both vehicles
follow the HCW transition with impulses at the start of each step, and each has
its own thrust limit (``*_cap_mps``) and delta-v budget (``*_budget_mps``).
Capture and breach are tested at ``substeps`` points inside every step. Every
episode advances through ``OrbitalGymEnv.step``.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from controllers import PURSUERS, controller_matrices, limit_impulse, pursuit_command
from pe_game import (
    DEFAULT_PLANNER,
    approach_penalty,
    clip_norm,
    initial_bank,
    make_environment,
    unit,
)

from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.env.types import Actions, BySide

GUARDS = ("coast", "direct", "pd", "lqr", "hcw", "mppi_coast", "mppi_approach")
BANDITS = ("coast", "direct", "lqr", "hcw", "evade", "mppi_direct", "mppi_lqr")
PLANNING = ("mppi_coast", "mppi_approach", "mppi_direct", "mppi_lqr")
AVOID_RANGE_M = 300.0


@dataclass(frozen=True)
class LBGGame:
    guard: str = "lqr"
    bandit: str = "direct"
    bandit_separation_m: float = 1000.0
    guard_separation_m: float = 200.0
    horizon_s: float = 1800.0
    guard_cap_mps: float = 0.1
    bandit_cap_mps: float = 0.1
    guard_budget_mps: float = 5.0
    bandit_budget_mps: float = 5.0
    dt_s: float = 10.0
    capture_m: float = 10.0
    breach_m: float = 50.0
    capture_speed_mps: float = math.inf
    breach_speed_mps: float = math.inf
    substeps: int = 10
    lookahead_s: float = 100.0

    def __post_init__(self):
        if self.guard not in GUARDS or self.bandit not in BANDITS:
            raise ValueError(f"Unknown policy {self.guard} or {self.bandit}")
        for name in (
            "bandit_separation_m",
            "guard_separation_m",
            "horizon_s",
            "guard_cap_mps",
            "dt_s",
            "capture_m",
            "breach_m",
            "lookahead_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("guard_budget_mps", "bandit_budget_mps", "bandit_cap_mps"):
            value = getattr(self, name)
            if math.isnan(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("capture_speed_mps", "breach_speed_mps"):
            if math.isnan(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive or positive infinity")
        if not math.isclose(
            self.horizon_s / self.dt_s, round(self.horizon_s / self.dt_s), abs_tol=1e-9
        ):
            raise ValueError("horizon must be an integer number of steps")

    def record(self):
        return {
            k: ("unlimited" if isinstance(v, float) and math.isinf(v) else v)
            for k, v in asdict(self).items()
        }

    def key(self, planner=DEFAULT_PLANNER):
        payload = {"game": "lbg", "config": self.record(), "planner": planner.record()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def plans(self):
        return self.guard in PLANNING, self.bandit in PLANNING

    @property
    def batch_signature(self):
        return (
            "lbg",
            self.dt_s,
            self.lookahead_s,
            self.capture_m,
            self.breach_m,
            self.substeps,
            self.plans,
        )

    def initial_states(self, episodes, seed):
        """Guard and bandit start at rest in independent uniform directions."""
        guard = initial_bank(episodes, seed + 1, self.guard_separation_m)
        bandit = initial_bank(episodes, seed, self.bandit_separation_m)
        return np.concatenate([guard, bandit], axis=1)


def first_event(taus, plus, radius, gate):
    """Index of the first in-step sample meeting the event, or the sample count if none."""
    states = taus @ plus
    inside = (jnp.linalg.norm(states[:, :3], axis=-1) < radius) & (
        jnp.linalg.norm(states[:, 3:], axis=-1) <= gate
    )
    return jnp.where(inside.any(), jnp.argmax(inside), taus.shape[0])


def events(taus, guard, bandit, capture, breach, capture_gate, breach_gate):
    """Capture and breach in one step; the earlier sample decides and a tie is a capture."""
    n = taus.shape[0]
    first_capture = first_event(taus, guard - bandit, capture, capture_gate)
    first_breach = first_event(taus, bandit, breach, breach_gate)
    caught = (first_capture < n) & (first_capture <= first_breach)
    breached = (first_breach < n) & (first_breach < first_capture)
    closest = jnp.min(jnp.linalg.norm((taus @ (guard - bandit))[:, :3], axis=-1))
    return caught, breached, closest


def bandit_heuristic(index, bandit, guard, matrices, cap, key):
    """Bandit feedback toward the lady: coast, direct, LQR, HCW intercept, or evasive."""
    toward = -unit(bandit[:3])
    away = unit(bandit[:3] - guard[:3])
    weight = jnp.maximum(0.0, 1.0 - jnp.linalg.norm(bandit[:3] - guard[:3]) / AVOID_RANGE_M)
    return jax.lax.switch(
        index,
        (
            lambda: jnp.zeros(3),
            lambda: cap * toward,
            lambda: -matrices[0] @ bandit,
            lambda: -matrices[1] @ bandit,
            lambda: cap * unit(toward + 2.0 * weight * away),
        ),
    )


def make_rollout(a, taus, gain, game_params, planner, guard_side):
    """Discounted cost of one impulse sequence for the planning side.

    The guard planner forecasts a coasting bandit or one thrusting straight at the
    lady; the bandit planner forecasts a guard in direct pursuit or under LQR.
    Stage cost: guard, normalized guard-bandit range; bandit, normalized range to
    the lady. A capture or breach adds the event bonus to the winner and the
    same penalty to the loser, and play then stops.
    """
    capture, breach = game_params

    def rollout(g0, b0, controls, own_rem, opp_cap, opp_rem, model, gates):
        capture_gate, breach_gate = gates
        scale = jnp.maximum(
            jnp.linalg.norm((g0 - b0)[:3] if guard_side else b0[:3]),
            capture if guard_side else breach,
        )

        def step(carry, u):
            g, b, own_left, opp_left, over, cost, weight = carry
            u = limit_impulse(u, jnp.inf, own_left)
            if guard_side:
                opp = jnp.where(model, -opp_cap * unit(b[:3]), jnp.zeros(3))
                opp = limit_impulse(opp, jnp.inf, opp_left)
                gp, bp = g.at[3:].add(u), b.at[3:].add(opp)
            else:
                rel = g - b
                opp = jnp.where(model, clip_norm(-gain @ rel, opp_cap), -opp_cap * unit(rel[:3]))
                opp = limit_impulse(opp, jnp.inf, opp_left)
                gp, bp = g.at[3:].add(opp), b.at[3:].add(u)
            caught, breached, _ = events(taus, gp, bp, capture, breach, capture_gate, breach_gate)
            caught, breached = caught & ~over, breached & ~over
            ng, nb = a @ gp, a @ bp
            win = caught if guard_side else breached
            loss = breached if guard_side else caught
            range_ = jnp.linalg.norm((ng - nb)[:3] if guard_side else nb[:3]) / scale
            stage = jnp.where(over | win | loss, 0.0, range_)
            stage = stage - planner.capture_bonus * win + planner.capture_bonus * loss
            if guard_side:
                stage = stage + jnp.where(
                    over | win | loss, 0.0, approach_penalty(ng - nb, capture_gate)
                )
            else:
                stage = stage + jnp.where(over | win | loss, 0.0, approach_penalty(nb, breach_gate))
            return (
                ng,
                nb,
                own_left - jnp.linalg.norm(u),
                opp_left - jnp.linalg.norm(opp),
                over | caught | breached,
                cost + weight * stage,
                weight * planner.discount,
            ), None

        init = (g0, b0, own_rem, opp_rem, jnp.array(False), jnp.array(0.0), jnp.array(1.0))
        final, _ = jax.lax.scan(step, init, controls)
        return final[5]

    return rollout


def make_mppi(rollout, planner):
    """MPPI step (Williams et al., 2017) on standardized costs for either LBG side."""

    def plan(key, g, b, nominal, own_cap, own_rem, opp_cap, opp_rem, model, gates):
        noise = jax.random.normal(key, (planner.samples, planner.segments, 3))
        noise = jnp.repeat(noise, planner.repeat, axis=1) * planner.noise * own_cap
        noise = noise.at[0].set(0.0)
        candidates = clip_norm(nominal[None] + noise, own_cap)
        candidates = candidates.at[1].set(0.0)
        costs = jax.vmap(rollout, in_axes=(None, None, 0, None, None, None, None, None))(
            g, b, candidates, own_rem, opp_cap, opp_rem, model, gates
        )
        z = (costs - costs.mean()) / (costs.std() + 1e-12)
        weights = jax.nn.softmax(-z / planner.temperature)
        updated = clip_norm(jnp.einsum("k,klj->lj", weights, candidates), own_cap)
        return updated[0], jnp.concatenate([updated[1:], updated[-1:]], axis=0)

    return plan


CLASSICAL_GUARD = {name: PURSUERS.index(name) for name in ("coast", "direct", "pd", "lqr", "hcw")}
CLASSICAL_BANDIT = {name: i for i, name in enumerate(("coast", "direct", "lqr", "hcw", "evade"))}


@lru_cache(maxsize=32)
def kernel(dt, steps, lookahead, capture, breach, substeps, planner, g_plans, b_plans):
    # The environment's own capture test is disabled; this game decides both events.
    env = make_environment(dt, dt * steps, 1e-9)
    matrices = controller_matrices(env.mean_motion, dt, lookahead)
    a = jnp.asarray(hcw_rtn_stm(env.mean_motion, dt))
    taus = jnp.stack(
        [
            jnp.asarray(hcw_rtn_stm(env.mean_motion, dt * (j + 1) / substeps))
            for j in range(substeps)
        ]
    )
    gain = matrices[0]
    evader_planner = planner.for_evader()
    g_plan = make_mppi(make_rollout(a, taus, gain, (capture, breach), planner, True), planner)
    b_plan = make_mppi(
        make_rollout(a, taus, gain, (capture, breach), evader_planner, False), evader_planner
    )
    template, _ = env.reset(jax.random.PRNGKey(0))
    gc, bc = env.guard_command_cls.zeros(1), env.bandit_command_cls.zeros(1)

    def episode(initial, params, episode_key):
        (
            g_index,
            b_index,
            g_cap,
            b_cap,
            g_budget,
            b_budget,
            horizon,
            g_model,
            b_model,
            capture_gate,
            breach_gate,
        ) = params
        gates = (capture_gate, breach_gate)
        state = template.replace(
            guards=template.guards.replace(rtn=initial[None, :6]),
            bandits=template.bandits.replace(rtn=initial[None, 6:]),
        )
        start_captured = (jnp.linalg.norm(initial[:3] - initial[6:9]) < capture) & (
            jnp.linalg.norm(initial[3:6] - initial[9:]) <= capture_gate
        )
        start_breached = (
            ~start_captured
            & (jnp.linalg.norm(initial[6:9]) < breach)
            & (jnp.linalg.norm(initial[9:]) <= breach_gate)
        )
        carry = dict(
            state=state,
            done=start_captured | start_breached,
            caught=start_captured,
            breached=start_breached,
            g_spent=jnp.array(0.0),
            b_spent=jnp.array(0.0),
            closest=jnp.linalg.norm(initial[:3] - initial[6:9]),
            end_time=jnp.where(start_captured | start_breached, 0.0, jnp.inf),
            g_nominal=jnp.zeros((planner.length, 3)),
            b_nominal=jnp.zeros((evader_planner.length, 3)),
        )

        def step(c, index):
            g, b = c["state"].guards.rtn[0], c["state"].bandits.rtn[0]
            g_key, b_key = jax.random.split(jax.random.fold_in(episode_key, index))
            g_left, b_left = g_budget - c["g_spent"], b_budget - c["b_spent"]
            relative = g - b
            if g_plans:
                g_cmd, g_nominal = g_plan(
                    g_key, g, b, c["g_nominal"], g_cap, g_left, b_cap, b_left, g_model > 0.5, gates
                )
            else:
                g_cmd = pursuit_command(g_index.astype(int), relative, matrices, g_cap, g_key, dt)
                g_nominal = c["g_nominal"]
            if b_plans:
                b_cmd, b_nominal = b_plan(
                    b_key, g, b, c["b_nominal"], b_cap, b_left, g_cap, g_left, b_model > 0.5, gates
                )
            else:
                b_cmd = bandit_heuristic(b_index.astype(int), b, g, matrices, b_cap, b_key)
                b_nominal = c["b_nominal"]
            g_dv = limit_impulse(g_cmd, g_cap, g_left)
            b_dv = limit_impulse(b_cmd, b_cap, b_left)
            actions = Actions(
                sides=BySide(guard=gc.replace(dv=g_dv[None]), bandit=bc.replace(dv=b_dv[None]))
            )
            out = env.step(jax.random.fold_in(jax.random.PRNGKey(1), index), c["state"], actions)
            done = c["done"]
            new = jax.tree.map(lambda n, o: jnp.where(done, o, n), out.state, c["state"])
            delivered = out.info["applied_dv"]
            dg = jnp.where(done, 0.0, jnp.linalg.norm(delivered.guard[0]))
            db = jnp.where(done, 0.0, jnp.linalg.norm(delivered.bandit[0]))
            gp = g.at[3:].add(delivered.guard[0])
            bp = b.at[3:].add(delivered.bandit[0])
            caught, breached, closest = events(
                taus, gp, bp, capture, breach, capture_gate, breach_gate
            )
            caught, breached = caught & ~done, breached & ~done
            ended = caught | breached
            steps_done = new.step >= jnp.rint(horizon / dt).astype(int)
            return dict(
                state=new,
                done=done | ended | steps_done,
                caught=c["caught"] | caught,
                breached=c["breached"] | breached,
                g_spent=c["g_spent"] + dg,
                b_spent=c["b_spent"] + db,
                closest=jnp.minimum(c["closest"], jnp.where(done, jnp.inf, closest)),
                end_time=jnp.where(ended, (index + 1) * dt, c["end_time"]),
                g_nominal=g_nominal,
                b_nominal=b_nominal,
            ), None

        final, _ = jax.lax.scan(step, carry, jnp.arange(steps))
        return {
            "outcome": (~final["breached"]).astype(int),
            "capture": final["caught"].astype(int),
            "breach": final["breached"].astype(int),
            "event_time_s": final["end_time"],
            "duration_s": final["state"].t,
            "guard_dv": final["g_spent"],
            "bandit_dv": final["b_spent"],
            "closest_m": final["closest"],
            "final_relative": final["state"].bandits.rtn[0] - final["state"].guards.rtn[0],
        }

    return jax.jit(jax.vmap(episode))


def parameters(game):
    return [
        0 if game.guard in PLANNING else CLASSICAL_GUARD[game.guard],
        0 if game.bandit in PLANNING else CLASSICAL_BANDIT[game.bandit],
        game.guard_cap_mps,
        game.bandit_cap_mps,
        game.guard_budget_mps,
        game.bandit_budget_mps,
        game.horizon_s,
        1.0 if game.guard == "mppi_approach" else 0.0,
        1.0 if game.bandit == "mppi_lqr" else 0.0,
        game.capture_speed_mps,
        game.breach_speed_mps,
    ]


def evaluate_lbg(games, initials, episode_ids, planner=DEFAULT_PLANNER, seed=0):
    first = games[0]
    if any(g.batch_signature != first.batch_signature for g in games):
        raise ValueError("A batch must share dt, lookahead, event radii, substeps and planners")
    initials = np.asarray(initials, dtype=float)
    if initials.shape != (len(games), 12):
        raise ValueError("One 12-element initial state per game")
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(
        jnp.asarray(episode_ids, dtype=jnp.uint32)
    )
    steps = round(max(g.horizon_s for g in games) / first.dt_s)
    fn = kernel(
        first.dt_s,
        steps,
        first.lookahead_s,
        first.capture_m,
        first.breach_m,
        first.substeps,
        planner,
        *first.plans,
    )
    params = jnp.asarray([parameters(g) for g in games])
    return jax.device_get(fn(jnp.asarray(initials), params, keys))
