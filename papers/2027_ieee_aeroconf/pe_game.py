"""One-on-one pursuit-evasion: the game, its sensing, and the MPPI and MCTS policies.

The pursuer is OrbitalGym's bandit and the evader its guard. The relative state
is x = [r_P - r_E; v_P - v_E] in the radial, along-track, cross-track frame.
Impulses apply at the start of each step, so x_next = A (x + [0; u_P - u_E]).
Each side has its own thrust limit (``*_cap_mps``, the largest impulse of one
step) and delta-v budget (``*_budget_mps``). Capture is tested at ``substeps``
points inside every step. Every episode advances through ``OrbitalGymEnv.step``.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import mctx
import numpy as np
from controllers import (
    EVADERS as FEEDBACK_EVADERS,
)
from controllers import (
    PURSUERS as FEEDBACK_PURSUERS,
)
from controllers import (
    controller_matrices,
    evasion_command,
    limit_impulse,
    pursuit_command,
)

from orbitalgym import OrbitalGymEnv, make_pursuit_evasion
from orbitalgym.config import VehicleParamsSpec
from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.env.types import Actions, BySide
from orbitalgym.reference_orbit import MU_EARTH, ReferenceOrbitState

PURSUERS = ("coast", "direct", "pd", "lqr", "hcw", "mppi_coast", "mppi_flee", "mcts")
EVADERS = ("coast", "random", "flee", "transverse", "mppi_direct", "mppi_lqr", "mcts")
PLANNING = ("mppi_coast", "mppi_flee", "mppi_direct", "mppi_lqr", "mcts")
REFERENCE_RADIUS_M = 7_000_000.0
# Planner rollouts penalize closing faster than gate + range / APPROACH_TIME_S when
# capture requires a relative speed at most ``gate``; with no speed condition the
# term is zero.
APPROACH_TIME_S = 100.0
SENSORS = ("full", "radial", "cone")
SENSING_FIELDS = (
    "sensor",
    "sensed",
    "sensor_range_m",
    "cone_half_angle_deg",
    "noise_fraction",
    "prior_sigma_m",
    "prior_sigma_mps",
)


@dataclass(frozen=True)
class Planner:
    """MPPI settings; one rollout segment holds one impulse.

    Both sides share the settings unless an ``evader_*`` override is positive.
    ``rollout_substeps`` sets the capture test inside planner rollouts; zero
    uses the game's own capture resolution.
    """

    samples: int = 256
    segments: int = 20
    repeat: int = 3
    noise: float = 0.5
    temperature: float = 0.1
    discount: float = 0.99
    capture_bonus: float = 10.0
    evader_samples: int = 0
    evader_segments: int = 0
    rollout_substeps: int = 0

    @property
    def length(self):
        return self.segments * self.repeat

    def record(self):
        """Settings that identify a run; overrides left at zero are omitted."""
        data = asdict(self)
        for name in ("evader_samples", "evader_segments", "rollout_substeps"):
            if not data[name]:
                data.pop(name)
        return data

    def for_evader(self):
        return replace(
            self,
            samples=self.evader_samples or self.samples,
            segments=self.evader_segments or self.segments,
            evader_samples=0,
            evader_segments=0,
        )


DEFAULT_PLANNER = Planner()


@dataclass(frozen=True)
class Game:
    pursuer: str = "direct"
    evader: str = "coast"
    separation_m: float = 300.0
    horizon_s: float = 1800.0
    pursuer_cap_mps: float = 0.1
    evader_cap_mps: float = 0.1
    pursuer_budget_mps: float = math.inf
    evader_budget_mps: float = math.inf
    dt_s: float = 10.0
    capture_m: float = 10.0
    substeps: int = 10
    lookahead_s: float = 100.0
    capture_speed_mps: float = math.inf
    sensor: str = "full"
    sensed: str = "both"
    sensor_range_m: float = math.inf
    cone_half_angle_deg: float = 180.0
    noise_fraction: float = 0.01
    prior_sigma_m: float = 0.0
    prior_sigma_mps: float = 0.0

    def __post_init__(self):
        if self.pursuer not in PURSUERS or self.evader not in EVADERS:
            raise ValueError(f"Unknown controller {self.pursuer} or {self.evader}")
        for name in (
            "separation_m",
            "horizon_s",
            "pursuer_cap_mps",
            "dt_s",
            "capture_m",
            "lookahead_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.evader_cap_mps) or self.evader_cap_mps < 0:
            raise ValueError("evader_cap_mps must be finite and nonnegative")
        for name in ("pursuer_budget_mps", "evader_budget_mps"):
            value = getattr(self, name)
            if math.isnan(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative or positive infinity")
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        if math.isnan(self.capture_speed_mps) or self.capture_speed_mps <= 0:
            raise ValueError("capture_speed_mps must be positive or positive infinity")
        if self.sensor not in SENSORS or self.sensed not in ("both", "pursuer", "evader"):
            raise ValueError(f"Unknown sensor {self.sensor} or sensed side {self.sensed}")
        if self.sensor == "radial" and self.cone_half_angle_deg != 180.0:
            raise ValueError("A radial sensor has no cone")
        if self.sensor == "cone" and not 0 < self.cone_half_angle_deg < 180:
            raise ValueError("A cone half-angle lies strictly between 0 and 180 degrees")
        if not math.isclose(
            self.horizon_s / self.dt_s, round(self.horizon_s / self.dt_s), abs_tol=1e-9
        ):
            raise ValueError("horizon must be an integer number of steps")

    def record(self):
        data = asdict(self)
        if math.isinf(self.capture_speed_mps):
            data.pop("capture_speed_mps")
        if self.sensor == "full":
            for name in SENSING_FIELDS:
                data.pop(name)
        return {
            k: ("unlimited" if isinstance(v, float) and math.isinf(v) else v)
            for k, v in data.items()
        }

    def key(self, planner=DEFAULT_PLANNER):
        payload = {"game": self.record(), "planner": planner.record()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def plans(self):
        """Planner kind for each side: "mppi", "mcts", or None for a feedback policy."""
        return tuple(
            ("mcts" if name == "mcts" else "mppi") if name in PLANNING else None
            for name in (self.pursuer, self.evader)
        )

    @property
    def senses(self):
        return self.sensor != "full"

    def initial_states(self, episodes, seed):
        """The pursuer starts at rest in a uniform direction at the separation."""
        return initial_bank(episodes, seed, self.separation_m)

    @property
    def batch_signature(self):
        """Settings that one compiled evaluation batch must share."""
        return (self.dt_s, self.lookahead_s, self.capture_m, self.substeps, self.plans, self.senses)


def initial_bank(episodes, seed, separation):
    """Initial relative states: uniform position directions at ``separation``, at rest."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    direction = np.random.default_rng(seed).normal(size=(episodes, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return np.column_stack([separation * direction, np.zeros((episodes, 3))])


def make_environment(dt, horizon, capture):
    """OrbitalGym pursuit-evasion environment about a circular reference orbit.

    The vehicle's own thrust limit is set far above any command, so that the
    per-side thrust limits and delta-v budgets of the game apply instead.
    """
    reference = ReferenceOrbitState(
        position_eci=jnp.array([REFERENCE_RADIUS_M, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, math.sqrt(MU_EARTH / REFERENCE_RADIUS_M), 0.0]),
    )
    vehicle = VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=1e6)
    cfg = make_pursuit_evasion(
        dt=dt,
        max_horizon_s=horizon,
        capture_distance_m=capture,
        reference_orbit=reference,
        guard_params=vehicle,
        bandit_params=vehicle,
    )
    return OrbitalGymEnv(cfg)


def clip_norm(u, cap):
    norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
    return u * jnp.minimum(1.0, cap / jnp.maximum(norm, 1e-30))


def unit(r):
    return r / jnp.maximum(jnp.linalg.norm(r), 1e-30)


def capture_test(taus, plus, capture, gate):
    """Closest sampled separation in a step, and whether any sample is a capture.

    ``taus`` holds the transition matrices to the sample times inside the step and
    ``plus`` the post-impulse relative state. A sample captures when the separation
    is below ``capture`` and the relative speed is at most ``gate``.
    """
    states = taus @ plus
    distance = jnp.linalg.norm(states[:, :3], axis=-1)
    speed = jnp.linalg.norm(states[:, 3:], axis=-1)
    return jnp.min(distance), jnp.any((distance < capture) & (speed <= gate))


def approach_penalty(x, gate):
    """Closing speed above the glideslope gate + range / APPROACH_TIME_S, in units of gate."""
    limit = gate + jnp.linalg.norm(x[:3]) / APPROACH_TIME_S
    return jnp.maximum(0.0, jnp.linalg.norm(x[3:]) - limit) / gate


def make_rollout(a, taus, gain, capture, planner, pursuer_side):
    """Discounted cost of one control sequence for the planning side.

    The opponent follows a forecast. For the pursuer planner the evader coasts
    or flees at its thrust limit (``model`` true). For the evader planner
    the pursuer thrusts directly toward the evader at its cap, or follows the
    clipped LQR law (``model`` true). Forecasts respect the remaining budget.
    """

    def rollout(x0, controls, own_rem, opp_cap, opp_rem, model, gate):
        scale = jnp.maximum(jnp.linalg.norm(x0[:3]), capture)

        def step(carry, u):
            x, own_left, opp_left, captured, cost, weight = carry
            u = limit_impulse(u, jnp.inf, own_left)
            los = unit(x[:3])
            if pursuer_side:
                opp = jnp.where(model, -opp_cap * los, jnp.zeros(3))
                du = u - limit_impulse(opp, jnp.inf, opp_left)
            else:
                opp = jnp.where(model, clip_norm(-gain @ x, opp_cap), -opp_cap * los)
                du = limit_impulse(opp, jnp.inf, opp_left) - u
            opp_used = jnp.linalg.norm(limit_impulse(opp, jnp.inf, opp_left))
            plus = x + jnp.concatenate([jnp.zeros(3), du])
            _, inside = capture_test(taus, plus, capture, gate)
            nxt = a @ plus
            hit = ~captured & inside
            distance = jnp.where(captured | hit, 0.0, jnp.linalg.norm(nxt[:3])) / scale
            sign = 1.0 if pursuer_side else -1.0
            bonus = planner.capture_bonus * jnp.where(hit, 1.0, 0.0)
            cost = cost + weight * (sign * distance - sign * bonus)
            if pursuer_side:
                cost = cost + weight * jnp.where(captured | hit, 0.0, approach_penalty(nxt, gate))
            return (
                nxt,
                own_left - jnp.linalg.norm(u),
                opp_left - opp_used,
                captured | hit,
                cost,
                weight * planner.discount,
            ), None

        init = (x0, own_rem, opp_rem, jnp.array(False), jnp.array(0.0), jnp.array(1.0))
        (_, _, _, _, cost, _), _ = jax.lax.scan(step, init, controls)
        return cost

    return rollout


def make_mppi(rollout, planner):
    """Model predictive path integral control (Williams et al., 2017) on standardized costs."""

    def plan(key, x, nominal, own_cap, own_rem, opp_cap, opp_rem, model, gate):
        noise = jax.random.normal(key, (planner.samples, planner.segments, 3))
        noise = jnp.repeat(noise, planner.repeat, axis=1) * planner.noise * own_cap
        noise = noise.at[0].set(0.0)
        candidates = clip_norm(nominal[None] + noise, own_cap)
        candidates = candidates.at[1].set(0.0)
        costs = jax.vmap(rollout, in_axes=(None, 0, None, None, None, None, None))(
            x, candidates, own_rem, opp_cap, opp_rem, model, gate
        )
        z = (costs - costs.mean()) / (costs.std() + 1e-12)
        weights = jax.nn.softmax(-z / planner.temperature)
        updated = clip_norm(jnp.einsum("k,klj->lj", weights, candidates), own_cap)
        shifted = jnp.concatenate([updated[1:], updated[-1:]], axis=0)
        return updated[0], shifted

    return plan


def make_filter(a):
    """Kalman filter on the relative state for one side.

    Each side knows its own delivered impulse. The opponent's impulse is unknown
    and enters as zero-mean process noise with per-axis variance equal to the
    squared opponent cap. A measurement is the true relative position with
    isotropic noise proportional to range, available when the opponent is within
    sensor range and inside a cone about the estimated line of sight.
    """
    h = jnp.hstack([jnp.eye(3), jnp.zeros((3, 3))])

    def update(estimate, cov, truth, key, max_range, cos_half_angle, fraction):
        distance = jnp.linalg.norm(truth[:3])
        cosine = jnp.dot(unit(estimate[:3]), unit(truth[:3]))
        visible = (distance <= max_range) & (cosine >= cos_half_angle)
        sigma = jnp.maximum(fraction * distance, 0.1)
        z = truth[:3] + sigma * jax.random.normal(key, (3,))
        noise = sigma**2 * jnp.eye(3)
        gain = cov[:, :3] @ jnp.linalg.inv(cov[:3, :3] + noise)
        joseph = jnp.eye(6) - gain @ h
        updated = estimate + gain @ (z - estimate[:3])
        updated_cov = joseph @ cov @ joseph.T + gain @ noise @ gain.T
        return (
            jnp.where(visible, updated, estimate),
            jnp.where(visible, updated_cov, cov),
            visible,
        )

    def predict(estimate, cov, own_dv, opponent_cap):
        plus = estimate + jnp.concatenate([jnp.zeros(3), own_dv])
        spread = cov.at[3:, 3:].add(opponent_cap**2 * jnp.eye(3))
        return a @ plus, a @ spread @ a.T

    return update, predict


MCTS_SIMULATIONS = 256
MCTS_DEPTH = 20
MCTS_ACTIONS = 9


def make_mcts(a, taus, gain, intercept, capture, planner, pursuer_side):
    """Monte Carlo tree search over nine held impulses against a random opponent policy.

    Actions: coast; the cap along each of the six RTN axes; the cap along the line
    of sight (toward the evader for the pursuer, away from the pursuer for the
    evader); and the clipped HCW intercept (pursuer) or a transverse burn (evader).
    Each edge holds its impulse for ``planner.repeat`` steps. The opponent's policy
    is a chance outcome drawn uniformly at every edge (Stochastic MuZero search in
    mctx): coasting or fleeing evaders for the pursuer's search, and coasting,
    direct, or HCW-intercept pursuers for the evader's. Edge rewards are the
    negated stage costs of the MPPI rollout. Nodes are valued by a Monte Carlo
    rollout of a default policy (HCW intercept for the pursuer, fleeing for the
    evader) against a drawn opponent, truncated at the planner lookahead.
    """
    sign = 1.0 if pursuer_side else -1.0
    axes = jnp.concatenate([jnp.eye(3), -jnp.eye(3)])
    edge_discount = planner.discount**planner.repeat
    outcomes = 2 if pursuer_side else 3
    chance_logits = jnp.where(jnp.arange(3) < outcomes, 0.0, -1e9)[None]

    def own_impulse(action, x, cap):
        los = unit(x[:3])
        if pursuer_side:
            last = clip_norm(-intercept @ x, cap)
        else:
            axis = jnp.where(
                jnp.abs(los[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0])
            )
            last = cap * unit(jnp.cross(-los, axis))
        options = jnp.concatenate([jnp.zeros((1, 3)), cap * axes, (-cap * los)[None], last[None]])
        return options[action]

    def opponent_impulse(choice, x, cap):
        los = unit(x[:3])
        if pursuer_side:
            return jnp.where(choice == 0, jnp.zeros(3), -cap * los)
        return jnp.select(
            [choice == 0, choice == 1], [jnp.zeros(3), -cap * los], clip_norm(-intercept @ x, cap)
        )

    def default_impulse(x, cap):
        return clip_norm(-intercept @ x, cap) if pursuer_side else -cap * unit(x[:3])

    def simulate(choice, node, length, own_policy, remaining):
        """Play ``length`` steps; only the first ``remaining`` add discounted cost."""
        x, own_left, opp_left, captured, scale, own_cap, opp_cap, gate = node

        def step(carry, index):
            x, own_left, opp_left, captured, cost, weight = carry
            u = limit_impulse(own_policy(x, own_cap), jnp.inf, own_left)
            opp = limit_impulse(opponent_impulse(choice, x, opp_cap), jnp.inf, opp_left)
            du = u - opp if pursuer_side else opp - u
            plus = x + jnp.concatenate([jnp.zeros(3), du])
            _, inside = capture_test(taus, plus, capture, gate)
            nxt = jnp.where(captured, x, a @ plus)
            hit = ~captured & inside
            distance = jnp.where(captured | hit, 0.0, jnp.linalg.norm(nxt[:3])) / scale
            stage = sign * distance - sign * planner.capture_bonus * jnp.where(hit, 1.0, 0.0)
            if pursuer_side:
                stage = stage + jnp.where(captured | hit, 0.0, approach_penalty(nxt, gate))
            active = (index < remaining) & ~captured
            return (
                nxt,
                own_left - jnp.linalg.norm(u),
                opp_left - jnp.linalg.norm(opp),
                captured | hit,
                cost + weight * jnp.where(active, stage, 0.0),
                weight * planner.discount,
            ), None

        init = (x, own_left, opp_left, captured, jnp.array(0.0), jnp.array(1.0))
        final, _ = jax.lax.scan(step, init, jnp.arange(length))
        return final[:5]

    def rollout_value(key, node, remaining):
        choice = jax.random.randint(key, (), 0, outcomes)
        *_, cost = simulate(choice, node, planner.length, default_impulse, remaining)
        return -cost

    def unpack(embedding):
        return tuple(field[0] for field in embedding)

    def pack(node, depth, action):
        return tuple(jnp.asarray(field)[None] for field in (*node, depth, action))

    def edge(choice, node, action):
        a_own = own_impulse(action, node[0], node[5])
        nx, nown, nopp, ncap, cost = simulate(
            choice, node, planner.repeat, lambda _x, _cap: a_own, planner.repeat
        )
        return (nx, nown, nopp, ncap, *node[4:]), cost

    def decision(params, key, action, embedding):
        *node, depth, _ = unpack(embedding)
        choice_key, leaf_key = jax.random.split(key)
        choice = jax.random.randint(choice_key, (), 0, outcomes)
        after, cost = edge(choice, tuple(node), action[0])
        remaining = planner.length - (depth + 1) * planner.repeat
        value = -cost + edge_discount * jnp.where(
            after[3], 0.0, rollout_value(leaf_key, after, remaining)
        )
        output = mctx.DecisionRecurrentFnOutput(
            chance_logits=chance_logits, afterstate_value=value[None]
        )
        return output, pack(node, depth, action[0])

    def chance(params, key, outcome, embedding):
        *node, depth, action = unpack(embedding)
        after, cost = edge(outcome[0], tuple(node), action)
        remaining = planner.length - (depth + 1) * planner.repeat
        value = jnp.where(after[3], 0.0, rollout_value(key, after, remaining))
        output = mctx.ChanceRecurrentFnOutput(
            action_logits=jnp.zeros((1, MCTS_ACTIONS)),
            value=value[None],
            reward=(-cost)[None],
            discount=jnp.where(after[3], 0.0, edge_discount)[None],
        )
        return output, pack(after, depth + 1, 0)

    def plan(key, x, own_cap, own_rem, opp_cap, opp_rem, gate):
        scale = jnp.maximum(jnp.linalg.norm(x[:3]), capture)
        node = (
            x,
            own_rem,
            opp_rem,
            jnp.array(False),
            scale,
            jnp.asarray(own_cap, x.dtype),
            jnp.asarray(opp_cap, x.dtype),
            jnp.asarray(gate, x.dtype),
        )
        root_key, search_key = jax.random.split(key)
        root = mctx.RootFnOutput(
            prior_logits=jnp.zeros((1, MCTS_ACTIONS)),
            value=rollout_value(root_key, node, planner.length)[None],
            embedding=pack(node, 0, 0),
        )
        output = mctx.stochastic_muzero_policy(
            params=None,
            rng_key=search_key,
            root=root,
            decision_recurrent_fn=decision,
            chance_recurrent_fn=chance,
            num_simulations=MCTS_SIMULATIONS,
            max_depth=2 * MCTS_DEPTH,
            dirichlet_fraction=0.0,
            temperature=0.0,
        )
        return own_impulse(output.action[0], x, own_cap)

    return plan


CLASSICAL_PURSUER = {
    name: FEEDBACK_PURSUERS.index(name) for name in ("coast", "direct", "pd", "lqr", "hcw")
}
CLASSICAL_EVADER = {
    name: FEEDBACK_EVADERS.index(name) for name in ("coast", "random", "flee", "transverse")
}


@lru_cache(maxsize=32)
def kernel(dt, steps, lookahead, capture, substeps, planner, p_plans, e_plans, senses=False):
    env = make_environment(dt, dt * steps, capture)
    matrices = controller_matrices(env.mean_motion, dt, lookahead)
    a = jnp.asarray(hcw_rtn_stm(env.mean_motion, dt))

    def interior(count):
        return jnp.stack(
            [jnp.asarray(hcw_rtn_stm(env.mean_motion, dt * (j + 1) / count)) for j in range(count)]
        )

    taus = interior(substeps)
    rollout_taus = interior(planner.rollout_substeps or substeps)
    gain = matrices[0]
    p_plan = make_mppi(make_rollout(a, rollout_taus, gain, capture, planner, True), planner)
    evader_planner = planner.for_evader()
    e_plan = make_mppi(
        make_rollout(a, rollout_taus, gain, capture, evader_planner, False), evader_planner
    )
    p_search = make_mcts(a, rollout_taus, gain, matrices[1], capture, planner, True)
    e_search = make_mcts(a, rollout_taus, gain, matrices[1], capture, evader_planner, False)
    kf_update, kf_predict = make_filter(a)
    template, _ = env.reset(jax.random.PRNGKey(0))
    gc, bc = env.guard_command_cls.zeros(1), env.bandit_command_cls.zeros(1)

    def episode(initial, params, episode_key):
        (
            p_index,
            e_index,
            p_cap,
            e_cap,
            p_budget,
            e_budget,
            horizon,
            p_model,
            e_model,
            p_sensed,
            e_sensed,
            max_range,
            cos_half_angle,
            fraction,
            prior_m,
            prior_mps,
            gate,
        ) = params
        prior_scale = jnp.concatenate([jnp.full(3, prior_m), jnp.full(3, prior_mps)])
        prior_cov = jnp.diag(jnp.maximum(prior_scale, 1e-3) ** 2)
        p_prior, e_prior = jax.random.normal(jax.random.fold_in(episode_key, 2**31), (2, 6))
        state = template.replace(
            guards=template.guards.replace(rtn=jnp.zeros((1, 6))),
            bandits=template.bandits.replace(rtn=initial[None]),
        )
        distance = jnp.linalg.norm(initial[:3])
        start_captured = (distance < capture) & (jnp.linalg.norm(initial[3:]) <= gate)
        zeros = jnp.zeros((planner.length, 3))
        evader_zeros = jnp.zeros((evader_planner.length, 3))
        carry = dict(
            state=state,
            done=start_captured,
            captured=start_captured,
            p_spent=jnp.array(0.0),
            e_spent=jnp.array(0.0),
            closest=distance,
            capture_time=jnp.where(start_captured, 0.0, jnp.inf),
            p_nominal=zeros,
            e_nominal=evader_zeros,
            p_est=initial + prior_scale * p_prior,
            e_est=initial + prior_scale * e_prior,
            p_cov=prior_cov,
            e_cov=prior_cov,
            p_seen=jnp.array(0.0),
            p_error=jnp.array(0.0),
        )

        def step(c, index):
            truth = c["state"].bandits.rtn[0] - c["state"].guards.rtn[0]
            p_key, e_key = jax.random.split(jax.random.fold_in(episode_key, index))
            p_left, e_left = p_budget - c["p_spent"], e_budget - c["e_spent"]
            if senses:
                pm_key, em_key = jax.random.split(jax.random.fold_in(episode_key, index + 2**30))
                p_est, p_cov, p_visible = kf_update(
                    c["p_est"], c["p_cov"], truth, pm_key, max_range, cos_half_angle, fraction
                )
                e_est, e_cov, _ = kf_update(
                    c["e_est"], c["e_cov"], truth, em_key, max_range, cos_half_angle, fraction
                )
                p_view = jnp.where(p_sensed > 0.5, p_est, truth)
                e_view = jnp.where(e_sensed > 0.5, e_est, truth)
            else:
                p_view = e_view = truth
            relative = p_view
            if p_plans == "mcts":
                p_cmd = p_search(p_key, relative, p_cap, p_left, e_cap, e_left, gate)
                p_nominal = c["p_nominal"]
            elif p_plans == "mppi":
                p_cmd, p_nominal = p_plan(
                    p_key,
                    relative,
                    c["p_nominal"],
                    p_cap,
                    p_left,
                    e_cap,
                    e_left,
                    p_model > 0.5,
                    gate,
                )
            else:
                p_cmd = pursuit_command(p_index.astype(int), relative, matrices, p_cap, p_key, dt)
                p_nominal = c["p_nominal"]
            if e_plans == "mcts":
                e_cmd = e_search(e_key, e_view, e_cap, e_left, p_cap, p_left, gate)
                e_nominal = c["e_nominal"]
            elif e_plans == "mppi":
                e_cmd, e_nominal = e_plan(
                    e_key, e_view, c["e_nominal"], e_cap, e_left, p_cap, p_left, e_model > 0.5, gate
                )
            else:
                e_cmd = evasion_command(e_index.astype(int), e_view, e_cap, e_key)
                e_nominal = c["e_nominal"]
            p_dv = limit_impulse(p_cmd, p_cap, p_left)
            e_dv = limit_impulse(e_cmd, e_cap, e_left)
            actions = Actions(
                sides=BySide(guard=gc.replace(dv=e_dv[None]), bandit=bc.replace(dv=p_dv[None]))
            )
            out = env.step(jax.random.fold_in(jax.random.PRNGKey(1), index), c["state"], actions)
            done = c["done"]
            new = jax.tree.map(lambda n, o: jnp.where(done, o, n), out.state, c["state"])
            delivered = out.info["applied_dv"]
            dp = jnp.where(done, 0.0, jnp.linalg.norm(delivered.bandit[0]))
            de = jnp.where(done, 0.0, jnp.linalg.norm(delivered.guard[0]))
            plus = truth + jnp.concatenate([jnp.zeros(3), delivered.bandit[0] - delivered.guard[0]])
            near, inside = capture_test(taus, plus, capture, gate)
            near = jnp.where(done, jnp.inf, near)
            hit = ~c["captured"] & ~done & inside
            captured = c["captured"] | hit
            t_next = (index + 1) * dt
            steps_done = new.step >= jnp.rint(horizon / dt).astype(int)
            if senses:
                p_est, p_cov = kf_predict(p_est, p_cov, delivered.bandit[0], e_cap)
                e_est, e_cov = kf_predict(e_est, e_cov, -delivered.guard[0], p_cap)
                active = ~done
                p_seen = c["p_seen"] + jnp.where(active & p_visible, 1.0, 0.0)
                p_error = c["p_error"] + jnp.where(
                    active, jnp.linalg.norm(p_view[:3] - truth[:3]), 0.0
                )
            else:
                p_est, p_cov, e_est, e_cov = c["p_est"], c["p_cov"], c["e_est"], c["e_cov"]
                p_seen, p_error = c["p_seen"], c["p_error"]
            return dict(
                state=new,
                done=done | captured | steps_done,
                captured=captured,
                p_spent=c["p_spent"] + dp,
                e_spent=c["e_spent"] + de,
                closest=jnp.minimum(c["closest"], near),
                capture_time=jnp.where(hit, t_next, c["capture_time"]),
                p_nominal=p_nominal,
                e_nominal=e_nominal,
                p_est=p_est,
                e_est=e_est,
                p_cov=p_cov,
                e_cov=e_cov,
                p_seen=p_seen,
                p_error=p_error,
            ), None

        final, _ = jax.lax.scan(step, carry, jnp.arange(steps))
        return {
            "outcome": final["captured"].astype(int),
            "capture_time_s": final["capture_time"],
            "duration_s": final["state"].t,
            "pursuer_dv": final["p_spent"],
            "evader_dv": final["e_spent"],
            "closest_m": final["closest"],
            "pursuer_detection_fraction": final["p_seen"] / jnp.maximum(final["state"].step, 1),
            "pursuer_mean_error_m": final["p_error"] / jnp.maximum(final["state"].step, 1),
            "final_relative": final["state"].bandits.rtn[0] - final["state"].guards.rtn[0],
        }

    return jax.jit(jax.vmap(episode))


def parameters(game):
    p_index = 0 if game.pursuer in PLANNING else CLASSICAL_PURSUER[game.pursuer]
    e_index = 0 if game.evader in PLANNING else CLASSICAL_EVADER[game.evader]
    return [
        p_index,
        e_index,
        game.pursuer_cap_mps,
        game.evader_cap_mps,
        game.pursuer_budget_mps,
        game.evader_budget_mps,
        game.horizon_s,
        1.0 if game.pursuer == "mppi_flee" else 0.0,
        1.0 if game.evader == "mppi_lqr" else 0.0,
        float(game.sensed in ("both", "pursuer")),
        float(game.sensed in ("both", "evader")),
        game.sensor_range_m,
        math.cos(math.radians(game.cone_half_angle_deg)),
        game.noise_fraction,
        game.prior_sigma_m,
        game.prior_sigma_mps,
        game.capture_speed_mps,
    ]


def evaluate_games(games, initials, episode_ids, planner=DEFAULT_PLANNER, seed=0):
    """Evaluate one episode per (game, initial state, episode id) triple in a single batch."""
    first = games[0]
    if any(g.batch_signature != first.batch_signature for g in games):
        raise ValueError("A batch must share dt, lookahead, capture, substeps and planner sides")
    initials = np.asarray(initials, dtype=float)
    episode_ids = np.asarray(episode_ids)
    if initials.shape != (len(games), 6) or episode_ids.shape != (len(games),):
        raise ValueError("One initial state and episode id per game")
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(
        jnp.asarray(episode_ids, dtype=jnp.uint32)
    )
    steps = round(max(g.horizon_s for g in games) / first.dt_s)
    fn = kernel(
        first.dt_s,
        steps,
        first.lookahead_s,
        first.capture_m,
        first.substeps,
        planner,
        *first.plans,
        first.senses,
    )
    params = jnp.asarray([parameters(g) for g in games])
    return jax.device_get(fn(jnp.asarray(initials), params, keys))


def with_substeps(game, substeps):
    return replace(game, substeps=substeps)
