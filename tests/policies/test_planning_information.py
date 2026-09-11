"""Independent searches must consume the acting observer's private information."""

from dataclasses import replace

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.belief.contact_aware import ContactAwareBelief
from orbitalgym.belief.flatten import belief_mean_to_flat_state
from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.policies.mcts import BeliefAdaptedMCTSPolicy, MCTSPolicy, ParticleRootMCTSPolicy
from orbitalgym.policies.mppi import MPPIPolicy
from orbitalgym.policies.zero import ZeroControl


@flax.struct.dataclass
class MeanBelief:
    mean: jax.Array


def resource_env(**kwargs):
    from orbitalgym.sampling.mass import ConstantMass

    cfg = make_lady_bandit_guard(n_guards=2, n_bandits=2, **kwargs)
    ic = replace(
        cfg.ic_sampler,
        bandit_sampler=replace(
            cfg.ic_sampler.bandit_sampler, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
        ),
    )
    return OrbitalGymEnv(replace(cfg, ic_sampler=ic))


def setup(side, kind, resources=False):
    env = OrbitalGymEnv(
        make_lady_bandit_guard(n_guards=2, n_bandits=2, shaping_gain=0.0, separation_cost=0.0)
    )
    if resources:
        from orbitalgym.registry import StateComponentKey

        components = (StateComponentKey.RTN, StateComponentKey.MASS)
        env = resource_env(
            guard_components=components,
            bandit_components=components,
            shaping_gain=0.0,
            separation_cost=0.0,
        )
    adapter = POMDPAdapter(env)
    state, _ = env.reset(jax.random.key(0))
    own_cls = env.guard_command_cls if side is Side.GUARD else env.bandit_command_cls
    opp_cls = env.bandit_command_cls if side is Side.GUARD else env.guard_command_cls

    def value(s):
        es = adapter.unpack(s)
        own = es.guards.rtn if side is Side.GUARD else es.bandits.rtn
        opp = es.bandits.rtn if side is Side.GUARD else es.guards.rtn
        return -jnp.sum((own[:, 0] - opp[0, 0]) ** 2)

    common = dict(
        env_model=adapter,
        side=side,
        command_cls=own_cls,
        n_vehicles=2,
        opponent_model=ZeroControl(n_vehicles=2, command_cls=opp_cls),
        teammate_model=ZeroControl(n_vehicles=2, command_cls=own_cls),
        coordination="independent",
    )
    if kind == "mppi":
        policy = MPPIPolicy(
            **common,
            template_env_state=state,
            n_samples=32,
            horizon=1,
            terminal_value_fn=value,
            temperature=0.01,
            noise_sigma=0.3,
        )
    else:
        grid = jnp.array([[-0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        inner = MCTSPolicy(
            **common,
            action_grid=grid,
            opponent_action_grid=grid,
            num_simulations=16,
            max_depth=1,
            leaf_value_fn=value,
        )
        cls = BeliefAdaptedMCTSPolicy if kind == "mcts" else ParticleRootMCTSPolicy
        policy = cls(inner_mcts=inner, template_env_state=state)
    mean = jnp.zeros((2, 4, 6)).at[:, :2, 0].set(1000.0)
    mean = mean.at[0, 2:, 0].set(2000.0).at[1, 2:, 0].set(-2000.0)
    if kind == "particle":
        belief = ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=4)(
            state, side, jax.random.key(1)
        )
        belief = belief.replace(particles=jnp.repeat(mean[:, :, None, :], 4, axis=2))
    else:
        belief = MeanBelief(mean)
    return env, adapter, state, policy, ContactAwareBelief(belief, jnp.zeros(2, dtype=bool))


@pytest.mark.parametrize("side", [Side.GUARD, Side.BANDIT])
@pytest.mark.parametrize("kind", ["mppi", "mcts", "particle"])
def test_independent_action_uses_own_row_and_ignores_other_observer(side, kind):
    _, _, _, policy, view = setup(side, kind)
    if kind == "particle":
        changed = view.inner.replace(particles=view.inner.particles.at[0, 2:, :, 0].set(-2000.0))
    else:
        changed = view.inner.replace(mean=view.mean.at[0, 2:, 0].set(-2000.0))
    act = jax.jit(lambda v: policy(None, v, jax.random.key(7), jnp.array(0.0))[0].dv)
    original = act(view)
    other = act(view.replace(inner=changed))
    np.testing.assert_allclose(original[1], other[1], atol=1e-6)
    assert original[0, 0] > 0.05
    assert original[1, 0] < -0.05
    assert other[0, 0] < -0.05


@pytest.mark.parametrize("side", [Side.GUARD, Side.BANDIT])
def test_root_selects_observer_under_jit_vmap(side):
    _, adapter, state, _, view = setup(side, "mppi")
    roots = jax.jit(
        jax.vmap(
            lambda i: belief_mean_to_flat_state(view.mean, side, adapter, state, observer_index=i)
        )
    )(jnp.arange(2))
    states = jax.vmap(adapter.unpack)(roots)
    opposing = states.bandits if side is Side.GUARD else states.guards
    np.testing.assert_array_equal(opposing.rtn[:, 0, 0], [2000.0, -2000.0])


@pytest.mark.parametrize("side", [Side.GUARD, Side.BANDIT])
def test_context_carries_only_current_own_telemetry_and_local_counters(side):
    from orbitalgym.belief.contact_aware import advance_planning_context, initial_planning_context
    from orbitalgym.dynamics.attitude import AttitudeParams
    from orbitalgym.registry import AttitudeDynamicsKey, StateComponentKey

    ap = AttitudeParams(inertia_diag=jnp.ones(3), omega_max=jnp.ones(3))
    components = (
        StateComponentKey.RTN,
        StateComponentKey.MASS,
        StateComponentKey.ATTITUDE,
        StateComponentKey.BODY_RATES,
    )
    env = resource_env(
        guard_components=components,
        bandit_components=components,
        catch_radius_m=50.0,
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=ap,
        bandit_attitude_params=ap,
    )
    adapter = POMDPAdapter(env)
    state, _ = env.reset(jax.random.key(0))
    own_name = "guards" if side is Side.GUARD else "bandits"
    own = getattr(state, own_name)
    current = state.replace(
        t=jnp.array(30.0),
        step=jnp.array(3),
        dwell_catch=jnp.array([99, 99]),
        dwell_breach=jnp.array([88, 88]),
        **{
            own_name: own.replace(
                propellant_mass=jnp.array([1.0, 2.0]),
                quat=jnp.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
            )
        },
    )
    ctx = initial_planning_context(env, current, side)
    opp_name = "bandits" if side is Side.GUARD else "guards"
    opp = getattr(current, opp_name)
    hidden_change = current.replace(
        **{opp_name: opp.replace(rtn=opp.rtn * 7.0, propellant_mass=jnp.ones(2) * 777.0)},
        dwell_catch=jnp.array([123, 456]),
        dwell_breach=jnp.array([11, 22]),
    )
    hidden_ctx = initial_planning_context(env, hidden_change, side)
    for before, after in zip(jax.tree.leaves(ctx), jax.tree.leaves(hidden_ctx), strict=True):
        np.testing.assert_array_equal(before, after)
    assert {"propellant_mass", "quat"} <= set(ctx.own_telemetry)
    assert not {"rt", "rtn", "eci"} & set(ctx.own_telemetry)
    mean = jnp.zeros((2, 4, 6)).at[:, :, 0].set(1000.0)
    # Only observer 0 believes the fleets are close; observer 1 has no catch.
    mean = mean.at[1, 2:, 0].set(5000.0)
    for _step in range(2):
        ctx = jax.jit(lambda c: advance_planning_context(c, mean, mean, side, env, state, current))(
            ctx
        )
    np.testing.assert_array_equal(ctx.dwell_catch, [[2, 2], [0, 0]])
    assert not bool(jnp.any(ctx.dwell_breach))

    def root(context, i, joint=False):
        return adapter.unpack(
            belief_mean_to_flat_state(
                mean, side, adapter, state, observer_index=i, planning_context=context, joint=joint
            )
        )

    local = root(ctx, 1)
    own_root = getattr(local, own_name)
    assert float(local.t) == 30.0 and int(local.step) == 3
    np.testing.assert_array_equal(own_root.propellant_mass, [own.propellant_mass[0], 2.0])
    np.testing.assert_array_equal(own_root.quat[1], [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(local.dwell_catch, [0, 0])
    joint = root(ctx, 0, True)
    np.testing.assert_array_equal(getattr(joint, own_name).propellant_mass, [1.0, 2.0])
    np.testing.assert_array_equal(joint.dwell_catch, [2, 2])
    changed = ctx.replace(
        own_telemetry={
            **ctx.own_telemetry,
            "propellant_mass": ctx.own_telemetry["propellant_mass"].at[0].set(999.0),
        }
    )
    np.testing.assert_array_equal(adapter.pack(local), adapter.pack(root(changed, 1)))
    # Losing the estimated geometry resets an existing hold.
    far = mean.at[:, 2:, 0].set(5000.0)
    ctx = advance_planning_context(ctx, far, far, side, env, state, current)
    np.testing.assert_array_equal(ctx.dwell_catch, [[0, 0], [0, 0]])
    inside = jnp.zeros_like(mean)
    ctx = advance_planning_context(ctx, inside, inside, side, env, state, current)
    np.testing.assert_array_equal(ctx.dwell_catch, [[1, 1], [1, 1]])
    np.testing.assert_array_equal(ctx.dwell_breach, [[1, 1], [1, 1]])


@pytest.mark.parametrize("side", [Side.GUARD, Side.BANDIT])
def test_belief_rollout_delivers_current_roots_and_persistent_estimated_dwell(side):
    from orbitalgym.env.types import BySide
    from orbitalgym.registry import StateComponentKey
    from orbitalgym.rollout import belief_rollout

    components = (StateComponentKey.RTN, StateComponentKey.MASS)
    env = resource_env(
        guard_components=components,
        bandit_components=components,
        catch_dwell_steps=100,
        max_horizon_s=500.0,
    )
    adapter = POMDPAdapter(env)
    template, _ = env.reset(jax.random.key(0))
    initial = template.replace(
        t=jnp.array(70.0),
        step=jnp.array(7),
        dwell_catch=jnp.array([42, 42]),
        dwell_breach=jnp.array([23, 23]),
    )
    # The local estimates keep both fleets co-located, irrespective of truth.
    mean = jnp.zeros((2, 4, 6)).at[:, :, 0].set(1000.0)

    class RecordRoot:
        def __call__(self, ps, view, key, t):
            root = adapter.unpack(
                belief_mean_to_flat_state(
                    view.mean,
                    side,
                    adapter,
                    template,
                    observer_index=1,
                    planning_context=view.planning_context,
                )
            )
            own = root.guards if side is Side.GUARD else root.bandits
            cls = env.guard_command_cls if side is Side.GUARD else env.bandit_command_cls
            return cls.zeros(2).replace(dv=jnp.ones((2, 3)) * 0.1), jnp.array(
                [root.t, own.propellant_mass[1], root.dwell_catch[0]]
            )

    zero_g = ZeroControl(n_vehicles=2, command_cls=env.guard_command_cls)
    zero_b = ZeroControl(n_vehicles=2, command_cls=env.bandit_command_cls)
    policies = BySide(
        guard=RecordRoot() if side is Side.GUARD else zero_g,
        bandit=RecordRoot() if side is Side.BANDIT else zero_b,
    )

    def init(cfg, es, key):
        return jnp.zeros(3)

    def belief_init(es, side, key):
        return MeanBelief(mean)

    def update(belief, obs, dv, side, key):
        return belief

    traj, _ = belief_rollout(
        env,
        policies,
        BySide(init, init),
        BySide(belief_init, belief_init),
        BySide(update, update),
        key=jax.random.key(9),
        n_steps=4,
        initial_state=initial,
    )
    records = (
        traj.sides.guard.policy_state if side is Side.GUARD else traj.sides.bandit.policy_state
    )
    # reset_from_state deliberately restarts the episode clock.
    np.testing.assert_array_equal(records[1:, 0], [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(records[1:, 2], [0.0, 1.0, 2.0])
    assert records[3, 1] < records[2, 1] < records[1, 1]


def test_flat_belief_rollout_keeps_legacy_policy_contract():
    from orbitalgym.env.types import BySide
    from orbitalgym.rollout import belief_rollout

    env, adapter, state, _, _ = setup(Side.GUARD, "mppi")
    belief = MeanBelief(adapter.pack(state))

    def init_belief(es, side, key):
        return belief

    def update_belief(b, obs, dv, side, key):
        return b

    def init_policy(cfg, es, key):
        return None

    policies = BySide(
        ZeroControl(command_cls=env.guard_command_cls, n_vehicles=2),
        ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=2),
    )
    traj, history = belief_rollout(
        env,
        policies,
        BySide(init_policy, init_policy),
        BySide(init_belief, init_belief),
        BySide(update_belief, update_belief),
        key=jax.random.key(0),
        n_steps=2,
    )
    np.testing.assert_array_equal(history.guard.mean[0], belief.mean)
    assert traj.env_state.t.shape == (2,)


@pytest.mark.parametrize("side", [Side.GUARD, Side.BANDIT])
@pytest.mark.parametrize("kind", ["mppi", "mcts", "particle"])
def test_actual_planners_use_current_self_fuel_without_teammate_telemetry(side, kind):
    from orbitalgym.belief.contact_aware import initial_planning_context

    env, _, state, policy, view = setup(side, kind, resources=True)
    own_name = "guards" if side is Side.GUARD else "bandits"
    own = getattr(state, own_name)
    current = state.replace(
        t=jnp.array(30.0),
        step=jnp.array(3),
        **{own_name: own.replace(propellant_mass=own.propellant_mass.at[1].set(0.0))},
    )
    ctx = initial_planning_context(env, current, side)
    current_view = view.replace(planning_context=ctx)
    # A manually current template is an independent reference for actor 1.
    reference = replace(policy, template_env_state=current)
    key = jax.random.key(7)
    action = jax.jit(lambda v: policy(None, v, key, jnp.array(30.0))[0].dv)
    actual = action(current_view)
    expected = reference(None, view, key, jnp.array(30.0))[0].dv
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-6)
    changed = ctx.replace(
        own_telemetry={
            **ctx.own_telemetry,
            "propellant_mass": ctx.own_telemetry["propellant_mass"].at[0].set(0.0),
        }
    )
    np.testing.assert_array_equal(actual[1], action(view.replace(planning_context=changed))[1])
    if kind == "mppi":
        # Fuel exhaustion removes action-dependent motion and changes the plan.
        assert not bool(jnp.allclose(actual[1], action(view)[1]))
