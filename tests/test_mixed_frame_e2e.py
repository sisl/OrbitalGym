"""End-to-end mixed-frame tests (Task 9.1+).

Verifies that running the same scenario with different truth dynamics produces
RTN trajectories consistent with the underlying physics:

- KEPLERIAN_ECI truth + HCW policy: unperturbed deputy relative motion should
  match pure HCW-truth to within HCW linearization tolerance over a fraction
  of one orbit.
"""

import jax
import jax.numpy as jnp
import numpy as np


def test_keplerian_truth_hcw_policy_relative_trajectory_close_to_pure_hcw():
    """Run the same scenario with two truth models. Relative motion of an unperturbed
    deputy under Keplerian truth (with the reference also Keplerian) should match
    HCW-truth to within HCW linearization tolerance over a fraction of an orbit."""
    from orbitalgym.env.core import OrbitalGymEnv
    from orbitalgym.env.types import Actions, BySide
    from orbitalgym.registry import DynamicsKey, StateComponentKey
    from orbitalgym.sampling.side import RelativeEllipse
    from orbitalgym.sampling.spec import ICSpec
    from tests.test_config_resolved_dynamics import _minimal_cfg

    # Non-trivial bounded relative orbit: deterministic per-vehicle phase so
    # both runs sample identical ICs (RTN is sampled directly; ECI is the
    # closed-form RTN-projected-to-ECI from the same numbers — frame-agnostic).
    sampler = RelativeEllipse(
        radial_ellipse_m=500.0,
        cross_track_m=300.0,
        along_track_offset_m=200.0,
        phase_rad=0.5,
    )
    ic_sampler = ICSpec(
        guard_sampler=sampler,
        bandit_sampler=sampler,
        validators=(),
        max_attempts=10,
    )

    common = dict(
        policy_dynamics=DynamicsKey.HCW_RTN,
        belief_dynamics=DynamicsKey.HCW_RTN,
        dt=10.0,
        max_horizon_s=600.0,
        ic_sampler=ic_sampler,
    )

    # Run 1: pure HCW truth.
    cfg_hcw = _minimal_cfg(truth_dynamics=DynamicsKey.HCW_RTN, **common)
    # Run 2: KEPLERIAN_ECI truth (mixed-frame).
    cfg_kep = _minimal_cfg(
        truth_dynamics=DynamicsKey.KEPLERIAN_ECI,
        guard_components=(StateComponentKey.ECI,),
        bandit_components=(StateComponentKey.ECI,),
        **common,
    )

    def _final_rtn(cfg):
        env = OrbitalGymEnv(cfg)
        actions = Actions(
            sides=BySide(
                guard=env.guard_command_cls.zeros(1),
                bandit=env.bandit_command_cls.zeros(1),
            )
        )
        state, _ = env.reset(jax.random.PRNGKey(0))
        for i in range(int(cfg.max_horizon_s / cfg.dt)):
            state = env.step(jax.random.PRNGKey(i), state, actions).state
        return state.guards.rtn

    rtn_hcw = _final_rtn(cfg_hcw)
    rtn_kep = _final_rtn(cfg_kep)
    # Sanity: trajectories must be non-trivially non-zero (otherwise the test
    # would pass vacuously).
    assert float(jnp.linalg.norm(rtn_hcw[0, :3])) > 100.0
    # 600 seconds is < 10% of one LEO orbit. Unperturbed two-body relative motion
    # should match HCW linearization to within ~10 m at this scale.
    np.testing.assert_allclose(rtn_kep, rtn_hcw, atol=10.0)


def test_j2_truth_drifts_relative_to_keplerian_truth():
    """Over 5 orbits, J2 truth produces visibly more along-track drift than
    Keplerian truth. (Both in mixed-frame mode: ECI truth, HCW belief/policy.)

    Uses an inclined (45 deg) reference orbit because J2's secular effect on
    along-track drift vanishes for the equatorial reference used by
    `_minimal_cfg`'s default; J2 needs a non-zero inclination to produce a
    secular RAAN/argument-of-perigee drift that translates into along-track
    motion of a deputy.
    """
    import jax
    import jax.numpy as jnp

    from orbitalgym.env.core import OrbitalGymEnv
    from orbitalgym.env.types import Actions, BySide
    from orbitalgym.reference_orbit import ReferenceOrbitState
    from orbitalgym.registry import DynamicsKey, StateComponentKey
    from orbitalgym.sampling.side import RelativeEllipse
    from orbitalgym.sampling.spec import ICSpec
    from tests.test_config_resolved_dynamics import _minimal_cfg

    # Inclined LEO reference orbit so J2 has a meaningful secular signature.
    mu = 3.986004418e14
    a = 6378137.0 + 500e3
    inc = jnp.deg2rad(45.0)
    speed = (mu / a) ** 0.5
    ref = ReferenceOrbitState(
        position_eci=jnp.array([a, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, speed * jnp.cos(inc), speed * jnp.sin(inc)]),
    )
    period_s = 2.0 * float(jnp.pi) * (a**3 / mu) ** 0.5  # ~LEO
    n_orbits = 5
    # Use a non-trivial relative orbit so along-track drift is meaningful.
    sampler = RelativeEllipse(
        radial_ellipse_m=500.0,
        cross_track_m=300.0,
        along_track_offset_m=200.0,
        phase_rad=0.5,
    )
    ic_sampler = ICSpec(
        guard_sampler=sampler, bandit_sampler=sampler, validators=(), max_attempts=10
    )

    common = dict(
        policy_dynamics=DynamicsKey.HCW_RTN,
        belief_dynamics=DynamicsKey.HCW_RTN,
        guard_components=(StateComponentKey.ECI,),
        bandit_components=(StateComponentKey.ECI,),
        ic_sampler=ic_sampler,
        reference_orbit=ref,
        dt=60.0,
        max_horizon_s=period_s * n_orbits,
    )

    def _final_along_track(cfg):
        env = OrbitalGymEnv(cfg)
        actions = Actions(
            sides=BySide(
                guard=env.guard_command_cls.zeros(1),
                bandit=env.bandit_command_cls.zeros(1),
            )
        )
        state, _ = env.reset(jax.random.PRNGKey(0))
        for i in range(int(cfg.max_horizon_s / cfg.dt)):
            state = env.step(jax.random.PRNGKey(i), state, actions).state
        return float(state.guards.rtn[0, 1])  # along-track component

    # Pin reference_orbit_dynamics=KEPLERIAN_ECI in both runs so the RTN frame
    # is anchored to the same (unperturbed) chief in both cases; otherwise the
    # default would propagate the chief under J2 too and we'd be measuring only
    # *differential* J2 across the small relative orbit (which is ~tens of m).
    cfg_kep = _minimal_cfg(
        truth_dynamics=DynamicsKey.KEPLERIAN_ECI,
        reference_orbit_dynamics=DynamicsKey.KEPLERIAN_ECI,
        **common,
    )
    cfg_j2 = _minimal_cfg(
        truth_dynamics=DynamicsKey.J2_ECI,
        reference_orbit_dynamics=DynamicsKey.KEPLERIAN_ECI,
        **common,
    )
    drift = abs(_final_along_track(cfg_j2) - _final_along_track(cfg_kep))
    assert drift > 100.0, (
        f"J2 should produce > 100 m along-track drift over 5 orbits; got {drift} m"
    )
