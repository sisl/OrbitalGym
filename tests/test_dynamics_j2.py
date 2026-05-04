import jax.numpy as jnp


def test_j2_eci_metadata():
    from orbital_game.dynamics import j2  # noqa
    from orbital_game.registry import DynamicsKey, DynamicsKind, Frame, resolve

    fn = resolve(DynamicsKey.J2_ECI)
    assert fn.frame is Frame.ECI
    assert fn.kind is DynamicsKind.ABSOLUTE


def test_j2_secular_node_precession_sign():
    """For a prograde, non-equatorial circular orbit, J2 drives RAAN westward
    (Ω̇ < 0). After several orbits, the orbit's RAAN should be < 0."""
    from orbital_game.dynamics import j2  # noqa
    from orbital_game.registry import DynamicsKey, resolve

    fn = resolve(DynamicsKey.J2_ECI)
    mu = 3.986004418e14
    a = 6378137.0 + 800e3
    inc = jnp.deg2rad(45.0)
    speed = (mu / a) ** 0.5
    # Inclined circular orbit: position on ascending node, velocity tilted by inc.
    state0 = jnp.array([[a, 0.0, 0.0, 0.0, speed * jnp.cos(inc), speed * jnp.sin(inc)]])
    dv = jnp.zeros((1, 3))
    period = 2 * jnp.pi * jnp.sqrt(a**3 / mu)
    n_sub_per_orbit = 720
    n_orbits = 5
    dt = float(period / n_sub_per_orbit)
    state = state0
    for _ in range(n_orbits * n_sub_per_orbit):
        state = fn(state, dv, params=None, dt=dt)

    # RAAN from final state's angular momentum vector.
    r = state[0, :3]
    v = state[0, 3:]
    h = jnp.cross(r, v)
    n_vec = jnp.cross(jnp.array([0.0, 0.0, 1.0]), h)
    raan_final = float(jnp.arctan2(n_vec[1], n_vec[0]))
    # Initial RAAN is 0; westward precession gives raan_final < 0.
    assert raan_final < 0, f"RAAN should precess westward under J2; got {raan_final}"
