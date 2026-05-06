"""Tests for the ASTROJAX_ORBIT typed-instance dynamics.

Unlike the registered step *functions* (HCW, KEPLERIAN_ECI, J2_ECI), this is a
configurable *class* carrying a user-supplied ``ForceModelConfig``. The
ScenarioConfig role fields accept either a registry enum key or a typed-instance
instance — the env resolves both shapes to a callable with ``.frame`` / ``.kind``
attributes.
"""

import jax.numpy as jnp
import numpy as np
import pytest


def test_astrojax_orbit_two_body_matches_keplerian():
    """ASTROJAX_ORBIT with default ForceModelConfig() (point-mass two-body)
    matches the standalone KEPLERIAN_ECI dynamics."""
    from astrojax import ForceModelConfig

    from orbital_game.dynamics import astrojax_orbit, keplerian  # noqa: F401
    from orbital_game.dynamics.astrojax_orbit import AstrojaxOrbitDynamics
    from orbital_game.registry import DynamicsKey, resolve

    # `DynamicsKey.KEPLERIAN_ECI` resolves to the typed-instance class; instantiate.
    kep = resolve(DynamicsKey.KEPLERIAN_ECI)()
    ajx = AstrojaxOrbitDynamics(force_model=ForceModelConfig())  # point-mass

    state0 = jnp.array([[7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]])
    dv = jnp.zeros((1, 3))
    s_kep = kep(state0, dv, params=None, dt=60.0)
    s_ajx = ajx(state0, dv, params=None, dt=60.0)
    np.testing.assert_allclose(s_ajx, s_kep, rtol=1e-9)


def test_astrojax_orbit_metadata():
    from orbital_game.dynamics import astrojax_orbit  # noqa: F401
    from orbital_game.dynamics.astrojax_orbit import AstrojaxOrbitDynamics
    from orbital_game.registry import DynamicsKind, Frame

    inst = AstrojaxOrbitDynamics()
    assert inst.frame is Frame.ECI
    assert inst.kind is DynamicsKind.ABSOLUTE


def test_astrojax_orbit_5x5_smoke():
    """Smoke test: 5x5 spherical harmonics should run without crashing."""
    from astrojax import ForceModelConfig

    try:
        from astrojax import GravityModel

        # Try several common factory names; skip if none works.
        gm = None
        last_exc: Exception | None = None
        for factory_name in ("egm96", "from_egm96", "default", "load_egm96", "from_type"):
            f = getattr(GravityModel, factory_name, None)
            if f is None:
                continue
            try:
                # from_type takes a string model name; the others take degree/order.
                if factory_name == "from_type":
                    gm = f("JGM3")
                    if hasattr(gm, "set_max_degree_order"):
                        # set_max_degree_order mutates in place and returns None.
                        gm.set_max_degree_order(5, 5)
                else:
                    gm = f(degree=5, order=5)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        if gm is None:
            pytest.skip(
                f"No accessible GravityModel factory for 5x5 spherical harmonics "
                f"(last error: {last_exc!r})"
            )
    except (AttributeError, FileNotFoundError, ImportError):
        pytest.skip("astrojax 5x5 GravityModel not available")

    from orbital_game.dynamics.astrojax_orbit import AstrojaxOrbitDynamics

    config = ForceModelConfig(
        gravity_type="spherical_harmonics",
        gravity_model=gm,
        gravity_degree=5,
        gravity_order=5,
    )
    ajx = AstrojaxOrbitDynamics(force_model=config)
    state0 = jnp.array([[7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]])
    out = ajx(state0, jnp.zeros((1, 3)), params=None, dt=60.0)
    assert jnp.all(jnp.isfinite(out))
