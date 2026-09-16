"""Persisted built-ins load without prior construction in a fresh interpreter."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.eval.bank import sample_bank, save_bank
from orbitalgym.registry import StateComponentKey


def _fresh(code, *args):
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    # Bound JAX CPU resources in children, including when other suites run.
    bootstrap = """
import os
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(bootstrap) + textwrap.dedent(code), *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    del env  # Do not expose the inherited environment in assertion diagnostics.
    assert result.returncode == 0, result.stdout + result.stderr


def test_fresh_pe_process_loads_and_replays_lbg_bank(tmp_path):
    cfg = make_lady_bandit_guard(guard_components=(StateComponentKey.RTN,))
    states = sample_bank(OrbitalGymEnv(cfg), n_episodes=2, seed=17)
    path = tmp_path / "shared-bank.h5"
    save_bank(path, cfg, states, seed=17)
    _fresh(
        """
        import sys
        import h5py
        import jax
        import jax.numpy as jnp
        import numpy as np
        import orbitalgym
        from orbitalgym import OrbitalGymEnv, make_pursuit_evasion
        from orbitalgym.eval.bank import load_bank, bank_episode
        orbitalgym.set_precision(jnp.float64)
        # Construct only PE before reading the LBG file; no LBG factory call.
        env = OrbitalGymEnv(make_pursuit_evasion())
        cfg, states = load_bank(sys.argv[1])
        assert type(cfg.game).__name__ == "LadyBanditGuard"
        assert type(cfg.termination_fn).__name__ == "LbgEventTermination"
        with h5py.File(sys.argv[1]) as bank:
            np.testing.assert_array_equal(states.guards.rtn, bank["states/guards.rtn"][...])
            np.testing.assert_array_equal(states.bandits.rtn, bank["states/bandits.rtn"][...])
        start = bank_episode(states, 0)
        replayed, _ = env.reset_from_state(start, jax.random.PRNGKey(0))
        np.testing.assert_array_equal(replayed.guards.rtn, start.guards.rtn)
        np.testing.assert_array_equal(replayed.bandits.rtn, start.bandits.rtn)
    """,
        path,
    )


def test_all_registered_builtin_key_families_resolve_on_cold_import():
    _fresh("""
        from orbitalgym import registry as r
        families = (
            r.DynamicsKey, r.AttitudeDynamicsKey, r.PolicyKey, r.ObservationFnKey,
            r.RewardFnKey, r.TerminationFnKey, r.SideSamplerKey, r.MassSamplerKey,
            r.AttitudeSamplerKey, r.ValidatorKey, r.BeliefInitializerKey,
            r.BeliefUpdaterKey, r.BeliefSyncKey, r.LinkKey,
        )
        failures = []
        for family in families:
            for key in family:
                try:
                    assert callable(r.resolve(key))
                except KeyError:
                    failures.append(f"{family.__name__}.{key.name}")
        assert not failures, failures
    """)


def test_lazy_loading_preserves_overrides_and_rejects_unknown_keys():
    _fresh("""
        from enum import StrEnum
        from orbitalgym.registry import RewardFnKey, register, resolve
        @register(RewardFnKey.ZERO)
        def custom_zero(*args):
            return 123
        # A different key from the same built-in module must not overwrite it.
        assert callable(resolve(RewardFnKey.DISTANCE_TO_REFERENCE_ORBIT))
        assert resolve(RewardFnKey.ZERO) is custom_zero
        class ExternalKey(StrEnum):
            UNKNOWN = "untrusted.module.path"
        try:
            resolve(ExternalKey.UNKNOWN)
        except KeyError:
            pass
        else:
            raise AssertionError("Unknown custom key was accepted")
    """)
