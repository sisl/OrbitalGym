"""Build ``examples/mcts_cpu_vs_mps.ipynb`` programmatically with nbformat.

Run from the repo root:

    uv run python examples/_build_mcts_cpu_vs_mps_notebook.py

Dedicated CPU/MPS comparison for the JAX-native ``MCTSPolicy``. The whole
search is one ``lax.fori_loop``-based JIT region (mctx is fully traced),
so the device backend (MPS / GPU) pays for itself on the simulation
budget — unlike the previous root-UCB pseudo-MCTS, which serialized one
JIT call per UCB iteration through the Python host.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s)


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s)


cells: list[nbf.NotebookNode] = []


cells.append(
    md(
        """# MCTSPolicy: CPU vs MPS speedup

The protocol unification turned MCTS from "Python loop calling JIT'd leaf
rollouts" (one host↔device round-trip per iteration) into a single
on-device search via `mctx.gumbel_muzero_policy` — `lax.fori_loop`
internals, JIT-friendly recurrent_fn, fully on the device.

This notebook quantifies the resulting speedup. Expectation: MPS pulls
ahead at higher `num_simulations` because more work amortizes a fixed
JIT-compile cost.
"""
    )
)


cells.append(
    code(
        """# Configure JAX BEFORE any jax import. We probe whether jax-mps is installed
# by attempting to import its provider; if it works we enable cpu+mps,
# otherwise we run CPU-only (the comparison just becomes a CPU benchmark).
import importlib.util
import os

if importlib.util.find_spec('jax_mps') is not None:
    os.environ['JAX_PLATFORMS'] = 'cpu,mps'
    os.environ['JAX_DEFAULT_DEVICE'] = 'cpu'
else:
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import sys
from pathlib import Path
_here = Path.cwd()
if (_here / 'src' / 'orbital_game').is_dir():
    _repo_root = _here
elif (_here.parent / 'src' / 'orbital_game').is_dir():
    _repo_root = _here.parent
else:
    raise RuntimeError(f'Could not locate orbital-game repo root from cwd={_here}')
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

%load_ext autoreload
%autoreload 2

import time
import jax
import jax.numpy as jnp
import numpy as np

import orbital_game
orbital_game.set_precision(jnp.float32)

from orbital_game import OrbitalGameEnv, Side
from orbital_game.adapters.pomdp import POMDPAdapter
from orbital_game.policies.mcts import MCTSPolicy
from orbital_game.policies.uniform_random import UniformRandomDiscretePolicy
from examples.lbg_ring_intercept import build_scenario, RingInterceptParams

print('JAX', jax.__version__, '| default backend:', jax.default_backend())
print('CPU devices:', jax.devices('cpu'))
try:
    mps = jax.devices('mps')
    HAS_MPS = True
    print('MPS devices:', mps)
except RuntimeError:
    HAS_MPS = False
    print('MPS not available — running CPU-only.')
    print('Install with: pip install orbital-game[mps]   (Apple Silicon only)')
"""
    )
)


cells.append(md("## 1. Build a small scenario"))

cells.append(
    code(
        """params = RingInterceptParams(
    n_bandits=2,
    ring_radius_m=2000.0,
    guard_ring_radius_m=200.0,
    breach_radius_m=5.0,
    catch_radius_m=50.0,
    dt=10.0,
    max_horizon_s=600.0,
    seed=0,
)
cfg = build_scenario(params)
env = OrbitalGameEnv(cfg)
adapter = POMDPAdapter(env)

WET = 12.0 + 3.0  # cold-gas wet mass
DV_MAX = 3.6 * cfg.dt / WET

angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
grid_2d = DV_MAX * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
action_grid = jnp.asarray(np.concatenate([grid_2d, np.zeros((1, 2))], axis=0), dtype=jnp.float32)

print(f'states_dim = {adapter.states_dim} | A = {action_grid.shape[0]}')
"""
    )
)


cells.append(md("## 2. Build MCTSPolicy under each backend"))

cells.append(
    code(
        """def build_mcts(env_, adapter_, num_simulations: int) -> MCTSPolicy:
    opp = UniformRandomDiscretePolicy(
        action_grid=action_grid,
        n_vehicles=cfg.n_bandits,
        command_cls=env_.bandit_command_cls,
    )
    return MCTSPolicy(
        env_model=adapter_,
        side=Side.GUARD,
        action_grid=action_grid,
        opponent_model=opp,
        opponent_action_grid=action_grid,
        num_simulations=num_simulations,
        n_vehicles=cfg.n_guards,
        command_cls=env_.guard_command_cls,
    )

def time_n_calls(policy, s_flat, t, n_warmup: int = 1, n_calls: int = 5):
    \"\"\"Warm-up call (compile) + n_calls timed calls.\"\"\"
    keys = jax.random.split(jax.random.PRNGKey(0), n_warmup + n_calls)
    @jax.jit
    def _call(s, k, tt):
        return policy(None, s, k, tt)
    # Warm up.
    for i in range(n_warmup):
        cmd, _ = _call(s_flat, keys[i], t)
        cmd.dv.block_until_ready()
    # Time.
    t0 = time.perf_counter()
    for i in range(n_warmup, n_warmup + n_calls):
        cmd, _ = _call(s_flat, keys[i], t)
        cmd.dv.block_until_ready()
    return (time.perf_counter() - t0) / n_calls
"""
    )
)


cells.append(md("## 3. Sweep `num_simulations` — CPU vs MPS"))

cells.append(
    code(
        """state, _ = env.reset(jax.random.PRNGKey(0))
sweep = [16, 32, 64, 128]

print(f'{"num_sims":>8} | {"CPU (ms)":>10} | {"MPS (ms)":>10} | {"speedup":>8}')
print('-' * 48)
for num_sims in sweep:
    # CPU
    cpu_pol = build_mcts(env, adapter, num_sims)
    s_flat_cpu = adapter.pack(state)
    cpu_ms = 1000 * time_n_calls(cpu_pol, s_flat_cpu, state.t)

    if HAS_MPS:
        with jax.default_device(jax.devices('mps')[0]):
            env_mps = OrbitalGameEnv(cfg)
            adapter_mps = POMDPAdapter(env_mps)
            state_mps, _ = env_mps.reset(jax.random.PRNGKey(0))
            mps_pol = build_mcts(env_mps, adapter_mps, num_sims)
            s_flat_mps = adapter_mps.pack(state_mps)
            mps_ms = 1000 * time_n_calls(mps_pol, s_flat_mps, state_mps.t)
        speedup = cpu_ms / mps_ms
        print(f'{num_sims:>8} | {cpu_ms:>10.1f} | {mps_ms:>10.1f} | {speedup:>7.2f}x')
    else:
        print(f'{num_sims:>8} | {cpu_ms:>10.1f} | {"-":>10} | {"-":>8}')
"""
    )
)


cells.append(
    md(
        """## 4. Notes

- The single-JIT design is the load-bearing claim. If MPS doesn't beat CPU
  at higher `num_simulations`, that's a regression — investigate before
  declaring the demo healthy. Common culprits: stale JIT cache, an
  inadvertent host-roundtrip in `recurrent_fn`, or float64 leakage
  triggering an MPS fallback.
- Per-call cost should grow roughly linearly in `num_simulations` (work
  per simulation ≈ one `recurrent_fn` invocation = one env step + one
  opponent-model evaluation).
- The numbers above are illustrative; absolute timings depend on the
  device, the scenario size (states_dim, action grid), and `mctx`'s
  internals. The *trend* (MPS catches up and overtakes CPU at scale) is
  the take-away.
"""
    )
)


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
)


def main():
    out = Path(__file__).parent / "mcts_cpu_vs_mps.ipynb"
    nbf.write(nb, out.as_posix())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
