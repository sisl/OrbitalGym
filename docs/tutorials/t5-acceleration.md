# T5 — Run on GPU / MPS

**Time:** ~15 minutes.
**You will:** install GPU- or MPS-backed JAX and vmap a rollout over a
batch of seeds. The realistic perf win for this codebase is
**parallel-over-seeds**, not single-rollout speedup.
**Prerequisite:** [T1 — First rollout (RT 2D)](t1-first-rollout.md).

## Install

For NVIDIA CUDA:

```bash
pip install orbital-game[cuda12]   # or [cuda13]
```

For Apple Silicon Metal (MPS):

```bash
pip install orbital-game[mps]
```

These extras pull in the matching `jax`-with-device wheel.

## Verify the device

```python
--8<-- "tests/docs/test_tut_t5_acceleration.py:imports"
```

```python
--8<-- "tests/docs/test_tut_t5_acceleration.py:devices"
```

You should see `CudaDevice(id=0)` (CUDA) or `MetalDevice(id=0)` (MPS).
If you see only `CpuDevice`, the extra didn't take — check
`pip show jax` and `pip show jaxlib` for matching device-suffixed
versions.

## A single rollout often loses on GPU

The whole `rollout_single_agent` already compiles to one
`jax.lax.scan`. Single-rollout dispatch over PCIe / Metal Shared
Memory typically loses to CPU in-process for small problem sizes. Run
a benchmark before assuming the GPU helps.

## The headline: vmap over seeds

The realistic speedup comes from running many rollouts in parallel.
`vmap` over the seed axis adds a leading batch dimension to every leaf
of the resulting `Trajectory`:

```python
--8<-- "tests/docs/test_tut_t5_acceleration.py:vmap-rollout"
```

After the vmap:

- `trajs.sides.guard.reward` has shape `(B, T)`.
- `trajs.env_state.guards.rtn` has shape `(B, T, N_g, 6)`.
- The full `BATCH → TIME → VEHICLE → FEATURE` axis order applies.

## Read off batched results

```python
--8<-- "tests/docs/test_tut_t5_acceleration.py:read-batched"
```

`final_rewards` has shape `(B,)` — the cumulative reward of each
seed's episode. `final_rewards.std()` is your unbiased estimate of
return variance, useful for plotting confidence bands.

## Caveats

- **Float64 is the default** in this package (set at import time via
  `astrojax.config.set_dtype(jnp.float64)`). On consumer GPUs that's
  emulated and slow. For throughput-focused experiments, call
  `astrojax.config.set_dtype(jnp.float32)` after importing
  `orbital_game` — but verify your physics still produces sensible
  trajectories at single precision.
- **JIT cache warmup** — the first call to a `jax.jit`-compiled
  function compiles it. Time a second call for the steady-state
  number.
- **Host transfer cost** — `float(arr)` blocks until GPU→CPU transfer
  completes. Pull arrays to host once, after the loop.

## What's next

You've completed the tutorial track. From here:

- Browse [Game guides](../games/index.md) for what each game rewards.
- Use a how-to recipe: [vmap rollouts over seeds](../how-to/vmap-rollouts.md)
  is the one-page reference for the pattern above.
- Read [In depth → Symmetric core](../in-depth/symmetric-core.md) for
  the full type-and-shape model.
