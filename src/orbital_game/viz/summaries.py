"""Summary plots: reward curves and propellant-mass curves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_reward_curve(reward: jax.Array, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot reward vs. step.

    reward shape:
      - `(T,)`   — single trajectory, drawn as a single line
      - `(B, T)` — batched; individual traces at alpha=0.2, bolded mean overlay
    """
    if ax is None:
        _, ax = plt.subplots()
    r = jnp.asarray(reward)
    if r.ndim == 1:
        ax.plot(r)
    elif r.ndim == 2:
        for i in range(r.shape[0]):
            ax.plot(r[i], alpha=0.2)
        ax.plot(r.mean(axis=0), linewidth=2.0, label="mean")
    else:
        raise ValueError(f"Expected 1D or 2D reward, got shape {r.shape}")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.grid(True)
    return ax


def plot_mass_curve(mass: jax.Array, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot propellant mass per vehicle vs. step.

    mass shape: `(T, N)` where T is time and N is vehicle index.
    """
    if ax is None:
        _, ax = plt.subplots()
    m = jnp.asarray(mass)
    for i in range(m.shape[1]):
        ax.plot(m[:, i], label=f"v{i}")
    ax.set_xlabel("step")
    ax.set_ylabel("propellant mass (kg)")
    ax.grid(True)
    return ax
