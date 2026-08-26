"""Negative-information policy types for the particle filter.

The PF consults a `NegativeInfoMode` to decide how to update particle
weights on non-detections from visibility-gated sensor channels:

  Off  — no negative-information update; weights unchanged on non-visible
         pairs. Reproduces pre-feature behavior.
  Hard — particles whose hypothesized state lies inside the sensor gate
         are crushed (log_eps penalty); particles outside are untouched.
  Soft — sigmoid falloff at the gate boundary, with per-channel softness.

`Soft.softness_per_channel` carries one positive softness value per
Observation channel (positional indexing). Entries for non-gated channels
are required but ignored — pass any positive placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass


class NegativeInfoMode:
    """Base class for PF negative-information policies."""


@dataclass(frozen=True)
class Off(NegativeInfoMode):
    """Disable negative-information updates. PF behaves as today."""


@dataclass(frozen=True)
class Hard(NegativeInfoMode):
    """Hard cutoff: particles inside the gate get crushed on non-detection."""


@dataclass(frozen=True)
class Soft(NegativeInfoMode):
    """Smooth sigmoid cutoff at the gate boundary.

    Parameters
    ----------
    softness_per_channel:
        Positive softness value per Observation channel, in the channel's
        native score units (meters for range-limited, radians for cone).
        Entries for non-gated channels are required but ignored — pass
        any positive placeholder so positional indexing stays aligned
        with the observation tuple.
    """

    softness_per_channel: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.softness_per_channel:
            raise ValueError(
                "Soft.softness_per_channel must be non-empty (one entry per Observation channel)"
            )
        if any(x <= 0 for x in self.softness_per_channel):
            raise ValueError(
                f"Soft softness values must be positive, got {self.softness_per_channel}"
            )


def softness_from_sigma(sigma: float, ratio: float = 1.0) -> float:
    """Convenience: scale softness by sensor measurement std.

    ratio=1 gives a one-sigma transition width; larger ratios give wider
    (more conservative) edges; smaller ratios approach Hard behavior.
    """
    return ratio * sigma
