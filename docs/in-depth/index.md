# In depth

Conceptual deep-dives into the major components, with worked examples
and "how to extend" sections.

- [Symmetric core & data shapes](symmetric-core.md) — sides, `BySide`,
  `Actions`, `StepOutput`, `Trajectory`, axis ordering, indexing
  recipes. **Read this first.**
- [State layout & adding a Power component](state-layout.md) — components,
  field maps, end-to-end example of adding a battery state-of-charge.
- [Action components](action-components.md) — per-side `Command`
  composition, the `(IMPULSIVE_MANEUVER,)` default, the LBG-with-comms recipe.
- [Dynamics](dynamics.md) — HCW-RT vs HCW-RTN, actuator composition,
  registering a new dynamics module.
- [Observations](observations.md) — channel struct, scope rules, the
  four bundled channels, writing a custom one.
- [Belief](belief.md) — `(N_obs, N_total, d)` shape, KF / EKF math,
  `belief_rollout`, custom updaters.
- [MCTS policy](mcts.md) — JAX-native Gumbel MuZero / PUCT search via `mctx`;
  opponent-model pattern; CPU vs MPS speedup notes.
