# Where Planning Changes the Outcome of Orbital Games

Code for the experiments, figures, and quoted statistics of

> D. Eddy, M. Al-Husseini, and M. J. Kochenderfer, "Where Planning Changes the Outcome of Orbital Games," IEEE Aerospace Conference, 2027.

The paper studies where the choice of policy, and where vehicle capability, decides two close-range games: pursuit-evasion (PE) and lady-bandit-guard (LBG). Every script here runs with [`uv`](https://docs.astral.sh/uv/) from a checkout of this repository and uses the `orbitalgym` package of the same commit.

## Experiments

Each experiment of the paper's Tables 4 and 5 has one script, named after it. A script defines one or more *runs*; a run is a grid of configurations, each simulated on the same 128 initial conditions and seeds.

| Experiment | Script | Runs | Used in |
|---|---|---|---|
| PE-1: capability | `pe_1_capability.py` | `pe_1`, `pe_1_fine`, `pe_1_seed2`, `pe_1_n512` | Figure 3(d–f), Results |
| PE-2: budget size | `pe_2_budget_size.py` | `pe_2`, `pe_2_fine` | Figure 3(a–c, g–i) |
| PE-3: geometry | `pe_3_geometry.py` | `pe_3`, `pe_3_fine` | Figure 5 |
| PE-4: information | `pe_4_information.py` | `pe_4`, `pe_4_capability` | Figure 6 |
| PE-5: lookahead | `pe_5_lookahead.py` | `pe_5` | Figure 7 |
| PE-6: evader forecast | `pe_6_evader_forecast.py` | `pe_6` | Results |
| PE-7: capture definition | `pe_7_capture_definition.py` | `pe_7`, `pe_7_endpoint` | Results |
| PE-8: search | `pe_8_search.py` | `pe_8`, `pe_8_hcw_lookahead` | Table 6 |
| LBG-1: capability | `lbg_1_capability.py` | `lbg_1`, `lbg_1_fine`, `lbg_1_n512` | Figure 4, Results |
| LBG-2: geometry | `lbg_2_geometry.py` | `lbg_2` | Results |
| LBG-3: event definition | `lbg_3_event_definition.py` | `lbg_3` | Results |

The `_fine` runs evaluate an experiment on a finer grid of the same parameters for the contour maps. `pe_1_seed2`, `pe_1_n512`, and `lbg_1_n512` repeat a capability experiment on independent initial conditions. The docstring of each script describes its runs and gives its run time.

Run an experiment on a GPU from the repository root:

```bash
uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_1_capability.py pe_1
```

Without run names a script executes all of its runs. A run writes one Parquet file of encounter records per configuration to `results/<run>/cells/` and resumes from the files already there. When it finishes it writes `summary/<run>.csv`, one row per configuration. Neither the records nor the summaries are committed.

All runs together take about 7.5 hours on one NVIDIA H100. Without `--with "jax[cuda12]"` the scripts run on the CPU, which is practical only for the runs without MPPI or MCTS.

`diagnostics.py` runs the controlled tests behind the mechanisms that the Results section states, and writes `summary/diagnostics/<name>.json`.

## Figures and quoted statistics

`figures.py` and `quoted_numbers.py` read the summaries that the experiment scripts and `diagnostics.py` write, so run those first. None of these scripts needs a GPU:

```bash
uv run papers/2027_ieee_aeroconf/figures.py
```

```bash
uv run papers/2027_ieee_aeroconf/quoted_numbers.py
```

```bash
uv run papers/2027_ieee_aeroconf/figure_1_method_overview.py
```

```bash
uv run papers/2027_ieee_aeroconf/figure_2_game_progression.py
```

| Script | Output |
|---|---|
| `figure_1_method_overview.py` | Figure 1, from `figure_data/method_overview.npz` |
| `figure_2_game_progression.py` | Figure 2, from `figure_data/game_progression.npz` |
| `figures.py` | Figures 3 to 7 and Table 6, from `summary/` |
| `quoted_numbers.py` | The statistics quoted in the Results section, from `summary/` |
| `policy_gain_uncertainty.py` | Standard errors of the win fractions and policy gains, from `results/` |
| `listing_1_pursuit_evasion.py` | Listing 1 |

Figures are written to `figures/`.

## Shared modules

| Module | Contents |
|---|---|
| `pe_game.py` | The pursuit-evasion game, its Kalman filter and sensors, and the MPPI and MCTS policies |
| `lbg_game.py` | The lady-bandit-guard game and the MPPI policy of either side |
| `controllers.py` | Direct, HCW intercept, random, and evasion laws; the PD and LQR laws are `orbitalgym.policies.heuristic.feedback` |
| `baseline.py` | The baseline scenario, the ratios, and the capability sweep |
| `runner.py` | Batching, encounter records, resumption, and summaries |

Both games advance every encounter through `OrbitalGymEnv.step`.

## Names

The code and the paper name some quantities differently.

| Code | Paper |
|---|---|
| `*_cap_mps` | Thrust limit, the largest velocity change of one step |
| `*_budget_mps` | Delta-v budget |
| `hcw` | HCW intercept |
| `flee`, `mppi_flee` | Flee, MPPI-Flee |
| `mppi_approach` | MPPI-Approach |
| `capture_speed_mps`, `breach_speed_mps` | Velocity-matching condition $v_e$ |
| `substeps` | Points $S$ at which an event is tested inside a step |

In pursuit-evasion the pursuer is OrbitalGym's bandit and the evader its guard. The paper's ratios divide the guard's capability by the bandit's. The pursuit-evasion grids are built from the evader-over-pursuer ratio, which is that ratio. The lady-bandit-guard grids are built from the bandit-over-guard ratio, its reciprocal; `figures.py` and `quoted_numbers.py` convert.

## Reproducing the paper's numbers

The paper's results were produced on one NVIDIA H100 in double precision with `jax` 0.11.1 and `numpy` 2.5.2. To pin them:

```bash
uv run --with "jax[cuda12]==0.11.1" --with "numpy==2.5.2" papers/2027_ieee_aeroconf/pe_1_capability.py pe_1
```

Other versions or devices can change continuous results in the last digits, and with them the outcome of an occasional encounter.

## Tests

```bash
uv run --with pandas --with pyarrow pytest papers/2027_ieee_aeroconf/tests
```

The tests cover the capture and breach tests, the budgets, sensing, the planners, the number of configurations of every run, and the listing.
