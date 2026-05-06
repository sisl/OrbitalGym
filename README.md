<p align="center">
    <em>orbital-game — JAX-native decision-making for orbital scenarios</em>
</p>

<p align="center">
  <a href="https://github.com/sisl/orbital-game/actions/workflows/ci.yml"><img src="https://github.com/sisl/orbital-game/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://sisl.github.io/orbital-game/"><img src="https://img.shields.io/badge/docs-latest-blue" alt="Documentation"></a>
</p>

---

A JAX-native framework for orbital game scenarios. The goal is to provide a high-performance research and education framework for orbital game decision-making. It is built on top of JAX and [astrojax](https://github.com/duncaneddy/astrojax) for differentiable dynamics, and is intended as a starting point for further development and research.

Orbital game is any two-side decision making problem where two sides compete to complete an objective. Four scenarios ship in the box: Lady-Bandit-Guard, Pursuit-Evasion, Sun-Blocking, and Observation-Blocking. 

The same core drives single-agent (Gymnasium-shaped), multi-agent (PettingZoo-shaped), and POMDP-shaped views of the same underlying game to enable integration with the rich ecosystem of tools built around those interfaces. The core is also designed to be easily extended with new scenarios, new dynamics, and new interfaces.

## Install

```bash
pip install orbital-game
# or
uv add orbital-game
```

If you want JAX accelerated on GPU or Apple Silicon, install the matching extra:

```bash
# NVIDIA CUDA 12
pip install orbital-game[cuda12]
# NVIDIA CUDA 13
pip install orbital-game[cuda13]
# Apple Silicon Metal (via jax-mps)
pip install orbital-game[mps]
```

The Gymnasium and PettingZoo adapters live behind their own extras (the POMDP adapter has no external dependency and is always importable):

```bash
pip install orbital-game[gymnasium,pettingzoo]
```

## Quickstart

This project uses [`just`](https://github.com/casey/just) as a command runner and [`uv`](https://docs.astral.sh/uv/) for Python package management.

```bash
# Install just (macOS)
brew install just

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install everything (dev deps + adapter extras)
just install
```

See the [Getting started guide](https://duncaneddy.github.io/orbital-game/getting-started/) to walk from a fresh checkout to a rendered trajectory plot.

## Development

Every recipe has an equivalent raw command you can run directly.

| Task | `just` recipe | Raw command |
|------|---------------|-------------|
| Install dev deps + adapter extras | `just install` | `uv sync --dev --extra gymnasium --extra pettingzoo` |
| Run tests | `just test` | `uv run pytest tests/ -v` |
| Test with coverage | `just test-cov` | `uv run pytest --cov=orbital_game --cov-report=term-missing` |
| Format code | `just fmt` | `uv run ruff format` |
| Lint (auto-fix) | `just lint` | `uv run ruff check --fix` |
| Type check | `just typecheck` | `uv run pyrefly check` |
| All quality checks | `just check` | Runs fmt + lint + typecheck |
| Build docs | `just docs-build` | `uv run zensical build --clean` |
| Serve docs | `just docs-serve` | `uv run zensical serve --clean` |

## License

The code in this repository is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.
