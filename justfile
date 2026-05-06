default:
    @just --list

# Install dev deps
install:
    uv sync --dev

# Run the test suite
test:
    uv run pytest tests/ -v

# Run tests with coverage
test-cov:
    uv run pytest --cov=orbital_game --cov-report=term-missing

# Format code
fmt:
    uv run ruff format

# Lint with auto-fix
lint:
    uv run ruff check --fix

# Type check
typecheck:
    uv run pyrefly check

# Format + lint + typecheck
check: fmt lint typecheck

# Build the docs site
docs-build:
    uv run zensical build --clean

# Serve the docs site locally
docs-serve:
    uv run zensical serve --clean

# Execute all example notebooks end-to-end (uses the project venv's Jupyter).
notebooks:
    #!/usr/bin/env bash
    set -e
    for nb in examples/games/*.ipynb examples/workflow_*.ipynb; do
        echo "=== $nb ==="
        .venv/bin/jupyter nbconvert --to notebook --execute --output /tmp/_out.ipynb "$nb"
    done
