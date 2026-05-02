default:
    @just --list

# Install dev deps + adapter extras
install:
    uv sync --dev --extra gymnasium --extra pettingzoo

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
