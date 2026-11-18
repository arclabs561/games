# Lint, type-check and test contract for the games repo. See README.md for per-game usage.

export TY_UV := "scripts"

default:
    just --list

# Lint all Python (notebooks excluded).
check:
    uv run ruff check

# Format all Python in place.
fmt:
    uv run ruff format

# Check formatting without writing.
fmt-check:
    uv run ruff format --check

# Type-check the project and each PEP 723 script in its own environment.
typecheck:
    uv run ty check

# Run the tests.
test:
    uv run pytest

# Everything CI runs.
ci: check fmt-check typecheck test
