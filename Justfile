# Minimal lint/test contract for games repo.
# See README.md for per-game usage.

default:
    just --list

# Lint all Python under repo root (excludes notebooks).
check:
    ruff check .

# Format all Python (in-place).
fmt:
    ruff format .

# Check formatting without writing.
fmt-check:
    ruff format --check .

# Run pytest (smoke + any future tests).
test:
    python -m pytest tests/ -q
