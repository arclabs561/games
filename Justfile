# Lint, type-check and test contract for the games repo. See README.md for per-game usage.

export TY_UV := "scripts"

default:
    just --list

# Lint all Python, notebooks included.
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

# Rerun every notebook and compare its outputs with the committed ones.
notebooks:
    uv run pytest --nbval --sanitize-with tests/nbval_sanitize.cfg $(git ls-files '*.ipynb')

# Rerun every notebook and save its new outputs.
execute:
    for nb in $(git ls-files '*.ipynb'); do uv run jupyter execute --inplace "$nb"; done

# Everything CI runs, except the script lock and z3 checks.
ci: check fmt-check typecheck test notebooks
