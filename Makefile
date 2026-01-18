PHONY: check
check:
	uv run ruff check .
	uv run ruff format --check .
	uvx ty check

.PHONY: fix
fix:
	uv run ruff check --fix .
	uv run ruff format .
