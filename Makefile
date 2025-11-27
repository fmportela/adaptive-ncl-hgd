.PHONY: install lint type-check

# install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh
install:
	uv sync --link-mode=copy
	.venv/bin/pre-commit install

lint:
	uv run ruff check toolkit/ --fix 

type-check:
	uv run mypy toolkit/**/*.py
