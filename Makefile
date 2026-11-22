.PHONY: install check lint test

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest -q

check: lint test
