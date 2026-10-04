.PHONY: setup test lint clean

setup:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
