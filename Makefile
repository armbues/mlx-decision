.PHONY: setup clean

setup:
	pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
