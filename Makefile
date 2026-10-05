.PHONY: setup test lint build publish-test publish clean

VERSION := $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' src/mlx_decision/__init__.py)

setup:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

build:
	rm -rf dist
	python -m build
	twine check --strict dist/*

publish-test: build
	@read -p "Upload mlx-decision $(VERSION) to TestPyPI? [y/N] " answer && [ "$$answer" = y ]
	twine upload --repository testpypi dist/*

publish: build
	@git diff --quiet HEAD || { echo "uncommitted changes: commit them before publishing"; exit 1; }
	@read -p "Upload mlx-decision $(VERSION) to PyPI? This cannot be undone. [y/N] " answer && [ "$$answer" = y ]
	twine upload --repository pypi dist/*

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
