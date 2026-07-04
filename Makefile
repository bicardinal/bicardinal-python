.PHONY: help install dev clean format pybuild upload-pypi test

PYTHON := python3


help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make dev          - Run development server"
	@echo "  make clean        - Clean up cache and temp files"
	@echo "  make format       - Format code content"
	@echo "  make upload-pypi  - Update pypi package"
	@echo "  make pybuild      - Python build"
	@echo "  make test         - Test"

install:
	pip install -e .

dev:
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -r wheelhouse

format:
	autoflake bicardinal --remove-all-unused-imports --quiet --in-place -r --exclude third_party
	isort bicardinal --force-single-line-imports
	black bicardinal
	autoflake tests --remove-all-unused-imports --quiet --in-place -r --exclude third_party
	isort tests --force-single-line-imports
	black tests

pybuild:
	rm -rf dist
	python -m build
	twine check dist/*

upload-testpypi:
	python -m twine upload -r testpypi dist/* --verbose

upload-pypi:
	python -m twine upload dist/* --verbose

test:
	pytest .
