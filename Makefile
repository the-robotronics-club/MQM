.PHONY: install test test-live lint typecheck format-check dev services services-down

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements-dev.txt

test:
	.venv/bin/python -m pytest -q

test-live:
	RUN_LIVE_STACK=1 .venv/bin/python -m pytest tests/test_live_stack.py -q

lint:
	.venv/bin/ruff check api dashboard inference research tests tools

typecheck:
	.venv/bin/mypy

format-check:
	.venv/bin/ruff format --check api dashboard inference research tests tools

services:
	docker compose up -d --wait

services-down:
	docker compose down

dev:
	./scripts/dev.sh
