.PHONY: install install-dev test lint format typecheck run-api run-ui docker-up

install:
	pip install -r Project/requirements.txt

install-dev:
	pip install -r Project/requirements.txt
	pip install -r requirements-dev.txt

test:
	PYTHONPATH=Project pytest -q tests

lint:
	ruff check .

format:
	ruff format .

typecheck:
	PYTHONPATH=Project mypy Project tests

run-api:
	PYTHONPATH=Project uvicorn api.app:app --reload --port 8000

run-ui:
	streamlit run Project/app.py

docker-up:
	docker compose up --build
