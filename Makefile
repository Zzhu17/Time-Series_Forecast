.PHONY: install install-dev test test-models lint format typecheck run-api run-ui docker-up

install:
	pip install -r Project/requirements.txt

install-dev:
	pip install -r Project/requirements.txt
	pip install -r requirements-dev.txt

test:
	PYTHONPATH=Project pytest -q tests

test-models:
	PYTHONPATH=Project pytest -q \
		tests/test_model_xgboost.py \
		tests/test_model_randomforest.py \
		tests/test_model_arima.py \
		tests/test_model_prophet.py \
		tests/test_model_lstm.py \
		tests/test_model_informer.py

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
