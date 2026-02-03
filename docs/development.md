# Development

## Setup
```bash
pip install -r Project/requirements.txt
pip install -r requirements-dev.txt
```

## Local Run
```bash
PYTHONPATH=Project uvicorn api.app:app --reload --port 8000
streamlit run Project/app.py
```

## Tests
```bash
PYTHONPATH=Project pytest -q tests
```

## Lint and Format
```bash
ruff check .
ruff format .
```
