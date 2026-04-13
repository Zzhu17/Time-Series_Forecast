# Contributing

Thanks for contributing! This project targets clean, reproducible forecasting pipelines with a simple API.

## Quickstart

```bash
pip install -r Project/requirements.txt
pip install -r requirements-dev.txt
```

Run the API:

```bash
PYTHONPATH=Project uvicorn api.app:app --reload --port 8000
```

Run Streamlit UI:

```bash
streamlit run Project/app.py
```

## Development Guidelines

- Keep changes small and focused.
- Prefer ASCII and avoid large generated artifacts in git.
- Run tests before PRs:
  ```bash
  PYTHONPATH=Project pytest -q tests
  ```
- Use consistent formatting; avoid reformatting unrelated files.

## Submitting Changes

1. Create a feature branch.
2. Make changes with clear commit messages.
3. Open a PR describing what changed and why.

## Reporting Issues

Include:
- model name and data size
- repro steps
- logs or error messages

