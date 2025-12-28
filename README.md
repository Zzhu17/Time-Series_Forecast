# Time-Series_Forecast

Streamlit-based playground for time-series forecasting.

## Quickstart

```bash
cd Project
pip install -r requirements.txt
streamlit run app.py
```

## Repo hygiene (GitHub-friendly)

This repo generates training outputs locally (models, scalers, plots, snapshots). They are intentionally ignored by git:

- `Project/artifacts/`
- `Project/output/`

If you previously committed large files (e.g. `Project/artifacts/informer_model.pth`), they were removed from tracking and added to `.gitignore`.

### Optional: completely remove large files from git history

If you want to shrink the remote repository (history rewrite required):

```bash
pip install git-filter-repo
git filter-repo --path Project/artifacts/ --invert-paths
git push --force --all
git push --force --tags
```

Alternatively, use Git LFS for `*.pth` / `*.pkl` if you need to version model artifacts.
