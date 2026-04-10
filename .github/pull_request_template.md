## Summary

- 

## Validation

- [ ] Tests added/updated for behavior changes
- [ ] Relevant docs updated (`README.md` / `docs/*`)

## Registry Change Checklist (if `Project/models/registry.py` changed)

- [ ] `SUPPORTED_MODELS` listed models are resolvable in `list_model_catalog()`
- [ ] `trainable` / `buildable` / `forecastable` semantics are internally consistent
- [ ] Hybrid naming follows `<residual_model>+<base_model>` (e.g., `xgboost+informer`)
- [ ] Backward compatibility paths remain valid (if any)
