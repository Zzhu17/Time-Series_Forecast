from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def assert_7tuple_contract(out: Any, model_name: str) -> None:
    assert isinstance(out, tuple), f"{model_name}: trainer output must be tuple"
    assert len(out) == 7, f"{model_name}: trainer output must be 7-tuple"

    val_true, val_pred, test_true, test_pred, model, _test_df, best_params = out
    assert model is not None, f"{model_name}: model should not be None"
    assert isinstance(best_params, (dict, list, tuple, type(None)))

    if val_true is not None and val_pred is not None:
        assert len(val_true) == len(val_pred), f"{model_name}: val y_true/yhat mismatch"
    if test_true is not None and test_pred is not None:
        assert len(test_true) == len(test_pred), f"{model_name}: test y_true/yhat mismatch"
