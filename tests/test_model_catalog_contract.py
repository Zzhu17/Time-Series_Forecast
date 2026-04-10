import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes.models import list_models  # noqa: E402
from models.registry import SUPPORTED_MODELS  # noqa: E402
from services.model_service import list_model_catalog  # noqa: E402


def test_catalog_is_generated_from_supported_models():
    names_from_constant = {str(item["name"]).lower() for item in SUPPORTED_MODELS}
    names_from_catalog = {str(item["name"]).lower() for item in list_model_catalog()}
    assert names_from_catalog == names_from_constant


def test_catalog_models_have_executable_path():
    catalog = list_model_catalog()
    for item in catalog:
        assert (
            bool(item.get("buildable")) or bool(item.get("trainable")) or bool(item.get("forecastable"))
        ), f"model '{item.get('name')}' must map to at least one executable path"


def test_listed_models_are_resolvable_in_catalog():
    catalog = list_model_catalog()
    names_from_catalog = {str(item["name"]).strip().lower() for item in catalog}
    listed_names = {
        str(item["name"]).strip().lower()
        for item in SUPPORTED_MODELS
        if bool(item.get("listed", True))
    }
    assert listed_names.issubset(names_from_catalog)


def test_capability_semantics_are_not_conflicting():
    catalog = list_model_catalog()
    for item in catalog:
        name = str(item.get("name") or "")
        buildable = bool(item.get("buildable"))
        trainable = bool(item.get("trainable"))
        forecastable = bool(item.get("forecastable"))
        assert buildable or trainable or forecastable, f"model '{name}' must expose at least one capability"
        if buildable:
            assert trainable or forecastable, f"buildable model '{name}' must be trainable or forecastable"
        if trainable:
            assert forecastable, f"trainable model '{name}' must be forecastable"


def test_hybrid_model_names_follow_convention():
    hybrid_pattern = re.compile(r"^[a-z0-9_]+\+[a-z0-9_]+$")
    for item in SUPPORTED_MODELS:
        name = str(item.get("name") or "").strip().lower()
        if "+" in name:
            assert hybrid_pattern.match(name), (
                f"hybrid model name '{name}' must follow '<residual_model>+<base_model>' convention, "
                "e.g. 'xgboost+informer'"
            )


def test_models_endpoint_exposes_capability_flags():
    models = [item.dict() for item in list_models()]
    assert models

    for item in models:
        assert "listed" in item
        assert "trainable" in item
        assert "buildable" in item
