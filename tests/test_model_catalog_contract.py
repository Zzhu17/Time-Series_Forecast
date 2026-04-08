import sys
from pathlib import Path

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


def test_models_endpoint_exposes_capability_flags():
    models = [item.dict() for item in list_models()]
    assert models

    for item in models:
        assert "listed" in item
        assert "trainable" in item
        assert "buildable" in item
