from __future__ import annotations

import importlib
import sys
from types import ModuleType


def load_pipeline_module(module_name: str = "services.pipeline") -> ModuleType:
    """
    Always return a fresh `services.pipeline` module.
    Streamlit reruns can keep module cache; we reload explicitly.
    """
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        return sys.modules[module_name]
    return importlib.import_module(module_name)

