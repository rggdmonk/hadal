import importlib

import pytest


@pytest.mark.parametrize("module_name", ["faiss", "numpy", "torch", "transformers"])
def test_basic_imports(module_name):
    """Test if imports `import <module_name>` work."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} can't be imported!")
