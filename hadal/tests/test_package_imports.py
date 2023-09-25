import pytest


def test_hadal_import():
    """Test if import `import hadal` works."""
    try:
        import hadal  # noqa: F401
    except ImportError:
        pytest.fail("hadal can't be imported!")


def test_hadal_imports():
    """Test if imports `from hadal import *` work."""
    try:
        from hadal import HuggingfaceAutoModel  # noqa: F401, I001
        from hadal import FaissSearch  # noqa: F401
        from hadal import MarginBasedPipeline  # noqa: F401
        from hadal import MarginBased  # noqa: F401
    except ImportError:
        pytest.fail("hadal can't be imported!")
