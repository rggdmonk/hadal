"""Tools for testing."""
import pytest
import torch


def prepare_test_on_devices(test_devices: list[str] | None = None) -> pytest.mark.parametrize:
    """Get a pytest.mark.parametrize decorator for testing on test_devices."""
    if test_devices is None:
        test_devices = ["cpu", "cuda"]
    if torch.cuda.is_available() is False:
        test_devices.remove("cuda")

    return pytest.mark.parametrize("test_device", [torch.device(device) for device in test_devices], ids=test_devices)


pytest_test_on_devices = prepare_test_on_devices()
