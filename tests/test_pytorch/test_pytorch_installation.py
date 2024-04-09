import pytest
from ..conftest import plot_pyvista, TEST_SPEED, REQUIREMENT_LEVEL, Requirements

pytestmark = pytest.mark.skipif(REQUIREMENT_LEVEL.value < Requirements.OPTIONAL.value, reason="This test needs higher requirements.")


def test_pytorch_install():
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() > 0