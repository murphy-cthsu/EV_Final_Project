import torch
import pytest


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


@pytest.fixture
def device():
    return torch.device("cpu")
