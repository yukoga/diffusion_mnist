import pytest
import torch
from diffusion_mnist.model.diffusion_model import DiffusionModel


@pytest.fixture
def sample_input_size():
    return 784


@pytest.fixture
def sample_hidden_size():
    return 128


@pytest.fixture
def sample_batch_size():
    return 32


@pytest.fixture
def sample_x_t(sample_batch_size, sample_input_size):
    return torch.randn(sample_batch_size, sample_input_size)


@pytest.fixture
def sample_t(sample_batch_size):
    return torch.randint(0, 1000, (sample_batch_size,))  # Example timesteps


def test_diffusion_model_init(sample_input_size, sample_hidden_size):
    model = DiffusionModel(
        sample_input_size, sample_hidden_size)
    assert isinstance(model.linear1, torch.nn.Linear)
    assert isinstance(model.linear2, torch.nn.Linear)
    assert isinstance(model.linear3, torch.nn.Linear)
    assert model.linear1.in_features == sample_input_size + sample_hidden_size
    assert model.linear1.out_features == sample_hidden_size
    assert model.linear2.in_features == sample_hidden_size
    assert model.linear2.out_features == sample_hidden_size
    assert model.linear3.out_features == sample_input_size


def test_diffusion_model_forward(
    sample_input_size,
    sample_hidden_size,
    sample_x_t,
    sample_t,
    sample_batch_size
):
    model = DiffusionModel(sample_input_size, sample_hidden_size)
    output = model(sample_x_t, sample_t)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (sample_batch_size, sample_input_size)
