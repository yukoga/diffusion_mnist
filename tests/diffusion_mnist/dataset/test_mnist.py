import pytest
from diffusion_mnist.dataset.mnist import Dataset, MNISTDataLoader


def test_mnist_dataset():
    dataset = Dataset()
    mnist_dataset = dataset()
    assert len(mnist_dataset) > 0
    assert mnist_dataset[0][0].shape == (784,)


def test_mnist_dataloader():
    dataloader = MNISTDataLoader()
    mnist_dataloader = dataloader()
    batch = next(iter(mnist_dataloader))
    assert len(mnist_dataloader) > 0
    assert batch[0].shape == (128, 784)
