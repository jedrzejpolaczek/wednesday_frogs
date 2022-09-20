import tensorflow
from model.datasets.frogs.cifar10.cifar10_frogs import get_frogs


def get_dataset(height: int, width: int, channels: int, buffer_size: int, batch_size: int) -> tensorflow.data.Dataset:
    """ TODO: add docstring """
    x_train, _ = get_frogs(height, width, channels)

    dataset = tensorflow.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)

    return dataset
