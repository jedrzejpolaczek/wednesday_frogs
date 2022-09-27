import tensorflow

# Local imports
from model.datasets.frogs.cifar10.cifar10_frogs import get_frogs
from utils import get_json_data


def get_dataset() -> tensorflow.data.Dataset:
    """
    Convert numpy.ndarray of images into dataset of images.

    return (tensorflow.data.Dataset): dataset of images
    """
    training_data = get_json_data('training_config.json')

    height = training_data["height"]
    width = training_data["width"]
    channels = training_data["channels"]
    buffer_size = training_data["buffer_size"]
    batch_size = training_data["batch_size"]

    x_train, _ = get_frogs(height, width, channels)

    dataset = tensorflow.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)

    return dataset
