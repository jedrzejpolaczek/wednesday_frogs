import tensorflow


def get_frogs(height: int, width: int, channels: int) -> tuple:
    """
    Download cifar10 dataset, extraxt only frogs from there and reshape them to get images as numpy.ndarray of (5000, 32, 32, 3).

    height (int): value represented hight of image measured in pixels.
    width (int): value represented width of image measured in pixels.
    channels (int): number of possible colorues in image.

    return (numpy.ndarray): array of size (5000, 32, 32, 3) as 5000 images of size 32x32 pixels with 3 colours.
    """
    (x_train, y_train), (_, _) = tensorflow.keras.datasets.cifar10.load_data()

    x_train = x_train[y_train.flatten() == 6]

    x_train = x_train.reshape(
        (x_train.shape[0],) +
        (height, width, channels)
    ).astype('float32') / 255

    return x_train, y_train
