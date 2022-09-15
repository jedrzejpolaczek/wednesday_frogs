import tensorflow

def get_frogs(height, width, channels):
    (x_train, y_train), (_, _) = tensorflow.keras.datasets.cifar10.load_data()

    x_train = x_train[y_train.flatten() == 6]

    x_train = x_train.reshape(
        (x_train.shape[0],) +
        (height, width, channels)
    ).astype('float32') / 255

    return x_train, y_train