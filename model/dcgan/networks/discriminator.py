import keras
from keras import layers
from loguru import logger
import tensorflow

# ---------------------
# --- DISCRIMINATOR ---
# ---------------------

def create_discriminator(height: int, width: int, channels: int):
    """ 
    Simple CNN image classifier.
    
    height (int): TODO add description
    width (int): TODO add description
    channels (int): TODO add description
    """
    # DISCRIMINATOR MODEL DECLARATION
    model = tensorflow.keras.Sequential()

    # DISCRIMINATOR MODEL DEFINITION
    # INPUT LAYER
    model.add(keras.Input(shape=(height, width, channels)))

    # HIDDEN LAYERS
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())

    # OUTPUT LAYER
    model.add(layers.Dense(1, activation='sigmoid'))

    # DISCRIMINATOR MODEL OPTIMIZATION
    # Optimization function is returning by function discriminator_optimizer. 
    # We will add it later when we will be putting everything together to create GAN.
    
    # DISCRIMINATOR MODEL COMPILATION
    # We will compile all models at once

    logger.debug(f"Discriminator network: \n {model.summary()}")

    return model


# TODO: add typing
def get_discriminator_loss(real_output, fake_output):
    """ TODO: add docstring """
    cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tensorflow.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tensorflow.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss


def get_discriminator_optimizer():
    """ TODO: add docstring """
    return tensorflow.keras.optimizers.Adam(1e-4)