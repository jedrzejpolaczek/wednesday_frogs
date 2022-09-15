import keras
from keras import layers
from loguru import logger

# -----------------
# --- GENERATOR ---
# -----------------

def create_generator(latent_dim, height, width, channels):
    # INPUT LAYER
    generator_input = keras.Input(shape=(latent_dim,))

    # HIDDEN LAYERS
    x = layers.Dense(height * width * channels)(generator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((height, width, channels))(x)
    # assert x.shape == (None, 224, 224, 32)  # Note: None is the batch size

    x = layers.Conv2DTranspose(256, 5, padding='same')(x)
    # TODO: add assert x.output_shape == (None, 224, 224, 32)
    x = layers.BatchNormalization()(x)
    X = layers.LeakyReLU()(x)

    x = layers.Conv2D(124, 4, strides=(1, 1), padding='same', use_bias=False)(x)
    # TODO: add assert x.output_shape == (None, 224, 224, 32)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, 5, strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(32, 5, strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    # TODO: add assert x.output_shape == (None, 224, 224, 32)
    x = layers.BatchNormalization()(x)

    # OUTPUT LAYER
    x = layers.Conv2DTranspose(channels, 7, activation='tanh', padding='same')(x)
    # TODO: add assert x.output_shape == (None, 224, 224, 32)

    # GENERATOR MODEL DECLARATION
    generator = keras.models.Model(generator_input, x)
    
    # GENERATOR MODEL OPTIMIZATION
    # N/A
    # GENERATOR MODEL COMPILATION
    # N/A

    logger.info(generator.summary())

    return generator
