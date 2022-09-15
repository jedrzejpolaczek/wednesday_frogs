import keras
from keras import layers
from loguru import logger

# -----------------
# --- GENERATOR ---
# -----------------

def create_generator(latent_dim, height, width, channels):
    """
    The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). 
    Start with a Dense layer that takes this seed as input, 
    then upsample several times until you reach the desired image size of 28x28x1. 
    Warning: The tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh.

    Note: description from tensorflow GAN example.
    """
    # INPUT LAYER
    generator_input = keras.Input(shape=(latent_dim,))

    # HIDDEN LAYERS
    x = layers.Dense(height * width * channels)(generator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((height, width, channels))(x)

    x = layers.Conv2DTranspose(256, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    X = layers.LeakyReLU()(x)

    x = layers.Conv2D(124, 4, strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, 5, strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(32, 5, strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    # OUTPUT LAYER
    x = layers.Conv2DTranspose(channels, 7, activation='tanh', padding='same')(x)  # Fixme: magic number

    # GENERATOR MODEL DECLARATION
    generator = keras.models.Model(generator_input, x)
    
    # GENERATOR MODEL OPTIMIZATION
    # N/A
    # GENERATOR MODEL COMPILATION
    # N/A

    generator.output_shape == (None, height, width, channels)
    logger.info(generator.summary())

    return generator
