import keras
from keras import layers
from loguru import logger
import tensorflow

# -----------------
# --- GENERATOR ---
# -----------------

def create_generator(noise_dim: int, height: int, width: int, channels: int):
    """
    The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). 
    Start with a Dense layer that takes this seed as input, 
    then upsample several times until you reach the desired image size of 32x32x3. 
    Warning: The tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh.

    Note: description from tensorflow GAN example.

    noise_dim (int): your input shape has only one dimension, you don't need to give it as a tuple, you give input_dim as a scalar number.
    height (int): TODO add description
    width (int): TODO add description
    channels (int): TODO add description
    """
    # GENERATOR MODEL DECLARATION
    model = tensorflow.keras.Sequential()

    # GENERATOR MODEL DEFINITION
    # INPUT LAYER
    # model.add(keras.Input(shape=(noise_dim,)))

    # HIDDEN LAYERS
    # model.add(layers.Dense(height * width * channels))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Reshape((height, width, channels)))

    # model.add(layers.Conv2DTranspose(256, 5, padding='same'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Conv2D(124, 4, strides=(1, 1), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Conv2DTranspose(64, 5, strides=(1, 1), padding='same', use_bias=False))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Conv2DTranspose(32, 5, strides=(1, 1), padding='same', use_bias=False))
    # model.add(layers.LeakyReLU())
    # model.add(layers.BatchNormalization())

    # OUTPUT LAYER
    # model.add(layers.Conv2DTranspose(channels, 7, activation='tanh', padding='same'))  # TODO: Fixme: 7 is amagic number
    # assert model.output_shape == (None, height, width, channels)
    
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(layers.Dense(n_nodes, input_dim=noise_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    model.add(layers.Conv2D(3, (3,3), activation='tanh', padding='same'))

    # GENERATOR MODEL OPTIMIZATION
    # Optimization function is returning by function generator_optimizer. 
    # We will add it later when we will be putting everything together to create GAN.
    
    # GENERATOR MODEL COMPILATION
    # We will compile all models at once

    logger.debug(f"Generator network: \n {model.summary()}")

    return model


# TODO: add typing
def get_generator_loss(fake_output):
    """
    TODO: add docstring
    
    fake output (TODO add type): TODO add description
    """
    cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy(tensorflow.ones_like(fake_output), fake_output)


def get_generator_optimizer():
    """ TODO: add docstring """
    return tensorflow.keras.optimizers.Adam(1e-4)
