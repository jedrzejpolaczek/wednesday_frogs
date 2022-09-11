import keras
from keras import layers

# -----------------
# --- GENERATOR ---
# -----------------

def create_generator(latent_dim, channels):
    # INPUT LAYER
    generator_input = keras.Input(shape=(latent_dim,))

    # HIDDEN LAYERS
    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((32, 32, 32))(x)

    x = layers.Conv2D(256, 5, padding='same')(x)
    X = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 4, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # OUTPUT LAYER
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

    # GENERATOR MODEL DECLARATION
    generator = keras.models.Model(generator_input, x)

    # GENERATOR MODEL OPTIMIZATION
    # N/A
    # GENERATOR MODEL COMPILATION
    # N/A

    return generator
