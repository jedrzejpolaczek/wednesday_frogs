import keras
from keras import layers
from loguru import logger

# ---------------------
# --- DISCRIMINATOR ---
# ---------------------

def create_discriminator(height, width, channels):
    """ Simple CNN image classifier."""
    # INPUT LAYER
    discriminator_input = layers.Input(shape=(height, width, channels))

    # HIDDEN LAYERS
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, 4, strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(32, 4, strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(16, 4, strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)

    # OUTPUT LAYER
    x = layers.Dense(1, activation='sigmoid')(x)

    # DISCRIMINATOR MODEL DECLARATION
    discriminator = keras.models.Model(discriminator_input, x)

    # DISCRIMINATOR MODEL OPTIMIZATION
    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008,
        clipvalue=1.0,
        decay=1e-8
    )

    # DISCRIMINATOR MODEL COMPILATION
    discriminator.compile(
        optimizer=discriminator_optimizer,
        loss='binary_crossentropy'
    )

    logger.info(discriminator.summary())

    return discriminator
