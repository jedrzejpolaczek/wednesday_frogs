import keras
from loguru import logger

# -----------
# --- GAN ---
# -----------

def create_gan(discriminator, generator, noise_dim):
    discriminator.trainable = False

    # INPUT LAYER
    gan_input = keras.Input(shape=(noise_dim,))

    # HIDDEN LAYERS
    pass

    # OUTPUT LAYER
    gan_output = discriminator(generator(gan_input))

    # GAN MODEL DECLARATION
    gan = keras.models.Model(gan_input, gan_output)

    # GAN MODEL OPTIMIZATION
    gan_optimizer = keras.optimizers.RMSprop(
        lr=0.0004,
        clipvalue=1.0,
        decay=1e-8
    )

    # GAN MODEL COMPILATION
    gan.compile(
        optimizer=gan_optimizer,
        loss='binary_crossentropy'
    )

    logger.info(discriminator.summary())

    return gan
