from typing import Callable

import tensorflow
from loguru import logger

# Local imports
from utils import get_json_data


@tensorflow.function  # Making it compiled
def train_step(
    images: tensorflow.Tensor,
    generator: tensorflow.keras.Model,
    discriminator: tensorflow.keras.Model,
    get_generator_loss: Callable,
    get_discriminator_loss: Callable,
    generator_optimizer: Callable,
    discriminator_optimizer: Callable,
) -> None:
    """
    Train networks by getting their output, calculating their losses, gradients and update weights.

    images (tensorflow.Tensor): tensor of images.
    generator (tensorflow.keras.Model): model of generator.
    discriminator (tensorflow.keras.Model): model of discriminator.
    get_generator_loss (Callable): function to calculate loss for generator.
    get_discriminator_loss (Callable): function to calculate loss for discriminator.
    generator_optimizer (Callable): function to get instation of optimizer for generator.
    discriminator_optimizer (Callable): function to get instation of optimizer for discriminator.
    """
    logger.debug("Load variables.")
    training_data = get_json_data('training_config.json')

    batch_size = training_data["batch_size"]
    noise_dim = training_data["noise_dim"]
    noise = tensorflow.random.normal([batch_size, noise_dim])

    with tensorflow.GradientTape() as generator_tape, tensorflow.GradientTape() as discriminator_tape:
        generated_images = generator(noise, training=True)
        logger.debug(f"Generated images: {generated_images}")
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        logger.debug(f"Calculate outputs. \nREAL: {real_output} \nFAKE: {fake_output}")

        gen_loss = get_generator_loss(fake_output)
        disc_loss = get_discriminator_loss(real_output, fake_output)
        logger.debug(f"Calculate losses. \nGENERATOR: {gen_loss} \nDISCRIMINATOR: {disc_loss}")

    gradients_of_generator = generator_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = discriminator_tape.gradient(disc_loss, discriminator.trainable_variables)
    logger.debug(f"Calculate gradients. \nGENERATOR: {gradients_of_generator} \nDISCRIMINATOR: {gradients_of_discriminator}")

    generator_optimizer.apply_gradients(
        zip(
            gradients_of_generator, 
            generator.trainable_variables
        )
    )
    discriminator_optimizer.apply_gradients(
        zip(
            gradients_of_discriminator, 
            discriminator.trainable_variables
        )
    )
