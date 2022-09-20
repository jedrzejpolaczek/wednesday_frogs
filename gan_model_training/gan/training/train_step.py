import tensorflow
from loguru import logger


@tensorflow.function
def train_step(
    images,
    generator,
    discriminator,
    generator_loss,
    discriminator_loss,
    generator_optimizer,
    discriminator_optimizer,
    batch_size,
    noise_dim
):
    """ TODO: add docstring"""
    noise = tensorflow.random.normal([batch_size, noise_dim])

    with tensorflow.GradientTape() as generator_tape, tensorflow.GradientTape() as discriminator_tape:
        generated_images = generator(noise, training=True)
        logger.debug(f"Generated images: {generated_images}")

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        logger.debug(f"Calculate outputs. \nREAL: {real_output} \nFAKE: {fake_output}")

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
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
