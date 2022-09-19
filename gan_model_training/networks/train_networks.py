# import numpy as np
import tensorflow
import os
# from datetime import datetime
from loguru import logger
import time
# from IPython import display
import matplotlib.pyplot as plt


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
    noise = tensorflow.random.normal([batch_size, noise_dim])

    with tensorflow.GradientTape() as generator_tape, tensorflow.GradientTape() as discriminator_tape:
        logger.debug("Generate image for step.")
        generated_images = generator(noise, training=True)
        logger.debug(f"Generated images: {generated_images}")

        logger.debug("Calculate outputs.")
        real_output = discriminator(images, training=True)
        logger.debug(f"Real output: {real_output}")
        fake_output = discriminator(generated_images, training=True)
        logger.debug(f"Fake output: {fake_output}")

        logger.debug("Calculate losses.")
        gen_loss = generator_loss(fake_output)
        logger.debug(f"Generator loss: {gen_loss}")
        disc_loss = discriminator_loss(real_output, fake_output)
        logger.debug(f"Discriminator loss: {disc_loss}")

    logger.debug("Calculate gradients.")
    gradients_of_generator = generator_tape.gradient(gen_loss, generator.trainable_variables)
    logger.debug(f"Gradients of generator: {gradients_of_generator}")
    gradients_of_discriminator = discriminator_tape.gradient(disc_loss, discriminator.trainable_variables)
    logger.debug(f"Gradients of discriminator: {gradients_of_discriminator}")

    logger.debug("Appling gradients to networks.")
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


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train(
    dataset, 
    epochs,
    generator,
    discriminator,
    generator_loss,
    discriminator_loss,
    generator_optimizer,
    discriminator_optimizer,
    checkpoint,
    checkpoint_prefix,
    seed,
    batch_size,
    noise_dim
):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(
                image_batch,
                generator,
                discriminator,
                generator_loss,
                discriminator_loss,
                generator_optimizer,
                discriminator_optimizer,
                batch_size,
                noise_dim
            )

        # Produce images for the GIF as you go
        # display.clear_output(wait=True)
        # generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        logger.info('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                         epochs,
    #                         seed)
    predictions = generator(seed, training=False)
    img = tensorflow.keras.utils.array_to_img(predictions[0] * 255., scale=False)
    img.save(os.path.join('', 'WEDNESDAY_FROG.png'))
