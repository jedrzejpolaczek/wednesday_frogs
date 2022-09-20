from loguru import logger
import time

from gan.training.train_step import train_step
from gan.utils.generate_and_save_images import generate_and_save_images


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
    """ TODO: add docstring"""
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

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            # Produce images for each 10 epochs just for us
            generate_and_save_images(generator, seed, checkpoint_prefix, f"frog_in_training_{epoch}")
 
        logger.info(f"Time for epoch {epoch + 1} is {time.time()-start} sec. Total progress {int((epoch*100)/epochs)}%")

    # Generate after the final epoch
    generate_and_save_images(generator, seed, "", "WEDNESDAY_FROG")