from loguru import logger
import time

# Local imports
from model.dcgan.training.train_step import train_step
from utils import (generate_images, save_image)


# TODO: add typing
def train(
    dataset, 
    epochs,
    generator,
    discriminator,
    get_generator_loss,
    get_discriminator_loss,
    generator_optimizer,
    discriminator_optimizer,
    model_save_dir,
    images_save_dir,
    seed,
    batch_size,
    noise_dim
):
    """ TODO: add docstring"""
    for epoch in range(epochs):
        start = time.time()

        # Train model for each image in dataset
        # TODO: make it nicer to eye
        for image_batch in dataset:
            train_step(
                image_batch,
                generator,
                discriminator,
                get_generator_loss,
                get_discriminator_loss,
                generator_optimizer,
                discriminator_optimizer,
                batch_size,
                noise_dim
            )

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            generator.save(f"{model_save_dir}.h5")
            
            # Produce images for each 10 epochs just for us
            images = generate_images(generator, seed)
            save_image(images_save_dir, f"frog_in_training_{epoch}", images)
 
        logger.info(f"Time for epoch {epoch + 1} is {time.time()-start} sec. Total progress {int((epoch*100)/epochs)}%")

    # Generate image after the final epoch
    images = generate_images(generator, seed)
    save_image("", "WEDNESDAY_FROG", images)
