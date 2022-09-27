import time
from typing import Callable

import tensorflow
from loguru import logger

# Local imports
from model.dcgan.training.train_step import train_step
from utils import (get_json_data, generate_images, save_image)


def train(
    dataset: tensorflow.data.Dataset, 
    generator: tensorflow.keras.Model,
    discriminator: tensorflow.keras.Model,
    get_generator_loss: Callable,
    get_discriminator_loss: Callable,
    generator_optimizer: Callable,
    discriminator_optimizer: Callable,
):
    """
    Train network for each image in dataset.
    
    dataset (tensorflow.data.Dataset): dataset of images.
    generator (tensorflow.keras.Model): model of generator.
    discriminator (tensorflow.keras.Model): model of discriminator.
    get_generator_loss (Callable): function to calculate loss for generator.
    get_discriminator_loss (Callable): function to calculate loss for discriminator.
    generator_optimizer (Callable): function to get instation of optimizer for generator.
    discriminator_optimizer (Callable): function to get instation of optimizer for discriminator.
    """
    training_data = get_json_data('training_config.json')

    epochs = training_data["iterations"]
    images_save_dir = training_data["images_save_dir"]
    model_save_dir = training_data["model_save_dir"]

    noise_dim = training_data["noise_dim"]
    examples_to_generate = training_data["examples_to_generate"]
    seed = tensorflow.random.normal([examples_to_generate, noise_dim])

    for epoch in range(epochs):
        start = time.time()

        # Train model for each image in dataset
        for image_batch in dataset:
            train_step(
                image_batch,
                generator,
                discriminator,
                get_generator_loss,
                get_discriminator_loss,
                generator_optimizer,
                discriminator_optimizer,
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
