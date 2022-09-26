import os
import tensorflow
from loguru import logger

# Local imports
from model.dcgan.networks.generator import (create_generator, get_generator_loss, get_generator_optimizer)
from model.dcgan.networks.discriminator import (create_discriminator, get_discriminator_loss, get_discriminator_optimizer)
from model.dcgan.training.train import train
from model.datasets.frogs.cifar10.get_frogs_dataset import get_dataset
from utils import get_json_data


def run_model_training():
    """ TODO: add docstring """
    # First fings first, lets check if we will be using GPU
    if tensorflow.test.gpu_device_name(): 
        logger.debug('Default GPU Device:{}'.format(tensorflow.test.gpu_device_name()))
    else:
        logger.debug("Please install GPU version of tensorflow!")

    # -----------------------
    logger.info("LOADING VARIABLES")
    # TODO: make it nicer to eye
    training_data = get_json_data('training_config.json')

    noise_dim = training_data["noise_dim"]
    height = training_data["height"]
    width = training_data["width"]
    channels = training_data["channels"]

    epochs = training_data["iterations"]
    buffer_size = training_data["buffer_size"]
    batch_size = training_data["batch_size"]
    images_save_dir = training_data["images_save_dir"]
    model_save_dir = training_data["model_save_dir"]
    examples_to_generate = training_data["examples_to_generate"]
    seed=tensorflow.random.normal([examples_to_generate, noise_dim])

    # -----------------------
    logger.info("GENERATOR DEFINITION")
    generator = create_generator(noise_dim)
    gen_optimizer = get_generator_optimizer()

    logger.info("DISCRIMINATOR DEFINITION")
    discriminator = create_discriminator(height, width, channels)
    disc_optimizer = get_discriminator_optimizer()

    # -----------------------
    logger.info("PREPARING DATASET")
    train_dataset = get_dataset(height, width, channels, buffer_size, batch_size)

    # -----------------------
    logger.info("CREATING CHECKPOINT")
    checkpoint = tensorflow.train.Checkpoint(
        generator_optimizer=gen_optimizer,
        discriminator_optimizer=disc_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    # -----------------------
    logger.info("TRAINING NETWORKS")

    # TODO: make it nicer to eye
    train(
        train_dataset, 
        epochs,
        generator,
        discriminator,
        get_generator_loss,
        get_discriminator_loss,
        gen_optimizer,
        disc_optimizer,
        model_save_dir,
        images_save_dir,
        seed,
        batch_size,
        noise_dim
    )
