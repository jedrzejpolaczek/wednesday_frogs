import tensorflow
from loguru import logger

# Local imports
from model.dcgan.networks.generator import (create_generator, get_generator_loss, get_generator_optimizer)
from model.dcgan.networks.discriminator import (create_discriminator, get_discriminator_loss, get_discriminator_optimizer)
from model.dcgan.training.train import train
from model.datasets.frogs.cifar10.get_frogs_dataset import get_dataset


def run_model_training():
    """ Train model and save it to .h5 file. """
    
    # First fings first, lets check if we will be using GPU
    if tensorflow.test.gpu_device_name(): 
        logger.debug('Default GPU Device:{}'.format(tensorflow.test.gpu_device_name()))
    else:
        logger.debug("Please install GPU version of tensorflow!")

    # -----------------------
    logger.info("GENERATOR DEFINITION")
    generator = create_generator()
    gen_optimizer = get_generator_optimizer()

    logger.info("DISCRIMINATOR DEFINITION")
    discriminator = create_discriminator()
    disc_optimizer = get_discriminator_optimizer()

    # -----------------------
    logger.info("PREPARING DATASET")
    train_dataset = get_dataset()

    # -----------------------
    logger.info("TRAINING NETWORKS")
    train(
        train_dataset, 
        generator,
        discriminator,
        get_generator_loss,
        get_discriminator_loss,
        gen_optimizer,
        disc_optimizer,
    )
