import tensorflow
from loguru import logger

from gan.networks.generator import (create_generator, generator_loss, generator_optimizer)
from gan.networks.discriminator import (create_discriminator, discriminator_loss, discriminator_optimizer)
from gan.train import train
from datasets.cifar10_frogs import get_frogs
# TODO: fixme from utils.configuration import get_json_data


def run_training():
    # First fings first, lets check if we will be using GPU
    if tensorflow.test.gpu_device_name(): 
        logger.debug('Default GPU Device:{}'.format(tensorflow.test.gpu_device_name()))
    else:
        logger.debug("Please install GPU version of tensorflow!")

    # -----------------------
    logger.info("LOADING VARIABLES")
    # TODO: fixme training_data = get_json_data('training_config.json')

    logger.info("Opening JSON file.")
    json_file = open('training_config.json')

    import json    
    logger.info("Returns JSON object as a dictionary.")
    training_data = json.load(json_file)
    #TODO: UP-------------------------------------

    noise_dim = training_data["latent_dim"]
    height = training_data["height"]
    width = training_data["width"]
    channels = training_data["channels"]

    epochs = training_data["iterations"]
    buffer_size = 10  # TODO: move it to JSON
    batch_size = training_data["batch_size"]
    save_dir = training_data["save_dir"]
    # start = training_data["start"]
    num_examples_to_generate = 16
    seed=tensorflow.random.normal([num_examples_to_generate, noise_dim])

    # -----------------------
    logger.info("GENERATOR DEFINITION")
    generator = create_generator(noise_dim, height, width, channels)
    gen_optimizer = generator_optimizer()

    logger.info("DISCRIMINATOR DEFINITION")
    discriminator = create_discriminator(height, width, channels)
    disc_optimizer = discriminator_optimizer()

    # logger.info("GAN DEFINITION")
    # gan = create_gan(discriminator, generator, noise_dim)

    # -----------------------
    logger.info("PREPARING DATASET")
    x_train, _ = get_frogs(height, width, channels)
    train_dataset = tensorflow.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)

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

    train(
        train_dataset, 
        epochs,
        generator,
        discriminator,
        generator_loss,
        discriminator_loss,
        gen_optimizer,
        disc_optimizer,
        checkpoint,
        save_dir,
        seed,
        batch_size,
        noise_dim
    )
