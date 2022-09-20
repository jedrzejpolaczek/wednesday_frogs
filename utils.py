from loguru import logger
import json
import os
import tensorflow
from datetime import datetime
from xmlrpc.client import Boolean
from model.networks.generator import create_generator


def get_json_data(file_path: str) -> dict:
    """ 
    Read JSON dict from file.

    file_path (str): path to JSON file.
    
    return dict: dict based on read JSON file.
    """
    logger.info("Opening JSON file.")
    json_file = open(file_path)
    
    logger.info("Returns JSON object as a dictionary.")
    data = json.load(json_file)

    return data


def generate_images(model, seed, json_path: str="training_config.json"):
    """ TODO: add docstring"""
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    seed = get_seed(json_path)
    predictited_images = model(seed, training=False)

    return predictited_images


def get_seed(json_path: str="training_config.json"):
    model_data = get_json_data(json_path)

    noise_dim = model_data["noise_dim"]
    examples_to_generate = model_data["examples_to_generate"]
    
    seed = tensorflow.random.normal([examples_to_generate, noise_dim])

    return seed


def save_image(save_dir, name, images):
    """ TODO: add docstring"""
    img = tensorflow.keras.utils.array_to_img(images[0] * 255., scale=False)
    img.save(os.path.join(save_dir, f'{name}.png'))


def is_it_wednesday() -> Boolean:
    """ 
    Check if it is Wednesday.
    
    return boolean: True if it is, false if not.
    """
    # If today is Wednesday (0 = Mon, 1 = Tue, 2 = Wen ...)
    is_it_wednesday = datetime.today().weekday() == 2
    logger.debug("Is it Wednesdat? : " + str(is_it_wednesday))
    
    return is_it_wednesday


def load_model(json_path: str="training_config.json") -> tensorflow.Model:
    """ TODO: Add docstring """
    model_data = get_json_data(json_path)

    noise_dim = model_data["noise_dim"]
    height = model_data["height"]
    width = model_data["width"]
    channels = model_data["channels"]
    generator = create_generator(noise_dim, height, width, channels)

    checkpoint_dir = model_data["checkpoint_dir"]
    generator.load_weights(checkpoint_dir)

    return generator
