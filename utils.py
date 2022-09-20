from loguru import logger
import json
import os
import tensorflow
from datetime import datetime
from xmlrpc.client import Boolean


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


def generate_images(model, test_input):
    """ TODO: add docstring"""
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictited_images = model(test_input, training=False)

    return predictited_images


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
