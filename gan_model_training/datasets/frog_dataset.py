import os
import tensorflow

from PIL import Image
from loguru import logger

def get_frogs(height, width, channels, images_dir_path='gan_model_training\\datasets\\frog-dataset\\data-64'):
    # list to store files
    x_train_images = []

    # Iterate directory
    for path in os.listdir(images_dir_path):
        # check if current path is a file
        image_dir = os.path.join(images_dir_path, path)
        if os.path.isfile(image_dir):
            logger.info("Processing image " + path)
            image = Image.open(image_dir)
            image_array = tensorflow.keras.utils.img_to_array(image, data_format=None, dtype=None)
            x_train_images.append(image_array)
    
    return x_train_images
