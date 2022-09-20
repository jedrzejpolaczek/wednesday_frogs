import tensorflow
from loguru import logger

from email_frog.sender import send_email
from utils import (generate_images, save_image, is_it_wednesday)


def send_email_on_wednesday(model_dir_path: str) -> None:
    """ 
    Send email with generated image on each Wednesday. 
    
    model_dir_path (str): path to saved model checkpoint.
    """
    if is_it_wednesday():
        logger.info("It is Wednesday!")

        # TODO: fix thing below in comment
        # logger.info("Loading model.")
        # # I am keeping the loading of the model here due to potential model update reasons
        # generator = tensorflow.keras.models.load_model(model_dir_path)

        # logger.info("Generating new frog.")
        # frog_images = generate_images(generator, )

        # logger.info("Saving frog image on hard drive.")
        # save_image(frog_images)

        logger.info("Beginning procedure of sending emails.")
        send_email()

    else:
        logger.info("It is not Wednesday.")
        logger.info(" :( ")
