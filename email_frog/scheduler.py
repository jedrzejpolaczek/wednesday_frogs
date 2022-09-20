from loguru import logger

from email_frog.sender import send_email
from utils import (is_it_wednesday, generate_images, save_image)


# TODO: add typing
def send_email_on_wednesday(generator) -> None:
    """ 
    Send email with generated image on each Wednesday. 
    
    model_dir_path (str): path to saved model checkpoint.
    """
    if is_it_wednesday():
        logger.info("It is Wednesday!")

        logger.info("Generating new frog.")
        frog_images = generate_images(generator)

        logger.info("Saving frog image on hard drive.")
        save_image(frog_images)

        logger.info("Beginning procedure of sending emails.")
        send_email()

    else:
        logger.info("It is not Wednesday.")
        logger.info(" :( ")
