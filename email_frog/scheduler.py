import tensorflow
from loguru import logger

# Local imports
from email_frog.sender import send_email
from utils import (is_it_wednesday, generate_images, save_image)


def send_email_on_wednesday(generator: tensorflow.keras.Model) -> None:
    """ 
    Send email with generated image on each Wednesday. 
    
    generator (tensorflow.keras.Model): model of generator.
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
