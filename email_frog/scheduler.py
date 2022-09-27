from loguru import logger

# Local imports
from email_frog.sender import send_email
from utils import (is_it_wednesday, generate_images, save_image, get_seed, load_model)


def send_email_on_wednesday(image_name: str) -> None:
    """ 
    Send email with generated image on each Wednesday. 
    
    image_name (str): name for generated frog image.
    """
    if is_it_wednesday():
        logger.info("It is Wednesday!")

        logger.debug("Loading model.")
        model = load_model()

        logger.debug("Generate seed.")
        seed = get_seed()

        logger.debug("Generate image.")
        images = generate_images(model, seed)

        logger.debug("Save image.")
        save_image(save_dir="", name=image_name, images=images)

        logger.debug("Sending emails...")
        send_email()

    else:
        logger.info("It is not Wednesday.")
        logger.info(" :( ")
