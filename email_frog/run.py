from loguru import logger

from email_frog.scheduler import send_email_on_wednesday

from utils import (load_model, generate_images, save_image)

def run_email_sender():
    """ Run sending emails with generated frog. """
    logger.info("Loading model.")
    generator = load_model()

    logger.info("Starting to wait for Wednesday...")
    send_email_on_wednesday(generator)
