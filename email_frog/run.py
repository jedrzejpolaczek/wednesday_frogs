from loguru import logger

from email_frog.scheduler import send_email_on_wednesday

from utils import get_json_data


def run_email_sender():
    """ Run sending emails with generated frog. """
    logger.info("Loading model.")
    email_config_data = get_json_data("email_config.json")

    logger.info("Waiting for Wednesday...")
    send_email_on_wednesday(email_config_data["image_name"])
