import sys
from loguru import logger

from model.run import run_model_training
from email_frog.run import run_email_sender
from discord.run import run_bot

# TODO: add args
# TODO: make it configurable from args
logger.remove()
logger.add(sys.stderr, level="INFO")

# TODO: add logic here based on args
# run_model_training()
# run_email_sender()
# run_bot()


