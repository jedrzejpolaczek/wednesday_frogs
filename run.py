import sys
from loguru import logger

# from email.sender import send_email_on_wednesday

logger.remove()
logger.add(sys.stderr, level="INFO")

# Fixme: while(1):
# send_email_on_wednesday()


from model.run import run_model_training
from email_frog.run import run_email_sender

# run_model_training()
run_email_sender()
