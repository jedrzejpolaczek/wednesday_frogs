import sys
from loguru import logger

# from email.sender import send_email_on_wednesday

logger.remove()
logger.add(sys.stderr, level="INFO")

# Fixme: while(1):
# send_email_on_wednesday()



