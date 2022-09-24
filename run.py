import sys
from loguru import logger

from model.dcgan.run import run_model_training
from email_frog.run import run_email_sender
from discord.run import run_bot
from utils import get_args


args = get_args()

logger.remove()
logger.add(sys.stderr, level=args.log_level)

if args.run == "train":
    run_model_training()
elif args.run == "email":
    run_email_sender()
elif args.run == "discord":
    run_bot()
else:
    logger.error("Invalid run type. You can only choose type_of_run bewteen train, email or discord!")
