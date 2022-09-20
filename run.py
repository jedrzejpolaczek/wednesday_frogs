import sys
from loguru import logger

from model.run import run_model_training
from email_frog.run import run_email_sender
from discord.run import run_bot
from utils import get_args


args = get_args()

logger.remove()
logger.add(sys.stderr, level=args.logger_level)

if args.type_of_run == "train":
    run_model_training()
elif args.type_of_run == "email":
    run_email_sender()
elif args.type_of_run == "discord":
    run_bot()
else:
    logger.error("Invalid run type. You can only choose type_of_run bewteen train, email or discord!")
