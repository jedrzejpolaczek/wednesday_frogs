import os
import datetime
import discord
from discord.ext import tasks
from loguru import logger

from utils import (get_json_data, is_it_wednesday, load_model, get_seed, generate_images, save_image)


def run_bot():
    """ TODO: add docstrings """
    discord_config_data = get_json_data("discord_config.json")

    discord_token = discord_config_data["discord_token"]
    guild_name = discord_config_data["guild_name"]  # DŹGAŃSKO
    channel_name = discord_config_data["channel_name"]  # sroda
    image_name = discord_config_data["image_name"]

    client = discord.Client()


    @client.event
    async def on_ready():
        guild = discord.utils.get(client.guilds, name=guild_name)
        channel = discord.utils.get(guild.text_channels, name=channel_name)

        logger.info(
            f'{client.user} is connected to the following guild: '
            f'{guild.name} (id: {guild.id})'
        )

        wednesday_check.start(channel, image_name)


    @tasks.loop(hours=2)
    async def wednesday_check(channel, image_name):
        if is_it_wednesday():
            logger.debug("Loading model.")
            model = load_model()

            logger.debug("Generate seed.")
            seed = get_seed()

            logger.debug("Generate image.")
            images = generate_images(model, seed)

            logger.debug("Save image")
            save_image(save_dir="", name=image_name, images=images)

            logger.debug("Post frog")
            await channel.send("It's Wednesday my dudes!")
            await channel.send(file=discord.File(image_name))
        
        else:
            await channel.send("It's not Wednesday yet :(")

    client.run(discord_token)

    return client
