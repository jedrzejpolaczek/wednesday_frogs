from loguru import logger
import json


def get_json_data() -> dict:
    """ 
    Read JSON dict from file.
    
    return dict: dict based on read JSON file.
    """
    logger.info("Opening JSON file.")
    json_file = open('config.json')
    
    logger.info("Returns JSON object as a dictionary.")
    data = json.load(json_file)

    return data
