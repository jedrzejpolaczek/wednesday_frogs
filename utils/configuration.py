from loguru import logger
import json


def get_json_data(file_path: str) -> dict:
    """ 
    Read JSON dict from file.

    file_path (str): path to JSON file.
    
    return dict: dict based on read JSON file.
    """
    logger.info("Opening JSON file.")
    json_file = open(file_path)
    
    logger.info("Returns JSON object as a dictionary.")
    data = json.load(json_file)

    return data
