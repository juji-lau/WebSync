"""
File for logging and debugging
"""
import logging

logger = logging.getLogger("WebSync Logger")
logger.setLevel(logging.DEBUG)

# console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Make a formatter and set it
formatter = logging.Formatter('MESSAGE DESCR: %(levelname)s, %(filename)s in function: %(funcName)s \n COTENT: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_logger(level:int = 10):
    """ Gets logger with the level specified; default DEBUG"""
    logger.setLevel(level)
    return logger