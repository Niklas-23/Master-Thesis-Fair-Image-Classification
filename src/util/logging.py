import logging
import sys
import os


def create_logger(name: str = "Training", level=logging.INFO, filename=None):
    if name in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not os.path.exists(f"logs/{name}"):
        os.makedirs(f"logs/{name}")

    if filename is None:
        file_handler = logging.FileHandler(f'logs/{name}/benchmark.log', mode="w")
    else:
        file_handler = logging.FileHandler(f'logs/{name}/{filename}.log', mode="w")
    file_handler.setLevel(level)

    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)

    logger.addHandler(file_handler)

    return logger
