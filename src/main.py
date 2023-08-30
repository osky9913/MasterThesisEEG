import logging

from config import config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Config loaded")
    logging.info("DASP: %s", config["DASP"])
    logging.info("DASP_range: %s", config["DASP_range"])
