import pandas as pd
import os
import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset():
    try:
        if os.path.exists(config.DATASET_PATH):
            df = pd.read_csv(config.DATASET_PATH)
            logger.info("Dataset loaded successfully")
            return df
        else:
            logger.error("Dataset file not found")
            raise FileNotFoundError("Dataset file not found")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
