import os
import sys
import pickle
import pandas as pd
import numpy as np
from Student_Performance_Analyser.logger import logger
from Student_Performance_Analyser.exception import CustomException


def save_object(obj, filepath:str):
    try:
        logger.info(f"Saving object at path: {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)