import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
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
    
def evaluate_models(X_train, Y_train, X_test, Y_test, models:dict):
    try:
        logger.info("Initiating model evaluation")
        model_report = {}
        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")
            model.fit(X_train, Y_train)
            logger.info(f"Model training completed")
            Y_pred = model.predict(X_test)
            r2 = r2_score(Y_test, Y_pred)
            model_report[model_name] = r2
            logger.info(f"Model evaluation completed for model: {model_name}")
        return model_report
    except Exception as e:
        raise CustomException(e, sys)