import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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
    
def evaluate_models(X_train, Y_train, X_test, Y_test, models:dict, params:dict):
    try:
        logger.info("Initiating model evaluation")
        model_report = {}
        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")
            param = params.get(model_name)
            if not param:
                logger.info(f"No hyperparameters to tune for {model_name}")
                model.fit(X_train, Y_train)
            else:
                logger.info(f"Performing GridSearchCV for {model_name}")
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param,
                    cv=3,
                    n_jobs=3,
                    verbose=1
                )
                gs.fit(X_train, Y_train)
                
                # Set the best parameters found
                model.set_params(**gs.best_params_)
                model.fit(X_train, Y_train)
                logger.info(f"Best parameters for {model_name}: {gs.best_params_}")
            logger.info(f"Model training completed")
            Y_pred = model.predict(X_test)
            r2 = r2_score(Y_test, Y_pred)
            model_report[model_name] = r2
            logger.info(f"Model evaluation completed for model: {model_name}")
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(filepath:str):
    try:
        logger.info(f"Loading object from path: {filepath}")
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info("Object loaded successfully")
        return obj
    except Exception as e:
        raise CustomException(e, sys)