import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from Student_Performance_Analyser.logger import logger
from Student_Performance_Analyser.exception import CustomException
from Student_Performance_Analyser.utils.common import save_object , evaluate_models
from Student_Performance_Analyser.config.configuration import params



@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Initiating data split")
            X_train, Y_train, X_test, Y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            model_report:dict = evaluate_models(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models, params=params)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                logger.info(f"Best model {best_model_name} has r2 score less than 0.6. Hence, not saving the model")
                raise CustomException("Model score less than 0.6", sys)
            
            logger.info(f"Best model is {best_model_name} with r2 score {best_model_score}")

            save_object(
                obj=best_model,
                filepath=self.model_trainer_config.trained_model_path
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(Y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)