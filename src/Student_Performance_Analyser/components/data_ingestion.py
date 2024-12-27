import os
import sys
from Student_Performance_Analyser.logger import logger
from Student_Performance_Analyser.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from Student_Performance_Analyser.components.data_transformation import DataTransformation, DataTransformationConfig
from Student_Performance_Analyser.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logger.info("Initiating data ingestion")
            df = pd.read_csv(r"C:/mayuresh/end_to_end_practice_project/student_performace_analysis/assets/stud.csv")
            logger.info("Data ingestion successful")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info("Raw data saved successfully")
            logger.info(f"Train Test Split Initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info(f"Train Test Split Completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
    print(r2_score)