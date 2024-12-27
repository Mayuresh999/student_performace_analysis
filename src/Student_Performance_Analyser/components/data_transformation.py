import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from Student_Performance_Analyser.logger import logger
from Student_Performance_Analyser.exception import CustomException
from Student_Performance_Analyser.utils.common import save_object
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            logger.info("Initiating data transformation")
            numeric_features = ["writing_score", "reading_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logger.info("Data transformation for numerical features successful")
            categocical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logger.info("Data transformation for categorical features successful")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_transformer', numeric_transformer, numeric_features),
                    ('cat_transformer', categocical_transformer, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path:str, test_data_path:str):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logger.info("Data read completed")
            
            preprocessor = self.get_data_transformer_obj()
            target_column = "math_score"
            numeric_features = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_train_df = train_df[target_column]
           
            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_test_df = test_df[target_column]
            logger.info("Initiating data transformation")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]
            logger.info("Data transformation completed")

            save_object(preprocessor, self.transformation_config.preprocessor_obj_file_path)

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)