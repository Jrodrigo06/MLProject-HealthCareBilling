import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer ##Handles missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler ##OneHot for Categorical Variables
## And Standard Scaler uses Z-Score Normilization for numerical values

from src.exception import customException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocesser.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_columns = ["age", "bmi", "children"]
            categorical_columns = ["sex", "region" ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), ##Replace missing values with median
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(
                    steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")) ##Replace missing values with most frequent
                    ("one_hot_encoder", OneHotEncoder())
                    ("scaler", StandardScaler())
                    ]
                )
            
            logging.info("Categorical columns encoding completed")

    

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                    ("cat_pipeline", cat_pipeline, categorical_columns)

                ]

            )

            return preprocessor
            
        except Exception as E:
            raise customException(E, sys)
        
    def iniate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "charges"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            targest_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

        except:
            pass