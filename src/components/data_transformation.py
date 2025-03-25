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

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_columns = ["age", "bmi_smoker_interaction"]
            categorical_columns = ["sex", "region" ]
            non_transformed_columns = ["smoker_binary"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), ##Replace missing values with median
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(
                    steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")), ##Replace missing values with most frequent
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                    ]
                )
            
            logging.info("Categorical columns encoding completed")

    

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    ("none_transformed", "passthrough", non_transformed_columns)  # Leave smoker_binary untouched
                ]

            )

            return preprocessor
            
        except Exception as E:
            raise customException(E, sys)
        
    def iniate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df['smoker_binary'] = train_df['smoker'].map({'yes': 1, 'no': 0})
            train_df['bmi_smoker_interaction'] = train_df['bmi'] * train_df['smoker_binary']

            test_df['smoker_binary'] = test_df['smoker'].map({'yes': 1, 'no': 0})
            test_df['bmi_smoker_interaction'] = test_df['bmi'] * test_df['smoker_binary']

            test_df = test_df.drop(columns=['bmi', 'children', 'smoker',],axis=1)
            train_df = train_df.drop(columns=['bmi', 'children', 'smoker'],axis=1)

            logging.info("Reading train and test completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "charges"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise customException(e,sys)