import sys
import pandas as pd
from src.exception import customException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise customException(e, sys)

class CustomData:
    def __init__(self, age : int, 
                 sex : str,
                 bmi : float, 
                 smoker_binary : int, 
                 region : str):
        self.age = age
        self.sex = sex
        self.bmi = float(bmi)
        self.smoker_binary = smoker_binary
        self.bmi_smoker_interaction = smoker_binary * self.bmi
        self.region = region

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age" : [self.age],
                "sex" : [self.sex],
                "bmi_smoker_interaction" : [self.bmi_smoker_interaction],
                "smoker_binary" : [self.smoker_binary],
                "region" : [self.region]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise customException(e,sys)
