import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from .data_ingestion import DataIngestion
from .model_trainer import ModelTrainer
@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            cat_features = ['SeniorCitizen', 'Partner', 'Dependents', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod'] 

            numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder())
                ]
            )
            logging.info('Numerical columns standard scaling completed')
            logging.info('categorical columns coding completed')
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numeric_features),
                    ('cat_pipeline',cat_pipeline,cat_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')
            
            train_df.TotalCharges = pd.to_numeric(train_df.TotalCharges, errors='coerce')
            test_df.TotalCharges = pd.to_numeric(test_df.TotalCharges, errors='coerce')

            preprocessing_obj = self.get_data_transformation_object()
            input_features_train = train_df.drop(columns=['Churn','PhoneService', 'gender','MultipleLines','customerID'])
            target_train = train_df['Churn']
            
            input_features_test = test_df.drop(columns=['Churn','PhoneService', 'gender','MultipleLines','customerID'])
            target_test = test_df['Churn']
            label_encoder = LabelEncoder()
            target_train = label_encoder.fit_transform(target_train)
            target_test = label_encoder.transform(target_test)
            logging.info(
                'Applying preprocessing on training and testing dataframes'
            )
            
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessing_obj.fit_transform(input_features_test)
            
            train_arr = np.c_[input_features_train_arr,np.array(target_train)]
            test_arr = np.c_[input_features_test_arr,np.array(target_test)]
            
            logging.info('Saved preprocessing object')
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
