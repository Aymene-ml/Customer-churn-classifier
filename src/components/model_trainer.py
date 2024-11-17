import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig.trained_model_file_path
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Split training and test input data')
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            classifiers = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Support Vector Classifier': SVC(probability=True, random_state=42),
                'LightGBM': LGBMClassifier(),
            }
            
            results:list=evaluate_models(X_train,y_train,X_test,y_test,classifiers)
            best_result = max(results,key=lambda x:x['test f1 score'])
            best_model_scores = (
                f"{round(best_result['test f1 score'] * 100, 2)}%",
                f"{round(best_result['test accuracy'] * 100, 2)}%"
            )
            best_model_name = best_result['classifier']
            best_model = classifiers[best_model_name]
            
            logging.info('Best found model on test dataset')
            
            save_object(self.model_trainer_config,best_model)
            return best_model_scores
        except Exception as e:
            raise CustomException(e,sys)
        
