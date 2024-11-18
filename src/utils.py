import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,classifiers):
    try:
        results = []
        for name,clf in classifiers.items():
            clf.fit(X_train,y_train)
            y_test_pred = clf.predict(X_test)
            val_acc = accuracy_score(y_test,y_test_pred)
            val_f1 = f1_score(y_test,y_test_pred,average='weighted')
            results.append(
                {
                    'classifier':name,
                    'test accuracy': val_acc,
                    'test f1 score': val_f1
                }
            )
        return results
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)