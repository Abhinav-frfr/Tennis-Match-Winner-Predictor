import os 
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException

def save_object(file_path: str, obj: object) -> None:
    """
    Save an object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The object to be saved.

    Raises:
        Exception: If there is an error while saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise Exception(f"Error occurred while saving object: {e}")
    
def evaluate_models(X, y, X_test, y_test, models,params):
    """
    Evaluate multiple machine learning models and return their performance scores.

    Args:
        X (array-like): Training features.
        y (array-like): Training labels.
        X_test (array-like): Testing features.
        y_test (array-like): Testing labels.
        models (dict): A dictionary of model names and their corresponding model instances.
        params (dict): A dictionary of hyperparameters for each model.

    Returns:
        dict: A dictionary containing model names as keys and their performance scores as values.
    """
    try:
        model_report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X,y)

            model.set_params(**gs.best_params_)
            model.fit(X,y)

            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)
            model_report[list(models.keys())[i]] = test_model_score

        return model_report

    except Exception as e:
        raise Exception(f"Error occurred while evaluating models: {e}")

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)      