import os 
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings

from src.exceptions import CustomException
from src.logger import logging
from src.utils import evaluate_models

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression Classifier": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                ##"Decision Tree Classifier": DecisionTreeClassifier(),
                ##"Random Forest Classifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(), 
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            params = {

            "Logistic Regression Classifier": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"],
                "penalty": ["l2"]
            },

            "K-Neighbors Classifier": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            },

            # "Decision Tree Classifier": {
            #     "criterion": ["gini", "entropy", "log_loss"],
            #     "max_depth": [None, 5, 10, 20],
            #     "min_samples_split": [2, 5, 10],
            #     "min_samples_leaf": [1, 2, 5]
            # },

            # "Random Forest Classifier": {
            #     "n_estimators": [100, 200, 300],
            #     "max_depth": [None, 10, 20],
            #     "min_samples_split": [2, 5, 10],
            #     "min_samples_leaf": [1, 2, 5],
            #     "max_features": ["sqrt", "log2"]
            # },

            "XGBClassifier": {
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            },

            "CatBoosting Classifier": {
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [100, 300, 500]
            },

            "AdaBoost Classifier": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
            }
        }

            model_report: dict = evaluate_models(X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset: {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            accuracy=accuracy_score(y_test,predicted)
            return accuracy 

        except Exception as e:
            raise CustomException(e, sys)