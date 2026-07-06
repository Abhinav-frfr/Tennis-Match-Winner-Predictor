import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for performing data transformation on the dataset.
        It defines the numerical and categorical columns, creates pipelines for preprocessing, and saves the 
        preprocessor object as a pickle file for future use.
        
        '''

        logging.info("Entered the data transformation method or component")
        
        try:
            numerical_columns=['Best of', 'Player1_H2H_WinPct', 'RecentWinPct_1', 'RecentWinPct_2','Surface_WinPct_1', 'Surface_WinPct_2', 'Higher_Rank_Player','Abs_Rank_Diff']
            categorical_columns=['Court', 'Surface', 'Round', 'Season']


            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()


            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", numeric_transformer)
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", oh_transformer),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns Encoded: {categorical_columns}")
            logging.info(f"Numerical columns Scaled: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessing object created and returned.")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Winner_Encoded"
            numerical_columns=['Best of', 'Player1_H2H_WinPct', 'RecentWinPct_1', 'RecentWinPct_2','Surface_WinPct_1', 'Surface_WinPct_2', 'Higher_Rank_Player','Abs_Rank_Diff']

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)
            

if __name__ == "__main__":
    obj = DataTransformation()
    obj.get_data_transformer_object()