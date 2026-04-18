import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformer:
    def __init__(self):
        self.data_trans_config = DataTransformationConfig()
    def get_data_transformer_obj(self):
        try:
            numerical_columns=['age', 'sleep_hours', 'social_media_hours', 'reading_score', 'writing_score', 'science_score', 'final_exam_score', 'total_score', 'average']
            categorical_columns=['gender', 'internet_access', 'study_environment', 'pass_fail']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder())
                    
                ]
            )
            logging.info("numerical columns scaling completed")
            logging.info("categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:

            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data completed")
            logging.info("obtaining pre processing object")
            preprocessor_obj = self.get_data_transformer_obj()
            target_column_name = "math_score"
            numerical_columns=['age', 'sleep_hours', 'social_media_hours', 'reading_score', 'writing_score', 'science_score', 'final_exam_score', 'total_score', 'average']
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            logging.info("applying preprocessing object on train and test dataframe")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"saved preprocessing object")
            save_object(
                file_path = self.data_trans_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_trans_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)