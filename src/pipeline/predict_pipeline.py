import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class predictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


# 🔥 CUSTOM DATA CLASS
class CustomData:
    def __init__(self,
                 gender: str,
                 internet_access: str,
                 study_environment: str,
                 pass_fail: str,
                 age: int,
                 sleep_hours: float,
                 social_media_hours: float,
                 reading_score: float,
                 writing_score: float,
                 science_score: float,
                 final_exam_score: float):

        self.gender = gender
        self.internet_access = internet_access
        self.study_environment = study_environment
        self.pass_fail = pass_fail
        self.age = age
        self.sleep_hours = sleep_hours
        self.social_media_hours = social_media_hours
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.science_score = science_score
        self.final_exam_score = final_exam_score

    def get_data_as_data_frame(self):
        try:
            # 🔥 COMPUTE DERIVED FEATURES
            total_score = (
                self.reading_score +
                self.writing_score +
                self.science_score +
                self.final_exam_score
            )

            average = total_score / 4

            # 🔥 CREATE DATAFRAME (ORDER MUST MATCH TRAINING)
            custom_data_input_dict = {
                "age": [self.age],
                "sleep_hours": [self.sleep_hours],
                "social_media_hours": [self.social_media_hours],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
                "science_score": [self.science_score],
                "final_exam_score": [self.final_exam_score],
                "total_score": [total_score],      # ✅ ADDED
                "average": [average],              # ✅ ADDED
                "gender": [self.gender],
                "internet_access": [self.internet_access],
                "study_environment": [self.study_environment],
                "pass_fail": [self.pass_fail]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)