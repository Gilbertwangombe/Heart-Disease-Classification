import os
import pickle
import logging
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging as custom_logger
from src.utils import save_object

custom_logger.basicConfig(level=logging.INFO)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
        custom_logger.info(f"Created directory: {os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path)}")

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            categorical_columns = [
                'sex',
                'cp',
                'fbs',
                'restecg',
                'exang',
                'slope',
                'ca',
                'thal'
            ]
            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            custom_logger.info(f"Categorical Columns: {categorical_columns}")
            custom_logger.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            custom_logger.info("Reading the train and test file")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'target'
            numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            custom_logger.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            custom_logger.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        custom_logger.error(f"Error loading object: {e}")

# Example usage
if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        custom_logger.info(f"Train data path: {train_data}")
        custom_logger.info(f"Test data path: {test_data}")

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data, test_data)

        custom_logger.info(f"Train array shape: {train_arr.shape}")
        custom_logger.info(f"Test array shape: {test_arr.shape}")
        custom_logger.info(f"Preprocessor object file path: {preprocessor_obj_file_path}")

    except Exception as e:
        custom_logger.error(f"Error in the main process: {e}")
