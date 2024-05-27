import logging

# Update logging configuration to include print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import pickle
import sys
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow

from src.utils import evaluate_models
# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.components.data_transformation import load_object
from src.exception import CustomException

logging.basicConfig(level=logging.INFO)

@dataclass
class ModelTrainerConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    model_save_path: str = os.path.join('artifacts', 'model.pkl')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def load_data(self):
        try:
            train_df = pd.read_csv(self.model_trainer_config.train_data_path)
            test_df = pd.read_csv(self.model_trainer_config.test_data_path)
            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, train_df):
        try:
            X_train = train_df.drop(columns=['target'], axis=1)
            y_train = train_df['target']

            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
            }

            for model_name, model in models.items():
                model.fit(X_train, y_train)

            return models
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, models, test_df):
        try:
            X_test = test_df.drop(columns=['target'], axis=1)
            y_test = test_df['target']

            model_report = {}
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred)

                model_report[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc
                }

            return model_report
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self, models):
        try:
            with open(self.model_trainer_config.model_save_path, 'wb') as file:
                pickle.dump(models, file)
            logging.info(f"Model saved at: {self.model_trainer_config.model_save_path}")
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        model_trainer = ModelTrainer()
        
        train_df, test_df = model_trainer.load_data()
        logging.info("Data loaded successfully.")
        
        models = model_trainer.train_model(train_df)
        logging.info("Models trained successfully.")
        
        model_report = model_trainer.evaluate_model(models, test_df)
        for model_name, metrics in model_report.items():
            logging.info(f"Model: {model_name}, Metrics: {metrics}")
        
        model_trainer.save_model(models)
        logging.info("Models saved successfully.")
        
    except CustomException as ce:
        logging.error(f"Custom Exception occurred: {ce}")
