import os
import sys
import mlflow
import traceback
from src.utils import load_object
from src.exception import CustomException
from src.logger import logger
import pandas as pd


class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        """
        This function is used to make prediction.
        """
        with mlflow.start_run(nested=True):
            try:
                logger.info("Attempting to make prediction...")
                preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")
                model_file_path = os.path.join("artifacts", "model.pkl")

                preprocessor = load_object(preprocessor_file_path)
                model = load_object(model_file_path)

                data_scaled = preprocessor.transform(features)
                pred = model.predict(data_scaled)

                return pred
            except Exception as e:
                mlflow.log_param("Prediction_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()), "prediction_traceback.txt"
                )
                logger.error(f"Exception occured while trying to make prediction: {e}")
                raise CustomException(e, sys)

class Custom_Data:
    def __init__(self,
        depth: float,
        table: float,
        volume: float,
        log_carat: float,
        cut: object,
        color: object,
        clarity: object,
    ):
        self.depth = depth
        self.table = table
        self.volume = volume
        self.log_carat = log_carat
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def gather_data_as_dataframe(self):
        """
        This function is used to create the dataframe with the custom data.
        """
        with mlflow.start_run(nested=True):
            try:
                logger.info("Attempting to create custom DataFrame...")

                custom_data_dict = {
                    "depth": [self.depth],
                    "table": [self.table],
                    "volume": [self.volume],
                    "log_carat": [self.log_carat],
                    "cut": [self.cut],
                    "color": [self.color],
                    "clarity": [self.clarity]
                }


                df = pd.DataFrame(custom_data_dict)

                logger.info("Custom Data Successfully Gathered... ")

                return df
            except Exception as e:
                mlflow.log_param("DataFrame_Creation_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()), "dataframe_creation_traceback.txt"
                )
                logger.error(
                    f"Exception occured while trying to create custom Dataframe: {e}"
                )
                raise CustomException(e, sys)
