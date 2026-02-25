import os
import sys
import mlflow
import traceback
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logger
import numpy as np


def save_object(file_path: str, obj):
    """
    This function is used to save the pickle file to the location provided.
    arg1: file_path is str
    arg2: pickle file
    """
    with mlflow.start_run(nested=True):
        try:
            logger.info("Attempting to save the pickle file....")

            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file=file_path, mode="wb") as file_obj:
                pickle.dump(obj, file_obj)

            logger.info("Pickle File Saved Successfully... ")

        except Exception as e:
            mlflow.log_param("Pickle_Save_Exception", str(e))
            mlflow.log_text(
                "".join(traceback.format_exc()), "pickle_save_traceback.txt"
            )
            logger.error(f"Exception occured while trying to save the object file: {e}")
            raise CustomException(e, sys)


def load_object(file_path: str):
    """
    This function is used to load the object file from the saved location.
    arg1: file path in str
    """
    with mlflow.start_run(nested=True):
        try:
            logger.info("Attempting to load the pickle file...")

            with open(file=file_path, mode="rb") as file_obj:
                load_obj = pickle.load(file_obj)

                logger.info("Pickle File Successfully Loaded...")

                return load_obj
        except Exception as e:
            mlflow.log_param("Pickle_Load_Exception", str(e))
            mlflow.log_text(
                "".join(traceback.format_exc()), "pickle_load_traceback.txt"
            )
            logger.error(f"Exception occured while trying to load the pickle file: {e}")
            raise CustomException(e, sys)


def eval_model(X_train, X_test, y_train, y_test, models):
    """
    This function is used to evaluate the model on specific given metrics.
    """
    with mlflow.start_run(nested=True):
        try:
            report = {}

            for model_name, model in models.items():
                logger.info(f"{model_name}_evaluation started...")
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r_square = r2_score(y_test, y_pred)

                report[model_name] = {
                    "Mean Squared Error": mse,
                    "Root Mean Squared Error": rmse,
                    "Mean Absolute Error": mae,
                    "R2_Score": r_square,
                }

            logger.info(f"{model_name}_evaluation_completed.")

            return report
        except Exception as e:
            mlflow.log_param("Model_Eval_Exception", str(e))
            mlflow.log_text("".join(traceback.format_exc()), "model_eval_traceback.txt")
            logger.error(f"Exception occured while trying to evaluate the model: {e}")
            raise CustomException(e, sys)


def remove_outlier_iqr(data, column):
    """
    This function is used to remove the outlier for the dataframe and the column provided.
    arg1: DataFrame that need to be used.
    arg2: Column from that dataset from which you need to remove the outlier.
    """
    with mlflow.start_run(nested=True):
        try:
            logger.info(f"Attempting to remove the outlier form column: {column}")

            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)

            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

        except Exception as e:
            mlflow.log_param("Outlier_Removal_Exception", str(e))
            mlflow.log_text(
                "".join(traceback.format_exc()), "outlier_removal_traceback.txt"
            )
            logger.error(f"Exception occured while trying to remove the outlier: {e}")
            raise CustomException(e, sys)
