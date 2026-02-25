import os
import sys
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
import mlflow
import traceback
from src.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.transformation_config = DataTransformationConfig()

    def gather_transformation_obj(self):
        """
        This function is used to gather the transformation object through pipeline
        """
        with mlflow.start_run(nested=True):
            try:
                logger.info("Attempting to Transform the data through pipeline...")
                num_column = ["depth", "table", "volume", "log_carat"]
                cat_column = ["cut", "color", "clarity"]

                cut_cat = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
                color_cat = ["D", "E", "F", "G", "H", "I", "J"]
                clarity_cat = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

                num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

                cat_pipeline = Pipeline(
                    steps=[
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=[cut_cat, color_cat, clarity_cat]
                            ),
                        )
                    ]
                )

                preprocessor = ColumnTransformer(
                    [
                        ("num_pipeline", num_pipeline, num_column),
                        ("cat_pipeline", cat_pipeline, cat_column),
                    ]
                )

                logger.info("Transformation Pipeline Successfully Created...")

                return preprocessor
            except Exception as e:
                mlflow.log_param("Transformation_Pipeline_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()),
                    "transformation_pipeline_traceback.txt",
                )
                logger.error(
                    f"Exception occured while trying to use transformation pipeline for transformation: {e}"
                )
                raise CustomException(e, sys)

    def initiate_transformation(self, train_path, test_path):
        """
        This function is used to initiate the data transformation through pipeline created.
        arg1: train dataset path in str
        arg2: test dataset path in str
        """
        with mlflow.start_run(nested=True):
            try:
                logger.info("Data Transformation Started...")
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                train_df["log_price"] = np.log1p(train_df["price"])
                test_df["log_price"] = np.log1p(test_df["price"])

                train_df["log_carat"] = np.log1p(train_df["carat"])
                test_df["log_carat"] = np.log1p(test_df["carat"])

                train_df["volume"] = train_df["x"] * train_df["y"] * train_df["z"]
                test_df["volume"] = test_df["x"] * test_df["y"] * test_df["z"]

                target = ["log_price"]
                drop_column = ["log_price", "price", "carat", "x", "y", "z"]

                input_feature_train_df = train_df.drop(drop_column, axis=1)
                input_target_train_df = train_df[target]

                input_feature_test_df = test_df.drop(drop_column, axis=1)
                input_target_test_df = test_df[target]

                logger.info("Obtaining Preprocessor Object...")
                preprocessor_obj = self.gather_transformation_obj()

                input_feature_train_arr = preprocessor_obj.fit_transform(
                    input_feature_train_df
                )
                input_feature_test_arr = preprocessor_obj.transform(
                    input_feature_test_df
                )

                train_arr = np.c_[
                    input_feature_train_arr, np.array(input_target_train_df)
                ]
                test_arr = np.c_[input_feature_test_arr, np.array(input_target_test_df)]

                save_object(
                    file_path=self.transformation_config.preprocessor_file_path,
                    obj=preprocessor_obj,
                )

                mlflow.log_artifact(
                    self.transformation_config.preprocessor_file_path,
                    "preprocessor.pkl",
                )

                logger.info("Data Transformation Completed Successfully....")

                return (train_arr, test_arr)

            except Exception as e:
                mlflow.log_param("Data_Transformation_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()), "data_transformation_traceback.txt"
                )
                logger.error(f"Exception occured while trying to transform data: {e}")
                raise CustomException(e, sys)
