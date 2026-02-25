import os
import sys
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import traceback
from src.utils import remove_outlier_iqr


@dataclass
class DataIngestionConfig:
    train_set_path: str = os.path.join("artifacts", "train.csv")
    test_set_path: str = os.path.join("artifacts", "test.csv")
    raw_set_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        """
        This function is used to ingest the dataset and use train test split. Once done the dataset will be saved at the artifacts folder.
        """
        with mlflow.start_run(nested=True):
            try:
                logger.info("fetching the dataset...")

                df = pd.read_csv("https://raw.githubusercontent.com/abhijitpaul0212/GemstonePricePrediction/refs/heads/master/notebooks/data/gemstone.csv")

                logger.info("Removing the id column")
                df.drop("id", axis=1, inplace=True)

                logger.info(f"Shape of the dataframe: {df.shape}")
                logger.info("Saving the raw dataframe into the artifact folder...")
                os.makedirs(
                    os.path.dirname(self.ingestion_config.raw_set_path), exist_ok=True
                )

                df.to_csv(self.ingestion_config.raw_set_path, index=False)

                for col in df.select_dtypes(exclude="O").columns:
                    df = remove_outlier_iqr(df,col)

                logger.info("Dropping Duplicates...")
                df.drop_duplicates(keep="first", inplace=True)

                logger.info("Train Test Split...")
                train_set, test_set = train_test_split(
                    df, test_size=0.30, random_state=42
                )

                logger.info(
                    "Saving the train and test set into the artifacts folder..."
                )
                train_set.to_csv(
                    self.ingestion_config.train_set_path, index=False, header=True
                )
                test_set.to_csv(
                    self.ingestion_config.test_set_path, index=False, header=True
                )

                mlflow.log_artifact(
                    self.ingestion_config.raw_set_path, artifact_path="raw.csv"
                )
                mlflow.log_artifact(
                    self.ingestion_config.train_set_path, artifact_path="train.csv"
                )
                mlflow.log_artifact(
                    self.ingestion_config.test_set_path, artifact_path="test.csv"
                )

                logger.info("Data Ingestion Completed successfully...")

                return (
                    self.ingestion_config.train_set_path,
                    self.ingestion_config.test_set_path,
                )

            except Exception as e:
                mlflow.log_param("Data_Ingestion_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()), "data_ingestion_traceback.txt"
                )
                logger.error(f"Exception occured while trying to ingest the data: {e}")
                raise CustomException(e, sys)
