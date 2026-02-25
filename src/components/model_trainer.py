import os
import sys
from src.exception import CustomException
from src.logger import logger
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from dataclasses import dataclass
from src.utils import save_object
import mlflow
import traceback
from src.utils import eval_model


@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.trainer_config = ModelTrainerConfig()

    def initiate_trainer(self, train_arr, test_arr):
        """
        This function is used to train the model in specific metrics and params.
        arg1: train arr made from data transformation
        arg2: test arr made from data transformation
        """
        with mlflow.start_run(nested=True):
            try:
                logger.info("Creating X,Y train and test dataset")

                X_train, X_test, y_train, y_test = (
                    train_arr[:, :-1],
                    test_arr[:, :-1],
                    train_arr[:, -1],
                    test_arr[:, -1],
                )

                models = {
                    "LinearRegression": LinearRegression(fit_intercept=True, n_jobs= None),
                    "Ridge": Ridge(alpha=1, max_iter= 5, solver='saga'),
                    "DecisionTreeRegressor": DecisionTreeRegressor(criterion='poisson', max_depth=10, min_samples_leaf=4, min_samples_split=4, splitter='best'),
                    "RandomForestRegressor": RandomForestRegressor(),
                    "SVR": SVR(),
                }

                report = eval_model(X_train, X_test, y_train, y_test, models)
                best_model_name = max(
                    report, key=lambda model_name: report[model_name]["R2_Score"]
                )
                best_model = models[best_model_name]
                best_model_score = report[best_model_name]["R2_Score"]

                print(
                    f"Best Model found: {best_model} and best model score is: {best_model_score}"
                )
                print(
                    "\n--------------------------------------------------------------------------\n"
                )

                save_object(
                    file_path=self.trainer_config.model_file_path, obj=best_model
                )

                mlflow.log_artifact(
                    self.trainer_config.model_file_path, artifact_path="model.pkl"
                )
                mlflow.log_param("Best Model", best_model)
                mlflow.log_param("Best Model Score", best_model_score)

                logger.info("Model Training Completed Successfully...")

            except Exception as e:
                mlflow.log_param("Model_Trainer_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()), "model_trainer_traceback.txt"
                )
                logger.error(f"Exception occured while trying to train the model: {e}")
                raise CustomException(e, sys)
