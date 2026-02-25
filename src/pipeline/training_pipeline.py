from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_ingestion()
    transformation = DataTransformation()
    train_arr, test_arr = transformation.initiate_transformation(train_path, test_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_trainer(train_arr, test_arr)
