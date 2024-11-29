import os
import sys
print(sys.path)
import pandas as pd
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifacts", "train.csv")
    test_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")  # Corrected typo

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initializationData(self):
        logger.info("Data Ingestion Initialization")
        try:
            # Read data from CSV
            data = pd.read_csv("notebook/data/stud.csv")
            logger.info("Reading data from file")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)

            # Save raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info("Data saved to raw_data_path")
            
            # Split data into train and test
            logger.info("Train Test Split Starts")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)  # Fixed here
            logger.info("Train data saved to train_path")

            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)  # Fixed here
            logger.info("Test data saved to test_path")

            logger.info("Data ingestion is completed successfully")
            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )
        except Exception as e:
            logger.error("Data ingestion failed")
            raise CustomException("Data ingestion failed", e)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initializationData()
