import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # SimpleImputer is used to replace missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import save_object
from src.exception import CustomException
from src.logger import logger

@dataclass
class DataTransformationConfig:
    preprocess_ob_file_path = os.path.join("artifacts", "preprocess_ob.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is used to get the data transformation object
        '''
        try:
            numeric_features = ['reading_score','writing_score']
            categorical_features= [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipeline = Pipeline(
                steps =[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('std_scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps =[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    ('std_scaler',StandardScaler(with_mean=False))
                ]
            )

            logger.info(f"Categorical Features: {categorical_features}")
            logger.info(f"Numeric Features: {numeric_features}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numeric_features),
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )
            return preprocessor

        except Exception as e:
            logger.error("Data Transformation failed")
            raise CustomException("Data Transformation failed", e, sys)
    
    def initialize_data_transformation(self, train_path, test_path):
        '''
        This function is used to initialize the data transformation
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            numeric_column = ['reading_score','writing_score']

            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logger.info("Applying preprocessor on train data and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # np.c_ is used to concatenate the arrays column-wise
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("Saving preprocessing object successfully")

            save_object(
                file_path=self.data_transformation_config.preprocess_ob_file_path,
                obj=preprocessing_obj    
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_ob_file_path,
            )

        except Exception as e:
            logger.error("Data Transformation Initialization failed")
            raise CustomException(f"Data Transformation Initialization failed: {str(e)}")

