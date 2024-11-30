import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from dataclasses import dataclass
import warnings

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object,evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logger.info("Model Training Initialization-Split train and test data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],train_array[:,-1],
                test_array[:,:-1],test_array[:,-1]
            )

            models ={
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "SVR": SVR()
            }

            params={
                "Decision Tree": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "Ridge": {
                    #  "alpha": [0.01, 0.1, 1, 10]
                          },
                "Lasso": {
                    #  "alpha": [0.01, 0.1, 1, 10]
                          },
                "KNN": {
                    #  "n_neighbors": np.arange(1, 10)
                     },
                "XGBoost":{
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost":{
                    # 'learning_rate':[.1,.01,0.5,.001],
                    # # 'loss':['linear','square','exponential'],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "SVR": {
                    # 'C': [0.1, 1, 10, 100, 1000],
                    # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    # 'kernel': ['rbf', 'poly', 'sigmoid']
                }
                
            }
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] #nested list comprehension

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                        raise CustomException("No best model found")


            save_object(
                file_path =self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)
            return r2

        except Exception as e:
            logger.error("Model Training Initialization failed")
            raise CustomException("Model Training Initialization failed", sys)